#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime_m = 0;
__device__ __managed__ u32 gtime_c = 0;
__device__ __managed__ int gpwd[4] = {0, -1, -1, -1};   // index list

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->VOLUME_SIZE = VOLUME_SIZE;
  fs->BLOCK_SIZE = BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init SUPERBLOCK_SIZE
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++)
  {
    fs->volume[i] = 0;
  }
  // init FCBs
  for (int i = fs->SUPERBLOCK_SIZE; i < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES * fs->FCB_SIZE; i++)
  {
    fs->volume[i] = 0;
  }
  // init the root dir
  fs_open(fs, "root\0", G_PWD);
  // fs_gsys(fs, CD, "\0");
}

__device__ int str_len(char str[])
{
	char *p = str;
	int count = 0;
	while (*p++ != '\0')
	{
		count++;
	}
	return count+1;
}

__device__ bool num_in_list(int num, int* list) 
{
	for (int i=0; i<50; i++) {
		if (num == list[i]) {
      return true;
    }
	}
  return false;
}

__device__ void printf_list(int* list) 
{
	for (int i=0; i<50; i++) {
		printf("%d  ", list[i]);
	}
	printf("\n----------------------\n");
	return;
}

// init FCB
//   FCB:
//   0-19   name
//	 20     LSB valid bit
//   21     LSB directory
//	 22-23  start address
//	 24-27  size
//	 28-29  created time
//   30-31  modified time

__device__ u32 fs_open(FileSystem *fs, char *file_name, int op)
{
  int target_fcb_index = -1;
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    if ( (fcb_get_validbit(fs, i)) && (cmp_str(fcb_get_name(fs, i), file_name)) 
    && (pwd_file_is_under_curr_dir(fs, file_name) || fcb_check_dir(fs, i)) )
    {
      target_fcb_index = i;
      break;
    }
  }
  
  if (target_fcb_index == -1)
  { // not find the target
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      if (!fcb_get_validbit(fs, i))
      { // find the fcb that is invalid
        target_fcb_index = i;
        break;
      }
    }
    if (target_fcb_index == -1)
    {
      printf("The FCB blocks are already full!\n");
      return 0;
    }
    fcb_clear(fs, target_fcb_index);
    fcb_set_name(fs, target_fcb_index, file_name);
    fcb_set_validbit(fs, target_fcb_index);
    int start = 0xffff;
    fcb_set_start(fs, target_fcb_index, start);
    fcb_set_size(fs, target_fcb_index, 0);
    fcb_set_createtime(fs, target_fcb_index, gtime_c);
    gtime_c++;
  /////////////////////////////////////////
  // newly append

    if (op == G_PWD) {
      fcb_set_dir(fs, target_fcb_index);
      fcb_set_modifytime(fs, target_fcb_index,  gtime_m);
      gtime_m ++;
      // fcb_set_size(fs, target_fcb_index, 1024);
      int start_index_in_fileblocks = allocate(fs, fs->BLOCK_SIZE);
      fcb_set_start(fs, target_fcb_index, start_index_in_fileblocks);
      for (int i=start_index_in_fileblocks; i<start_index_in_fileblocks+fs->BLOCK_SIZE; i++) {
        vcb_set(fs, i);
      }
    }
    if (op == G_READ || op == G_WRITE || op == G_PWD) { // a new file is added, update the the directory file
      // update content in fcb of the directory file
      int current_pwd_index = pwd_get();
      // int size = ( fcb_get_size(fs, current_pwd_index)+str_len(file_name)+1);
      // fcb_set_size(fs, current_pwd_index, size);
      fcb_set_modifytime(fs, current_pwd_index, gtime_m);
      gtime_m ++;
      // update content in the directory file
      // u32 handler = fs_open(fs, fcb_get_name(fs, current_pwd_index), G_PWD);
      u32 handler = 0;
      handler |= (1<<31);
      handler |= (1<<30);
      handler |= ((u16)current_pwd_index << 16);
      int start = fcb_get_start(fs, current_pwd_index);
      handler |= (u16)start;
      // printf("here!\n");
      fs_write(fs, (uchar*)file_name, 1, handler);
    }
  /////////////////////////////////////////
  }
  // after all of these, we have already get or allocate the FCB
  // The handler consists of three part, read/write bit + FCB index + start block
  u32 handler = 0;
  if (op == G_READ)
  {
    handler |= (1 << 31);
  }
  else if (op == G_WRITE)
  {
    handler |= (1 << 30);
  } 
  else if (op == G_PWD)
  {
    handler |= (1 << 31);
    handler |= (1 << 30);
  }
  handler |= (target_fcb_index << 16);                 // FCB index
  handler |= (u16)fcb_get_start(fs, target_fcb_index); // start block index
  return handler;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  bool out_w = fp & (1 << 30); // check the valid bit
  bool out_r = fp & (1 << 31);
  int start_index = fp & 0xffff;          // retrieve the start file block
  int FCB_index = (fp & 0xfff0000) >> 16; // retrieve the FCB block index
  if (out_r && !out_w) // read for normal file, read into the output
  {

    int file_size = fcb_get_size(fs, FCB_index);

    if (start_index == 0xffff || file_size == 0)
    {
      printf("The file block hasn't been allocate, cannot read, bro!\n");
      return;
    }
    int read_size;
    if (size > file_size)
    {
      read_size = file_size;
    }
    else
    {
      read_size = size;
    }
    // start to read
    int start_addr = fs->FILE_BASE_ADDRESS + start_index * fs->BLOCK_SIZE;
    // printf("start_index = %d\n", start_index);
    for (int i = 0; i < size; i++)
    {
      output[i] = fs->volume[start_addr + i];
      printf("");

    }
    return;
  }
  /////////////////////////////////////////
  // newly append
  else if (out_r && out_w) { // read a directory file
    int* out_ptr = (int*) output;
    int new_addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * start_index;
    
    for (int i = 0; i < 50; i++)
    {
      // read the index of the file fcb from the directory file blocks
      u16 *index_ptr = (u16 *)&fs->volume[new_addr + 2*i];
      if (*index_ptr != 0) {                                
        *out_ptr = (int)*index_ptr;
      }    
      else {
        *out_ptr = 0;
      }
      out_ptr ++;
    }
    return;
  }
  /////////////////////////////////////////
  else
  {
    printf("Invalid operation, this is a read operation, bro!\n");
    return;
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  bool out_w = fp & (1 << 30); // check the valid bit
  bool out_r = fp & (1 << 31);
  
  int tag = size; // tag = 1 means write in, 0 means remove out from the directory file
  if (out_r && out_w) // this means operating on a directory file
  {
    
    // there are two fcb blocks involved. The directory fcb and the fcb of the input file
    // we want to write the index of the file fcb into the directory file

    int start_index = fp & 0xffff;          // retrieve the start of directory file
    int directory_FCB_index = (fp & 0xfff0000) >> 16; // retrieve the FCB block index

    
    if (tag == 1) { // write in data
      int file_FCB_index = fcb_use_name_retrieve_latest_index(fs, (char *)input);
      // write in data in file block
      int new_addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * start_index;
      for (int i = 0; i < 50; i++)
      {
        // write the index of the file fcb into the directory file
        u16 *index_ptr = (u16 *)&fs->volume[new_addr + 2*i];
        if (*index_ptr == 0) { // since fcb 0 has already occupied by the root directory file, 
                                // therefore there should be no other fcb with index 0
          *index_ptr = (u16)file_FCB_index;
          break;
        }    
      }
      // update data in fcb
      int size = fcb_get_size(fs, directory_FCB_index) + str_len((char *)input);
      fcb_set_size(fs, directory_FCB_index, size);
      fcb_set_modifytime(fs, directory_FCB_index, gtime_m);
      gtime_m ++;
    }
    else if (tag == 0) // remove filename 
    {
      // remove filename from the file block
      int addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * start_index;

      // get the fcb index to be crossed out from the directory file blocks
      int fcb_index_to_be_removed = -1;
      char* filename = (char*) input;
      for (int k=0; k<fs->FCB_ENTRIES; k++) { // [3, 7]
        if (pwd_file_is_under_curr_dir_index(fs, k) && 
        fcb_get_validbit(fs, k) && cmp_str(filename, fcb_get_name(fs, k))){
          fcb_index_to_be_removed = k;
        }
      }

      // cross out the targeted fcb index
      for (int i = 0; i < 50; i++)
      {
        u16 *index_ptr = (u16 *)&fs->volume[addr + 2*i];
        // printf("%d done!\n", (int)(*index_ptr));
        if (*index_ptr == (u16)fcb_index_to_be_removed) { 
          *index_ptr = 0;
          break;
        }    
        index_ptr ++;
      }

      // update data in fcb
      int size = fcb_get_size(fs, directory_FCB_index) - str_len((char *)input);
      fcb_set_size(fs, directory_FCB_index, size);
      fcb_set_modifytime(fs, directory_FCB_index, gtime_m);
      gtime_m ++;
    }
  }
  else if (out_w && !out_r) // just write operation for normal file operation
  {
    int start_index = fp & 0xffff;          // retrieve the start file block
    int FCB_index = (fp & 0xfff0000) >> 16; // retrieve the FCB block index


    int need_block_num = (size - 1 + fs->BLOCK_SIZE) / fs->BLOCK_SIZE;
    int pre_size = fcb_get_size(fs, FCB_index);
    int occupied_block_num = (pre_size - 1 + fs->BLOCK_SIZE) / fs->BLOCK_SIZE;

    if (pre_size != 0 && start_index != 0xffff) // not empty! clear VCB and file blocks!
    {
      int start = fcb_get_start(fs, FCB_index);
      clear_VCB_fileblocks(fs, start, occupied_block_num);
    }

    // start to allocate space
    int new_start_index = allocate(fs, need_block_num);
    // printf("%d = allocate(fs, %d)\n", new_start_index, need_block_num);
    if (new_start_index == -1)
    {
      // compact and then allocate again
      compact(fs);
     
      new_start_index = allocate(fs, need_block_num);
      // printf("new_start_index = %d\n", new_start_index);
      if (new_start_index == -1) // still cannot allocate, give out error!
      {
        printf("Error in fswrite! the disk is full, cannot allocate anymore!\n");
        return 1;
      }
    }
    
    // VCB has already been updated in the allocate fu
    // update FCB
    fcb_set_start(fs, FCB_index, new_start_index);
    fcb_set_size(fs, FCB_index, size);
    fcb_set_modifytime(fs, FCB_index, gtime_m);
    gtime_m++;

    // write into the file blocks
    int new_addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * new_start_index;
    for (int i = 0; i < size; i++)
    {
      fs->volume[new_addr + i] = input[i];
    }
  
    return 0;
  }
  else
  {
    printf("Invalid operation, this is a write operation, bro!\n");
    return 1;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == LS_D) // sorted by modified time
  {
    printf("===sort by modified time===\n");
    for (int time = gtime_m; time >= 0; time--)
    {
      int target_fcb_index = -1;
      for (int i = 0; i < fs->FCB_ENTRIES; i++)
      {
        if ((fcb_get_validbit(fs, i)) && (fcb_get_modifytime(fs, i) == time))
        {
          target_fcb_index = i;
          break;
        }
      }
      // printf("target_fcb_index = %d and time = %d\n", target_fcb_index, time);
      // printf_ls(fs);
      if (target_fcb_index != -1 && pwd_file_is_under_curr_dir_index(fs, target_fcb_index ) && (target_fcb_index != 0) )
      {
        char *filename;
        filename = fcb_get_name(fs, target_fcb_index);
        if (fcb_check_dir(fs, target_fcb_index)) {
          printf("%s d\n", filename);
        }
        else {
          printf("%s\n", filename);
        }
      }
    }
  }

  else if (op == LS_S)// sorted by size
  {
    printf("===sort by file size===\n");
    int pre_tar_size = 2000000; // 2,000,000 > 1MB
    while (true)
    {
      int list_for_equal_size[50];
      int num = 0;
      int target_size = -1;

      // iterate all valid FCBs to find the biggest file that < pre_tar_size
      for (int i = 0; i < fs->FCB_ENTRIES; i++)
      {
        int temp_size = fcb_get_size(fs, i);
        if ((fcb_get_validbit(fs, i)) && (temp_size < pre_tar_size))
        {
          if (temp_size > target_size)
          {
            target_size = temp_size;
            num = 1;
            list_for_equal_size[0] = i;
          }
          else if (temp_size == target_size)
          {
            num++;
            list_for_equal_size[num - 1] = i;
          }
          else
          {
            continue;
          }
        }
      }

      if (target_size == -1)
      {
        return; // no more smaller file can be found
      }
      pre_tar_size = target_size;

      // now start to print the file (with the same size) based on creating time
      int list_for_creating_time[50];
      for (int i = 0; i < num; i++)
      {
        list_for_creating_time[i] = fcb_get_createtime(fs, list_for_equal_size[i]);
      }
      for (int k = 0; k < gtime_c; k++)
      {
        for (int i = 0; i < num; i++)
        {
          if (list_for_creating_time[i] == k)
          {
            int FCB_index = list_for_equal_size[i];
            bool tag = pwd_file_is_under_curr_dir_index(fs, FCB_index );
            if (tag && (FCB_index != 0)) {
              if (fcb_check_dir(fs, FCB_index)){
                printf("%s %d d\n", fcb_get_name(fs, FCB_index), fcb_get_size(fs, FCB_index));
              }
              else {
                printf("%s %d\n", fcb_get_name(fs, FCB_index), fcb_get_size(fs, FCB_index));
              }
            }
          }
        }
      }
    }
  }
  else if (op == PWD) {
    printf_pwd(fs);
  }
  else if (op == CD_P) {
    for(int i=3; i>0; i--){
      if (gpwd[i] != -1) {
        gpwd[i] = -1;
        return;
      }
    }
  }
  return;
}

__device__ void remove_file(FileSystem *fs, char *file_name) {
  int target_fcb_index = -1;
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      if (pwd_file_is_under_curr_dir_index (fs, i))
      {
        if (cmp_str(fcb_get_name(fs, i), file_name)) {
          target_fcb_index = i;
          break;
        }
      }
    }
    
    if (target_fcb_index == -1 )
    {
      printf("The file you want to remove do not exists under the current directory!\n");
    }
    else
    {
      // remove the file from the directory fileblocks and update the directory FCB
      u32 handler = fs_open(fs, fcb_get_name(fs, pwd_get()), G_PWD);
      fs_write(fs, (uchar*)file_name, 0, handler);

      int start = fcb_get_start(fs, target_fcb_index);
      int size = fcb_get_size(fs, target_fcb_index);
      int occupied_block_num = (size + fs->BLOCK_SIZE - 1) / fs->BLOCK_SIZE;
      clear_VCB_fileblocks(fs, start, occupied_block_num);
      fcb_clear(fs, target_fcb_index);

    }
    return;
}

__device__ void cd_index(FileSystem *fs, int index) {
  if (pwd_file_is_under_curr_dir_index(fs, index)) {
    if (!fcb_check_dir(fs, index)){
      printf("CD false! The %s is not a directory! \n", fcb_get_name(fs, index));
      return;
    }
    // get to the target directory
    for (int i=1; i<4; i++) {
      if (gpwd[i] == -1) {
        gpwd[i] = index;
        return;
      }
    }
  }
  else {
    printf("CD false! No dir names as %s under the current directory\n");
    return;
  }
}
__device__ void cd_name(FileSystem *fs, char *file_name) {
  int index = -1;
  for (int i=0; i<fs->FCB_ENTRIES; i++) {
    if (pwd_file_is_under_curr_dir_index(fs, i) && cmp_str(file_name, fcb_get_name(fs, i)) && fcb_check_dir(fs, i)) {
      index = i;
      break;
    }
  }
  cd_index(fs, index);
  return;
}

__device__ void fs_gsys(FileSystem *fs, int op, char *file_name)
{
  /* Implement rm operation here */
  if (op == RM)
  {
    remove_file(fs, file_name);
    return;
  }
  else if (op == MKDIR) {
    fs_open(fs, file_name, G_PWD);
    return;
  }
  else if (op == CD) {
    cd_name(fs, file_name);
  }
 
  else if (op == RM_RF) {
    // record temporary directory index
    int original_dir_index;
    for (int i=3; i>0; i--){
      if (gpwd[i] != -1) {
        original_dir_index = gpwd[i];
        break;
      }
    }
    // cd into the target dir
    int tar_dir_index = -1;
    for (int i=0; i<fs->FCB_ENTRIES; i++) {
      if (pwd_file_is_under_curr_dir_index(fs, i) && cmp_str(file_name, fcb_get_name(fs, i)) && fcb_check_dir(fs, i)) {
        tar_dir_index = i;
        break;
      }
    }

    int pre_tar_dir = -1;
    while (true)
    {
      if (pre_tar_dir == tar_dir_index) {
        break;
      }
      else {
        pre_tar_dir = tar_dir_index;
      }
      cd_index(fs, tar_dir_index);
      int ls[50];
      ls_get(fs, ls);
      for (int i=0; i<50; i++) {
        int temp_index = ls[i];
        if (temp_index != 0) {
          if (fcb_check_dir(fs, temp_index)) // is directory
          {
            tar_dir_index = temp_index; // next goes to this dir
          }
          else { // is ordinary file
            // remove the file from the directory fileblocks and update the directory FCB
            remove_file(fs, fcb_get_name(fs, temp_index));
          }
        }
      }
    }
    while (true) {
      int pre_dir_index; // /soft
      for (int i=3; i>0; i--){
        if (gpwd[i] != -1) {
          pre_dir_index = gpwd[i];
          break;
        }
      }
      fs_gsys(fs, CD_P);

      int temp_dir_index; // /app
      for (int i=3; i>0; i--){
        if (gpwd[i] != -1) {
          temp_dir_index = gpwd[i];
          break;
        }
      }
      // remove the dir file /soft
      int ls[50];
      ls_get(fs, ls);
      for (int i=0; i<50; i++) {
        int temp_index = ls[i];
        if (temp_index != 0 && fcb_check_dir(fs, temp_index) && (temp_index = pre_dir_index) ) {
          remove_file(fs, fcb_get_name(fs, temp_index));
          break;
        }
      }
      if (temp_dir_index == original_dir_index) {
        return;
      }
    }
    return;

  }
  else
  {
    printf("this is the fs_gsy(3) operation, but your command is wrong, bro!\n");
    return;
  }
}

// utils

// functions for pwd

// retrieve the fcb index of current dir
__device__ int pwd_get(void) {
  for (int i=3; i>=0; i--) {
    if (gpwd[i] != -1) {
      return gpwd[i];
    }
  }
  printf("Current directory is invalid!\n");
  return -1;
}


__device__ bool pwd_file_is_under_curr_dir(FileSystem *fs, char* file_name) {
  int pwd_fcb_index = pwd_get(); // get the fcb index of the current directory file
  int addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * fcb_get_start(fs, pwd_fcb_index);
  bool tag = false;  
  for (int i = 0; i < 50; i++)
  {
    // read the index of the file fcb from the directory file blocks
    u16 *index_ptr = (u16 *)&fs->volume[addr + 2*i];
    if (*index_ptr != 0) {                                
      int index = (int)*index_ptr;
      if (cmp_str(file_name, fcb_get_name(fs, index))) {
        tag = true;
        break;
      }
    }    
  }
  return tag;
}

__device__ bool pwd_file_is_under_curr_dir_index(FileSystem *fs, int file_index) {
  int pwd_fcb_index = pwd_get(); // get the fcb index of the current directory file
  int addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * fcb_get_start(fs, pwd_fcb_index);
  bool tag = false;  
  for (int i = 0; i < 50; i++)
  {
    // read the index of the file fcb from the directory file blocks
    u16 *index_ptr = (u16 *)&fs->volume[addr + 2*i];
    if (*index_ptr != 0) {                                
      int index = (int)*index_ptr;
      if (index == file_index) {
        tag = true;
        break;
      }
    }    
  }
  return tag;
}

__device__ void printf_pwd(FileSystem *fs) {
  bool tag = false;
    for(int i=1; i<4; i++){
      if (gpwd[i] != -1) {
        tag = true;
        printf("/%s", fcb_get_name(fs, gpwd[i]));
      }
    }
    if (!tag) printf("/");
    printf("\n");
}


__device__ void printf_ls(FileSystem *fs) {
  int pwd_fcb_index = pwd_get(); // get the fcb index of the current directory file
  int addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * fcb_get_start(fs, pwd_fcb_index);
  for (int i = 0; i < 50; i++)
  {
    // read the index of the file fcb from the directory file blocks
    u16 *index_ptr = (u16 *)&fs->volume[addr + 2*i];
    if (*index_ptr != 0) {                                
      int index = (int)*index_ptr;
      printf("%d ", index);
    }    
  }
  printf("\n");
  return;
}

// return a 50 entry list which stores index of fcb
__device__ void ls_get(FileSystem *fs, int* list) {
  int pwd_fcb_index = pwd_get();
  u32 fp = fs_open(fs, fcb_get_name(fs, pwd_fcb_index), G_PWD);
  fs_read(fs, (uchar*) list, NULL, fp);
  return;
}
// major functions
__device__ void compact(FileSystem *fs)
{
  int target_start = 0;
  while (true)
  {
    /* code */
    int start = fs->FCB_ENTRIES*fs->FCB_SIZE;
    int index = -1;
    for (int i=0; i<fs->FCB_ENTRIES; i++)
    {
      if (fcb_get_validbit(fs, i)) {
        int start_temp = fcb_get_start(fs, i);
        int size_temp = fcb_get_size(fs, i);
        if ( (start_temp != 0xffff) && (start_temp >= target_start) && (start_temp < start))
        {
          start = start_temp;
          index = i;
          
        }
      }
    }

    if (start == fs->FCB_ENTRIES*fs->FCB_SIZE) { // doesn't find blocks needed to be compacted
      return;
    }
    else {
      int size = fcb_get_size(fs, index);
      if (target_start != start) { // there is fragmentation between two consecutive blocks  
        // start to compact!
        // generate a handler to use fs_read
        u32 handler = 0;
        handler |= (1 << 31);
        handler |= (index << 16);                 // FCB index
        handler |= (u16)fcb_get_start(fs, index); // start block index
        uchar buffer[1024];
        fs_read(fs, buffer, size, handler);

        // clear VCB and file blocks
        int occupied_blocks = (size + fs->BLOCK_SIZE -1) / fs->BLOCK_SIZE;
        clear_VCB_fileblocks(fs, start, occupied_blocks);
        // set new VCB
        for (int k=target_start; k<target_start+occupied_blocks; k++) {
          vcb_set(fs, k);
        }
        // update FCB
        fcb_set_start(fs, index, target_start);

        // copy the file blocks
        int start_addr = fs->FILE_BASE_ADDRESS + target_start * fs->BLOCK_SIZE;
        for (int i = 0; i < size; i++)
        {
          fs->volume[start_addr + i] = buffer[i];
        }

        target_start += occupied_blocks;
      }
      else {
        target_start += size;
        continue;
      }

    }
  }
  
  return;
}

__device__ int allocate(FileSystem *fs, int num)
{
  int continuous_empty_VCB = 0;
  for (int i = 0; i < fs->SUPERBLOCK_SIZE * 8; i++)
  {
    if (!vcb_get(fs, i))
    { // empty
      continuous_empty_VCB++;
    }
    else
    {
      continuous_empty_VCB = 0;
      continue;
    }
    if (continuous_empty_VCB == num)
    {
      int start_index = i - num + 1;
      for (int t = start_index; t < start_index + num; t++)
      {
        vcb_set(fs, t);
      }
      return start_index;
    }
  }
  return -1;
}

__device__ void clear_VCB_fileblocks(FileSystem *fs, int start, int size)
{
  for (int i = start; i < start + size; i++)
  {
    vcb_clear(fs, i);
    fb_clear(fs, i);
  }
  return;
}

// file blocks
__device__ void fb_clear(FileSystem *fs, int index)
{
  int start_index = fs->FILE_BASE_ADDRESS + index * fs->BLOCK_SIZE;
  for (int i = start_index; i < start_index + fs->BLOCK_SIZE; i++)
  {
    fs->volume[i] = 0;
  }
  return;
}

// VCB
__device__ void printf_VCB(FileSystem *fs, int start, int end)
{
  for (int i = start; i < end + 1; i++)
  {
    printf("%d", vcb_get(fs, i));
  }
  printf("\n");
  return;
}

__device__ void printf_bin(int num)
{
  int i, j, k;
  unsigned char *p = (unsigned char *)&num + 3;

  for (i = 0; i < 4; i++)
  {
    j = *(p - i);
    for (int k = 7; k >= 0; k--)
    {
      if (j & (1 << k))
        printf("1");
      else
        printf("0");
    }
    printf(" ");
  }
  printf("\r\n");
  return;
}

__device__ void vcb_set(FileSystem *fs, int block_index)
{
  if (block_index < 0 || block_index >= (fs->SUPERBLOCK_SIZE * 8))
  {
    printf("superblock setting overflow\n");
    return;
  }
  int row = block_index / 8;
  int column = block_index % 8;
  uchar mask;
  mask = (1 << column);
  fs->volume[row] |= mask;
  return;
}

__device__ void vcb_clear(FileSystem *fs, int block_index)
{
  if (block_index < 0 || block_index >= (fs->SUPERBLOCK_SIZE * 8))
  {
    printf("superblock clearing overflow\n");
    return;
  }
  int row = block_index / 8;
  int column = block_index % 8;
  uchar mask;
  mask = (1 << column);
  mask = ~mask;
  fs->volume[row] &= mask;
  return;
}

__device__ bool vcb_get(FileSystem *fs, int block_index)
{
  if (block_index < 0 || block_index >= (fs->SUPERBLOCK_SIZE * 8))
  {
    printf("superblock getting overflow\n");
    return;
  }
  int row = block_index / 8;
  int column = block_index % 8;
  uchar mask;
  mask = (1 << column);
  bool out = fs->volume[row] & mask; // out=1 occupied
  return out;
}

// FCB
__device__ int fcb_use_name_retrieve_index(FileSystem *fs, char *name){
  int target_fcb_index = -1;
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    // printf("%s v.s. %s\n",name, fcb_get_name(fs, i));
    if ((fcb_get_validbit(fs, i)) && (cmp_str(fcb_get_name(fs, i), name)))
    {
      target_fcb_index = i;
      break;
    }
  }
  if (target_fcb_index == -1) // not find the target
  { 
    printf("target file do not exist!\n");
  }
  return target_fcb_index;
}

__device__ int fcb_use_name_retrieve_latest_index(FileSystem *fs, char *name){
  int target_fcb_index = -1;
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    // printf("%s v.s. %s\n",name, fcb_get_name(fs, i));
    if ((fcb_get_validbit(fs, i)) && (cmp_str(fcb_get_name(fs, i), name)) && fcb_get_createtime(fs, i) == (gtime_c-1))
    {
      target_fcb_index = i;
    }
  }
  if (target_fcb_index == -1) // not find the target
  { 
    printf("target file do not exist!\n");
  }
  return target_fcb_index;
}

__device__ void printf_fcb(FileSystem *fs, int index)
{
  char *getname = fcb_get_name(fs, index);
  printf("=== the %d fcb ===\n", index);
  printf("name: %s\n", getname);
  printf("valid: %d\n", fcb_get_validbit(fs, index));
  printf("dir: %d\n", fcb_check_dir(fs, index));
  printf("start: %d\n", fcb_get_start(fs, index));
  printf("size: %d\n", fcb_get_size(fs, index));
  printf("create: %d\n", fcb_get_createtime(fs, index));
  printf("modify: %d\n", fcb_get_modifytime(fs, index));
  printf("---------------\n");
  return;
}

__device__ bool cmp_str(char *str1, char *str2)
{
  while (*str1 != '\0' && *str2 != '\0' && *str1 == *str2)
  {
    str1++;
    str2++;
  }
  if (*str1 == '\0' && *str2 == '\0')
  {
    return true;
  }
  else
  {
    return false;
  }
}

__device__ void fcb_clear(FileSystem *fs, int index)
{
  int start = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index;
  for (int i = start; i < start + 32; i++)
  {
    fs->volume[i] = 0;
  }
}

__device__ char *fcb_get_name(FileSystem *fs, int index)
{

  uchar *ptr = &(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index]);
  char *out = (char *)ptr;
  // printf("i am fine\n");
  return out;
}

__device__ void fcb_set_name(FileSystem *fs, int index, char *file_name)
{
  // set new file name
  uchar temp = *file_name;
  char *temp_pt = file_name;
  int cnt = 0;
  while (temp != '\0')
  {
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + cnt] = temp;
    temp_pt++;
    cnt++;
    temp = *temp_pt;
    if (cnt == fs->MAX_FILENAME_SIZE)
    {
      printf("file name exceed the limit\n");
    }
  }
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + cnt] = '\0';
  return;
}

__device__ void fcb_set_validbit(FileSystem *fs, int index)
{
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 20] = 0x01;
  return;
}

__device__ void fcb_clear_validbit(FileSystem *fs, int index)
{
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 20] = 0x00;
  return;
}

__device__ bool fcb_get_validbit(FileSystem *fs, int index)
{
  if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 20] == 0x00)
  {
    return false;
  }
  return true;
}

__device__ void fcb_set_dir(FileSystem *fs, int index)
{
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 21] = 0x01;
  return;
}

__device__ bool fcb_check_dir(FileSystem *fs, int index)
{
  if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 21] == 0x00)
  {
    return false;
  }
  return true;
}

__device__ void fcb_set_start(FileSystem *fs, int index, int start)
{
  u16 in = start;
  u16 *start_ptr = (u16 *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 22];
  *start_ptr = in;
  return;
}

__device__ int fcb_get_start(FileSystem *fs, int index)
{
  u16 *start_ptr = (u16 *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 22];

  int out = *start_ptr;
  return out;
}

__device__ void fcb_set_size(FileSystem *fs, int index, int size)
{
  int *size_ptr = (int *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 24];
  *size_ptr = size;
  return;
}

__device__ int fcb_get_size(FileSystem *fs, int index)
{
  int *size_ptr = (int *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 24];
  return *size_ptr;
}

__device__ void fcb_set_createtime(FileSystem *fs, int index, u32 createtime)
{
  u16 in = createtime;
  u16 *create_time_ptr = (u16 *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 28];
  *create_time_ptr = in;
  return;
}

__device__ int fcb_get_createtime(FileSystem *fs, int index)
{
  u16 *create_time_ptr = (u16 *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 28];
  int out = *create_time_ptr;
  return out;
}

__device__ void fcb_set_modifytime(FileSystem *fs, int index, u32 modifytime)
{
  u16 in = modifytime;
  u16 *modifytime_ptr = (u16 *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 30];
  *modifytime_ptr = in;
  return;
}

__device__ int fcb_get_modifytime(FileSystem *fs, int index)
{
  u16 *modifytime_ptr = (u16 *)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 30];
  int out = *modifytime_ptr;
  return out;
}

