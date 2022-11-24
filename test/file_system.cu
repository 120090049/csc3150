#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime_m = 0;
__device__ __managed__ u32 gtime_c = 0;

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

  for (int i=0; i<fs->FCB_ENTRIES; i++) {
    if ( (fcb_get_validbit(fs, i) ) && ( cmp_str(fcb_get_name(fs, i), file_name)) ) {
      target_fcb_index = i;
      break;
    }
  }
  if (target_fcb_index == -1) { // not find the target
    for (int i=0; i<fs->FCB_ENTRIES; i++) {
      if ( !fcb_get_validbit(fs, i) ) { // find the fcb that is invalid
        target_fcb_index = i;
        break;
      }
    }
    if (target_fcb_index == -1){
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
    gtime_c ++;
  }
  // after all of these, we have already get or allocate the FCB
  // The handler consists of three part, read/write bit + FCB index + start block
  u32 handler = 0;
  if (op == G_READ){
    handler |= (1<<31);
  }
  if (op == G_WRITE) {
    handler |= (1<<30);
  }
  handler |= (target_fcb_index << 16); // FCB index
  handler |= (u16)fcb_get_start(fs, target_fcb_index); // start block index
  return handler;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  bool out = fp & (1 << 31); // check the valid bit
  if (out) {
    int start_index = fp & 0xffff; // retrieve the start file block
    int FCB_index = (fp & 0xfff0000) >> 16; // retrieve the FCB block index

    int file_size = fcb_get_size(fs, FCB_index);

    if (start_index == 0xffff || file_size == 0) {
      printf("The file block hasn't been allocate, cannot read, bro!\n");
      return;
    }
    int read_size;
    if (size > file_size) {
      read_size = file_size;
    }
    else {
      read_size = size;
    }
    // start to read
    int start_addr = fs->FILE_BASE_ADDRESS + start_index * fs->BLOCK_SIZE;
    for (int i = 0; i < size; i++) {
      output[i] = fs->volume[start_addr + i];
      // printf("%d and %c", i, output[i]);
    }
    return;
  }
  else {
    printf("Invalid operation, this is a read operation, bro!\n");
    return;
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  bool out = fp & (1 << 30); // check the valid bit
  if (out) {
    int start_index = fp & 0xffff; // retrieve the start file block
    int FCB_index = (fp & 0xfff0000) >> 16; // retrieve the FCB block index
    // printf("start_index: %d\n", start_index);
    // printf("FCB_index: %d\n", FCB_index);

    int need_block_num = (size - 1 + fs->BLOCK_SIZE) / fs->BLOCK_SIZE;
    int pre_size = fcb_get_size(fs, FCB_index);
    int occupied_block_num = (pre_size -1 + fs->BLOCK_SIZE) / fs->BLOCK_SIZE;
    
    if (pre_size != 0 && start_index != 0xffff)  // not empty! clear VCB and file blocks!
    { 
      int start = fcb_get_start(fs, FCB_index);
      clear_VCB_fileblocks(fs, start, occupied_block_num);
    }
    
    // start to allocate space
    int new_start_index = allocate(fs, need_block_num);
    if (new_start_index == -1) {
      // compact and allocate again
      compact(fs);
      new_start_index = allocate(fs, need_block_num);
      if (new_start_index == -1) // still cannot allocate, give out error!
      {
        printf("Error in fswrite! the disk is full, cannot allocate anymore!\n");
        return 1;
      }
    }
    else {
      // VCB has already been updated in the allocate fu
      // update FCB
      fcb_set_start(fs, FCB_index, new_start_index);
      fcb_set_size(fs, FCB_index, size);
      fcb_set_modifytime(fs, FCB_index, gtime_m);
      gtime_m ++;
      
      // write into the file blocks
      int new_addr = fs->FILE_BASE_ADDRESS + fs->BLOCK_SIZE * new_start_index;
      for (int i = 0; i < size; i++) {
        fs->volume[new_addr + i] = input[i];
      }
    }
    return 0;
  }
  else {
    printf("Invalid operation, this is a write operation, bro!\n");
    return 1;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == LS_D) // sorted by modified time
  {
    printf("===sorted by modified time===\n");
    for (int time = gtime_m; time >=0; time--)
    {
      int target_fcb_index = -1;
      for (int i=0; i<fs->FCB_ENTRIES; i++) {
        if ( (fcb_get_validbit(fs, i) ) && ( fcb_get_modifytime(fs, i) == time ) ) {
          target_fcb_index = i;
          break;
        }
      }
      if (target_fcb_index != -1) {
        char* filename;
        filename = fcb_get_name(fs, target_fcb_index);
        printf("%s\n", filename);
      }
    }
  }
  else // sorted by size
  {
    printf("===sorted by file size===\n");
    int pre_tar_size = 2000000; // 2,000,000 > 1MB
    while (true)
    {
      int list_for_equal_size[50];
      int num = 0;
      int target_size = -1;

      // iterate all valid FCBs to find the biggest file that < pre_tar_size
      for (int i=0; i<fs->FCB_ENTRIES; i++) {
        int temp_size = fcb_get_size(fs, i);
        if ( (fcb_get_validbit(fs, i) ) && ( temp_size < pre_tar_size ) ) {
          if (temp_size > target_size) {
            target_size = temp_size;
            num =  1;
            list_for_equal_size[0] = i;
          }
          else if (temp_size == target_size) {
            num ++;
            list_for_equal_size[num-1] = i;
          }
          else {
            continue;
          }
        }
      }

      if (target_size == -1) {
        return; //no more smaller file can be found
      }
      pre_tar_size = target_size;
      
      // now start to print the file (with the same size) based on creating time
      int list_for_creating_time[50];
      for (int i=0; i<num; i++) {
        list_for_creating_time[i] = fcb_get_createtime(fs, list_for_equal_size[i]);
      }
      for (int k=0; k<gtime_c; k++) {
        for (int i=0; i<num; i++) {
          if (list_for_creating_time[i] == k) {
            int FCB_index = list_for_equal_size[i];
            printf("%s %d\n", fcb_get_name(fs, FCB_index), fcb_get_size(fs, FCB_index));
          }
        }
      }
    } 
  }
  return;
  
}

__device__ void fs_gsys(FileSystem *fs, int op, char *file_name)
{
  /* Implement rm operation here */
  if (op == RM) {
    int target_fcb_index = -1;
    for (int i=0; i<fs->FCB_ENTRIES; i++) {
      if ( (fcb_get_validbit(fs, i) ) && ( cmp_str(fcb_get_name(fs, i), file_name)) ) {
        target_fcb_index = i;
        break;
      }
    }
    if (target_fcb_index == -1) {
      printf ("The file you want to remove do not exists!\n");
    }
    else {
      int start = fcb_get_start(fs, target_fcb_index);
      int size = fcb_get_size(fs, target_fcb_index);
      int occupied_block_num = (size + fs->BLOCK_SIZE -1) / fs->BLOCK_SIZE;
      clear_VCB_fileblocks(fs, start, occupied_block_num);
      fcb_clear(fs, target_fcb_index);
    }
    return;
  }
  else {
    printf("this is the RM operation, but your command is wrong, bro!\n");
    return;
  }
}






// utils

__device__ void compact(FileSystem *fs) { 
  return;
}

__device__ int allocate(FileSystem *fs, int num) {
  int continuous_empty_VCB = 0;
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
    if (!vcb_get(fs, i)) {
      continuous_empty_VCB ++;
    }
    else {
      continuous_empty_VCB = 0;
      continue;
    }
    if (continuous_empty_VCB == num) {
      int start_index = i-num+1;
      for (int t=start_index; t<start_index + num; t++){
        vcb_set(fs, t);
      }
      return start_index;
    }
  }
  return -1;
}

__device__ void clear_VCB_fileblocks (FileSystem *fs, int start, int size) {
  for (int i = start; i < start+size; i++)
  {
    vcb_clear(fs, i);
    fb_clear(fs, i);
  }
  return;
}

//file blocks
__device__ void fb_clear (FileSystem *fs, int index) {
  int start_index = fs->FILE_BASE_ADDRESS + index * fs->BLOCK_SIZE;
  for (int i=start_index; i<start_index+fs->BLOCK_SIZE; i++)
  {
    fs->volume[i] = 0; 
  }
  return;
}



// VCB
__device__ void printf_VCB(FileSystem *fs, int start, int end){
  for (int i=start; i<end+1; i++) {
    printf("%d", vcb_get(fs, i));
  }
  printf("\n");
  return;
}

__device__ void printf_bin(int num)
{
	int i, j, k;
	unsigned char *p = (unsigned char*)&num + 3;

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

__device__ void vcb_set(FileSystem *fs, int block_index) {
	if (block_index < 0 || block_index >= (fs->SUPERBLOCK_SIZE)) {
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

__device__ void vcb_clear(FileSystem *fs, int block_index) {
  if (block_index < 0 || block_index >= (fs->SUPERBLOCK_SIZE)) {
    printf("superblock setting overflow\n");
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

__device__ bool vcb_get(FileSystem *fs, int block_index) {
  if (block_index < 0 || block_index >= (fs->SUPERBLOCK_SIZE)) {
    printf("superblock setting overflow\n");
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
__device__ void printf_fcb(FileSystem *fs, int index) {
	char* getname = fcb_get_name(fs, index);
	printf("name: %s\n", getname);
	printf("valid: %d\n", fcb_get_validbit(fs, index));
	printf("dir: %d\n", fcb_check_dir(fs, index));
	printf("start: %d\n", fcb_get_start(fs, index));
	printf("size: %d\n", fcb_get_size(fs, index));
	printf("create: %d\n", fcb_get_createtime(fs,index));
	printf("modify: %d\n", fcb_get_modifytime(fs, index));
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

__device__ void fcb_clear(FileSystem *fs, int index) {
  int start = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index;
  for (int i = start; i < start + 32 ; i++)
  {
    fs->volume[i] = 0; 
  }
}


__device__ char * fcb_get_name(FileSystem *fs, int index) {
  uchar * ptr = &(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index]);
	char * out = (char*) ptr;
  return out;
} 

__device__ void fcb_set_name(FileSystem *fs, int index, char *file_name) {
  //set new file name
  uchar temp = *file_name;
	char * temp_pt = file_name;
  int cnt = 0;
  while (temp != '\0') {
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + cnt] = temp;
    temp_pt ++;
    cnt++;
    temp = *temp_pt;
    if (cnt == fs->MAX_FILENAME_SIZE) {
      printf("file name exceed the limit\n");
    }
  }
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + cnt] = '\0';
  return;
} 

__device__ void fcb_set_validbit(FileSystem *fs, int index) {
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 20] = 0x01;
  return;
} 

__device__ void fcb_clear_validbit(FileSystem *fs, int index) {
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 20] = 0x00;
  return;
} 

__device__ bool fcb_get_validbit(FileSystem *fs, int index) {
  if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 20] == 0x00)
  {
    return false;
  } 
  return true;  
} 

__device__ void fcb_set_dir(FileSystem *fs, int index) {
  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 21] = 0x01;
  return;
} 

__device__ bool fcb_check_dir(FileSystem *fs, int index) {
  if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 21] == 0x00)
  {
    return false;
  } 
  return true;  
} 

__device__ void fcb_set_start(FileSystem *fs, int index, int start) {
  u16 in = start;
  u16* start_ptr = (u16*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 22];
  *start_ptr = in;
  return;
} 

__device__ int fcb_get_start(FileSystem *fs, int index) {
  u16* start_ptr = (u16*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 22];
  
  int out = *start_ptr;
  return out;
} 

__device__ void fcb_set_size(FileSystem *fs, int index, int size) {
  int* size_ptr = (int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 24];
  *size_ptr = size;
  return;
} 

__device__ int fcb_get_size(FileSystem *fs, int index) {
  int* size_ptr = (int*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 24];
  return *size_ptr;
} 

__device__ void fcb_set_createtime(FileSystem *fs, int index, u32 createtime) {
  u16 in = createtime;
  u16* create_time_ptr = (u16*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 28];
  *create_time_ptr = in;
  return;
} 

__device__ int fcb_get_createtime(FileSystem *fs, int index) {
  u16* create_time_ptr = (u16*)&fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 28];
  int out = *create_time_ptr;
  return out;
} 

__device__ void fcb_set_modifytime(FileSystem *fs, int index, u32 modifytime) {
  u16 in = modifytime;
  u16* modifytime_ptr = (u16*) &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 30];
  *modifytime_ptr = in;
  return;
} 

__device__ int fcb_get_modifytime(FileSystem *fs, int index) {
  u16* modifytime_ptr = (u16*) &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * index + 30];
  int out = *modifytime_ptr;
  return out;
} 