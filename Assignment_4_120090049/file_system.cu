#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int VCB,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->VCB = VCB;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->BLOCK_SIZE = BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init VCB
  for (int i = 0; i < fs->VCB; i++)
  {
    fs->volume[i] = 0;
  }
  // init FCB
  //   FCB:
  //   0-19   name
  //	 20     nothing
  //   21     (all in the 21 byte least significent 3 bit) valid bit + directory or not + empty or not
  //          initial state = 0000 0001
  //	 22-23  start address
  //	 24-27  size (valid bit at 22)
  //	 28-29  created time
  //   30-31  modified time
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    fs->volume[fs->VCB + fs->FCB_SIZE * i + 21] = 0x1; // set to initial state = 001 (not valid + not directory + is empty)
  }
}

__device__ int fcb_find(FileSystem *fs, char *file_name)
{
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    fs->volume[fs->VCB + fs->FCB_SIZE * i + 21] = 0x1;
  }
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

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  /* Implement open operation here */
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  /* Implement read operation here */
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  /* Implement write operation here */
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  /* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  /* Implement rm operation here */
}
