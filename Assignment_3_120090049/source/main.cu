﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

// page size is 32bytes
#define PAGE_SIZE (1 << 5)
// 4 KB in page table
#define INVERT_PAGE_TABLE_SIZE (1 << 12)
// 8 KB in page table
#define INVERT_SWAP_TABLE_SIZE (1 << 13)
// 32 KB in shared memory
#define PHYSICAL_MEM_SIZE (1 << 15)
// 128 KB in global memory
#define STORAGE_SIZE (1 << 17)

//// count the pagefault times
__device__ __managed__ int pagefault_num = 0;
// head number in page table (used for LRU)
__device__ __managed__ int head_num_pt = 0xfff;

// data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

// memory allocation for virtual_memory
// secondary memory
__device__ __managed__ uchar swap_space[STORAGE_SIZE]; // 128K
// page table 1KB
extern __shared__ u32 pt[];

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size);

__global__ void mykernel(int input_size)
{
  // memory allocation for virtual_memory
  // take shared memory as physical memory
  // swap table
  __shared__ u16 st[4096];                          // 2KB
  __shared__ uchar physical_mem[PHYSICAL_MEM_SIZE]; // 32K

  VirtualMemory vm;

  vm_init(&vm, physical_mem, swap_space,
          pt, st,
          &pagefault_num, &head_num_pt,
          PAGE_SIZE,
          INVERT_PAGE_TABLE_SIZE, INVERT_SWAP_TABLE_SIZE,
          PHYSICAL_MEM_SIZE, STORAGE_SIZE,
          PHYSICAL_MEM_SIZE / PAGE_SIZE, STORAGE_SIZE / PAGE_SIZE);

  // user program the access pattern for testing paging
  user_program(&vm, input, results, input_size);
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
  FILE *fp;
  fp = fopen(fileName, "wb");
  fwrite(buffer, 1, bufferSize, fp);
  fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
  FILE *fp;

  fp = fopen(fileName, "rb");
  if (!fp)
  {
    printf("***Unable to open file %s***\n", fileName);
    exit(1);
  }

  // Get file length
  fseek(fp, 0, SEEK_END);
  int fileLen = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (fileLen > bufferSize)
  {
    printf("****invalid testcase!!****\n");
    printf("****software warrning: the file: %s size****\n", fileName);
    printf("****is greater than buffer size****\n");
    exit(1);
  }

  // Read file contents into buffer
  fread(buffer, fileLen, 1, fp);
  fclose(fp);

  return fileLen;
}

int main()
{
  cudaError_t cudaStatus;
  int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

  /* Launch kernel function in GPU, with single thread
  and dynamically allocate INVERT_PAGE_TABLE_SIZE bytes of share memory,
  which is used for variables declared as "extern __shared__" */
  mykernel<<<1, 1, INVERT_PAGE_TABLE_SIZE>>>(input_size);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "mykernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return 0;
  }

  cudaDeviceSynchronize();
  cudaDeviceReset();

  write_binaryFile(OUTFILE, results, input_size);

  printf("pagefault number is %d\n", pagefault_num);

  return 0;
}
