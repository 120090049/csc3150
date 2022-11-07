#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

struct VirtualMemory
{
  uchar *physical_mem;
  uchar *swap_space;
  u32 *invert_page_table;
  u16 *swap_table;
  int *pagefault_num_ptr;
  int *head_num_pt;

  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int INVERT_SWAP_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PT_ENTRIES;
  int ST_ENTRIES;
};

// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, u16 *swap_page_table,
                        int *pagefault_num_ptr, int *head_num_pt,
                        int PAGESIZE,
                        int INVERT_PAGE_TABLE_SIZE, int INVERT_SWAP_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PT_ENTRIES, int ST_ENTRIES);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);

#endif
