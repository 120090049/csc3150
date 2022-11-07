#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_invert_page_table(VirtualMemory *vm)
{
  for (int i = 0; i < vm->PT_ENTRIES; i++)
  {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
  }
  return;
}

__device__ void init_invert_swap_table(VirtualMemory *vm)
{
  for (int i = 0; i < vm->ST_ENTRIES; i++)
  {
    vm->swap_table[i] = 0x8000; // invalid := MSB is 1
  }
  return;
}

__device__ void vm_init(VirtualMemory *vm, uchar *physical_mem, uchar *swap_space,
                        u32 *invert_page_table, u16 *swap_table,
                        int *pagefault_num_ptr, int *head_num_pt,
                        int PAGESIZE,
                        int INVERT_PAGE_TABLE_SIZE, int INVERT_SWAP_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PT_ENTRIES, int ST_ENTRIES)
{
  // init variables
  vm->physical_mem = physical_mem;
  vm->swap_space = swap_space;
  vm->invert_page_table = invert_page_table;
  vm->swap_table = swap_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;
  vm->head_num_pt = head_num_pt;
  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->INVERT_SWAP_TABLE_SIZE = INVERT_SWAP_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PT_ENTRIES = PT_ENTRIES;
  vm->ST_ENTRIES = ST_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
  init_invert_swap_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr)
{
  /* Complate vm_read function to read single element from data buffer */
  int page_id = (0b111111111111100000 & addr) >> 5;
  u32 offset = (0b11111 & addr);
  // printf("\n\npage id = %d and page offset = %d\n", page_id, offset);
  u32 physical_addr;
  uchar read_result;
  // index for entry
  int index = -1;
  int i;
  // find whether page id is in the page table
  for (i = 0; i < vm->PT_ENTRIES; i++)
  {
    if ((vm->invert_page_table[i] & 0x80000000) == 0x0)
    {
      int current_page_id = (vm->invert_page_table[i] & 0x0ffff000) >> 12;
      if (current_page_id == page_id)
      {
        index = i;
        break;
      }
    }
  }
  if (index != -1) // find
  {
    // step1 read
    physical_addr = (index << 5) | offset;
    read_result = vm->physical_mem[physical_addr];

    // step2 update LRU
    int pre_LRU = (0xfff & vm->invert_page_table[index]);
    vm->invert_page_table[index] &= 0xfffff000;
    vm->invert_page_table[index] |= *(vm->head_num_pt);
    *(vm->head_num_pt) = index;

    // step3
    for (i = 0; i < vm->PT_ENTRIES; i++)
    {
      if ((vm->invert_page_table[i] & 0x80000000) == 0x0)
      {
        int current_LRU = (vm->invert_page_table[i] & 0xfff);
        if (current_LRU == index)
        {
          vm->invert_page_table[i] &= 0xfffff000;
          vm->invert_page_table[i] |= pre_LRU;
          break;
        }
      }
    }
    // printf("find in page table and physical_addr = %d\n", physical_addr);
  }
  else
  {
    *(vm->pagefault_num_ptr) += 1;
    // step1 find page id in swap table
    // 1-1 look into the swap table
    int st_index = -1;
    int k;
    for (k = 0; k < vm->ST_ENTRIES; k++)
    {
      if ((vm->swap_table[k] & 0x8000) == 0) // non-empty entries
      {
        if ((vm->swap_table[k] & 0x1fff) == page_id)
        {
          st_index = k;
          break;
        }
      }
    }

    // step2 update page table
    // find LRU = fff
    int j;
    int LRU_index; // this shoud be the frame id
    for (j = 0; j < vm->PT_ENTRIES; j++)
    {
      if ((vm->invert_page_table[j] & 0x00000fff) == 0xfff) // find LRU
      {
        LRU_index = j;
        break;
      }
    }
    // printf("LRU_index = %d\n", LRU_index);
    // get the page id to be swapped
    int origin_page_id = (vm->invert_page_table[LRU_index] &= 0x0ffff000) >> 12;
    // write in page id in the page table
    vm->invert_page_table[LRU_index] &= 0xf0000fff; // start to operate on the page id
    vm->invert_page_table[LRU_index] |= (page_id << 12);

    // update the LRU bit
    vm->invert_page_table[LRU_index] &= 0xfffff000;
    vm->invert_page_table[LRU_index] |= *(vm->head_num_pt);
    *(vm->head_num_pt) = LRU_index;

    // step3
    for (k = 0; k < vm->PT_ENTRIES; k++)
    {
      if ((vm->invert_page_table[k] & 0x80000000) == 0x0)
      {
        int current_LRU = (vm->invert_page_table[k] & 0xfff);
        if (current_LRU == LRU_index)
        {
          vm->invert_page_table[k] &= 0xfffff000;
          vm->invert_page_table[k] |= 0xfff;
          // printf("KKKK = %d\n", k);
          // printf("*(vm->head_num_pt) = %d\n", *(vm->head_num_pt));
          break;
        }
      }
    }

    // step4 swap data and write original page id into the swap table
    for (int t = 0; t < vm->PAGESIZE; t++)
    {
      uchar temp = vm->physical_mem[LRU_index * (vm->PAGESIZE) + t];
      vm->physical_mem[LRU_index * (vm->PAGESIZE) + t] = vm->swap_space[st_index * vm->PAGESIZE + t];
      vm->swap_space[st_index * vm->PAGESIZE + t] = temp;
    }
    vm->swap_table[st_index] &= 0x8000;
    vm->swap_table[st_index] |= origin_page_id;
    // printf("read addr %d and fault %d\n", LRU_index, *vm->pagefault_num_ptr);
    // printf("jiaohuanchenggong %c\n\n", vm->physical_mem[LRU_index * vm->PAGESIZE]);
    // step5 read
    physical_addr = (LRU_index << 5) | offset;
    read_result = vm->physical_mem[physical_addr];
    // printf("find in swap table, physical_addr = %d\n", physical_addr);
  }

  return read_result;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value)
{

  int page_id = (0b111111111111100000 & addr) >> 5;
  u32 offset = (0b11111 & addr);
  u32 physical_addr;
  // index for entry
  int index = -1;
  int i;
  // find whether page id is in the page table
  for (i = 0; i < vm->PT_ENTRIES; i++)
  {
    if ((vm->invert_page_table[i] & 0x80000000) == 0x0)
    {
      // page id in invert table (22bit-10bit) 0b 00111 11111 11111 00000 00000
      int current_page_id = (vm->invert_page_table[i] & 0x0ffff000) >> 12;
      if (current_page_id == page_id)
      {
        index = i;
        break;
      }
    }
  }
  if (index != -1) // find the page id in the page table
  {
    // step1 write
    physical_addr = (index << 5) | offset;
    vm->physical_mem[physical_addr] = value;

    // step2 update LRU
    int pre_LRU = (0xfff & vm->invert_page_table[index]);
    vm->invert_page_table[index] &= 0xfffff000;
    vm->invert_page_table[index] |= *(vm->head_num_pt);
    *(vm->head_num_pt) = index;

    // step3
    for (i = 0; i < vm->PT_ENTRIES; i++)
    {
      if ((vm->invert_page_table[i] & 0x80000000) == 0x0)
      {
        // page id in invert table (22bit-10bit) 0b 00111 11111 11111 00000 00000
        int current_LRU = (vm->invert_page_table[i] & 0xfff);
        if (current_LRU == index)
        {
          vm->invert_page_table[i] &= 0xfffff000;
          vm->invert_page_table[i] |= pre_LRU;
          break;
        }
      }
    }
  }
  else // not in page table
  {
    *(vm->pagefault_num_ptr) += 1;
    // find whether the page table is full
    int empty_index = -1;
    int j;
    for (j = 0; j < vm->PT_ENTRIES; j++)
    {
      if ((vm->invert_page_table[j] & 0x80000000) == 0x80000000) // find empty page
      {
        empty_index = j;
        break;
      }
    }

    if (empty_index != -1) // there is an empty page
    {
      // step1 write in page id in the page table
      vm->invert_page_table[empty_index] &= 0x7fffffff; // clear the invalid bit
      vm->invert_page_table[empty_index] &= 0xf0000fff; // start to operate on the page id
      vm->invert_page_table[empty_index] |= (page_id << 12);

      // step2 write into the physical memory
      physical_addr = (empty_index << 5) | offset;
      vm->physical_mem[physical_addr] = value;

      // step3 update the LRU
      vm->invert_page_table[empty_index] &= 0xfffff000;
      vm->invert_page_table[empty_index] |= *(vm->head_num_pt);
      *(vm->head_num_pt) = empty_index;
    }

    else // there is no empty page
    {
      // step1 find LRU = fff
      int j;
      int LRU_index;
      for (j = 0; j < vm->PT_ENTRIES; j++)
      {
        if ((vm->invert_page_table[j] & 0x00000fff) == 0xfff) // find LRU
        {
          LRU_index = j;
          break;
        }
      }
      // get the page id to be swapped
      int origin_page_id = (vm->invert_page_table[LRU_index] &= 0x0ffff000) >> 12;
      //    write in page id in the page table
      vm->invert_page_table[LRU_index] &= 0xf0000fff; // start to operate on the page id
      vm->invert_page_table[LRU_index] |= (page_id << 12);

      // step2 update the LRU bit
      vm->invert_page_table[LRU_index] &= 0xfffff000;
      vm->invert_page_table[LRU_index] |= *(vm->head_num_pt);
      *(vm->head_num_pt) = LRU_index;

      // step3 update the fff LRU
      int k;
      for (k = 0; k < vm->PT_ENTRIES; k++)
      {
        if ((vm->invert_page_table[k] & 0x80000000) == 0x0)
        {
          int current_LRU = (vm->invert_page_table[k] & 0xfff);
          if (current_LRU == LRU_index)
          {
            vm->invert_page_table[k] &= 0xfffff000;
            vm->invert_page_table[k] |= 0xfff;
            break;
          }
        }
      }
      // step4 swap the page from secondary memory to physical memory
      // 4-1 look into the swap table
      int st_index = 0;
      for (k = 0; k < vm->ST_ENTRIES; k++)
      {
        if ((vm->swap_table[k] & 0x8000) == 0x8000) // empty entries
        {
          st_index = k;
          break;
        }
      }
      vm->swap_table[st_index] &= 0x7fff; // clear the invalid bit
      vm->swap_table[st_index] &= 0x8000;
      // printf("page id = %x\n", origin_page_id);
      vm->swap_table[st_index] |= origin_page_id;
      for (int t = 0; t < vm->PAGESIZE; t++)
      {
        vm->swap_space[st_index * vm->PAGESIZE + t] = vm->physical_mem[LRU_index * (vm->PAGESIZE) + t];
      }
      // step5 write into the physical memory

      physical_addr = (LRU_index << 5) | offset;
      vm->physical_mem[physical_addr] = value;
      // printf("write addr %d and page fault %d\n", LRU_index, *vm->pagefault_num_ptr);
      // printf("jiaohuanchenggong %c\n\n", vm->physical_mem[LRU_index * vm->PAGESIZE]);
    }
  }
  return;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size)
{
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = offset; i < offset + input_size; i++)
  {
    results[i - offset] = vm_read(vm, i);
  }
}
