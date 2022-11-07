#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

// printf("------------------\n");
// for (int i = 0; i < 100; i++)
// {
//   printf("%x\n", vm->swap_table[i]);
// }
// printf("------------------\n");

// for (int i = 0; i < input_size / 4; i++)
// {
//   results[i] = vm->physical_mem[i];
// }
// for (int i = 0; i < input_size * 3 / 4; i++)
// {
//   results[i + input_size / 4] = vm->swap_space[i];
// }

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//                              int input_size)
// {
//   for (int i = 0; i < input_size; i++)
//     vm_write(vm, i, input[i]);

//   for (int i = 0; i < input_size / 4; i++)
//   {
//     int result = vm_read(vm, i);
//   }

//   for (int i = 0; i < input_size / 4; i++)
//     results[i] = vm->physical_mem[i];

//   // for (int i = 0; i < input_size / 4; i++)
//   //   results[i + 1 * input_size / 4] = vm->swap_space[i + 1 * input_size / 4];
//   // for (int i = 0; i < input_size / 4; i++)
//   //   results[i + 2 * input_size / 4] = vm->swap_space[i + 2 * input_size / 4];
//   // for (int i = 0; i < input_size / 4; i++)
//   //   results[i + 3 * input_size / 4] = vm->swap_space[i + 0 * input_size / 4];

//   printf("------------------\n");
//   for (int i = 1020; i < 1024; i++)
//   {
//     printf("%x\n", (vm->invert_page_table[i] & 0x0ffff000) >> 12);
//   }
//   printf("------------------\n");

//   printf("------------------\n");
//   for (int i = 0; i < 4; i++)
//   {
//     printf("%x\n", vm->swap_table[i]);
//   }
//   printf("------------------\n");
// }

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//                              int input_size)
// {
//   for (int i = 0; i < input_size; i++)
//     vm_write(vm, i, input[i]);

//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
//   for (int i = input_size - 1; i >= input_size - 32769; i--)
//     int value = vm_read(vm, i);
//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
//   // int value = vm_read(vm, 0);
//   // value = vm_read(vm, 32);
//   for (int i = input_size / 4; i < input_size / 2; i++)
//   {
//     int result = vm_read(vm, i);
//     printf("%d\n", i);
//   }
//   // vm_snapshot(vm, results, 0, input_size);
//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);

//   printf("------------------\n");
//   for (int i = 1020; i < 1024; i++)
//   {
//     printf("%x\n", (vm->invert_page_table[i] & 0x0ffff000) >> 12);
//   }
//   printf("------------------\n");

//   printf("------------------\n");
//   for (int i = 0; i < 4; i++)
//   {
//     printf("%x\n", vm->swap_table[i]);
//   }
//   printf("------------------\n");
// }

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//                              int input_size)
// {
//   // write the data.bin to the VM starting from address 32*1024 (1K-5K)
//   for (int i = 0; i < input_size; i++)
//     vm_write(vm, 32 * 1024 + i, input[i]);
//   printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
//   // write (1K-2K) data  to the VM starting from 0  write to (0-1K)
//   for (int i = 0; i < 32 * 1023; i++)
//     vm_write(vm, i, input[i + 32 * 1024]);
//   printf("pagefault number is %d\n", *vm->pagefault_num_ptr);

//   // page_table
//   // for (int i = 0; i < 1024; i++)
//   // {
//   //   int index = (vm->invert_page_table[i] &= 0x0ffff000) >> 12;
//   //   printf("index = %d\n", index);
//   // }
//   for (int i = 0; i < input_size; i++)
//   {
//     int result = vm_read(vm, i + 32 * 1024);
//   }

//   // vm_snapshot(vm, results, 32 * 1024, input_size / 4);
//   // printf("-----------------------\n\n");
//   // for (int i = 0; i < 1024; i++)
//   // {
//   //   int index = (vm->invert_page_table[i] &= 0x0ffff000) >> 12;
//   //   printf("index = %d\n", index);
//   // }

//   // swap_table
//   // printf("------------------\n");
//   // for (int i = 0; i < 4096; i++)
//   // {
//   //   printf("%d\n", vm->swap_table[i]);
//   // }

//   // printf("------------------\n");
//   // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
//   // printf("-----------------------\n\n");
//   // vm_snapshot(vm, results, 32 * 1024, input_size / 4);
//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
//   // vm_snapshot(vm, results, input_size / 2, input_size / 4);
//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
//   // vm_snapshot(vm, results, 3 * input_size / 4, input_size / 4);
//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
//   // vm_snapshot(vm, results, input_size, input_size / 4);
//   // printf("pagefault number is %d\n", *vm->pagefault_num_ptr);
// }

// // expected page fault num: 9215

// // printf("physical_addr = %d\n", physical_addr);
// // printf("value = %x\n", vm->physical_mem[physical_addr]);
// // printf("LRU: %d\n\n", (vm->invert_page_table[empty_index] & 0xfff));
// // printf("*(vm->pagefault_num_ptr) = %d\n\n\n", *(vm->pagefault_num_ptr));

// // __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
// //                              int input_size)
// // {
// //   for (int i = 0; i < input_size; i++)
// //     vm_write(vm, i, input[i]);

// //   for (int i = input_size - 1; i >= input_size - 32769; i--)
// //     int value = vm_read(vm, i);

// //   vm_snapshot(vm, results, 0, input_size);
// // }

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size)
{
  // write the data.bin to the VM starting from address 32*1024
  for (int i = 0; i < input_size; i++)
    vm_write(vm, 32 * 1024 + i, input[i]);
  // write (32KB-32B) data  to the VM starting from 0
  for (int i = 0; i < 32 * 1023; i++)
    vm_write(vm, i, input[i + 32 * 1024]);
  // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
  vm_snapshot(vm, results, 32 * 1024, input_size);
}