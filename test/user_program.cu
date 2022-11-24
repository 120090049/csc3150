#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>



__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {


	char name[20] = "clp\0";
	
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	printf_bin(fp);
	printf_fcb(fs, 0);


// FCB
	// printf("--------------------\n");
	// fcb_set_name(fs, 10, name);
	// fcb_set_validbit(fs, 10);
	// fcb_clear_validbit(fs, 10);
	// fcb_set_dir(fs, 10);
	// fcb_set_start(fs, 10, 1024);
	
	// fcb_set_size(fs,10, 123456789);
	// u32 gtime = 5;
	// fcb_set_createtime(fs, 10, gtime);
	// gtime ++;
	// fcb_set_modifytime(fs,10, gtime);

	// fcb_clear(fs, 10);
	// printf_fcb(fs, 10);
	
	

// VCB
	// int* super_block_ptr = (int*) &fs->volume[0]; // every 4 is a VCB
	// printf_bin(*super_block_ptr);
	// vcb_set(fs, 0);
	// printf_bin(*super_block_ptr);
	// vcb_set(fs, 0);
	// vcb_set(fs, 0);
	// printf_bin(*super_block_ptr);
	// vcb_clear(fs, 0);
	// printf_bin(*super_block_ptr);
	// printf("%d\n", vcb_get(fs, 0));
	// printf("%d\n", vcb_get(fs, 0));
	// vcb_clear(fs, 0);
	// printf_bin(*super_block_ptr);

	// printf("%d\n", vcb_get(fs, 0));
	// printf("%d\n", vcb_get(fs, 0));
	// vcb_set(fs, 0);
	// printf("%d\n", vcb_get(fs, 0));
	// printf("Done\n");
	return;

}
