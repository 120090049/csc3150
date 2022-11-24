#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>



__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {


// 	char name[20] = "clp\0";
// 	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
// 	printf_bin(fp);
// 	fs_write(fs, input, 64, fp);
// 	printf_fcb(fs, 0);
// printf("--------------------\n");
// 	fp = fs_open(fs, "b.txt\0", G_WRITE);
// 	printf_bin(fp);
// 	fs_write(fs, input, 64, fp);
// 	printf_fcb(fs, 1);

// printf("--------------------\n");
// 	fp = fs_open(fs, "t.txt\0", G_WRITE);
// 	printf_bin(fp);
// 	fs_write(fs, input, 64, fp);
// 	printf_fcb(fs, 0);

// 	printf("%d\n", vcb_get(fs, 0));
// 	printf("%d\n", vcb_get(fs, 1));
// 	printf("%d\n", vcb_get(fs, 2));
// 	printf("%d\n", vcb_get(fs, 3));
// 	printf("%d\n", vcb_get(fs, 4));
// 	printf("%d\n", vcb_get(fs, 5));
// 	printf("%d\n", vcb_get(fs, 6));
// 	printf("%d\n", vcb_get(fs, 7));
// 	printf("%d\n", vcb_get(fs, 8));
// 	uchar a[64];
// 	uchar b[32];
// 	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);  // t 0-64  0 1   0 0
// 	fs_write(fs, input, 64, fp);
// 	printf_fcb(fs, 0);
// 	printf_VCB(fs, 0, 10);
// printf("--------------------\n");

// 	fp = fs_open(fs, "b.txt\0", G_WRITE);   	// b 32-64  2   1 1
// 	fs_write(fs, input + 32, 32, fp);
// 	printf_fcb(fs, 1);
// 	printf_VCB(fs, 0, 10);
// printf("--------------------\n");

// 	fp = fs_open(fs, "t.txt\0", G_WRITE);    // t 32-64     0   0 2
// 	fs_write(fs, input + 32, 32, fp);
// 	printf_fcb(fs, 0);
// 	printf_VCB(fs, 0, 10);
// printf("--------------------\n");

// 	fp = fs_open(fs, "t.txt\0", G_READ);   // t read 32-64  0   0 2
// 	fs_read(fs, a, 32, fp);
// 	printf_fcb(fs, 0);
// 	printf_VCB(fs, 0, 10);
// 	// printf("read result:\n %s\n", a);
// printf("--------------------\n");
// 	// fs_gsys(fs,LS_D);
// 	// fs_gsys(fs, LS_S);

// 	fp = fs_open(fs, "b.txt\0", G_WRITE);  // b      64-76  1   1 3
// 	fs_write(fs, input + 64, 12, fp);
// 	printf_fcb(fs, 1);
// 	printf_VCB(fs, 0, 10);
// printf("--------------------\n");



	return;

}
