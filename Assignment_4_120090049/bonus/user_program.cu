﻿#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>


__device__ void printf_list(int* list) 
{
	for (int i=0; i<50; i++) {
		printf("%d  ", list[i]);
	}
	printf("\n----------------------\n");
	return;
}
__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {

	// /////////////////////// Bonus Test Case ///////////////
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys(fs, LS_D);  //1
	fs_gsys(fs, LS_S);	//4
	fs_gsys(fs, MKDIR, "app\0");
	fs_gsys(fs, LS_D); //7
	fs_gsys(fs, LS_S); //11
	fs_gsys(fs, CD, "app\0");
	fs_gsys(fs, LS_S); //15  /app
	fp = fs_open(fs, "a.txt\0", G_WRITE);
	fs_write(fs, input + 128, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 256, 32, fp);
	fs_gsys(fs, MKDIR, "soft\0"); 
	fs_gsys(fs, LS_S);	//16 /app
	fs_gsys(fs, LS_D);	//20
	fs_gsys(fs, CD, "soft\0");
	fs_gsys(fs, PWD);   //24  /app/soft
	fp = fs_open(fs, "A.txt\0", G_WRITE);  
	fs_write(fs, input + 256, 64, fp);
	fp = fs_open(fs, "B.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fp = fs_open(fs, "C.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fp = fs_open(fs, "D.txt\0", G_WRITE);
	fs_write(fs, input + 256, 1024, fp);
	fs_gsys(fs, LS_S);  //25 B.txt 1024 C.txt 1024 D.txt 1024 A.txt 64
	fs_gsys(fs, CD_P);  //  /app
	fs_gsys(fs, LS_S);  //30  a.txt 64  b.txt 32   soft 24 d
	fs_gsys(fs, PWD);   //34 /app
	fs_gsys(fs, CD_P);  
	fs_gsys(fs, LS_S);	//35  t.txt 32  b.txt 32   app 17 d
	fs_gsys(fs, CD, "app\0"); // cd /app
	// fs_gsys(fs, RM_RF, "soft\0");
	// fs_gsys(fs, LS_S);  //39
	// fs_gsys(fs, CD_P);
	// fs_gsys(fs, LS_S);  //42

// 	u32 fp = fs_open(fs, "a.txt\0", G_WRITE);
// 	fs_write(fs, input, 3, fp);
	
// 	fp = fs_open(fs, "b.txt\0", G_WRITE);
// 	fs_write(fs, input, 3, fp);



// 	fs_gsys(fs, RM, "b.txt\0");


// 	fs_gsys(fs, PWD); 
	
// 	fs_gsys(fs, MKDIR, "clp\0");

// 	fs_gsys(fs, CD, "clp\0");
// 	fs_gsys(fs, PWD); 

// 	fs_gsys(fs, MKDIR, "clp1\0");
// 	printf_fcb (fs, 1);
// 	printf_fcb (fs, 2);
// 	printf_fcb (fs, 3);

// 	fs_gsys(fs, CD, "clp1\0");
	
// 	fs_gsys(fs, PWD); 

// 	fs_gsys(fs, CD_P); 
// fs_gsys(fs, PWD); 
// 	printf("Done!\n");
	return;

}
