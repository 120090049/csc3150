#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>



__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {

u32 fp = fs_open(fs, "32-block-0", G_WRITE);
    fs_write(fs, input, 32, fp);
    for (int j = 0; j < 1023; ++j) {
        char tag[] = "1024-block-????";
        int i = j;
        tag[11] = static_cast<char>(i / 1000 + '0');
        i = i % 1000;
        tag[12] = static_cast<char>(i / 100 + '0');
        i = i % 100;
        tag[13] = static_cast<char>(i / 10 + '0');
        i = i % 10;
        tag[14] = static_cast<char>(i + '0');
        fp = fs_open(fs, tag, G_WRITE);
        fs_write(fs, input + j * 1024, 1024, fp);
    }
    fs_gsys(fs, RM, "32-block-0");
    // now it has one 32byte at first, 1023 * 1024 file in the middle
    printf_VCB(fs, 0, 31);
    printf_VCB(fs, 32, 63);

    fp = fs_open(fs, "1024-block-1023", G_WRITE);
    printf("triggering gc\n");
    fs_write(fs, input + 1023 * 1024, 1024, fp);

    // fs_gsys(fs, LS_D);
    // for (int j = 0; j < 1024; ++j) {
    //     char tag[] = "1024-block-????";
    //     int i = j;
    //     tag[11] = static_cast<char>(i / 1000 + '0');
    //     i = i % 1000;
    //     tag[12] = static_cast<char>(i / 100 + '0');
    //     i = i % 100;
    //     tag[13] = static_cast<char>(i / 10 + '0');
    //     i = i % 10;
    //     tag[14] = static_cast<char>(i + '0');
    //     fp = fs_open(fs, tag, G_READ);
    //     fs_read(fs, output + j * 1024, 1024, fp);
    // }
    printf("Done!\n");
	
	return;

}

	// printf_VCB(fs, 0, 20);
	// for (int i=0; i<13; i++){
	// 	printf_fcb(fs, i);

	// }
	// printf("\n\n\n"); 