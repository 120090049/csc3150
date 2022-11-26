#include <stdlib.h>
#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>

void printf_bin(int num)
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
}

int main(){
	
    for (int j = 0; j < 1023; ++j) {  // 1024-block-0000 --> 1024-block-1022
        char tag[] = "1024-block-????";
        int i = j;
        tag[11] = static_cast<char>(i / 1000 + '0');
        i = i % 1000;
        tag[12] = static_cast<char>(i / 100 + '0');
        i = i % 100;
        tag[13] = static_cast<char>(i / 10 + '0');
        i = i % 10;
        tag[14] = static_cast<char>(i + '0');
        printf("%s\n", tag);
    }
    // fs_gsys(fs, RM, "32-block-0");
    // // now it has one 32byte at first, 1023 * 1024 file in the middle

    // fp = fs_open(fs, "1024-block-1023", G_WRITE);
    // printf("triggering gc\n");
    // fs_write(fs, input + 1023 * 1024, 1024, fp);


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
	
	printf("DOne\n");
}