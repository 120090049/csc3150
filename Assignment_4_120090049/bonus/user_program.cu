#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>


// __device__ void printf_list(int* list) 
// {
// 	for (int i=0; i<50; i++) {
// 		printf("%d  ", list[i]);
// 	}
// 	printf("\n----------------------\n");
// 	return;
// }
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
	// printf_fcb(fs, 4);
	// printf_fcb(fs, 5);
	// printf_fcb(fs, 6);
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

	// printf_ls(fs);
	// printf_fcb(fs, 3);
	// printf_fcb(fs, 4);
	// printf_fcb(fs, 5);
	// fs_gsys(fs, RM, "soft\0");

	fs_gsys(fs, RM_RF, "soft\0");
	fs_gsys(fs, LS_S);  //39
	fs_gsys(fs, CD_P);
	fs_gsys(fs, LS_S);  //42

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
	printf("Done!\n");
	return;

}











//  else if (op == -1) {
//     int gpwd_store[4];
//     for (int i=0; i<4; i++){
//       gpwd_store[i] = gpwd[i];
//     }

//     int dir_fcb_index = -1;
//     for (int i = 0; i < fs->FCB_ENTRIES; i++)
//     {
//       if (pwd_file_is_under_curr_dir_index (fs, i))
//       {
//         if (cmp_str(fcb_get_name(fs, i), file_name) && fcb_check_dir(fs, i)) {
//           dir_fcb_index = i;
//           break;
//         }
//       }
//     }
    
//     if (dir_fcb_index == -1 )
//     {
//       printf("The directory you want to remove do not exists under the current directory!\n");
//     }
//     else {
//       int next_fcb_index = -1;
//       bool tag = false;
//       while (true)
//       {   
//         // before entering the dir to be removed
//         int dir_parent_index = pwd_get(); 
//         int dir_index_tobe_removed = dir_fcb_index;
//         char* dir_name_tobe_removed = fcb_get_name(fs, dir_index_tobe_removed);
//         // check whether to quit
//         if (next_fcb_index != dir_fcb_index && next_fcb_index != -1) {
//           dir_fcb_index = next_fcb_index;
//         }
//         else if (tag){ // there is no more directory need to be removed
//           break;
//         }
        

//         // cd into the directory 
//         for (int i=1; i<4; i++) {
//           if (gpwd[i] == -1) {
//             gpwd[i] = dir_fcb_index;
//             break;
//           }
//         }
//         tag = true;
//         int ls[50];
//         ls_get(fs, ls);
//         for (int i=0; i<50; i++) {
//           int temp_index = ls[i];
//           if (temp_index != 0) {
//             if (fcb_check_dir(fs, temp_index)) // is directory
//             {
//               next_fcb_index = temp_index; // next goes to this dir
//             }
//             else { // is ordinary file
//               // remove the file from the directory fileblocks and update the directory FCB
//               u32 handler = fs_open(fs, fcb_get_name(fs, pwd_get()), G_PWD);
//               fs_write(fs, (uchar*)fcb_get_name(fs, temp_index), 0, handler);

//               int start = fcb_get_start(fs, temp_index);
//               int size = fcb_get_size(fs, temp_index);
//               int occupied_block_num = (size + fs->BLOCK_SIZE - 1) / fs->BLOCK_SIZE;
//               clear_VCB_fileblocks(fs, start, occupied_block_num);
//               fcb_clear(fs, temp_index);
//             }
//           }
//         }
        
//         // remove this directory file
//         // cd back to parent
//         int temp;
//         for (int i=3; i>0; i--) {
//           if (gpwd[i] != -1) {
//             int temp = gpwd[i];
//             gpwd[i] = -1;
//             break;
//           }
//         }
//         u32 handler = fs_open(fs, fcb_get_name(fs, dir_parent_index), G_PWD);
//         fs_write(fs, (uchar*)dir_name_tobe_removed, 0, handler);
        
//         int start = fcb_get_start(fs, dir_index_tobe_removed);
//         int size = fcb_get_size(fs, dir_index_tobe_removed);
//         int occupied_block_num = (size + fs->BLOCK_SIZE - 1) / fs->BLOCK_SIZE;
//         clear_VCB_fileblocks(fs, start, occupied_block_num);
//         fcb_clear(fs, dir_index_tobe_removed);
//         for (int i=1; i<4; i++) {
//           if (gpwd[i] != -1) {
//             gpwd[i] = temp;
//             break;
//           }
//         }
        
//       }
//     }
    
//     for (int i=0; i<4; i++){
//       gpwd[i] = gpwd_store[i];
//       // printf("%d  ", gpwd[i]);
//     }
//   }