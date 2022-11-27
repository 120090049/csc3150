#ifndef FILE_SYSTEM_H
#define FILE_SYSTEM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdio.h>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

#define G_READ 0
#define G_WRITE 1
#define G_PWD 2

// __device__ void fs_gsys(FileSystem *fs, int op);
#define LS_D 0
#define LS_S 1
#define CD_P 5
#define PWD 6

// __device__ void fs_gsys(FileSystem *fs, int op, char *s);
#define RM 2
#define RM_RF 7
#define MKDIR 3
#define CD 4


struct FileSystem
{
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int VOLUME_SIZE;
	int BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
						int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
						int BLOCK_SIZE, int MAX_FILENAME_SIZE,
						int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);



///////////////////

// utils

// functions for pwd
__device__ int pwd_get(void);
__device__ void ls_get(FileSystem *fs, int* list);

// major functions
__device__ void compact(FileSystem *fs);

__device__ int allocate(FileSystem *fs, int num);
__device__ void clear_VCB_fileblocks (FileSystem *fs, int start, int size);

//file blocks
__device__ void fb_clear (FileSystem *fs, int index);

// VCB
__device__ void printf_VCB(FileSystem *fs, int start, int end);
__device__ void printf_bin(int num);
__device__ void vcb_set(FileSystem *fs, int block_index) ;
__device__ void vcb_clear(FileSystem *fs, int block_index) ;
__device__ bool vcb_get(FileSystem *fs, int block_index) ;

// FCB
__device__ int fcb_use_name_retrieve_index(FileSystem *fs, char *name);
__device__ void printf_fcb(FileSystem *fs, int index);
__device__ bool cmp_str(char *str1, char *str2);

__device__ void fcb_clear(FileSystem *fs, int index);
__device__ char* fcb_get_name(FileSystem *fs, int index);
__device__ void fcb_set_name(FileSystem *fs, int index, char *file_name);

__device__ void fcb_set_validbit(FileSystem *fs, int index);
__device__ void fcb_clear_validbit(FileSystem *fs, int index);
__device__ bool fcb_get_validbit(FileSystem *fs, int index);

__device__ void fcb_set_dir(FileSystem *fs, int index);
__device__ bool fcb_check_dir(FileSystem *fs, int index);

__device__ void fcb_set_start(FileSystem *fs, int index, int start);
__device__ int fcb_get_start(FileSystem *fs, int index);

__device__ void fcb_set_size(FileSystem *fs, int index, int size);
__device__ int fcb_get_size(FileSystem *fs, int index);

__device__ void fcb_set_createtime(FileSystem *fs, int index, u32 createtime);
__device__ int fcb_get_createtime(FileSystem *fs, int index);

__device__ void fcb_set_modifytime(FileSystem *fs, int index, u32 modifytime);
__device__ int fcb_get_modifytime(FileSystem *fs, int index);

#endif