#include <stdlib.h>
#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>

int strlen(char str[])
{
	char *p = str;
	int count = 0;
	while (*p++ != '\0')
	{
		count++;
	}
	return count+1;
}

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
bool cmp_str(char *str1, char *str2)
{
  while (*str1 != '\0' && *str2 != '\0' && *str1 == *str2)
  {
    str1++;
    str2++;
  }
  if (*str1 == '\0' && *str2 == '\0')
  {
    return true;
  }
  else
  {
    return false;
  }
}
int main(){
	
  char name1[10] = "abc\0";
char name2[10] = "abc";
char* str1 = name1;
char* str2 = name2;
	while (*str1 != '\0' && *str2 != '\0' && *str1 == *str2)
  {
    str1++;
    str2++;
  }
  if (*str1 == '\0' && *str2 == '\0')
  {
    printf("T");
  }
  else
  {
    printf("F");
  }
}