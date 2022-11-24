#include <stdlib.h>
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
    int a = 258;	
    printf_bin(a);
}