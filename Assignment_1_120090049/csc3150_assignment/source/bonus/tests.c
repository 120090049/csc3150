#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
 
void print_result(FILE *fp)
{
        char buf[100];
 
        if(!fp) {
                return;
        }
        printf("\n>>>\n");
        // while(memset(buf, 0, sizeof(buf)), fgets(buf, sizeof(buf) - 1, fp) != 0 ) {
        //         printf("%s", buf);
        // }
        fgets(buf, sizeof(buf) - 1, fp);
        printf("%s", buf);
        fgets(buf, sizeof(buf) - 1, fp);
        printf("%s", buf);
        printf("\n<<<\n");
}
 
int main(void)
{   
    // printf("----")
    // printf("|")
    // ─ ├ ┬ └
    
}
void test01()
{    //1.使用strtok()实现分割
	char str[] = "(world)";
	char* str1 = strtok(str, "("); // 0
    str1 = strtok(str1, ")"); // 0
    // str1 = strtok(NULL, " "); // 1
	printf("%s\n", str1);


	 
	
}
