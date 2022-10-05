# include <stdio.h>
# include <stdlib.h>
void main (void)
{
    int num = 112340;
    char str[8];
    sprintf(str, "%d", num);
    printf("%s", str);
}