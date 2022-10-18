 // Student的_num和Person的_num构成隐藏关系，可以看出这样代码虽然能跑，但是非常容易混淆
#include <string>
#include <iostream>
using namespace std;
#include "test1.h"
#include "test2.h"





int main(void)
{
    Person s1;
    s1.Print1();
    s1.student->Print2();
    s1.Print1();
};
