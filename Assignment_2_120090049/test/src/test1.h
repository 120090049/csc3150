#ifndef TEST1_H
#define TEST1_H

#include <string>
#include <iostream>
#include "test2.h"
namespace clp
{
    class Person
    {
    public :
        int num = 111; // 身份证号
        Student *student;
        Person() {
            student = new Student();
        }
        void Print1()
        {
            std::cout<<" 学号:"<<this->num<<endl;
        }
    };

}

#endif