#ifndef TEST2_H
#define TEST2_H
#include <string>
#include <iostream>
#include "test1.h"
using namespace clp;
class Student : public Person
{
public:
    void Print2()
    {
        this->num ++;
        std::cout<<" 学号:"<<this->num<<endl;
    }

};
#endif