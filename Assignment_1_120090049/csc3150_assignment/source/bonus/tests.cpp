#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
int main(void)
{   
    int clp = 10;
    stringstream ss;
    ss << clp;
    cout << ss.str().length();
}

