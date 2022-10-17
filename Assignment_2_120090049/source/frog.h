#ifndef FROG_H
#define FROG_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#include <iostream>
using namespace std;


#define ROW 10
#define COLUMN 50 

class Frog {

public:
    int x;
    int y;
    char (*map)[COLUMN];
    
	Frog(char mapp[ROW+10][COLUMN]) {
        this->map = mapp;
    }

    void init_frog(int x, int y)
    {
        this->x = x;
        this->y = y; // y means the actual line which frog is in
        this->map[x][y] = '0';
        // printf("update!\n");
        // cout << map <<endl;
    }

    void up(){
        this->y --;
    }
    void down(){
        this->y ++;
    }
    void left(){
        this->x --;
    }
    void right(){
        this->x ++;
    }

    int move(char key){
        // int pre_x, pre_y;
        // pre_x = this->x;
        // pre_y = this->y;
        switch (key)
        {
            case 'W':
            case 'w':
                this->up();
                break;

            case 'S':
            case 's':
                this->down();
                break;
            
            case 'A':
            case 'a':
                this->left();
                break;
            
            case 'D':
            case 'd':
                this->right();
                break;
            default:
                break;
        }

        if (this->y == 10) // one the back
        {
            this->map[this->x][this->y] = '0';
            return 0;
        }
        if (this->y == 0) // reach the other side
        {
            this->map[this->x][this->y] = '0';
            return 1;
        }
    }
};


#endif //CSC3150_ASSIGNMENT_2_FROG_H
