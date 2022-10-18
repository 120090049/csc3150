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
    // int (*logs_pos)[2];

	Frog(char mapp[ROW+10][COLUMN], int log[9][2]) {
        this->map = mapp;
        // this->logs_pos = log;
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
        this->x --;
    }
    void down(){
        this->x ++;
    }
    void left(){
        this->y --;
    }
    void right(){
        this->y ++;
    }
    // x-row y-column
    int move(char key){
        int pre_x, pre_y;
        pre_x = this->x;
        pre_y = this->y;
        switch (key)
        {
            case 'W':
            case 'w':
                if (this->x > 0){
                    this->up();
                }
                break;

            case 'S':
            case 's':
                if (this->x < 10){
                    this->down();
                }
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

        if (this->x == 10) // one the back
        {
            this->map[pre_x][pre_y] = '|';
            this->map[this->x][this->y] = '0';
            return 0;
        }
        else if (this->x == 0) // reach the other side and WIN !!!
        {
            this->map[this->x][this->y] = '0';
            return 1;
        }
        else{

        }
    }
};


#endif //CSC3150_ASSIGNMENT_2_FROG_H
