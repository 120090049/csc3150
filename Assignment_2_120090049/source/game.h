#ifndef GAME_H
#define GAME_H

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


class Game {

public:
	int state; //0 alive //1 died //2 quit //3 win
    char (*map)[COLUMN];
	int logs_pos[9][2];

	// frog position
	int x, y;


	Game(char mapp[ROW+10][COLUMN]) {
        this->map = mapp;
		state = 0; //0 alive //1 died //2 quit //3
	}

	void initlize_game(void){
		this->x = 10;
		this->y = 24;
		// this->x = ROW;
		// this->y = (COLUMN -2) / 2;
		this->init_logs();
		this->update_logs();
		this->map[ROW][(COLUMN -2) / 2] = '0';
		this->update_screen();
	}

	void update_screen(void){
        // printf("%d", this->num);
        // if (!state){
        printf("\033[2J\033[1;1H");
        int i = 0;
        for( i = 0; i <= ROW; ++i)	
            puts( this->map[i] );
        printf("%d, %d", this->x, this->y);
		return;
	}
	//////////////////////////////////
	// logs
	void init_logs(void){
        int i;
        for (i = 1; i<ROW; i++){
            int length = rand() % 10 + 12;
            int position = rand() % COLUMN;
            this->logs_pos[i-1][0] = length;
            this->logs_pos[i-1][1] = position;
        }
        return;
    }
	
    int move_logs() {
        /*  Move the logs  */
        // we only need to update the logs_pos
        if (!state){
            int i; 
            // 0 right 1 left
            for( i = 0; i < 9; ++i ) {
                if (i % 2 == 1) // left
                {
                    if (this->x == i+1) {
                        this->y --;
                    }
                    this->logs_pos[i][1] -= 1;
                    if (this->logs_pos[i][1] < 0 ) this->logs_pos[i][1] += 49;
                    if (this->y < 0) return 1;
                }
                else // move right
                {
                    if (this->x == i+1) {
                        this->y ++;
                    }
                    this->logs_pos[i][1] += 1;
                    if (this->logs_pos[i][1] >= 49 ) this->logs_pos[i][1] -= 49;
                    if (this->y >=49 ) return 1;
                }
            }
        }
        return 0;
    }

	// this function is used to update the maps according to the logs_pos
    void update_logs(void){
        int i , j ; 
        for( i = 1; i < ROW; ++i ){	
            // for( j = 0; j < COLUMN - 1; ++j )	
            // printf("%d & %d\n", logs_pos[i-1][0], logs_pos[i-1][1]);
            // sleep(1);
            // case1 |   ====  |
            int start = this->logs_pos[i-1][1];
            int end = this->logs_pos[i-1][1] + this->logs_pos[i-1][0];
            
            // clear the previous log
            for( j = 0; j < 49; j++ ){
                this->map[i][j] = ' ';
            }
            
            if (  end < COLUMN ) {
                for( j = start; j < end; j++ ){
                    this->map[i][j] = '=';
                }
            }
            // case2 |==   ====|
            if ( end >= COLUMN) {
                int new_end = end - 49;
                for (j=0; j<new_end; j++){
                    this->map[i][j] = '=';
                }
                for (j=start; j<COLUMN-1; j++){
                    this->map[i][j] = '=';
                }
            }
        }	
        this->map[this->x][this->y] = '0';
        
    }
	///////////////////////////////////
	// frog
	 // x-row y-column
    int frog_move(char key){
        if (!state){
            int pre_x, pre_y;
            pre_x = this->x;
            pre_y = this->y;
            switch (key)
            {
                case 'W':
                case 'w':
                    if (this->x > 0){
                        this->x --;
                    }
                    if (this->map[this->x][this->y] == ' '){
                        this->update_screen();
                        sleep(1);
                        state = 1;
                        return 1;
                    } 
                    break;

                case 'S':
                case 's':
                    if (this->x < 10){
                        this->x ++;
                    }
                    if (this->map[this->x][this->y] == ' ') {
                        this->map[this->x][this->y] == '0';
                        this->update_screen();
                        state = 1;
                        return 1;
                    }
                    break;
                
                case 'A':
                case 'a':
                    this->y --;
                    if (this->map[this->x][this->y] == ' '){
                        this->map[this->x][this->y] == '0';
                        this->update_screen();
                        state = 1;
                        return 1;
                    } 
                    break;
                
                case 'D':
                case 'd':
                    this->y ++;
                    if (this->map[this->x][this->y] == ' '){
                        this->map[this->x][this->y] == '0';
                        this->update_screen();
                        state = 1;
                        return 1;
                    }
                    break;
                default:
                    break;
            }

            this->map[this->x][this->y] = '0';
            if (this->x == 10) // one the back
            {
                this->map[pre_x][pre_y] = '|';
            }
            if (this->x == 9 && pre_x == 10) // leave the back
            {
                this->map[pre_x][pre_y] = '|';
            }
            else if (this->x == 0) // reach the other side and WIN !!!
            {
                this->state = 3;
                return 3;
            }
        }
        return 0;
    }
};


#endif //CSC3150_ASSIGNMENT_2_UTIL_H
