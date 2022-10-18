#ifndef LOGS_H
#define LOGS_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50

class Logs {
public:
    char (*map)[COLUMN];
	int (*logs_pos)[2];

	Logs(char mapp[ROW+10][COLUMN], int log[9][2]) {
        this->map = mapp;
        this->logs_pos = log;
	}
    
    void init_logs(void){
        int i;
        for (i = 1; i<=ROW; i++){
            int length = rand() % 9 + 8;
            int position = rand() % COLUMN;
            this->logs_pos[i-1][0] = length;
            this->logs_pos[i-1][1] = position;
            // this->logs_pos[i-1][0] = 9;
            // this->logs_pos[i-1][1] = 48;
        }
        return;
    }
    
    // 0-48 and 0=49
    void move_logs() {
        /*  Move the logs  */
        // we only need to update the logs_pos
        
        int i; 
        // 0 right 1 left
        for( i = 0; i < 9; ++i ) {
            if (i % 2 == 1) // left
            {
                this->logs_pos[i][1] -= 1;
                if (this->logs_pos[i][1] < 0 ) this->logs_pos[i][1] += 49;
            }
            else // move right
            {
                this->logs_pos[i][1] += 1;
                if (this->logs_pos[i][1] >= 49 ) this->logs_pos[i][1] -= 49;
            }
        }
        return;
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
    }
    
};


#endif 
