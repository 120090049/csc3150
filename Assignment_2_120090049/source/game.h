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

#include "logs.h"
#include "frog.h"

#define ROW 10
#define COLUMN 50


class Game {

public:
	int state;
    char (*map)[COLUMN];
	Logs *logs;
	Game(char mapp[ROW+10][COLUMN]) {
        map = mapp;
		state = 0; //0 alive //1 quit //2 died
		logs = new Logs(this->map);
	}

    // Game(int* clp) {
    //     num = *clp;
	// 	state = 0; //0 alive //1 quit //2 died
	// }

	void initlize_game(void){
		this->logs->init_logs();
		this->logs->update_logs();
		
		this->update_screen();
	}

	void update_screen(void){
        // printf("%d", this->num);
		printf("\033[2J\033[1;1H");
		int i = 0;
		for( i = 0; i <= ROW; ++i)	
			puts( map[i] );
		return;
	}

};


#endif //CSC3150_ASSIGNMENT_2_UTIL_H
