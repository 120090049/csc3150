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
	int (*logs_pos)[2];

	Logs *logs;
	Frog *frog;
	Game(char mapp[ROW+10][COLUMN], int log[9][2]) {
        this->map = mapp;
		this->logs_pos = log;
		state = 0; //0 alive //1 quit //2 died
		logs = new Logs(this->map, this->logs_pos);
		frog = new Frog(this->map, this->logs_pos);
	}

    // Game(int* clp) {
    //     num = *clp;
	// 	state = 0; //0 alive //1 quit //2 died
	// }

	void initlize_game(void){
		this->logs->init_logs();
		this->logs->update_logs();
		this->frog->init_frog(ROW, (COLUMN-2) / 2); // 10 25
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
