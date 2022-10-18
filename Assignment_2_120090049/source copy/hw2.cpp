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

#include "game.h"
#include "logs.h"

#define ROW 10
#define COLUMN 50 
#define TIME 50000

pthread_mutex_t map_mut;
pthread_mutex_t key_board_mut;
pthread_cond_t end_game;

static char map[ROW+10][COLUMN] ; 
static int log[9][2];

Game *game = new Game(map, log);

// struct Node{
// 	int x , y; 
// 	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
// 	Node(){} ; 
// } frog ; 



// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){

	/*  Move the logs  */
	while (true){
		// printf("1\n");
		pthread_mutex_lock(&map_mut);
		game->logs->move_logs();
		game->logs->update_logs();
		
		game->update_screen();
		pthread_mutex_unlock(&map_mut);
		usleep(TIME);
	}
	pthread_exit(NULL);
	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */


	/*  Print the map on the screen  */

}

void *keyboard_ctr(void *t){

	while (true){
		// printf("2\n");
		if (kbhit()){
			pthread_mutex_lock(&map_mut);
			char key = getchar();
			game->frog->move(key);
			pthread_mutex_unlock(&map_mut);
		}
		else
		{
			game->frog->move(' ');
			continue;
		}
		// printf("3\n");
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){
	pthread_t thread_logs;
	pthread_t thread_keyboard;
	pthread_attr_t attr;
	pthread_mutex_init(&map_mut, NULL);
	// pthread_mutex_init(&key_board_mut, NULL);
	pthread_cond_init(&end_game, NULL);
	// Initialize the river map and frog's starting position
	if (true){
		memset( map , 0, sizeof( map ) ) ;
		int i , j ; 
		for( i = 1; i < ROW; ++i ){	
			for( j = 0; j < COLUMN - 1; ++j )	
				map[i][j] = ' ' ;  
		}	

		for( j = 0; j < COLUMN - 1; ++j )	
			map[ROW][j] = map[0][j] = '|' ;

		for( j = 0; j < COLUMN - 1; ++j )	
			map[0][j] = map[0][j] = '|' ;

		// frog = Node( ROW, (COLUMN-1) / 2 ) ; // 10 
		// map[ROW][(COLUMN-1) / 2] = '0' ; 
	}

	

	
	//Print the map into screen
	game->initlize_game();
	// pthread_attr_init(&attr);

	/*  Create pthreads for wood move and frog control.  */
	pthread_create(&thread_keyboard, NULL, keyboard_ctr, NULL);
	pthread_create(&thread_logs, NULL, logs_move, NULL);
	
	/*  Display the output for user: win, lose or quit.  */
	pthread_cond_wait(&end_game, &map_mut);
	pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&map_mut);
    pthread_cond_destroy(&end_game);
	return 0;

}
