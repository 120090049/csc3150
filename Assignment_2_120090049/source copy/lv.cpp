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

//global variabies to see the status of the game
int WIN = 0; //will be 1 if game finish
int LOSE = 0; //will be 1 if game finish
int QUIT = 0; //will be 1 if quit

pthread_mutex_t mutex;

struct Node {
	int x, y;
	Node(int _x, int _y) : x(_x), y(_y){};
	Node(){};
} frog;

char map[ROW + 10][COLUMN];

int end(int x, int y);

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void)
{
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

	if (ch != EOF) {
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void initialize_log()
{
	int len = 15;
	for (int i = 1; i < ROW; i++) {
		int start = rand() % (30); //starting position

		for (int j = start; j < len + start; j++) {
			map[i][j] = '=';
		}
	}
}

void *frog_move(void *t)
{
	while ((!WIN) || (!LOSE) || (!QUIT)) {
		pthread_mutex_lock(&mutex);
		if (kbhit) {
			char ctrl = getchar(); //get the keyboard input
			if (ctrl == '\n')
				ctrl = getchar(); //remove the \n

			if (ctrl == 'A' || ctrl == 'a') { //left
				if (frog.y - 1 >= COLUMN - 1 ||
				    frog.y - 1 < 0 ||
				    map[frog.x][frog.y - 1] == ' ') {
					LOSE = 1;
					break;
				}

				if (frog.x == ROW) { //on the bank
					map[frog.x][frog.y] = '|';
				} else { //on the logs
					map[frog.x][frog.y] = '=';
				}
				frog.y -= 1;
				map[frog.x][frog.y] = '0';
			} else if (ctrl == 'D' || ctrl == 'd') { //right
				if (frog.y + 1 >= COLUMN + 1 ||
				    frog.y + 1 < 0 ||
				    map[frog.x][frog.y + 1] == ' ') {
					LOSE = 1;
					break;
				}
				if (frog.x == ROW) { //on the bank
					map[frog.x][frog.y] = '|';
				} else { //on the logs
					map[frog.x][frog.y] = '=';
				}
				frog.y += 1;
				map[frog.x][frog.y] = '0';
			} else if (ctrl == 'W' || ctrl == 'w') { //up
				if (map[frog.x - 1][frog.y] == ' ') {
					LOSE = 1;
					break;
				} else if (frog.x - 1 == 0) {
					WIN = 1;
					break;
				}
				if (frog.x == ROW) { //on the bank
					map[frog.x][frog.y] = '|';
				} else { //on the logs
					map[frog.x][frog.y] = '=';
				}
				frog.x -= 1;
				map[frog.x][frog.y] = '0';
			} else if (ctrl == 'S' || ctrl == 's') { //down
				if (map[frog.x + 1][frog.y] == ' ' ||
				    frog.x + 1 > ROW) {
					LOSE = 1;
					break;
				}
				if (frog.x == ROW) { //on the bank
					map[frog.x][frog.y] = '|';
				} else { //on the logs
					map[frog.x][frog.y] = '=';
				}
				frog.x -= 1;
				map[frog.x][frog.y] = '0';
			} else if (ctrl == 'Q' || ctrl == 'q') { //quit
				QUIT = 1;
			} else {
				printf("no such a key \n");
			}
			//clean the screen
			system("clear");
			for (int i = 0; i <= ROW; ++i) {
				puts(map[i]);
			}
		}
		pthread_mutex_unlock(&mutex);
	}
	pthread_exit(NULL);
}

void *logs_move(void *t)
{
	while (!WIN || !LOSE || !QUIT) {
		pthread_mutex_lock(&mutex);
		/*  Move the logs  */
		for (int i = 1; i <= ROW; i++) {
			if (i % 2 == 0) { //move right
				for (int j = COLUMN - 2; j >= 0; j--) {
					map[i][j] = map[i][j - 1];
				}
			} else { //left
			}
		}
		pthread_mutex_unlock(&mutex);
	}

	/*  Check keyboard hits, to change frog's position or quit the game. */

	/*  Check game's status  */

	/*  Print the map on the screen  */
}

int main(int argc, char *argv[])
{
	// Initialize the river map and frog's starting position
	memset(map, 0, sizeof(map));
	int i, j;
	for (i = 1; i < ROW; ++i) {
		for (j = 0; j < COLUMN - 1; ++j)
			map[i][j] = ' ';
	}

	for (j = 0; j < COLUMN - 1; ++j)
		map[ROW][j] = map[0][j] = '|';

	for (j = 0; j < COLUMN - 1; ++j)
		map[0][j] = map[0][j] = '|';

	frog = Node(ROW, (COLUMN - 1) / 2); //the frog is in the middle
	map[frog.x][frog.y] = '0'; //change the | to frog

	initialize_log();

	//Print the map into screen
	for (i = 0; i <= ROW; ++i)
		puts(map[i]);

	/*  Create pthreads for wood move and frog control.  */
	pthread_t frog_thread, log_thread; //9 logs and one frog
	pthread_mutex_init(&mutex, NULL);
	int rc;
	long tids[2] = { 0, 1 };
	// long i;

	rc = pthread_create(&log_thread, NULL, logs_move, (void *)tids[0]);
	if (rc) {
		printf("ERROR: CANNOT CREATE LOG_THREAD %d", i);
		exit(1);
	}
	rc = pthread_create(&frog_thread, NULL, frog_move, (void *)tids[1]);
	if (rc) {
		printf("ERROR: CANNOT CREATE FROG_THREAD %d", i);
		exit(1);
	}

	pthread_join(frog_thread, NULL);
	pthread_join(log_thread, NULL);

	/*  Display the output for user: win, lose or quit.  */
	system("clear"); //clear the screen

	if (WIN) {
		printf("You win the game!! \n");
	} else if (LOSE) {
		printf("You lose the game!! \n");
	} else if (QUIT) {
		printf("You exit the game!! \n");
	}

	pthread_mutex_destroy(&mutex);

	pthread_exit(NULL);
	return 0;
}
