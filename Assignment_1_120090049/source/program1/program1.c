#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <wait.h>

int main(int argc, char *argv[])
{
	/* fork a child process */
	int state;
	printf("Process start to fork\n");
	pid_t pid = fork();

	if (pid == -1) {
		perror("fork error!");
		exit(1);
	} else {
		if (pid == 0) {
			int i;
			char *arg[argc];
			printf("I'm the Child Process, my pid = %d\n",
			       getpid());
			arg[argc - 1] = NULL;
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			printf("Child process start to execute test program:\n");
			execve(arg[0], arg, NULL);
		}
		/* wait for child process terminates */
		else {
			printf("I'm the Parant Process, my pid = %d\n",
			       getpid());
			waitpid(-1, &state, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");
			// Normal exit
			if (WIFEXITED(state)) {
				int signal_num = WEXITSTATUS(state);
				printf("Normal termination with EXIT STATUS = %d\n",
				       signal_num);
			}
			// terminating signal
			else if (WIFSIGNALED(state)) {
				int signal_num = WTERMSIG(state);
				switch (signal_num) {
				// ./program1 ./abort #6
				case 6:
					printf("child process get SIGABRT signal\n");
					break;
				// ./program1 ./alarm #14
				case 14:
					printf("child process get SIGALRM signal\n");
					// printf("This is the SIGALRM signal\n");
					break;
				// ./program1 ./bus   #7
				case 7:
					printf("child process get SIGBUS signal\n");
					// printf("This is the SIGBUS signal\n");
					break;
				// ./program1 ./floating #8
				case 8:
					printf("child process get SIGFPE signal\n");
					// printf("This is the SIGFPE signal\n");
					break;
				// ./program1 ./hangup #1
				case 1:
					printf("child process get SIGHUP signal\n");
					// printf("This is the SIGHUP signal\n");
					break;
				// ./program1 ./illegal_instr #4
				case 4:
					printf("child process get SIGILL signal\n");
					// printf("This is the SIGILL signal\n");
					break;
				// ./program1 ./interrupt #2
				case 2:
					printf("child process get SIGINT signal\n");
					// printf("This is the SIGINT signal\n");
					break;
				// ./program1 ./kill #9
				case 9:
					printf("child process get SIGKILL signal\n");
					// printf("This is the SIGKILL signal\n");
					break;
				// ./program1 ./pipe #13
				case 13:
					printf("child process get SIGPIPE signal\n");
					// printf("This is the SIGPIPE signal\n");
					break;
				// ./program1 ./quit #3
				case 3:
					printf("child process get SIGQUIT signal\n");
					// printf("This is the SIGQUIT signal\n");
					break;
				// ./program1 ./segment_fault #11
				case 11:
					printf("child process get SIGSEGV signal\n");
					// printf("This is the SIGSEGV signal\n");
					break;
				// ./program1 ./terminate #15
				case 15:
					printf("child process get SIGTERM signal\n");
					// printf("This is the SIGTERM signal\n");
					break;
				// ./program1 ./trap #5
				case 5:
					printf("child process get SIGTRAP signal\n");
					// printf("This is the SIGTRAP signal\n");
					break;
				}
			}
			// stop signal
			else if (WIFSTOPPED(state)) {
				printf("child process get SIGSTOP signal\n");
			} else {
				printf("continue\n");
			}
			exit(0);
		}
	}
	return 0;
}
