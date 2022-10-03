#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <wait.h>


int main(int argc, char *argv[]){

	/* fork a child process */
	// char buf[50] = "Original test strings";
	int state;
	printf("Process start to fork\n");
	pid_t pid = fork();
	

	/* execute test program */ 
	if (pid == -1){
		perror("fork error!");
		exit(1);
	}
	else{
		if (pid==0){
			// strcpy(buf, "Test string are updated by child");
			int i;
			char *arg[argc];
			// printf("I'm the Child Process, my pid = %d, and my ppid = %d\n", getpid(), getppid());
			printf("I'm the Child Process, my pid = %d\n", getpid());
			arg[argc-1]=NULL;
			for (i=0; i<argc-1; i++){
				arg[i] = argv[i+1];
			}

			printf("Child process start to execute test program:\n");
			execve(arg[0], arg, NULL);
			
			// raise(SIGCHLD);
			// SIGCHLD 信号(17号信号—非可靠信号):
			// 僵尸进程: 子进程退出后, 操作系统发送 SIGCHLD 信号给父进程, 但是因为 SIGCHLD 信号的默认处理方式就是忽略, 因此在之前的程序中并没有感受到操作系统的通知, 因此只能固定的使用进程等待来避免产生僵尸进程, 但是在这个过程中父进程是一直阻塞的, 只能一直等待子进程退出
		}
		/* wait for child process terminates */
		else{
			printf("I'm the Parant Process, my pid = %d\n", getpid());
			
			waitpid(-1, &state, WUNTRACED); 
			// WUNTRACED reports on stopped child processes as well as terminated ones.

			printf("Parent process receives SIGCHLD signal\n");
			// Normal exit
			//WEXITSTATUS是一个检验子进程退出的正常还是非正常和返回值的宏
			//WIFEXITED(status) 这个宏用来指出子进程是否为正常退出的，如果是，它会返回一个非零值。
			//WEXITSTATUS(status) 当WIFEXITED返回非零值时，可以用这个宏来提取子进程的返回值，如果子进程调用exit(5)退出，WEXITSTATUS(status)就会返回5；
			//如果子进程调用exit(7)，WEXITSTATUS(status)就会返回7。请注意，如果进程不是正常退出的，也就是说，WIFEXITED返回0，这个值就毫无意义。
			if (WIFEXITED(state)){
				int signal_num = WEXITSTATUS(state);
				printf("Normal termination with EXIT STATUS = %d\n", signal_num);
			}

			// terminating signal
			else if(WIFSIGNALED(state)){
				int signal_num = WTERMSIG(state);
				// printf("Abnoraml exit signal: %d\n", signal_num);
				switch (signal_num){
					// ./program1 ./abort #6
					case 6:
						// printf("This is the SIGABRT signal\n");
						break;
					// ./program1 ./alarm #14
					case 14:
						// printf("This is the SIGALRM signal\n");
						break;
					// ./program1 ./bus   #7
					case 7:
						// printf("This is the SIGBUS signal\n");
						break;
					// ./program1 ./floating #8
					case 8:
						// printf("This is the SIGFPE signal\n");
						break;
					// ./program1 ./hangup #1
					case 1:
						// printf("This is the SIGHUP signal\n");
						break;
					// ./program1 ./illegal_instr #4
					case 4:
						// printf("This is the SIGILL signal\n");
						break;
					// ./program1 ./interrupt #2
					case 2:
						// printf("This is the SIGINT signal\n");
						break;
					// ./program1 ./kill #9
					case 9:
						// printf("This is the SIGKILL signal\n");
						break;
					// ./program1 ./pipe #13
					case 13:
						// printf("This is the SIGPIPE signal\n");
						break;
					// ./program1 ./quit #3
					case 3:
						// printf("This is the SIGQUIT signal\n");
						break;
					// ./program1 ./segment_fault #11
					case 11:
						// printf("This is the SIGSEGV signal\n");
						break;
					// ./program1 ./terminate #15
					case 15:
						// printf("This is the SIGTERM signal\n");
						break;
					// ./program1 ./trap #5
					case 5:
						// printf("This is the SIGTRAP signal\n");
						break;
					
				}
			}
			else if(WIFSTOPPED(state)){ // stop signal
                printf("child process stopped\n");            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
			exit(0);
		}
	}
	
	/* check child process termination status */
	
	return 0;
	
}
