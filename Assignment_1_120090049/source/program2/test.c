#include <unistd.h>
#include <stdio.h>
#include <signal.h>


int main(int argc,char* argv[]){
	int i=0;

	printf("--------USER PROGRAM--------\n");

	// SIGABRT 134
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGABRT program\n\n");
	// abort();
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// SIGALARM 14
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGALRM program\n\n");
	// alarm(2);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// SIGBUS 135
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGBUS program\n\n");
	// raise(SIGBUS);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");
	
	// // SIGFPE // 136
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGFPE program\n\n");
	// raise(SIGFPE);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");
	// raise(SIGFPE);
	
	// // SIGHUP 1
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGHUP program\n\n");
	// raise(SIGHUP);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// // SIGILL 132
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGILL program\n\n");
	// raise(SIGILL);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// SIGINT 2
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGINT program\n\n");
	// raise(SIGINT);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// SIGKILL 9
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGKILL program\n\n");
	// raise(SIGKILL);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");


	// SIGPIPE 13
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGPIPE program\n\n");
	// raise(SIGPIPE);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// SIGQUIT 131
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGQUIT program\n\n");
	// raise(SIGQUIT);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");

	// SIGSEGV 139
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGSEGV program\n\n");
	// raise(SIGSEGV);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");


	// SIGTERM  15
	printf("------------CHILD PROCESS START------------\n");
	printf("This is the SIGTERM program\n\n");
	raise(SIGTERM);
	sleep(5);
	printf("------------CHILD PROCESS END------------\n");

	// SIGTRAP 133
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGTRAP program\n\n");
	// raise(SIGTRAP);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");
	
	
	// normal program 25600
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the normal program\n\n");
	// printf("------------CHILD PROCESS END------------\n");


	// SIGSTOP
	// printf("------------CHILD PROCESS START------------\n");
	// printf("This is the SIGSTOP program\n\n");
	// raise(SIGSTOP);
	// sleep(5);
	// printf("------------CHILD PROCESS END------------\n");
	
	return 100;
}
