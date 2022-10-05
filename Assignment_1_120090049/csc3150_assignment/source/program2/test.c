#include <unistd.h>
#include <stdio.h>
#include <signal.h>


0
int main(int argc,char* argv[]){
	int i=0;

	printf("--------USER PROGRAM--------\n");
	// SIGABRT
	abort();

	// SIGALARM
	// alarm(2);

	// SIGBUS
	raise(SIGBUS);
	
	// SIGHUP 1

	// SIGINT 2
	// SIGQUIT nothing
	// SIGILL
	// SIGTRAP
	// SIGFPE
	// SIGKILL
	// SIGSEGV
	// SIGPIPE
	// SIGTERM  15

	// sleep(5);
	printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 100;
}
