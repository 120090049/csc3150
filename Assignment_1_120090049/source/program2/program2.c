#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>
#include <linux/signal.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

static struct task_struct *mythread;
static struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int notask_error;
};
static struct kernel_clone_args;
// export from the kernel
extern pid_t kernel_clone(struct kernel_clone_args *args); // kernel/fork.c
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);
extern struct filename *getname_kernel(const char *filename);
extern long do_wait(struct wait_opts *wo);

int myfork(void *);
int myexe(void);
void mywait(pid_t);

//implement fork function
int myfork(void *argc)
{
	// initialize the date structure related to signal handling
	// the handler keeps track of the shared pending signals
	struct k_sigaction *ksa_handler = &current->sighand->action[0];
	int k;
	for (k = 0; k < _NSIG; k++) {
		ksa_handler->sa.sa_handler = SIG_DFL;
		ksa_handler->sa.sa_flags = 0;
		ksa_handler->sa.sa_restorer = NULL;
		sigemptyset(&ksa_handler->sa.sa_mask);
		ksa_handler++;
	}
	// STEP 3 Fork a process
	/* fork a process using kernel_clone or kernel_thread */
	// initialize the kernel clone arguments
	struct kernel_clone_args kc_args = {
		// include in /include/linux/sched/task
		.flags =
			SIGCHLD, // How to clone the child process. When executing fork(), it is set as SIGCHILD.
		.stack =
			(unsigned long)&myexe, // Specifies the location of the stack used by the child process.
		.stack_size = 0, //Normally set as 0 because it is unused.
		.parent_tid =
			NULL, // Used for clone() to point to user space memory in parent process address space. It is set as NULL when executing fork();
		.child_tid =
			NULL, // Used for clone() to point to user space memory in child process address space. It is set as NULL when executing fork();
		.tls = 0, // Set thread local storage.
		.exit_signal = SIGCHLD,
	};

	/* execute a test program in child process */
	// go to STEP 4 (child process executes the test program)
	pid_t pid = kernel_clone(
		&kc_args); // Fork successfully: pid of child process
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       (int)current->pid);

	// STEP 5 wait until child process terminates
	mywait(pid);

	return 0;
}

// STEP 4 child process executes the test program
int myexe()
{
	msleep(1);
	int output;
	// char *file_path =
	// 	"/home/vagrant/csc3150/Assignment_1_120090049/source/program2/test"; 
	char *file_path = "/tmp/test"; // pointer of the file path
	struct filename *my_file_name = getname_kernel(file_path); 
	printk("[program2] : child process\n");
	output = do_execve(my_file_name, NULL, NULL);
	return 0;
}

void mywait(pid_t pid)
{
	static int status;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WUNTRACED | WEXITED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user *)&status;
	wo.wo_rusage = NULL;

	int a;
	a = do_wait(&wo);

	switch (wo.wo_stat) {
	// ./program1 ./abort #6
	case 134:
		printk("[program2] : get SIGTERM signal\n");
		break;
	// ./program1 ./alarm #14
	case 14:
		printk("[program2] : get SIGALARM signal\n");
		break;
	// ./program1 ./bus   #7
	case 135:
		printk("[program2] : get SIGBUS signal\n");
		break;
	// ./program1 ./floating #8
	case 136:
		printk("[program2] : get SIGFPE signal\n");
		break;
	// ./program1 ./hangup #1
	case 1:
		printk("[program2] : get SIGHUP signal\n");
		break;
	// ./program1 ./illegal_instr #4
	case 132:
		printk("[program2] : get SIGILL signal\n");
		break;
	// ./program1 ./interrupt #2
	case 2:
		printk("[program2] : get SIGINT signal\n");
		break;
	// ./program1 ./kill #9
	case 9:
		printk("[program2] : get SIGKILL signal\n");
		break;
	// ./program1 ./pipe #13
	case 13:
		printk("[program2] : get SIGPIPE signal\n");
		break;
	// ./program1 ./quit #3
	case 131:
		printk("[program2] : get SIGQUIT signal\n");
		break;
	// ./program1 ./segment_fault #11
	case 139:
		printk("[program2] : get SIGSEGV signal\n");
		break;
	// ./program1 ./terminate #15
	case 15:
		printk("[program2] : get SIGTERM signal\n");
		break;
	// ./program1 ./trap #5
	case 133:
		printk("[program2] : get SIGTRAP signal\n");
		break;
	case 25600:
		printk("[program2] : This is the normal program\n");
		break;
	case 4991:
		printk("[program2] : get SIGSTOP signal\n");
		break;
	}

	printk("[program2] : child process terminated\n");
	printk("[program2] : The return signal is %d\n", (wo.wo_stat));

	return;
}

static int __init program2_init(void)
{
	// STEP 1 kernel module initializes
	printk("[program2] : module_init Lingpeng Chen 120090049\n");
	printk("[program2] : module_init create kthread start\n");

	// STEP 2 create kernel thread
	/* create a kernel thread to run myfork */
	struct task_struct *myThread;
	myThread = kthread_create(&myfork, NULL, "MyThread");
	// wake up new thread if ok
	if (!IS_ERR(myThread)) {
		printk("[program2] : module_init kthread start\n");
		wake_up_process(myThread);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : module_exit./my\n");
}

module_init(program2_init);
module_exit(program2_exit);
