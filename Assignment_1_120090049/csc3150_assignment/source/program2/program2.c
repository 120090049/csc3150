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

MODULE_LICENSE("GPL");

static struct task_struct *mythread;
static struct wait_opts {
	enum pid_type wo_type;
	int	wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int	wo_stat;
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int	notask_error;
};
static struct kernel_clone_args;
// export from the kernel
extern pid_t kernel_clone(struct kernel_clone_args *args); // kernel/fork.c
extern int do_execve(struct filename *filename, const char __user *const __user *__argv, const char __user *const __user *__envp);
extern struct filename *getname_kernel(const char * filename);
extern long do_wait(struct wait_opts *wo);

int my_fork(void*);
int my_exec(void);
void my_wait(pid_t);

//implement fork function
int my_fork(void *argc){
	
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
// STEP 3 Fork a process
	/* fork a process using kernel_clone or kernel_thread */
	// initialize the kernel clone arguments
	struct kernel_clone_args kc_args = { // include in /include/linux/sched/task
		.flags = SIGCHLD, // How to clone the child process. When executing fork(), it is set as SIGCHILD.
		.stack = (unsigned long)&my_exec, // Specifies the location of the stack used by the child process.
		.stack_size = 0, //Normally set as 0 because it is unused.
		.parent_tid = NULL, // Used for clone() to point to user space memory in parent process address space. It is set as NULL when executing fork();
		.child_tid = NULL,  // Used for clone() to point to user space memory in child process address space. It is set as NULL when executing fork();
		.tls = 0, // Set thread local storage.
		.exit_signal = SIGCHLD,
	};
	/* execute a test program in child process */
// go to STEP 4 (child process executes the test program)
	pid_t pid = kernel_clone(&kc_args); // Fork successfully: pid of child process
	// printk("[program2] : The child process has pid= %d\n", getpid());
    // printk("[program2] : The parent process has pid= %d\n", getppid());
	printk("[program2] : The child process has pid= %d\n", pid);
    printk("[program2] : This is the parent process, pid= %d\n", (int) current->pid);

// STEP 5 wait until child process terminates 
	my_wait(pid);

	return 0;
}

// STEP 4 child process executes the test program
int my_exec(){
	int output;
	char* file_path = "/home/vagrant/csc3150/Assignment_1_120090049/csc3150_assignment/source/program2/test"; // pointer of the file path
	struct filename *my_file_name = getname_kernel(file_path);
	// printk("NAME!!!: %d", IS_ERR(my_file_name));
	// const char *const *__argv;
	// const char *const *__envp;
	printk("[program2] : child process");
	output = do_execve(my_file_name, NULL, NULL);
	// printk("OUTPUT!!!: %d", output);
	// return 0;
	if (!output) {
        return 0;
    } else {
        do_exit(output);
    }
}

void my_wait(pid_t pid){
	static int status;
	struct wait_opts wo;
	struct pid *wo_pid=NULL;
	enum pid_type type;
	type=PIDTYPE_PID;
	wo_pid=find_get_pid(pid);

	wo.wo_type=type;
	wo.wo_pid=wo_pid;
	wo.wo_flags=WEXITED;
	wo.wo_info=NULL;
	wo.wo_stat=(int __user*)&status;
	wo.wo_rusage=NULL;

	int a;
	a=do_wait(&wo);
	printk("[program2] : get SIGTERM signal: %d\n", (wo.wo_stat));
	// [ 3769.391604] [program2] : get SIGTERM signal
	// [ 3769.391605] [program2] : child process terminated
	printk("The return value is %d\n", (wo.wo_stat));
	

	put_pid(wo_pid);

	return;
}

static int __init program2_init(void){
// STEP 1 kernel module initializes
	printk("[program2] : module_init Lingpeng Chen 120090049\n");
	printk("[program2] : module_init create kthread start\n");
	
// STEP 2 create kernel thread
	/* create a kernel thread to run my_fork */
	struct task_struct *myThread;
	myThread = kthread_create(&my_fork, NULL, "MyThread");
	// wake up new thread if ok
	if(!IS_ERR(myThread)){
		printk("[program2] : module_init kthread start\n");
        wake_up_process(myThread);
	}
	
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
