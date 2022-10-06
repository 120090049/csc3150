#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#define BUFFSIZE_INFO 50

struct pid_ppid{
    int pid;
    int ppid;
    char name[32];
};
struct pid_ppid process_list[100];
void print_result(FILE *fp);
int get_pid();

int main(){
    int pid_num = get_pid();
    for (int i=0; i<pid_num; i++){
        printf("pid_name: %s\n", process_list[i].name);
        printf("pid: %d\n", process_list[i].pid);
        printf("ppid: %d\n\n", process_list[i].ppid);
    }
    printf("%d", pid_num);
}


// execute the "cat state" command and read the second and the fourth result
void exe_cmd(int pid, char* pid_name, int* ppid){
 
    char str[8];
    sprintf(str, "%d", pid);
    char dir[32] = "cd /proc/";
    strcat(dir, str);
    FILE *cmd_pt = NULL;
    strcat(dir, "; cat stat");
    cmd_pt = popen(dir, "r");
    if(!cmd_pt) {
        perror("popen");
        exit(EXIT_FAILURE);
    }
    char buf[64];
    fgets(buf, sizeof(buf) - 1, cmd_pt);
    // start to split
    char* temp = strtok(buf, " "); // pid
    temp = strtok(NULL, " "); // (name)

    strcpy(pid_name, temp); // get the name 
    

	// printf("%s\n", pid_name);
	temp = strtok(NULL, " "); // S
    temp = strtok(NULL, " "); // ppid
    *ppid = atoi(temp);

    pclose(cmd_pt);
}

int get_pid()
{   
    int pid_num = 0;
    DIR * dir;
    dir = opendir("/proc");
    struct dirent * ptr;
    while(ptr=readdir(dir))
    {
        //
        int pid=atoi(ptr->d_name);
        if (pid != 0){            
            char pid_name[32];
            int ppid;
            exe_cmd(pid, pid_name, &ppid);

            // remove the "()" of the "(pid_name)"
            char* temp = strtok(pid_name, "("); 
            temp = strtok(temp, ")"); 

            process_list[pid_num].pid = pid;
            strcpy(process_list[pid_num].name, temp);
            process_list[pid_num].ppid = ppid;
            pid_num ++;
        }
        else{
            continue;
        }
    }
    return pid_num;
}



