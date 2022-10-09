#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>

struct pid_ppid{
    int pid;
    int ppid;
    char name[32];
    // char status[8];
    
};
struct pid_ppid process_list[500];
void print_result(FILE *fp);
int get_pid();

int main(){
    int pid_num = get_pid();
    for (int i=0; i<pid_num; i++){
        printf("pid_name: %s\n", process_list[i].name);
        // printf("pid_status: %s\n", process_list[i].status);
        printf("pid: %d\n", process_list[i].pid);
        printf("ppid: %d\n\n", process_list[i].ppid);
       
    }
    printf("\n%d", pid_num);
}

void add()

// execute the "cat state" command and read the second and the fourth result
void exe_cmd(int pid, char* pid_name, char* ppid){
// void exe_cmd(int pid, char* pid_name, char* ppid, char* status){ 
    char str[8];
    sprintf(str, "%d", pid);
    char dir[32] = "cd /proc/";
    strcat(dir, str);
    FILE *cmd_pt = NULL;
    // strcat(dir, "; cat stat");
    strcat(dir, "; more status");
    cmd_pt = popen(dir, "r");
    if(!cmd_pt) {
        perror("popen");
        exit(EXIT_FAILURE);
    }
    char buf[64];
    fgets(buf, sizeof(buf) - 1, cmd_pt); // name
    strcpy(pid_name, buf);

    fgets(buf, sizeof(buf) - 1, cmd_pt);
    fgets(buf, sizeof(buf) - 1, cmd_pt); // state
    // strcpy(status, buf);
    fgets(buf, sizeof(buf) - 1, cmd_pt);
    fgets(buf, sizeof(buf) - 1, cmd_pt);
    fgets(buf, sizeof(buf) - 1, cmd_pt);
    fgets(buf, sizeof(buf) - 1, cmd_pt); // ppid
    strcpy(ppid, buf);

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
            char ppid_string[32];
            // char status[32];
            exe_cmd(pid, pid_name, ppid_string);

            // pid name
            // printf("%s\n", ppid_string);
            // printf("%s\n", ppid_status);
            sscanf(pid_name, "%*s\t%s",pid_name);

            sscanf(ppid_string, "%*s\t%s",ppid_string);
            int ppid = atoi(ppid_string);

            // sscanf(status, "%*s\t%s",status);
        
            // remove the "()" of the "(pid_name)"
            // char* temp = strtok(pid_name, "("); 
            // temp = strtok(temp, ")"); 

            process_list[pid_num].pid = pid;
            process_list[pid_num].ppid = ppid;
            // strcpy(process_list[pid_num].status, status);
            strcpy(process_list[pid_num].name, pid_name);
            pid_num ++;
        }
        else{
            continue;
        }
        
    }
    return pid_num;
}

// xxx--+--xxx--+--xxx
//              +--xxx

//      +--xxx--+--xxx


