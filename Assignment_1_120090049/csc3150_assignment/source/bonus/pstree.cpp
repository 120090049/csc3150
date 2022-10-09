#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <iostream>
using namespace std;

struct pid_ppid{
    int pid;
    int ppid;
    char name[32];
    // char status[8];
    
};
vector<int> thread_list;

struct pid_ppid process_list[500];
void print_result(FILE *fp);
int get_pid();
void add_to_thread(int pid);
int main(){
    // int pid_num = get_pid();
    add_to_thread(1819);
    // for (int i=0; i<pid_num; i++){
    //     printf("pid_name: %s\n", process_list[i].name);
    //     // printf("pid_status: %s\n", process_list[i].status);
    //     printf("pid: %d\n", process_list[i].pid);
    //     printf("ppid: %d\n\n", process_list[i].ppid);
       
    // }
    // printf("\n%d", pid_num);
}


// execute the "cat state" command and read the second and the fourth result
// void exe_cmd(int pid, char* pid_name, char* ppid, char* status){ 
void exe_cmd(int pid, char* pid_name, char* ppid){
    char str[8];
    sprintf(str, "%d", pid);
    char dir[32] = "cd /proc/";
    strcat(dir, str);
    FILE *cmd_pt = NULL;
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

void add_to_thread(int pid){
    char str_pid[8];
    sprintf(str_pid, "%d", pid);
    char dir[32] = "cd /proc/";
    strcat(dir, str_pid);
    strcat(dir, "/task; ls");
    FILE *cmd_pt = NULL;
    cmd_pt = popen(dir, "r");
    if(!cmd_pt) {
        perror("popen");
        exit(EXIT_FAILURE);
    }
    char buf[64];
    int pre = -1;
    while (1){
      
        fgets(buf, sizeof(buf) - 1, cmd_pt); // name
        int pid=atoi(buf);
        if (pre == pid) {break;};
        pre = pid;
        // if (buf[0] == '\0') { break; }
        // cout << pid << "|" << endl;
        thread_list.pushback
    }
    
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
            sscanf(pid_name, "%*s\t%s",pid_name);
            sscanf(ppid_string, "%*s\t%s",ppid_string);
            int ppid = atoi(ppid_string);
            process_list[pid_num].pid = pid;
            process_list[pid_num].ppid = ppid;
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


