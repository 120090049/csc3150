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
};
struct pid_ppid process_list[100];
void get_pid();

int main(){
    // get_pid();
    char pid_name[16];
    int ppid;
    exe_cmd(1, pid_name, &ppid);
}

// execute the "cat state" command and read the second and the fourth result
void exe_cmd(int pid, char* pid_name, int* ppid){
    // char buf_ps[1024];   
    // char ps[1024]={0};   
    // FILE *ptr;   
    // strcpy(ps, cmd); 
 
    char str[8];
    sprintf(str, "%d", pid);
    char dir[32] = "/proc/";
    strcat(dir, str);
    printf("%s", dir);   
    // if((ptr=popen(ps, "r"))!=NULL)   
    // {   
    //     while(fgets(buf_ps, 1024, ptr)!=NULL)   
    //     {   
    //        if(strlen(result)>1024)   
    //            break;   
    //     }   
    //     pclose(ptr);   
    //     ptr = NULL;   
    // }   
}

// void get_pid()
// {   
//     pid_num = 0;
//     DIR * dir;
//     dir = opendir("/proc");
//     struct dirent * ptr;
//     while(ptr=readdir(dir_ptr))
//     {
//         pid_num ++;
//         //memcpy(name,direntp->d_name,strlen(direntp->d_name)+1);
//         pid=atoi(ptr->d_name);
//         if(pid!=0)
//         {
//             process_list[pid_num].pid = pid;
            
//             sprintf(pidStr,"%d",pid);
//             strcat(process_path,pidStr);
//             strcat(process_path,stat);
//             //process_path中为指定process的对应stat文件路径

//             int ppid=getPPid(process_path);//返回-1表示解析出错
//             if(ppid!=-1)
//                 processInfos[number_process++].ppid=ppid;
//             else
//                 number_process++;

//             //重置process_path
//             process_path[6]=0;
//             //printf("%s\n",process_path);
//         }
//     }
//     printf("the number of process is %d\n",number_process);
// }



