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
#include <string>
#include <sstream>
using namespace std;

struct pid_ppid {
	int pid;
	int ppid;
	string name;
	struct pid_ppid *parent;
	vector<pid_ppid *> sons;
	// char status[8];
};

vector<int> thread_list;
vector<pid_ppid> process_list;
void print_result(FILE *fp);
int get_pid();
void add_to_thread(int pid);
void create_tree();
void print_tree(struct pid_ppid *node, vector<int> rec, bool direct,
		int show_num);

int main(int argc, char *argv[])
{
	int show_num;
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "-p") == 0)
			show_num = 0;
		else if (strcmp(argv[i], "-c") == 0)
			show_num = 1;
		// else if (strcmp(argv[i], "-g") == 0) show_num = 2;
		else {
			puts("errof command");
			exit(0);
		}
	}
	int pid_num = get_pid();

	// create tree
	create_tree();
	struct pid_ppid root_process;
	root_process = process_list[0];
	// struct pid_ppid son1 = (*root_process.sons[0]);
	// cout << son1.pid << endl;
	// struct pid_ppid son2 = (*root_process.sons[15]);
	// cout << son2.pid << endl;
	// struct pid_ppid son3 = (*son2.sons[0]);
	// cout << son3.name.length() << endl;
	vector<int> rec;
	cout << "-----------------------------------------" << endl;
	print_tree(&root_process, rec, 1, show_num);
}

void create_tree()
{
	for (int i = 0; i < process_list.size(); i++) {
		// cout << process_list[i].name;
		// find the parent node
		int pid = process_list[i].pid;
		int ppid = process_list[i].ppid;
		if (pid != 1) {
			for (int j = 0; j < process_list.size(); j++) {
				if (ppid == process_list[j].pid) {
					process_list[i].parent =
						&process_list[j];
					break;
				}
			}
		}
		// find the sons node
		for (int j = 0; j < process_list.size(); j++) {
			if (pid == process_list[j].ppid) {
				process_list[i].sons.push_back(
					&process_list[j]);
			}
		}
	}
}

void print_tree(struct pid_ppid *node, vector<int> rec, bool direct,
		int show_num)
{
	if (show_num == 0) { // pstree -g
		int len = node->name.length();
		int pid = node->pid;
		stringstream ss;
		ss << pid;
		rec.push_back(len + 3 + 2 + ss.str().length());
		if (direct) {
			cout << "+--" << node->name << "(" << ss.str() << ")";
		} else {
			if (rec.size() > 1) { //    |    |    +--node
				for (int i = 0; i < rec.size() - 2; i++) {
					if (i == 0) {
						for (int j = 0; j < rec[i];
						     j++) {
							cout << " ";
						}
					} else {
						for (int j = 1; j < rec[i];
						     j++) {
							cout << " ";
						}
					}
					cout << "|";
				}
				int clp;
				if (rec.size() == 2) {
					for (int k = 0; k < rec[rec.size() - 2];
					     k++) {
						cout << " ";
					}
				} else {
					for (int k = 1; k < rec[rec.size() - 2];
					     k++) {
						cout << " ";
					}
				}
				// cout <<endl <<  clp << "serssssssssssssssssssssss" << endl;
			}
			cout << "+--" << node->name << "(" << ss.str() << ")";
		}
		if (node->sons.empty()) {
			cout << endl;
		} else {
			for (int i = 0; i < node->sons.size(); i++) {
				if (i == 0) {
					print_tree(node->sons[i], rec, 1,
						   show_num);
				} else {
					print_tree(node->sons[i], rec, 0,
						   show_num);
				}
			}
		}
	} else if (show_num == 1) { // pstree -c
		int len = node->name.length();
		rec.push_back(len + 3);
		if (direct) {
			cout << "+--" << node->name;
		} else {
			if (rec.size() > 1) {
				for (int i = 0; i < rec.size() - 2; i++) {
					if (i == 0) {
						for (int j = 0; j < rec[i];
						     j++) {
							cout << " ";
						}
					} else {
						for (int j = 1; j < rec[i];
						     j++) {
							cout << " ";
						}
					}
					cout << "|";
				}
				int clp;
				if (rec.size() == 2) {
					for (int k = 0; k < rec[rec.size() - 2];
					     k++) {
						cout << " ";
					}
				} else {
					for (int k = 1; k < rec[rec.size() - 2];
					     k++) {
						cout << " ";
					}
				}
				// cout <<endl <<  clp << "serssssssssssssssssssssss" << endl;
			}
			cout << "+--" << node->name;
		}
		if (node->sons.empty()) {
			cout << endl;
		} else {
			for (int i = 0; i < node->sons.size(); i++) {
				if (i == 0) {
					print_tree(node->sons[i], rec, 1,
						   show_num); // direct
				} else {
					print_tree(node->sons[i], rec, 0,
						   show_num); // non-direct
				}
			}
		}
	} else if (show_num == 2) { // pstree -p
		int len = node->name.length();
		int pid = node->ppid;
		stringstream ss;
		ss << pid;
		rec.push_back(len + 3 + 2 + ss.str().length());
		if (direct) {
			cout << "+--" << node->name << "(" << ss.str() << ")";
		} else {
			if (rec.size() > 1) { //    |    |    +--node
				for (int i = 0; i < rec.size() - 2; i++) {
					if (i == 0) {
						for (int j = 0; j < rec[i];
						     j++) {
							cout << " ";
						}
					} else {
						for (int j = 1; j < rec[i];
						     j++) {
							cout << " ";
						}
					}
					cout << "|";
				}
				int clp;
				if (rec.size() == 2) {
					for (int k = 0; k < rec[rec.size() - 2];
					     k++) {
						cout << " ";
					}
				} else {
					for (int k = 1; k < rec[rec.size() - 2];
					     k++) {
						cout << " ";
					}
				}
				// cout <<endl <<  clp << "serssssssssssssssssssssss" << endl;
			}
			cout << "+--" << node->name << "(" << ss.str() << ")";
		}
		if (node->sons.empty()) {
			cout << endl;
		} else {
			for (int i = 0; i < node->sons.size(); i++) {
				if (i == 0) {
					print_tree(node->sons[i], rec, 1,
						   show_num);
				} else {
					print_tree(node->sons[i], rec, 0,
						   show_num);
				}
			}
		}
	}
}

// execute the "cat state" command and read the second and the fourth result
// void exe_cmd(int pid, char* pid_name, char* ppid, char* status){
void exe_cmd(int pid, char *pid_name, char *ppid)
{
	char str[8];
	sprintf(str, "%d", pid);
	char dir[32] = "cd /proc/";
	strcat(dir, str);
	FILE *cmd_pt = NULL;
	strcat(dir, "; more status");
	cmd_pt = popen(dir, "r");
	if (!cmd_pt) {
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

void add_to_thread(int pid)
{
	char str_pid[8];
	sprintf(str_pid, "%d", pid);
	char dir[32] = "cd /proc/";
	strcat(dir, str_pid);
	strcat(dir, "/task; ls");
	FILE *cmd_pt = NULL;
	cmd_pt = popen(dir, "r");
	if (!cmd_pt) {
		perror("popen");
		exit(EXIT_FAILURE);
	}
	char buf[64];
	int pre = -1;
	thread_list.push_back(0);
	while (1) {
		fgets(buf, sizeof(buf) - 1, cmd_pt); // name
		int pid = atoi(buf);
		if (pre == pid) {
			break;
		};
		thread_list.push_back(pid);

		pre = pid;
	}

	pclose(cmd_pt);
}

int get_pid()
{
	int pid_num = 0;
	DIR *dir;
	dir = opendir("/proc");
	struct dirent *ptr;
	while (ptr = readdir(dir)) {
		//
		int pid = atoi(ptr->d_name);
		if (pid != 0) {
			add_to_thread(pid);
			char pid_name[32];
			char ppid_string[32];
			// char status[32];
			// cout << pid << endl;
			exe_cmd(pid, pid_name, ppid_string);
			sscanf(pid_name, "%*s\t%s", pid_name);
			sscanf(ppid_string, "%*s\t%s", ppid_string);
			int ppid = atoi(ppid_string);
			struct pid_ppid temp;
			temp.pid = pid;
			temp.ppid = ppid;
			temp.name = string(pid_name);
			process_list.push_back(temp);
			pid_num++;
		} else {
			continue;
		}
	}
	int len = thread_list.size();
	int pre_pid = -10;
	int fake_pid = 0;
	for (int i = 0; i < len; i++) {
		int pid = thread_list[i];
		// 0 pid thread_pid thread_pid
		if (pid == 0) { // ignore 0 and continue
			fake_pid = thread_list[i + 1];
			i++; // ignore pid
			continue;
		}
		char pid_name[32];
		char ppid_string[32];
		// char status[32];

		exe_cmd(pid, pid_name, ppid_string);
		sscanf(pid_name, "%*s\t%s", pid_name);
		sscanf(ppid_string, "%*s\t%s", ppid_string);
		int ppid = atoi(ppid_string);
		struct pid_ppid temp;
		temp.pid = pid;
		temp.ppid = fake_pid;
		string temp_name;
		temp_name = "{";
		temp_name.append(string(pid_name));
		temp_name.append("}");
		temp.name = temp_name;
		process_list.push_back(temp);
		pid_num++;
		pre_pid = pid;
	}
	return pid_num;
}

// xxx--+--xxx--+--xxx
//              +--xxx

//      +--xxx--+--xxx
