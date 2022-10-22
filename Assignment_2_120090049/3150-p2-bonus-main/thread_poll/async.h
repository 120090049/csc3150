#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>


typedef struct my_item {
  void (*func) (int);
  int args;
  struct my_item *next;
} work_t;

typedef struct my_queue {
  work_t *head;
  pthread_cond_t task_arrive;    
  pthread_mutex_t queue_lock;  
} thread_pool_t;

void async_init(int);
void async_run(void (*fx)(int), int args);

#endif
