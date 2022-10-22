#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

static thread_pool_t thread_pool;

static void* worker_thread(void* nothing) // worker_thread is like bank counter clerk
{
    while(1){
        pthread_mutex_lock(&thread_pool.queue_lock);
        while(!thread_pool.head){ // if there is no works arrived, the clerk will suspend
            pthread_cond_wait(&thread_pool.task_arrive, &thread_pool.queue_lock);
        }
        work_t* work;
        work = thread_pool.head; // The first customer(request) walks to the counter (ready to be served)
        thread_pool.head = work->next; // The next customer becomes the first customer in the queue
        pthread_mutex_unlock(&thread_pool.queue_lock);
        work->func(work->args); // start to serve customer
        free(work); // service is done, the customer leaves
   }
    return NULL;
}

void async_init(int num_threads) {  // initialize the pool
    pthread_t threads_in_queue[num_threads];
    thread_pool.head = NULL;
    pthread_mutex_init(&(thread_pool.queue_lock),NULL); 
    pthread_cond_init(&(thread_pool.task_arrive),NULL);
    for(int i = 0; i < num_threads; i++){
        pthread_create(&threads_in_queue[i], NULL, worker_thread, NULL);  // create worker_thread
    }
    return;
}

void async_run(void (*hanlder)(int), int args) {  // add work to the thread pool

    work_t* work = (work_t*)malloc(sizeof(work_t));
    work->func = hanlder;
    work->args = args;
    work->next = NULL;
    pthread_mutex_lock(&thread_pool.queue_lock);

    work_t *end = thread_pool.head; // add work to the thread poll
    if (end == NULL){
        thread_pool.head = work;
    }
    else 
    {
        while (end->next) end = end->next; // go to the end of the thread poll
        end->next = work; // add the work to the thread poll
    }

    pthread_cond_signal(&thread_pool.task_arrive); // notify that the new task has arrived
    pthread_mutex_unlock(&thread_pool.queue_lock);
    return NULL;
}

// void async_run(void (*hanlder)(int), int args) {  // add work to the thread pool
    
//     work_t work;
//     work.func = hanlder;
//     work.args = args;
//     work.next = NULL;
   
//     pthread_mutex_lock(&thread_pool.queue_lock);

//     work_t *end = thread_pool.head; // add work to the thread poll
//     if (end == NULL){
//         thread_pool.head = &work;
//     }
//     else 
//     {
//         while (end->next) end = end->next; // go to the end of the thread poll
//         end->next = &work; // add the work to the thread poll
//     }

//     pthread_cond_signal(&thread_pool.task_arrive); // notify that the new task has arrived
//     pthread_mutex_unlock(&thread_pool.queue_lock);
//     return NULL;
// }