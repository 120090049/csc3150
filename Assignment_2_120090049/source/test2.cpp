#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_THREADS 3
#define TCOUNT      10
#define COUNT_LIMIT 10

int count = 0;
int thread_ids[3] = {0,1,2};
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

void *inc_count(void *idp){
    int i = 0;
    int taskid = 0;
    int *my_id = (int*)idp;

    for (i=0; i<TCOUNT; i++){
        pthread_mutex_lock(&count_mutex);
        taskid = count;
        count ++;

        if (count == COUNT_LIMIT){
            pthread_cond_signal(&count_threshold_cv);
            printf("Signal!\n");
        }
        printf("inc_count(): thread %d, count = %d, unlocking mutex\n", *my_id, count);
        pthread_mutex_unlock(&count_mutex);
        sleep(1);
    }

    printf("inc_count(): thread %d, Threadhold reached. \n", *my_id);
    pthread_exit(NULL);
}

// void *watch_count(void *idp)
// {
//     int *my_id = (int*)idp;
//     printf("Starting watch_count(): thread %d\n", *my_id);
//     pthread_mutex_lock(&count_mutex);
//     printf("wo jin lai le! \n");
//     while(count < COUNT_LIMIT){
//         printf("wo zai li mian le lai le! \n");
//         pthread_cond_wait(&count_threshold_cv, &count_mutex);
//         printf("watch_count(): thread %d Condition signal received.\n", *my_id);
//     }

//     count += 100;
//     pthread_mutex_unlock(&count_mutex);
//     pthread_exit(NULL);
// }

int main (int argc, char *argv[]){
    int i, rc;
    pthread_t threads[3];
    pthread_attr_t attr;
    
    pthread_mutex_init(&count_mutex, NULL);
    pthread_cond_init(&count_threshold_cv, NULL);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &attr, inc_count, (void *)&thread_ids[0]);
    pthread_create(&threads[1], &attr, inc_count, (void *)&thread_ids[1]);
    // pthread_create(&threads[2], &attr, watch_count, (void *)&thread_ids[2]);
    // for (i=0; i<NUM_THREADS; i++){
    //     pthread_join(threads[i], NULL);
    // }
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
    printf("Main(): Wait on %d threads. threads. Done.\n", NUM_THREADS);

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&count_mutex);
    pthread_cond_destroy(&count_threshold_cv);
    // pthread_exit(NULL);
    
    return 0;
}