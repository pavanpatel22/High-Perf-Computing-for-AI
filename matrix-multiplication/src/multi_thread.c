#include <pthread.h>
#include <stdlib.h>

typedef struct {
    int *A, *B, *C;
    int M, K, N;
    int row_start, row_end;
} ThreadData;

void* worker(void *arg) {
    ThreadData *d = (ThreadData*)arg;

    for (int i = d->row_start; i < d->row_end; i++) {
        for (int j = 0; j < d->N; j++) {
            d->C[i*d->N + j] = 0;
            for (int k = 0; k < d->K; k++) {
                d->C[i*d->N + j] +=
                    d->A[i*d->K + k] *
                    d->B[k*d->N + j];
            }
        }
    }
    return NULL;
}

void matmul_parallel(int *A, int *B, int *C,
                     int M, int K, int N, int threads) {

    pthread_t tids[threads];
    ThreadData data[threads];

    int rows = M / threads;

    for (int t = 0; t < threads; t++) {
        data[t] = (ThreadData){
            A, B, C, M, K, N,
            t * rows,
            (t == threads - 1) ? M : (t + 1) * rows
        };
        pthread_create(&tids[t], NULL, worker, &data[t]);
    }

    for (int t = 0; t < threads; t++) {
        pthread_join(tids[t], NULL);
    }
}
