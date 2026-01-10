#include <stdio.h>
#include <stdlib.h>
#include "../src/timer.h"

void matmul_parallel(int*, int*, int*, int, int, int, int);

int main() {
    int M = 2048, K = 2048, N = 2048;
    int *A = malloc(M*K*sizeof(int));
    int *B = malloc(K*N*sizeof(int));
    int *C = malloc(M*N*sizeof(int));

    for (int i = 0; i < M*K; i++) A[i] = rand() % 10;
    for (int i = 0; i < K*N; i++) B[i] = rand() % 10;

    int threads[] = {1, 4, 16, 32, 64, 128};

    for (int i = 0; i < 6; i++) {
        double start = now();
        matmul_parallel(A, B, C, M, K, N, threads[i]);
        double end = now();

        printf("Threads: %d | Time: %.3f s\n",
               threads[i], end - start);
    }

    free(A); free(B); free(C);
    return 0;
}
