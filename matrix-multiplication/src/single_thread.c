#include <stdio.h>

void matmul_single(int *A, int *B, int *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] = 0;
            for (int k = 0; k < K; k++) {
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}
