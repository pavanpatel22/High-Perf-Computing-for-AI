#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void matmul_single(int*, int*, int*, int, int, int);
void matmul_parallel(int*, int*, int*, int, int, int, int);

void check(int M, int K, int N) {
    int *A = malloc(M*K*sizeof(int));
    int *B = malloc(K*N*sizeof(int));
    int *C1 = malloc(M*N*sizeof(int));
    int *C2 = malloc(M*N*sizeof(int));

    for (int i = 0; i < M*K; i++) A[i] = rand() % 5;
    for (int i = 0; i < K*N; i++) B[i] = rand() % 5;

    matmul_single(A, B, C1, M, K, N);
    matmul_parallel(A, B, C2, M, K, N, 4);

    for (int i = 0; i < M*N; i++) {
        if (C1[i] != C2[i]) {
            printf("Mismatch detected\n");
            exit(1);
        }
    }

    free(A); free(B); free(C1); free(C2);
}

int main() {
    check(1,1,1);
    check(1,1,5);
    check(2,1,3);
    check(2,2,2);
    check(5,3,4);
    check(10,10,10);

    printf("All correctness tests passed.\n");
    return 0;
}
