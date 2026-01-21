#ifndef GEMM_H
#define GEMM_H

#include <cuda_runtime.h>

// Basic matmul: D = α * A * B + β * C
// Original version from the assignment
void launch_matmul(
    int m, int n, int k,
    float alpha,
    const float* d_A,
    const float* d_B,
    float beta,
    const float* d_C,
    float* d_D,
    cudaStream_t stream = 0
);

// Extended GEMM with transpose support and in-place update
// C ← α * op(A) * op(B) + β * C
void gemm(
    int m, int n, int k,
    float alpha,
    const float* d_A, bool transposeA,
    const float* d_B, bool transposeB,
    float beta,
    float* d_C,
    cudaStream_t stream = 0
);

// CUDA error checking helper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // GEMM_H