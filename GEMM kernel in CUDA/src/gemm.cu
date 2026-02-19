#include "gemm.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>

__global__ void gemm_naive_kernel(
    int m, int n, int k,
    float alpha,
    const float* __restrict__ A, bool tA,
    const float* __restrict__ B, bool tB,
    float beta,
    float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, n)

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    // op(A): m x k, op(B): k x n
    for (int p = 0; p < k; ++p) {
        // A_elem = op(A)[row, p]
        float a = tA ? A[p * m + row]  // A stored as k x m
                     : A[row * k + p]; // A stored as m x k

        // B_elem = op(B)[p, col]
        float b = tB ? B[col * k + p]  // B stored as n x k
                     : B[p * n + col]; // B stored as k x n

        sum += a * b;
    }

    float oldC = C[row * n + col];
    C[row * n + col] = alpha * sum + beta * oldC;
}

void gemm_cuda(
    int m, int n, int k,
    float alpha,
    const float* A, bool transposeA,
    const float* B, bool transposeB,
    float beta,
    float* C)
{
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    gemm_naive_kernel<<<grid, block>>>(
        m, n, k,
        alpha,
        A, transposeA,
        B, transposeB,
        beta,
        C
    );
    CUDA_CHECK(cudaGetLastError());
}
