#include "gemm.h"
#include <stdio.h>

// Helper function to get the 2D index in row-major order
__device__ int idx2d(int row, int col, int width) {
    return row * width + col;
}

// Kernel 1: Basic matmul from the assignment
// D = α * A * B + β * C
__global__ void matmul_kernel(
    int m, int n, int k,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    const float* C,
    float* D
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        
        for (int q = 0; q < k; ++q) {
            float a_val = A[row * k + q];
            float b_val = B[q * n + col];
            sum += a_val * b_val;
        }
        
        // D = alpha * A * B + beta * C
        D[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// Kernel 2: Extended GEMM with transpose support
// C ← α * op(A) * op(B) + β * C
__global__ void gemm_kernel(
    int m, int n, int k,
    float alpha,
    const float* A, bool transposeA,
    const float* B, bool transposeB,
    float beta,
    float* C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        
        for (int q = 0; q < k; ++q) {
            float a_val, b_val;
            
            // Get A element (with optional transpose)
            if (transposeA) {
                // A^T: access A[q][row] instead of A[row][q]
                a_val = A[q * m + row];
            } else {
                // A: access A[row][q]
                a_val = A[row * k + q];
            }
            
            // Get B element (with optional transpose)
            if (transposeB) {
                // B^T: access B[col][q] instead of B[q][col]
                b_val = B[col * k + q];
            } else {
                // B: access B[q][col]
                b_val = B[q * n + col];
            }
            
            sum += a_val * b_val;
        }
        
        // In-place update: C = alpha * (A op) * (B op) + beta * C
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

// Wrapper function for basic matmul
void launch_matmul(
    int m, int n, int k,
    float alpha,
    const float* d_A,
    const float* d_B,
    float beta,
    const float* d_C,
    float* d_D,
    cudaStream_t stream
) {
    // Validate dimensions
    if (m <= 0 || n <= 0 || k <= 0) {
        printf("Error: Invalid matrix dimensions\n");
        return;
    }
    
    // Set up thread blocks
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);
    
    // Launch the kernel
    matmul_kernel<<<gridSize, blockSize, 0, stream>>>(
        m, n, k, alpha, d_A, d_B, beta, d_C, d_D
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

// Wrapper function for extended GEMM
void gemm(
    int m, int n, int k,
    float alpha,
    const float* d_A, bool transposeA,
    const float* d_B, bool transposeB,
    float beta,
    float* d_C,
    cudaStream_t stream
) {
    // Calculate actual dimensions based on transpose flags
    int A_rows = transposeA ? k : m;
    int A_cols = transposeA ? m : k;
    int B_rows = transposeB ? n : k;
    int B_cols = transposeB ? k : n;
    
    // Validate dimensions
    if (A_cols != B_rows) {
        printf("Error: Dimension mismatch. A_cols (%d) != B_rows (%d)\n", A_cols, B_rows);
        return;
    }
    
    if (m <= 0 || n <= 0 || k <= 0) {
        printf("Error: Invalid matrix dimensions\n");
        return;
    }
    
    // Set up thread blocks
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);
    
    // Launch the kernel
    gemm_kernel<<<gridSize, blockSize, 0, stream>>>(
        m, n, k,
        alpha,
        d_A, transposeA,
        d_B, transposeB,
        beta,
        d_C
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}