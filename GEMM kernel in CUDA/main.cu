#include "gemm.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Fill matrix with random values
void fillRandom(float* matrix, int size, float range = 1.0f) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = range * (rand() / (float)RAND_MAX);
    }
}

// Print matrix (for debugging)
void printMatrix(const std::string& name, const float* matrix, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < std::min(rows, 4); ++i) {
        for (int j = 0; j < std::min(cols, 4); ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "...\n";
    }
    std::cout << std::endl;
}

// Compare two matrices (for verification)
bool compareMatrices(const float* A, const float* B, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Test 1: Basic matmul (original version)
void testBasicMatmul() {
    std::cout << "=== Test 1: Basic Matmul (D = αAB + βC) ===\n";
    
    const int M = 32;
    const int N = 32;
    const int K = 32;
    float alpha = 2.0f;
    float beta = 3.0f;
    
    // Host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_D(M * N);
    std::vector<float> h_D_ref(M * N);
    
    // Initialize with random values
    srand(42);
    fillRandom(h_A.data(), M * K);
    fillRandom(h_B.data(), K * N);
    fillRandom(h_C.data(), M * N);
    
    // Reference calculation on CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int q = 0; q < K; ++q) {
                sum += h_A[i * K + q] * h_B[q * N + j];
            }
            h_D_ref[i * N + j] = alpha * sum + beta * h_C[i * N + j];
        }
    }
    
    // Device matrices
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_matmul(M, N, K, alpha, d_A, d_B, beta, d_C, d_D);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result
    if (compareMatrices(h_D.data(), h_D_ref.data(), M * N)) {
        std::cout << "✓ Basic matmul test passed!\n";
    } else {
        std::cout << "✗ Basic matmul test failed!\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
}

// Test 2: Extended GEMM with transpose
void testExtendedGEMM() {
    std::cout << "\n=== Test 2: Extended GEMM with Transpose ===\n";
    
    const int M = 64;
    const int N = 64;
    const int K = 32;
    float alpha = 1.5f;
    float beta = 0.5f;
    
    // Create matrices with compatible dimensions for transposed case
    std::vector<float> h_A(M * K);           // M x K
    std::vector<float> h_B(K * N);           // K x N
    std::vector<float> h_C(M * N);           // M x N
    std::vector<float> h_C_result(M * N);    // Result from GPU
    std::vector<float> h_C_ref(M * N);       // Reference from CPU
    
    // Initialize
    srand(42);
    fillRandom(h_A.data(), M * K);
    fillRandom(h_B.data(), K * N);
    fillRandom(h_C.data(), M * N);
    
    // Test case 1: No transpose (AB)
    std::cout << "Test 2.1: C = α * A * B + β * C\n";
    std::copy(h_C.begin(), h_C.end(), h_C_ref.begin());
    
    // CPU reference for AB
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int q = 0; q < K; ++q) {
                sum += h_A[i * K + q] * h_B[q * N + j];
            }
            h_C_ref[i * N + j] = alpha * sum + beta * h_C_ref[i * N + j];
        }
    }
    
    // GPU computation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel (no transpose)
    gemm(M, N, K, alpha, d_A, false, d_B, false, beta, d_C);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_result.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    if (compareMatrices(h_C_result.data(), h_C_ref.data(), M * N)) {
        std::cout << "  ✓ No transpose test passed!\n";
    } else {
        std::cout << "  ✗ No transpose test failed!\n";
    }
    
    // Test case 2: Transpose A (A^T B)
    std::cout << "Test 2.2: C = α * A^T * B + β * C\n";
    
    // Reset C
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // For A^T B, we need to think about dimensions differently
    // A^T is K x M, B is K x N, so result is M x N
    // But A is M x K, so we access A^T[q][i] = A[i][q]
    
    // CPU reference for A^T B
    std::copy(h_C.begin(), h_C.end(), h_C_ref.begin());
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int q = 0; q < K; ++q) {
                // A^T[q][i] = A[i][q]
                sum += h_A[i * K + q] * h_B[q * N + j];
            }
            h_C_ref[i * N + j] = alpha * sum + beta * h_C_ref[i * N + j];
        }
    }
    
    // GPU computation with A transposed
    gemm(M, N, K, alpha, d_A, true, d_B, false, beta, d_C);
    
    CUDA_CHECK(cudaMemcpy(h_C_result.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (compareMatrices(h_C_result.data(), h_C_ref.data(), M * N)) {
        std::cout << "  ✓ A^T B test passed!\n";
    } else {
        std::cout << "  ✗ A^T B test failed!\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Test 3: Performance test
void testPerformance() {
    std::cout << "\n=== Test 3: Performance Test ===\n";
    
    const int M = 512;
    const int N = 512;
    const int K = 512;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    
    fillRandom(h_A.data(), M * K);
    fillRandom(h_B.data(), K * N);
    fillRandom(h_C.data(), M * N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm up
    gemm(M, N, K, alpha, d_A, false, d_B, false, beta, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iterations; ++i) {
        gemm(M, N, K, alpha, d_A, false, d_B, false, beta, d_C);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_time = milliseconds / iterations;
    float gflops = (2.0f * M * N * K) / (avg_time * 1e6);  // Convert ms to s
    
    std::cout << "Matrix size: " << M << "x" << N << " * " << K << "x" << N << "\n";
    std::cout << "Average time: " << avg_time << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "CUDA GEMM Implementation Assignment\n";
    std::cout << "====================================\n\n";
    
    // Run tests
    testBasicMatmul();
    testExtendedGEMM();
    testPerformance();
    
    std::cout << "\nAll tests completed!\n";
    
    return 0;
}