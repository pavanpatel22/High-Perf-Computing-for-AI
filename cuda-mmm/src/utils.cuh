#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CUDA_CHECK(call) do {                              \
  cudaError_t err = (call);                                \
  if (err != cudaSuccess) {                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
    std::exit(1);                                          \
  }                                                        \
} while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline void print_device() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("GPU: %s | SMs: %d | CC: %d.%d | GlobalMem: %.2f GB\n",
         p.name, p.multiProcessorCount, p.major, p.minor,
         (double)p.totalGlobalMem / (1024.0*1024.0*1024.0));
}

inline double gflops_sgemm(int m, int n, int k, double ms) {
  // SGEMM FLOPs: 2*m*n*k
  double flops = 2.0 * (double)m * (double)n * (double)k;
  return (flops / 1e9) / (ms / 1e3);
}

struct GPUTimer {
  cudaEvent_t start{}, stop{};
  GPUTimer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
  ~GPUTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void tic() { CUDA_CHECK(cudaEventRecord(start)); }
  float toc_ms() {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};
