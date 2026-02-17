#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cublas_v2.h>

#include "utils.cuh"
#include "kernels.cuh"

static void fill_random(std::vector<float>& x, uint32_t seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& v : x) v = dist(rng);
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    m = std::max(m, (double)std::fabs(a[i] - b[i]));
  }
  return m;
}

// Correct row-major GEMM wrapper for column-major cuBLAS.
// We want: C_row(MxN) = alpha * A_row(MxK) * B_row(KxN) + beta * C_row(MxN)
//
// Trick:
// - A_row (MxK) is stored like A_col^T (KxM) in column-major with ld = K
// - B_row (KxN) is stored like B_col^T (NxK) in column-major with ld = N
// Then compute C_col (NxM) = B_col (NxK) * A_col (KxM)
// Writing that into dC corresponds to C_row (MxN).
static void cublas_gemm(cublasHandle_t h, int M, int N, int K,
                        float alpha, const float* dA, const float* dB, float beta, float* dC)
{
  // Column-major GEMM dimensions:
  // C_col is (N x M)
  // B_col is (N x K)  (this is B_row interpreted as col-major)
  // A_col is (K x M)  (this is A_row interpreted as col-major)
  const int lda = K;  // rows of A_col (KxM)
  const int ldb = N;  // rows of B_col (NxK)
  const int ldc = N;  // rows of C_col (NxM)

  cublasStatus_t st = cublasSgemm(
      h,
      CUBLAS_OP_N, CUBLAS_OP_N,   // IMPORTANT: NO transposes here
      N, M, K,                   // m=N, n=M, k=K
      &alpha,
      dB, ldb,                   // B first
      dA, lda,                   // then A
      &beta,
      dC, ldc);

  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS SGEMM failed (status=%d)\n", (int)st);
    std::exit(1);
  }
}

static void launch_custom(int algo, int M, int N, int K,
                          float alpha, const float* dA, const float* dB, float beta, float* dC)
{
  if (algo == Algo::NAIVE) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, (int)block.x), ceil_div(M, (int)block.y));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    return;
  }
  if (algo == Algo::COALESCED) {
    constexpr int BS = 32;
    dim3 block(BS * BS);
    dim3 grid(ceil_div(N, BS), ceil_div(M, BS));
    sgemm_coalesced<BS><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    return;
  }
  if (algo == Algo::SMEM) {
    constexpr int BS = 32;
    dim3 block(BS * BS);
    dim3 grid(ceil_div(N, BS), ceil_div(M, BS));
    sgemm_smem_tiled<BS><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    return;
  }
  if (algo == Algo::BLOCKTILING_1D) {
    constexpr int TM = 8;   // each thread computes 8 cols
    constexpr int BS = 32;
    constexpr int BN = BS * TM;  // 256 cols per block
    dim3 block(BS * BS);         // 1024 threads
    dim3 grid(ceil_div(N, BN), ceil_div(M, BS));
    sgemm_1d_blocktiling<TM><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    return;
  }
  if (algo == Algo::BLOCKTILING_2D) {
    // Parameters: BM=BN=128, BK=8, TM=TN=8 => 16x16 threads
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);  // (16,16) = 256 threads
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_2d_blocktiling<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    return;
  }
  if (algo == Algo::VECTORIZED) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);  // (16,16)
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    return;
  }

  fprintf(stderr, "Unknown algo=%d\n", algo);
  std::exit(1);
}

static int get_arg_int(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], key, strlen(key)) == 0) {
      const char* eq = strchr(argv[i], '=');
      if (!eq) return def;
      return std::atoi(eq + 1);
    }
  }
  return def;
}
static float get_arg_float(int argc, char** argv, const char* key, float def) {
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], key, strlen(key)) == 0) {
      const char* eq = strchr(argv[i], '=');
      if (!eq) return def;
      return (float)std::atof(eq + 1);
    }
  }
  return def;
}

int main(int argc, char** argv) {
  print_device();

  int M = get_arg_int(argc, argv, "--m", 4096);
  int N = get_arg_int(argc, argv, "--n", 4096);
  int K = get_arg_int(argc, argv, "--k", 4096);
  int algo = get_arg_int(argc, argv, "--algo", 6);
  int iters = get_arg_int(argc, argv, "--iters", 50);
  int warmup = get_arg_int(argc, argv, "--warmup", 10);
  float alpha = get_arg_float(argc, argv, "--alpha", 1.0f);
  float beta  = get_arg_float(argc, argv, "--beta", 0.0f);

  printf("M=%d N=%d K=%d | algo=%d (%s) | iters=%d warmup=%d | alpha=%.3f beta=%.3f\n",
         M, N, K, algo, algo_name(algo), iters, warmup, alpha, beta);

  size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
  size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
  size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

  std::vector<float> hA((size_t)M * (size_t)K), hB((size_t)K * (size_t)N),
                     hC((size_t)M * (size_t)N), hCref((size_t)M * (size_t)N);

  fill_random(hA, 1);
  fill_random(hB, 2);
  fill_random(hC, 3);
  hCref = hC;

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    return 1;
  }

  // Compute reference with cuBLAS into hCref
  {
    float* dCtmp = nullptr;
    CUDA_CHECK(cudaMalloc(&dCtmp, bytesC));
    CUDA_CHECK(cudaMemcpy(dCtmp, hCref.data(), bytesC, cudaMemcpyHostToDevice));
    cublas_gemm(handle, M, N, K, alpha, dA, dB, beta, dCtmp);
    CUDA_CHECK(cudaMemcpy(hCref.data(), dCtmp, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCtmp));
  }

  // Warmup
  for (int i = 0; i < warmup; i++) {
    if (algo == Algo::CUBLAS) cublas_gemm(handle, M, N, K, alpha, dA, dB, beta, dC);
    else launch_custom(algo, M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Reset dC to original before timing
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

  // Timing
  GPUTimer timer;
  timer.tic();
  for (int i = 0; i < iters; i++) {
    if (algo == Algo::CUBLAS) cublas_gemm(handle, M, N, K, alpha, dA, dB, beta, dC);
    else launch_custom(algo, M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = timer.toc_ms();
  float ms_per = ms / iters;

  double gflops = gflops_sgemm(M, N, K, ms_per);
  printf("Time: %.4f ms/iter | Throughput: %.2f GFLOPs\n", ms_per, gflops);

  // Correctness
  std::vector<float> hOut((size_t)M * (size_t)N);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytesC, cudaMemcpyDeviceToHost));

  double mad = max_abs_diff(hOut, hCref);
  printf("Max abs diff vs cuBLAS: %.6e\n", mad);

  // Tolerance for fast-math SGEMM
  if (mad > 5e-2) {
    fprintf(stderr, "FAIL: diff too large\n");
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 2;
  } else {
    printf("PASS\n");
  }

  cublasDestroy(handle);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
