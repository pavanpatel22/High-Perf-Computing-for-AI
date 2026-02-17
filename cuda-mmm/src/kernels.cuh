#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"
#include <device_launch_parameters.h>

// ------------------------- Kernel 1: Naive -------------------------
// 2D block, each thread computes one C element
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; k++) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ------------------------- Kernel 2: Coalesced mapping -------------------------
// Use 1D thread indexing over a 32x32 block (1024 threads).
// Map threadIdx.x -> (rowInBlock, colInBlock) with div/mod to improve coalescing.
template<int BS>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float beta,
                               float* __restrict__ C)
{
  int tid = threadIdx.x; // 0..BS*BS-1
  int rowIn = tid / BS;
  int colIn = tid % BS;

  int row = blockIdx.y * BS + rowIn;
  int col = blockIdx.x * BS + colIn;
  if (row >= M || col >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; k++) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ------------------------- Kernel 3: Shared memory tiling -------------------------
template<int BS>
__global__ void sgemm_smem_tiled(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta,
                                float* __restrict__ C)
{
  __shared__ float As[BS][BS];
  __shared__ float Bs[BS][BS];

  int tid = threadIdx.x;            // 0..BS*BS-1
  int rowIn = tid / BS;
  int colIn = tid % BS;

  int row = blockIdx.y * BS + rowIn;
  int col = blockIdx.x * BS + colIn;

  float acc = 0.f;

  for (int k0 = 0; k0 < K; k0 += BS) {
    // Load A tile
    if (row < M && (k0 + colIn) < K) As[rowIn][colIn] = A[row * K + (k0 + colIn)];
    else As[rowIn][colIn] = 0.f;

    // Load B tile
    if ((k0 + rowIn) < K && col < N) Bs[rowIn][colIn] = B[(k0 + rowIn) * N + col];
    else Bs[rowIn][colIn] = 0.f;

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BS; kk++) {
      acc += As[rowIn][kk] * Bs[kk][colIn];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ------------------------- Kernel 4: 1D block tiling (multiple cols per thread) -------------------------
// Block computes BM=32 rows and BN=32*TM cols per block. Each thread computes TM outputs.
template<int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C)
{
  constexpr int BS = 32;
  constexpr int BN = BS * TM;

  __shared__ float As[BS][BS];      // 32x32
  __shared__ float Bs[BS][BN];      // 32x(32*TM)

  int tid = threadIdx.x;            // 0..1023
  int threadRow = tid / BS;         // 0..31
  int threadCol = tid % BS;         // 0..31

  int globalRow = blockIdx.y * BS + threadRow;
  int globalColBase = blockIdx.x * BN + threadCol * TM;

  float acc[TM];
  #pragma unroll
  for (int i = 0; i < TM; i++) acc[i] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BS) {
    // Load A tile: 1024 threads load 1024 elements
    if (globalRow < M && (k0 + threadCol) < K) As[threadRow][threadCol] = A[globalRow * K + (k0 + threadCol)];
    else As[threadRow][threadCol] = 0.f;

    // Load B tile: Bs is 32*BN elements. Each thread loads 8 elements if TM=8 (BN=256).
    // General: total = 32*BN; per-thread = (32*BN)/1024 = BN/32 = TM
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      int linear = tid + i * 1024;          // covers entire Bs tile
      int bRow = linear / BN;               // 0..31
      int bCol = linear % BN;               // 0..BN-1
      int gCol = blockIdx.x * BN + bCol;
      int gRowK = k0 + bRow;
      float v = 0.f;
      if (gRowK < K && gCol < N) v = B[gRowK * N + gCol];
      Bs[bRow][bCol] = v;
    }

    __syncthreads();

    // Compute TM outputs for this thread
    #pragma unroll
    for (int kk = 0; kk < BS; kk++) {
      float a = As[threadRow][kk];
      int bBase = threadCol * TM;
      #pragma unroll
      for (int i = 0; i < TM; i++) {
        acc[i] += a * Bs[kk][bBase + i];
      }
    }

    __syncthreads();
  }

  // Store
  if (globalRow < M) {
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      int cCol = globalColBase + i;
      if (cCol < N) {
        int idx = globalRow * N + cCol;
        C[idx] = alpha * acc[i] + beta * C[idx];
      }
    }
  }
}

// ------------------------- Kernel 5: 2D block tiling (register micro-tile) -------------------------
template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C)
{
  static_assert((BM % TM) == 0, "BM must be multiple of TM");
  static_assert((BN % TN) == 0, "BN must be multiple of TN");

  constexpr int THREADS_Y = BM / TM;  // 16 for BM=128, TM=8
  constexpr int THREADS_X = BN / TN;  // 16 for BN=128, TN=8

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int tx = threadIdx.x; // 0..THREADS_X-1
  int ty = threadIdx.y; // 0..THREADS_Y-1
  int tid = ty * blockDim.x + tx; // 0..255

  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++)
    #pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Cooperative load As: BM*BK elements
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = tid + i * (THREADS_X * THREADS_Y); // tid + i*256
      if (idx < BM * BK) {
        int r = idx / BK;
        int c = idx % BK;
        float v = 0.f;
        int gR = blockRow + r;
        int gC = k0 + c;
        if (gR < M && gC < K) v = A[gR * K + gC];
        As[r][c] = v;
      }
    }

    // Cooperative load Bs: BK*BN elements
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = tid + i * (THREADS_X * THREADS_Y);
      if (idx < BK * BN) {
        int r = idx / BN;
        int c = idx % BN;
        float v = 0.f;
        int gR = k0 + r;
        int gC = blockCol + c;
        if (gR < K && gC < N) v = B[gR * N + gC];
        Bs[r][c] = v;
      }
    }

    __syncthreads();

    // Compute
    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float aReg[TM];
      float bReg[TN];

      #pragma unroll
      for (int i = 0; i < TM; i++) {
        int r = ty * TM + i;
        aReg[i] = As[r][kk];
      }
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        int c = tx * TN + j;
        bReg[j] = Bs[kk][c];
      }

      #pragma unroll
      for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] += aReg[i] * bReg[j];
    }

    __syncthreads();
  }

  // Store
  int rowBase = blockRow + ty * TM;
  int colBase = blockCol + tx * TN;

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int r = rowBase + i;
    if (r < M) {
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        int c = colBase + j;
        if (c < N) {
          int idx = r * N + c;
          C[idx] = alpha * acc[i][j] + beta * C[idx];
        }
      }
    }
  }
}

// ------------------------- Kernel 6: Vectorized loads (float4) -------------------------
template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C)
{
static_assert(BK == 8, "This vectorized kernel assumes BK=8");
static_assert((BN % 4) == 0, "BN must be multiple of 4 for float4 loads");
// NOTE: N is runtime, so we can't static_assert on it.
// Vectorized loads still work safely because we guard (gBc + 3) < N.
// Best performance when N is multiple of 4.


  constexpr int THREADS_Y = BM / TM;  // 16
  constexpr int THREADS_X = BN / TN;  // 16

  // Shared
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int tx = threadIdx.x; // 0..15
  int ty = threadIdx.y; // 0..15
  int tid = ty * blockDim.x + tx; // 0..255

  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++)
    #pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile: BM*BK = 128*8 = 1024 floats = 256 float4
    // Map tid -> (row, half) where half selects 0..3 or 4..7
    int aRow = tid / 2;      // 0..127
    int aHalf = tid & 1;     // 0 or 1
    int aCol = aHalf * 4;    // 0 or 4
    float4 a4 = make_float4(0,0,0,0);
    int gAr = blockRow + aRow;
    int gAc = k0 + aCol;
    if (gAr < M && (gAc + 3) < K) {
      a4 = *reinterpret_cast<const float4*>(&A[gAr * K + gAc]);
    }
    // store
    As[aRow][aCol + 0] = a4.x;
    As[aRow][aCol + 1] = a4.y;
    As[aRow][aCol + 2] = a4.z;
    As[aRow][aCol + 3] = a4.w;

    // B tile: BK*BN = 8*128 = 1024 floats = 256 float4
    // Map tid -> (bkRow, bnCol4)
    int cols4 = BN / 4;      // 32
    int bRow = tid / cols4;  // 0..7
    int bCol4 = tid % cols4; // 0..31
    int bCol = bCol4 * 4;
    float4 b4 = make_float4(0,0,0,0);
    int gBr = k0 + bRow;
    int gBc = blockCol + bCol;
    if (gBr < K && (gBc + 3) < N) {
      b4 = *reinterpret_cast<const float4*>(&B[gBr * N + gBc]);
    }
    Bs[bRow][bCol + 0] = b4.x;
    Bs[bRow][bCol + 1] = b4.y;
    Bs[bRow][bCol + 2] = b4.z;
    Bs[bRow][bCol + 3] = b4.w;

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float aReg[TM];
      float bReg[TN];

      #pragma unroll
      for (int i = 0; i < TM; i++) {
        int r = ty * TM + i;
        aReg[i] = As[r][kk];
      }
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        int c = tx * TN + j;
        bReg[j] = Bs[kk][c];
      }

      #pragma unroll
      for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] += aReg[i] * bReg[j];
    }

    __syncthreads();
  }

  // Store
  int rowBase = blockRow + ty * TM;
  int colBase = blockCol + tx * TN;

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int r = rowBase + i;
    if (r < M) {
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        int c = colBase + j;
        if (c < N) {
          int idx = r * N + c;
          C[idx] = alpha * acc[i][j] + beta * C[idx];
        }
      }
    }
  }
}

// ------------------------- Launchers -------------------------
enum Algo {
  CUBLAS = 0,
  NAIVE = 1,
  COALESCED = 2,
  SMEM = 3,
  BLOCKTILING_1D = 4,
  BLOCKTILING_2D = 5,
  VECTORIZED = 6
};

inline const char* algo_name(int a) {
  switch(a) {
    case 0: return "cuBLAS";
    case 1: return "Kernel1_Naive";
    case 2: return "Kernel2_Coalesced";
    case 3: return "Kernel3_SMEM";
    case 4: return "Kernel4_1D_BlockTiling";
    case 5: return "Kernel5_2D_BlockTiling";
    case 6: return "Kernel6_Vectorized";
    default: return "Unknown";
  }
}
