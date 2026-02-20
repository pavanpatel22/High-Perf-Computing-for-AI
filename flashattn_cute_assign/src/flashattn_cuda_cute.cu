#include "flashattn_cuda_cute.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                 \
  cudaError_t err = call;                                     \
  if (err != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
    std::exit(1);                                             \
  }                                                           \
} while (0)

static __host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <typename T>
__device__ __forceinline__ float to_f32(T x);

template <>
__device__ __forceinline__ float to_f32<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_f32<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// FlashAttention Algorithm 1 (forward), CuTe-tiling version.
// One block = one query tile (ti) for one (bh).
// Threads = Br, one thread per query row in the tile.
// Uses predication (active) so __syncthreads is safe on last partial tile.
template <typename Tin>
__global__ void flashattn_forward_cute_kernel(
    const Tin* __restrict__ Q,
    const Tin* __restrict__ K,
    const Tin* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int N, int D, int Br, int Bc,
    bool causal
) {
  using namespace cute;

  const int ti = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;
  const int r  = (int)threadIdx.x;

  const int q0 = ti * Br;
  const int q_idx = q0 + r;
  const bool active = (r < Br) && (q_idx < N);

  const float scale = rsqrtf((float)D);
  const int Tc = ceil_div(N, Bc);

  const size_t bh_stride = (size_t)N * (size_t)D;
  const Tin* Qbh = Q + (size_t)bh * bh_stride;
  const Tin* Kbh = K + (size_t)bh * bh_stride;
  const Tin* Vbh = V + (size_t)bh * bh_stride;
  float* Obh = O + (size_t)bh * bh_stride;
  float* Lbh = L + (size_t)bh * (size_t)N;

  auto gQ = make_tensor(make_gmem_ptr(Qbh), make_shape(N, D), make_stride(D, 1));
  auto gK = make_tensor(make_gmem_ptr(Kbh), make_shape(N, D), make_stride(D, 1));
  auto gV = make_tensor(make_gmem_ptr(Vbh), make_shape(N, D), make_stride(D, 1));

  // Shared memory: Q[Br,D], K[Bc,D], V[Bc,D] in float
  extern __shared__ unsigned char smem_raw[];
  float* smem = reinterpret_cast<float*>(smem_raw);

  float* smem_Q = smem;
  float* smem_K = smem_Q + (size_t)Br * (size_t)D;
  float* smem_V = smem_K + (size_t)Bc * (size_t)D;

  auto sQ = make_tensor(make_smem_ptr(smem_Q), make_shape(Br, D), make_stride(D, 1));
  auto sK = make_tensor(make_smem_ptr(smem_K), make_shape(Bc, D), make_stride(D, 1));
  auto sV = make_tensor(make_smem_ptr(smem_V), make_shape(Bc, D), make_stride(D, 1));

  auto gQ_tile = local_tile(gQ, make_shape(Br, D), make_coord(q0, 0));

  for (int d = 0; d < D; ++d) {
    sQ(r, d) = active ? to_f32(gQ_tile(r, d)) : 0.0f;
  }
  __syncthreads();

  float m = -INFINITY;
  float l = 0.0f;

  float* out_row = active ? (Obh + (size_t)q_idx * (size_t)D) : nullptr;
  if (active) {
    for (int d = 0; d < D; ++d) out_row[d] = 0.0f;
  }
  __syncthreads();

  for (int tj = 0; tj < Tc; ++tj) {
    const int k0 = tj * Bc;
    const int kn = min(Bc, N - k0);

    auto gK_tile = local_tile(gK, make_shape(Bc, D), make_coord(k0, 0));
    auto gV_tile = local_tile(gV, make_shape(Bc, D), make_coord(k0, 0));

    for (int idx = r; idx < kn * D; idx += Br) {
      int kk = idx / D;
      int d  = idx - kk * D;
      sK(kk, d) = to_f32(gK_tile(kk, d));
      sV(kk, d) = to_f32(gV_tile(kk, d));
    }
    __syncthreads();

    if (active) {
      float row_max = -INFINITY;
      for (int c = 0; c < kn; ++c) {
        const int k_idx = k0 + c;
        float s = -INFINITY;

        if (!(causal && (k_idx > q_idx))) {
          float acc = 0.0f;
          for (int d = 0; d < D; ++d) acc += sQ(r, d) * sK(c, d);
          s = acc * scale;
        }
        row_max = fmaxf(row_max, s);
      }

      float m_new = fmaxf(m, row_max);

      float alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
      l *= alpha;
      for (int d = 0; d < D; ++d) out_row[d] *= alpha;

      for (int c = 0; c < kn; ++c) {
        const int k_idx = k0 + c;
        float s = -INFINITY;

        if (!(causal && (k_idx > q_idx))) {
          float acc = 0.0f;
          for (int d = 0; d < D; ++d) acc += sQ(r, d) * sK(c, d);
          s = acc * scale;
        }
        if (!isfinite(s)) continue;

        float p = expf(s - m_new);
        l += p;
        for (int d = 0; d < D; ++d) out_row[d] += p * sV(c, d);
      }

      m = m_new;
    }

    __syncthreads();
  }

  if (active) {
    float inv_l = 1.0f / l;
    for (int d = 0; d < D; ++d) out_row[d] *= inv_l;
    Lbh[q_idx] = m + logf(l);
  }
}

extern "C" void flashattn_forward_cute(
    const void* Q, const void* K, const void* V,
    float* O, float* L,
    int B, int H, int N, int D,
    int Br, int Bc,
    bool causal,
    int dtype
) {
  const int BH = B * H;
  const int Tr = ceil_div(N, Br);

  dim3 grid(Tr, BH, 1);
  dim3 block(Br, 1, 1);

  // dynamic shared memory bytes
  size_t shmem = (size_t)(Br + 2 * Bc) * (size_t)D * sizeof(float);


  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (shmem > (size_t)prop.sharedMemPerBlockOptin) {
    fprintf(stderr,
            "Requested dynamic shared memory %zu bytes exceeds device opt-in limit %d bytes\n",
            shmem, prop.sharedMemPerBlockOptin);
    std::exit(1);
  }

  if (dtype == 0) {
    CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_forward_cute_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));

    flashattn_forward_cute_kernel<float><<<grid, block, shmem>>>(
        (const float*)Q, (const float*)K, (const float*)V,
        O, L, N, D, Br, Bc, causal);

  } else if (dtype == 1) {
    CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_forward_cute_kernel<__half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));

    flashattn_forward_cute_kernel<__half><<<grid, block, shmem>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V,
        O, L, N, D, Br, Bc, causal);

  } else if (dtype == 2) {
    CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_forward_cute_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));

    flashattn_forward_cute_kernel<__nv_bfloat16><<<grid, block, shmem>>>(
        (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
        O, L, N, D, Br, Bc, causal);

  } else {
    fprintf(stderr, "flashattn_forward_cute: unsupported dtype=%d\n", dtype);
    std::exit(1);
  }

  CUDA_CHECK(cudaGetLastError());
}