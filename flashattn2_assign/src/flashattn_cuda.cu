// flashattn_cuda.cu
#include "flashattn_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        std::exit(1);                                          \
    }                                                          \
} while (0)

static __host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -------------------- type loads --------------------
template <typename T>
__device__ __forceinline__ float load_f32(const T* p);

template <>
__device__ __forceinline__ float load_f32<float>(const float* p) { return *p; }

template <>
__device__ __forceinline__ float load_f32<__half>(const __half* p) { return __half2float(*p); }

template <>
__device__ __forceinline__ float load_f32<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// -------------------- warp reduce helpers --------------------
__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

// Kernel design (correctness-first, more parallel):
// - blockDim.x = 32 lanes (one warp)
// - blockDim.y = WARPS_PER_BLOCK warps
// - each warp handles one query row
// - dot(Q,K) is parallelized across lanes over D
// - streaming softmax stats m,l maintained per row by lane0 and broadcast
template <typename Tin>
__global__ void flashattn2_forward_warp_kernel(
    const Tin* __restrict__ Q,
    const Tin* __restrict__ K,
    const Tin* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int N, int D,
    int Br, int Bc,
    bool causal
) {
    constexpr int WARPS = 8; // warps per block (256 threads)

    const int lane = (int)threadIdx.x; // 0..31
    const int warp = (int)threadIdx.y; // 0..WARPS-1

    const int Tr = ceil_div(N, Br);
    const int row_blocks = ceil_div(Br, WARPS);

    const int packed = (int)blockIdx.x;
    const int ti = packed / row_blocks;
    const int rb = packed - ti * row_blocks;

    const int bh = (int)blockIdx.y;

    const int q0 = ti * Br;
    const int q_idx = q0 + rb * WARPS + warp;
    if (q_idx >= N) return;

    const float scale = rsqrtf((float)D);

    const size_t bh_stride = (size_t)N * (size_t)D;
    const Tin* Qbh = Q + (size_t)bh * bh_stride;
    const Tin* Kbh = K + (size_t)bh * bh_stride;
    const Tin* Vbh = V + (size_t)bh * bh_stride;
    float* Obh = O + (size_t)bh * bh_stride;
    float* Lbh = L + (size_t)bh * (size_t)N;

    float m = -INFINITY;
    float l = 0.0f;

    const int Tc = ceil_div(N, Bc);

    for (int tj = 0; tj < Tc; ++tj) {
        const int k0 = tj * Bc;
        const int kn = min(Bc, N - k0);

        // 1) row_max over this K-block
        float row_max = -INFINITY;

        for (int c = 0; c < kn; ++c) {
            const int k_idx = k0 + c;
            float s = -INFINITY;

            if (!(causal && (k_idx > q_idx))) {
                float partial = 0.0f;
                const Tin* qrow = Qbh + (size_t)q_idx * (size_t)D;
                const Tin* krow = Kbh + (size_t)k_idx * (size_t)D;

                for (int d = lane; d < D; d += 32) {
                    partial += load_f32(qrow + d) * load_f32(krow + d);
                }

                float dot = warp_sum(partial);
                if (lane == 0) s = dot * scale;
                s = __shfl_sync(0xffffffff, s, 0);
            }

            row_max = fmaxf(row_max, s);
        }

        // 2) m_new = max(m_old, row_max)
        const float m_new = fmaxf(m, row_max);

        // 3) alpha = exp(m_old - m_new); rescale old accumulators
        const float alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
        l *= alpha;

        // Use O as Otilde scratch: per-row, lanes own disjoint d indices
        float* out = Obh + (size_t)q_idx * (size_t)D;

        // rescale scratch Otilde
        for (int d = lane; d < D; d += 32) out[d] *= alpha;

        // 4) accumulate this block:
        // IMPORTANT FIX: Only lane0 updates l (otherwise you'd multiply by 32).
        float l_add = 0.0f; // lane0 only

        for (int c = 0; c < kn; ++c) {
            const int k_idx = k0 + c;
            float s = -INFINITY;

            if (!(causal && (k_idx > q_idx))) {
                float partial = 0.0f;
                const Tin* qrow = Qbh + (size_t)q_idx * (size_t)D;
                const Tin* krow = Kbh + (size_t)k_idx * (size_t)D;

                for (int d = lane; d < D; d += 32) {
                    partial += load_f32(qrow + d) * load_f32(krow + d);
                }

                float dot = warp_sum(partial);
                if (lane == 0) s = dot * scale;
                s = __shfl_sync(0xffffffff, s, 0);
            }

            if (!isfinite(s)) continue;

            const float p = expf(s - m_new);

            // ✅ only lane 0 accumulates softmax denominator
            if (lane == 0) l_add += p;

            // accumulate p*V into out scratch for owned d indices
            const Tin* vrow = Vbh + (size_t)k_idx * (size_t)D;
            for (int d = lane; d < D; d += 32) {
                out[d] += p * load_f32(vrow + d);
            }
        }

        // lane0 updates l, then broadcast
        if (lane == 0) l += l_add;
        l = __shfl_sync(0xffffffff, l, 0);

        // 5) update m, then broadcast
        if (lane == 0) m = m_new;
        m = __shfl_sync(0xffffffff, m, 0);
    }

    // finalize: O = Otilde / l ; L = m + log(l)
    const float inv_l = 1.0f / l;
    float* out = Obh + (size_t)q_idx * (size_t)D;
    for (int d = lane; d < D; d += 32) out[d] *= inv_l;

    if (lane == 0) Lbh[q_idx] = m + logf(l);
}

template <typename Tin>
static void launch_flashattn2_forward(
    const Tin* dQ, const Tin* dK, const Tin* dV,
    float* dO, float* dL,
    int B, int H, int N, int D,
    int Br, int Bc,
    bool causal
) {
    const int BH = B * H;
    const int Tr = ceil_div(N, Br);
    constexpr int WARPS = 8;
    const int row_blocks = ceil_div(Br, WARPS);

    dim3 block(32, WARPS, 1);
    dim3 grid(Tr * row_blocks, BH, 1);

    // We use dO as scratch Otilde storage; must be zeroed before kernel.
    CUDA_CHECK(cudaMemset(dO, 0, (size_t)BH * (size_t)N * (size_t)D * sizeof(float)));

    flashattn2_forward_warp_kernel<Tin><<<grid, block>>>(
        dQ, dK, dV, dO, dL, N, D, Br, Bc, causal
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void flashattn2_forward_cuda(
    const void* Q, const void* K, const void* V,
    float* O, float* L,
    int B, int H, int N, int D,
    int Br, int Bc,
    bool causal,
    int dtype
) {
    // dtype: 0=f32, 1=f16, 2=bf16
    if (dtype == 0) {
        launch_flashattn2_forward<float>(
            (const float*)Q, (const float*)K, (const float*)V,
            O, L, B, H, N, D, Br, Bc, causal
        );
    } else if (dtype == 1) {
        launch_flashattn2_forward<__half>(
            (const __half*)Q, (const __half*)K, (const __half*)V,
            O, L, B, H, N, D, Br, Bc, causal
        );
    } else if (dtype == 2) {
        launch_flashattn2_forward<__nv_bfloat16>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            O, L, B, H, N, D, Br, Bc, causal
        );
    } else {
        fprintf(stderr, "flashattn2_forward_cuda: unsupported dtype=%d\n", dtype);
        std::exit(1);
    }
}