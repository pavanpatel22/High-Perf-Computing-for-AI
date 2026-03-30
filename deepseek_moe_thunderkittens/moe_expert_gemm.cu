/*
 * moe_expert_gemm.cu
 * WMMA tensor-core SwiGLU expert GEMM for DeepSeekMoE.
 * Target: NVIDIA B200  (sm_90a or sm_100)
 * Compile:
 *   nvcc -O3 -arch=sm_90a -std=c++17 moe_expert_gemm.cu -o moe_expert_gemm -lm
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ──────────────────────────────────────────────────────────────
// WMMA GEMM: C = A [M,K] @ B^T [N,K]  →  C [M,N]
// A, B: bf16 row-major    C: fp32 row-major
// ──────────────────────────────────────────────────────────────
__global__ void wmma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    int M, int N, int K
) {
    int warp_row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * WMMA_M;
    int warp_col =  blockIdx.y * WMMA_N;
    if (warp_row >= M || warp_col >= N) return;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K,
                   __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k + WMMA_K <= K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k,         K);
        wmma::load_matrix_sync(b_frag, B + warp_col * K + k,         K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N,
                             wmma::mem_row_major);
}

// ──────────────────────────────────────────────────────────────
// SwiGLU: out[i] = sigmoid(gate[i]) * up[i]
// ──────────────────────────────────────────────────────────────
__global__ void swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (1.0f / (1.0f + expf(-gate[i]))) * up[i];
}

// ──────────────────────────────────────────────────────────────
// F32 → BF16 conversion
// ──────────────────────────────────────────────────────────────
__global__ void f32_to_bf16_kernel(
    const float*       __restrict__ in,
    __nv_bfloat16*     __restrict__ out,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2bfloat16(in[i]);
}

// ──────────────────────────────────────────────────────────────
// Full expert forward using WMMA
//   input  [N, H] fp32
//   w_gate [I, H] fp32
//   w_up   [I, H] fp32
//   w_down [H, I] fp32
//   output [N, H] fp32
// ──────────────────────────────────────────────────────────────
void expert_forward_wmma(
    const float* d_input,
    const float* d_w_gate,
    const float* d_w_up,
    const float* d_w_down,
    float*       d_output,
    int N, int H, int I,
    cudaStream_t stream
) {
    const int THREADS = 256;

    __nv_bfloat16 *d_in_bf16, *d_wg_bf16, *d_wu_bf16, *d_wd_bf16, *d_mid_bf16;
    float         *d_gate_out, *d_up_out, *d_mid_out;

    cudaMalloc(&d_in_bf16,  N*H * sizeof(__nv_bfloat16));
    cudaMalloc(&d_wg_bf16,  I*H * sizeof(__nv_bfloat16));
    cudaMalloc(&d_wu_bf16,  I*H * sizeof(__nv_bfloat16));
    cudaMalloc(&d_wd_bf16,  H*I * sizeof(__nv_bfloat16));
    cudaMalloc(&d_mid_bf16, N*I * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gate_out, N*I * sizeof(float));
    cudaMalloc(&d_up_out,   N*I * sizeof(float));
    cudaMalloc(&d_mid_out,  N*I * sizeof(float));

    // Convert all inputs to bf16
    f32_to_bf16_kernel<<<(N*H+THREADS-1)/THREADS, THREADS, 0, stream>>>(d_input,  d_in_bf16, N*H);
    f32_to_bf16_kernel<<<(I*H+THREADS-1)/THREADS, THREADS, 0, stream>>>(d_w_gate, d_wg_bf16, I*H);
    f32_to_bf16_kernel<<<(I*H+THREADS-1)/THREADS, THREADS, 0, stream>>>(d_w_up,   d_wu_bf16, I*H);
    f32_to_bf16_kernel<<<(H*I+THREADS-1)/THREADS, THREADS, 0, stream>>>(d_w_down, d_wd_bf16, H*I);

    int warps_per_block = 4;
    dim3 block(warps_per_block * 32, 1);

    // GEMM: gate_out = input @ w_gate^T  [N, I]
    dim3 grid_NI(
        (N + WMMA_M * warps_per_block - 1) / (WMMA_M * warps_per_block),
        (I + WMMA_N - 1) / WMMA_N
    );
    wmma_gemm_kernel<<<grid_NI, block, 0, stream>>>(d_in_bf16, d_wg_bf16, d_gate_out, N, I, H);

    // GEMM: up_out = input @ w_up^T  [N, I]
    wmma_gemm_kernel<<<grid_NI, block, 0, stream>>>(d_in_bf16, d_wu_bf16, d_up_out, N, I, H);

    // SwiGLU
    swiglu_kernel<<<(N*I+THREADS-1)/THREADS, THREADS, 0, stream>>>(d_gate_out, d_up_out, d_mid_out, N*I);

    // Convert SwiGLU result to bf16 for down projection
    f32_to_bf16_kernel<<<(N*I+THREADS-1)/THREADS, THREADS, 0, stream>>>(d_mid_out, d_mid_bf16, N*I);

    // GEMM: output = mid @ w_down^T  [N, H]
    dim3 grid_NH(
        (N + WMMA_M * warps_per_block - 1) / (WMMA_M * warps_per_block),
        (H + WMMA_N - 1) / WMMA_N
    );
    wmma_gemm_kernel<<<grid_NH, block, 0, stream>>>(d_mid_bf16, d_wd_bf16, d_output, N, H, I);

    cudaFree(d_in_bf16); cudaFree(d_wg_bf16); cudaFree(d_wu_bf16);
    cudaFree(d_wd_bf16); cudaFree(d_mid_bf16);
    cudaFree(d_gate_out); cudaFree(d_up_out); cudaFree(d_mid_out);
}

// ──────────────────────────────────────────────────────────────
// CPU reference (for correctness check)
// ──────────────────────────────────────────────────────────────
void cpu_expert_ref(
    const float* input, const float* wg, const float* wu, const float* wd,
    float* output, int N, int H, int I
) {
    float* mid = (float*)calloc(N * I, sizeof(float));
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < I; i++) {
            float g = 0, u = 0;
            for (int h = 0; h < H; h++) {
                g += input[n*H+h] * wg[i*H+h];
                u += input[n*H+h] * wu[i*H+h];
            }
            mid[n*I+i] = (1.0f / (1.0f + expf(-g))) * u;
        }
    }
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            float s = 0;
            for (int i = 0; i < I; i++) s += mid[n*I+i] * wd[h*I+i];
            output[n*H+h] = s;
        }
    }
    free(mid);
}

int main() {
    printf("DeepSeekMoE WMMA Expert GEMM — B200\n\n");

    const int N = 32, H = 128, I = 64;

    float *h_in   = (float*)malloc(N*H*sizeof(float));
    float *h_wg   = (float*)malloc(I*H*sizeof(float));
    float *h_wu   = (float*)malloc(I*H*sizeof(float));
    float *h_wd   = (float*)malloc(H*I*sizeof(float));
    float *h_gpu  = (float*)calloc(N*H, sizeof(float));
    float *h_cpu  = (float*)calloc(N*H, sizeof(float));

    srand(42);
    for (int i = 0; i < N*H; i++) h_in[i] = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < I*H; i++) {
        h_wg[i] = (float)rand()/RAND_MAX - 0.5f;
        h_wu[i] = (float)rand()/RAND_MAX - 0.5f;
        h_wd[i] = (float)rand()/RAND_MAX - 0.5f;
    }

    float *d_in, *d_wg, *d_wu, *d_wd, *d_out;
    cudaMalloc(&d_in,  N*H*sizeof(float));
    cudaMalloc(&d_wg,  I*H*sizeof(float));
    cudaMalloc(&d_wu,  I*H*sizeof(float));
    cudaMalloc(&d_wd,  H*I*sizeof(float));
    cudaMalloc(&d_out, N*H*sizeof(float));
    cudaMemset(d_out, 0, N*H*sizeof(float));

    cudaMemcpy(d_in, h_in, N*H*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wg, h_wg, I*H*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wu, h_wu, I*H*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wd, h_wd, H*I*sizeof(float), cudaMemcpyHostToDevice);

    cpu_expert_ref(h_in, h_wg, h_wu, h_wd, h_cpu, N, H, I);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    expert_forward_wmma(d_in, d_wg, d_wu, d_wd, d_out, N, H, I, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_gpu, d_out, N*H*sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N*H; i++) {
        float e = fabsf(h_gpu[i] - h_cpu[i]);
        if (e > max_err) max_err = e;
    }
    printf("Max abs error (WMMA vs CPU): %.9f\n", max_err);
    printf("Correctness : %s\n\n", max_err < 1e-2f ? "PASS" : "FAIL");

    // Benchmark
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i = 0; i < 20; i++)
        expert_forward_wmma(d_in, d_wg, d_wu, d_wd, d_out, N, H, I, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(t0, stream);
    for (int i = 0; i < 1000; i++)
        expert_forward_wmma(d_in, d_wg, d_wu, d_wd, d_out, N, H, I, stream);
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 1000;
    printf("Expert WMMA forward: %.4f ms  (%.0f tok/s)\n", ms, N/(ms/1000.0f));

    cudaFree(d_in); cudaFree(d_wg); cudaFree(d_wu); cudaFree(d_wd); cudaFree(d_out);
    free(h_in); free(h_wg); free(h_wu); free(h_wd); free(h_gpu); free(h_cpu);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}