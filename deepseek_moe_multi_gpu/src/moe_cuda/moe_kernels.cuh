#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Each thread handles one output element of the linear layer.
__global__ void linear_kernel(
    const float* __restrict__ input,   // [N, in_dim]
    const float* __restrict__ weight,  // [out_dim, in_dim]
    float*       __restrict__ output,  // [N, out_dim]
    int N, int in_dim, int out_dim)
{
    int token = blockIdx.x;
    int o     = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= N || o >= out_dim) return;

    const float* x = input  + token * in_dim;
    const float* w = weight + o     * in_dim;
    float acc = 0.f;
    for (int i = 0; i < in_dim; i++) acc += x[i] * w[i];
    output[token * out_dim + o] = acc;
}

// SwiGLU: hidden[i] = sigmoid(gate[i]) * up[i]
__global__ void swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ hidden,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float g = gate[idx];
    hidden[idx] = (1.f / (1.f + expf(-g))) * up[idx];
}

// Weighted accumulate: out[token] += weight * expert_out[token]
__global__ void weighted_add_kernel(
    const float* __restrict__ expert_out,  // [N, H]
    float*       __restrict__ accum,       // [N, H]
    float        weight,
    int N, int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) return;
    accum[idx] += weight * expert_out[idx];
}

// Residual add: out = a + b + c
__global__ void residual_add3_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float*       __restrict__ out,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = a[idx] + b[idx] + c[idx];
}

// Run one MLP: input -> gate_proj -> up_proj -> swiglu -> down_proj
// Scratch must be [N * I * 3] floats.
inline void run_mlp_device(
    const float* d_input,
    const float* d_w_gate,
    const float* d_w_up,
    const float* d_w_down,
    float*       d_output,
    float*       d_scratch,   // [N*I*3]
    int N, int H, int I,
    cudaStream_t stream)
{
    float* d_gate = d_scratch;
    float* d_up   = d_scratch + N * I;
    float* d_hid  = d_scratch + N * I * 2;

    dim3 blk(256);

    // gate = input @ w_gate^T
    dim3 grd_gate(N, (I + 255) / 256);
    linear_kernel<<<grd_gate, blk, 0, stream>>>(d_input, d_w_gate, d_gate, N, H, I);

    // up = input @ w_up^T
    linear_kernel<<<grd_gate, blk, 0, stream>>>(d_input, d_w_up, d_up, N, H, I);

    // hid = swiglu(gate, up)
    int tot_I = N * I;
    swiglu_kernel<<<(tot_I+255)/256, 256, 0, stream>>>(d_gate, d_up, d_hid, tot_I);

    // output = hid @ w_down^T
    dim3 grd_down(N, (H + 255) / 256);
    linear_kernel<<<grd_down, blk, 0, stream>>>(d_hid, d_w_down, d_output, N, I, H);
}