#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Contiguous layout: [B, H, N, D] row-major
// Q,K,V,O are size B*H*N*D
// L is size B*H*N  (logsumexp per row)
void flashattn2_forward_cpu_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* L,
    int B, int H, int N, int D,
    int Br, int Bc,
    bool causal
);

#ifdef __cplusplus
}
#endif
