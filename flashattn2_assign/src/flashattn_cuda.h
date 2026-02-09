#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// dtype: 0=f32, 1=f16, 2=bf16
void flashattn2_forward_cuda(
    const void* Q, const void* K, const void* V,
    float* O, float* L,
    int B, int H, int N, int D,
    int Br, int Bc,
    bool causal,
    int dtype
);

#ifdef __cplusplus
}
#endif
