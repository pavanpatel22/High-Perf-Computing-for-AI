#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Naive attention for correctness checking: O = softmax(QK^T/sqrt(D)) V
void attention_naive_cpu_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D,
    bool causal
);

#ifdef __cplusplus
}
#endif
