#include "naive_attention.h"
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

static inline const float* ptr_qkv(const float* base, int bh, int N, int D, int n, int d) {
    return base + (size_t)bh * (size_t)N * (size_t)D + (size_t)n * (size_t)D + (size_t)d;
}
static inline float* ptr_o(float* base, int bh, int N, int D, int n, int d) {
    return base + (size_t)bh * (size_t)N * (size_t)D + (size_t)n * (size_t)D + (size_t)d;
}

void attention_naive_cpu_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D,
    bool causal
) {
    const int BH = B * H;
    const float scale = 1.0f / sqrtf((float)D);

    // Temporary score row
    float* scores = (float*)malloc((size_t)N * sizeof(float));
    float* probs  = (float*)malloc((size_t)N * sizeof(float));

    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < N; ++i) {
            // scores
            float m = -INFINITY;
            const float* qi = ptr_qkv(Q, bh, N, D, i, 0);
            for (int j = 0; j < N; ++j) {
                float s = 0.0f;
                if (causal && (j > i)) {
                    s = -INFINITY;
                } else {
                    const float* kj = ptr_qkv(K, bh, N, D, j, 0);
                    for (int d = 0; d < D; ++d) s += qi[d] * kj[d];
                    s *= scale;
                }
                scores[j] = s;
                if (s > m) m = s;
            }

            // softmax
            float l = 0.0f;
            for (int j = 0; j < N; ++j) {
                if (!isfinite(scores[j])) { probs[j] = 0.0f; continue; }
                float p = expf(scores[j] - m);
                probs[j] = p;
                l += p;
            }
            float inv_l = 1.0f / l;

            // O[i] = sum_j p_ij * V[j]
            float* out = ptr_o(O, bh, N, D, i, 0);
            memset(out, 0, (size_t)D * sizeof(float));
            for (int j = 0; j < N; ++j) {
                const float w = probs[j] * inv_l;
                if (w == 0.0f) continue;
                const float* vj = ptr_qkv(V, bh, N, D, j, 0);
                for (int d = 0; d < D; ++d) out[d] += w * vj[d];
            }
        }
    }

    free(scores);
    free(probs);
}
