#include "flashattn_cpu.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline size_t idx4(int b, int h, int n, int d, int H, int N, int D) {
    return (((size_t)b * (size_t)H + (size_t)h) * (size_t)N + (size_t)n) * (size_t)D + (size_t)d;
}
static inline size_t idx_l(int b, int h, int n, int H, int N) {
    return ((size_t)b * (size_t)H + (size_t)h) * (size_t)N + (size_t)n;
}

void flashattn2_forward_cpu_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* L,
    int B, int H, int N, int D,
    int Br, int Bc,
    bool causal
) {
    const int Tr = ceil_div(N, Br);
    const int Tc = ceil_div(N, Bc);
    const float scale = 1.0f / sqrtf((float)D);

    // Temporary buffers sized to the maximum tile sizes
    float* m = (float*)malloc((size_t)Br * sizeof(float));
    float* l = (float*)malloc((size_t)Br * sizeof(float));
    float* Otilde = (float*)malloc((size_t)Br * (size_t)D * sizeof(float));
    float* s_local = (float*)malloc((size_t)Bc * sizeof(float)); // score cache for one row over a key-block

    if (!m || !l || !Otilde || !s_local) {
        free(m); free(l); free(Otilde); free(s_local);
        return;
    }

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {

            for (int ti = 0; ti < Tr; ++ti) {
                const int q0 = ti * Br;
                const int qn = (q0 + Br <= N) ? Br : (N - q0);

                // init streaming stats for this query tile
                for (int r = 0; r < qn; ++r) {
                    m[r] = -INFINITY;
                    l[r] = 0.0f;
                    memset(Otilde + (size_t)r * (size_t)D, 0, (size_t)D * sizeof(float));
                }

                for (int tj = 0; tj < Tc; ++tj) {
                    const int k0 = tj * Bc;
                    const int kn = (k0 + Bc <= N) ? Bc : (N - k0);

                    for (int r = 0; r < qn; ++r) {
                        const int q_idx = q0 + r;

                        // 1) compute scores for this K-block and row_max
                        float row_max = -INFINITY;

                        for (int c = 0; c < kn; ++c) {
                            const int k_idx = k0 + c;

                            float s;
                            if (causal && (k_idx > q_idx)) {
                                s = -INFINITY;
                            } else {
                                float acc = 0.0f;
                                for (int d = 0; d < D; ++d) {
                                    acc += Q[idx4(b,h,q_idx,d,H,N,D)] * K[idx4(b,h,k_idx,d,H,N,D)];
                                }
                                s = acc * scale;
                            }

                            s_local[c] = s;
                            if (s > row_max) row_max = s;
                        }

                        // 2) m_new = max(m_old, row_max)
                        const float m_old = m[r];
                        const float m_new = (m_old > row_max) ? m_old : row_max;

                        // 3) rescale previous accumulators by alpha = exp(m_old - m_new)
                        const float alpha = isfinite(m_old) ? expf(m_old - m_new) : 0.0f;

                        l[r] *= alpha;
                        float* otilde_row = Otilde + (size_t)r * (size_t)D;
                        for (int d = 0; d < D; ++d) otilde_row[d] *= alpha;

                        // 4) accumulate this block
                        float l_new = l[r];
                        for (int c = 0; c < kn; ++c) {
                            const float s = s_local[c];
                            if (!isfinite(s)) continue;

                            const float p = expf(s - m_new);
                            l_new += p;

                            const int k_idx = k0 + c;
                            for (int d = 0; d < D; ++d) {
                                otilde_row[d] += p * V[idx4(b,h,k_idx,d,H,N,D)];
                            }
                        }

                        // 5) update stats
                        m[r] = m_new;
                        l[r] = l_new;
                    }
                }

                // finalize tile: O = Otilde / l ; L = m + log(l)
                for (int r = 0; r < qn; ++r) {
                    const int q_idx = q0 + r;
                    const float inv_l = 1.0f / l[r];

                    const float* otilde_row = Otilde + (size_t)r * (size_t)D;
                    for (int d = 0; d < D; ++d) {
                        O[idx4(b,h,q_idx,d,H,N,D)] = otilde_row[d] * inv_l;
                    }
                    L[idx_l(b,h,q_idx,H,N)] = m[r] + logf(l[r]);
                }
            }
        }
    }

    free(m);
    free(l);
    free(Otilde);
    free(s_local);
}
