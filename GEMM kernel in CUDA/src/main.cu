#include "gemm.cuh"
#include "cuda_utils.cuh"

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

static void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// CPU reference: C <- alpha * op(A) * op(B) + beta * C
static void gemm_cpu_ref(
    int m, int n, int k,
    float alpha,
    const float* A, bool tA,
    const float* B, bool tB,
    float beta,
    float* C)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                float a = tA ? A[p * m + i] : A[i * k + p];
                float b = tB ? B[j * k + p] : B[p * n + j];
                sum += a * b;
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
}

int main() {
    // Multiple sizes to verify correctness (including non-multiples of block size)
    struct Case { int m, n, k; };
    std::vector<Case> cases = {
        {128, 128, 128},
        {255, 129, 63},
        {64,  257, 17},
    };

    float alpha = 1.25f;
    float beta  = -0.75f;

    CUDA_CHECK(cudaSetDevice(0));

    for (auto cs : cases) {
        int m = cs.m, n = cs.n, k = cs.k;

        // For transpose flags, we store matrices in the needed layouts:
        // A_noT: m x k, A_T: k x m
        // B_noT: k x n, B_T: n x k
        std::vector<float> hA_noT((size_t)m * k);
        std::vector<float> hA_T  ((size_t)k * m);
        std::vector<float> hB_noT((size_t)k * n);
        std::vector<float> hB_T  ((size_t)n * k);
        std::vector<float> hC0   ((size_t)m * n);

        fill_random(hA_noT, 1);
        fill_random(hA_T,   2);
        fill_random(hB_noT, 3);
        fill_random(hB_T,   4);
        fill_random(hC0,    5);

        auto run_one = [&](bool tA, bool tB) {
            const float* hA = tA ? hA_T.data() : hA_noT.data();
            const float* hB = tB ? hB_T.data() : hB_noT.data();

            size_t bytesA = (size_t)(tA ? k * m : m * k) * sizeof(float);
            size_t bytesB = (size_t)(tB ? n * k : k * n) * sizeof(float);
            size_t bytesC = (size_t)m * n * sizeof(float);

            std::vector<float> hC_gpu = hC0;
            std::vector<float> hC_ref = hC0;

            float *dA=nullptr, *dB=nullptr, *dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, bytesA));
            CUDA_CHECK(cudaMalloc(&dB, bytesB));
            CUDA_CHECK(cudaMalloc(&dC, bytesC));

            CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dC, hC_gpu.data(), bytesC, cudaMemcpyHostToDevice));

            gemm_cuda(m, n, k, alpha, dA, tA, dB, tB, beta, dC);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(hC_gpu.data(), dC, bytesC, cudaMemcpyDeviceToHost));

            gemm_cpu_ref(m, n, k, alpha, hA, tA, hB, tB, beta, hC_ref.data());

            float err = max_abs_diff(hC_gpu, hC_ref);
            std::cout << "  tA=" << tA << " tB=" << tB
                      << " | max_abs_err=" << err << "\n";

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
        };

        std::cout << "\nCase m=" << m << " n=" << n << " k=" << k << "\n";
        run_one(false, false);
        run_one(true,  false);
        run_one(false, true);
        run_one(true,  true);
    }

    std::cout << "\nDone.\n";
    return 0;
}
