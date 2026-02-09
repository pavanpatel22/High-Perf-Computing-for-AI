#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

extern "C" {
#include "flashattn_cpu.h"
#include "naive_attention.h"
#include "flashattn_cuda.h"
}

#define CUDA_CHECK(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        std::exit(1);                                       \
    }                                                       \
} while (0)

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static int get_arg_int(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    }
    return def;
}
static bool get_arg_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == key) return true;
    return false;
}
static std::string get_arg_str(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return std::string(def);
}

static int parse_dtype(const std::string& s) {
    if (s == "f32") return 0;
    if (s == "f16") return 1;
    if (s == "bf16") return 2;
    return -1;
}

int main(int argc, char** argv) {
    const int N  = get_arg_int(argc, argv, "--N", 256);
    const int D  = get_arg_int(argc, argv, "--D", 64);
    const int Br = get_arg_int(argc, argv, "--Br", 64);
    const int Bc = get_arg_int(argc, argv, "--Bc", 64);
    const int B  = get_arg_int(argc, argv, "--B", 1);
    const int H  = get_arg_int(argc, argv, "--H", 1);
    const bool causal = get_arg_flag(argc, argv, "--causal");
    const std::string dtype_s = get_arg_str(argc, argv, "--dtype", "f16");
    const int dtype = parse_dtype(dtype_s);

    if (dtype < 0) {
        printf("Unsupported --dtype. Use: f32 | f16 | bf16\n");
        return 1;
    }

    const int BH = B * H;
    const size_t qkv_elems = (size_t)BH * (size_t)N * (size_t)D;
    const size_t out_elems = qkv_elems;
    const size_t lse_elems = (size_t)BH * (size_t)N;

    printf("Config: B=%d H=%d N=%d D=%d Br=%d Bc=%d causal=%d dtype=%s\n",
           B, H, N, D, Br, Bc, (int)causal, dtype_s.c_str());

    // Random init in float
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Qf(qkv_elems), Kf(qkv_elems), Vf(qkv_elems);
    for (size_t i = 0; i < qkv_elems; ++i) {
        Qf[i] = dist(rng);
        Kf[i] = dist(rng);
        Vf[i] = dist(rng);
    }

    // Naive reference always in float32
    std::vector<float> O_naive(out_elems, 0.0f);
    attention_naive_cpu_f32(Qf.data(), Kf.data(), Vf.data(), O_naive.data(), B, H, N, D, causal);

    // CPU flash (float32)
    std::vector<float> O_cpu(out_elems, 0.0f);
    std::vector<float> L_cpu(lse_elems, 0.0f);
    flashattn2_forward_cpu_f32(Qf.data(), Kf.data(), Vf.data(), O_cpu.data(), L_cpu.data(),
                               B, H, N, D, Br, Bc, causal);

    float err_cpu = max_abs_diff(O_naive, O_cpu);
    printf("Max |O_naive - O_cpu_flash| = %.6g\n", err_cpu);

    // Allocate device inputs in chosen dtype
    void *dQ = nullptr, *dK = nullptr, *dV = nullptr;
    float *dO = nullptr, *dL = nullptr;

    size_t in_bytes = 0;
    if (dtype == 0) in_bytes = qkv_elems * sizeof(float);
    if (dtype == 1) in_bytes = qkv_elems * sizeof(__half);
    if (dtype == 2) in_bytes = qkv_elems * sizeof(__nv_bfloat16);

    CUDA_CHECK(cudaMalloc(&dQ, in_bytes));
    CUDA_CHECK(cudaMalloc(&dK, in_bytes));
    CUDA_CHECK(cudaMalloc(&dV, in_bytes));
    CUDA_CHECK(cudaMalloc(&dO, out_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, lse_elems * sizeof(float)));

    // Host temp buffers for casting
    if (dtype == 0) {
        CUDA_CHECK(cudaMemcpy(dQ, Qf.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, Kf.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, Vf.data(), in_bytes, cudaMemcpyHostToDevice));
    } else if (dtype == 1) {
        std::vector<__half> Qh(qkv_elems), Kh(qkv_elems), Vh(qkv_elems);
        for (size_t i = 0; i < qkv_elems; ++i) {
            Qh[i] = __float2half(Qf[i]);
            Kh[i] = __float2half(Kf[i]);
            Vh[i] = __float2half(Vf[i]);
        }
        CUDA_CHECK(cudaMemcpy(dQ, Qh.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, Kh.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, Vh.data(), in_bytes, cudaMemcpyHostToDevice));
    } else {
        std::vector<__nv_bfloat16> Qb(qkv_elems), Kb(qkv_elems), Vb(qkv_elems);
        for (size_t i = 0; i < qkv_elems; ++i) {
            Qb[i] = __float2bfloat16(Qf[i]);
            Kb[i] = __float2bfloat16(Kf[i]);
            Vb[i] = __float2bfloat16(Vf[i]);
        }
        CUDA_CHECK(cudaMemcpy(dQ, Qb.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, Kb.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, Vb.data(), in_bytes, cudaMemcpyHostToDevice));
    }

    // Run CUDA flash
    flashattn2_forward_cuda(dQ, dK, dV, dO, dL, B, H, N, D, Br, Bc, causal, dtype);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> O_gpu(out_elems, 0.0f);
    CUDA_CHECK(cudaMemcpy(O_gpu.data(), dO, out_elems * sizeof(float), cudaMemcpyDeviceToHost));

    float err_gpu = max_abs_diff(O_naive, O_gpu);
    printf("Max |O_naive - O_gpu_flash| = %.6g\n", err_gpu);

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    CUDA_CHECK(cudaFree(dL));

    // Tolerance: fp16/bf16 inputs create slightly larger error vs naive float
    const float tol = (dtype == 0) ? 1e-4f : 5e-3f;
    if (err_cpu < 1e-4f && err_gpu < tol) {
        printf("PASS (cpu tol=1e-4, gpu tol=%.1e)\n", tol);
        return 0;
    } else {
        printf("FAIL (cpu tol=1e-4, gpu tol=%.1e)\n", tol);
        return 1;
    }
}
