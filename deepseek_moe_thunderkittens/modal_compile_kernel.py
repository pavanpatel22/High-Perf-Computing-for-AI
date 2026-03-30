import modal
import os
import json

# ── CUDA kernel embedded directly (no external .cu file needed) ──────────────
CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/*
 * SwiGLU MoE Expert GEMM Kernel
 *
 * Each CUDA block handles ONE token routed to ONE expert.
 * Shared memory holds the intermediate SwiGLU activations.
 *
 * Inputs:
 *   x          [N, H]    - input token embeddings  (fp32)
 *   w_gate     [E, I, H] - gate projection weights (fp32)
 *   w_up       [E, I, H] - up   projection weights (fp32)
 *   w_down     [E, H, I] - down projection weights (fp32)
 *   expert_ids [N]       - expert index per token
 *   weights    [N]       - router softmax weight per token
 *   out        [N, H]    - output (fp32, must be zeroed before call)
 */
__global__ void moe_expert_gemm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    const int*   __restrict__ expert_ids,
    const float* __restrict__ weights,
    float*       __restrict__ out,
    int N, int H, int I, int E
) {
    int tok = blockIdx.x;
    int tid = threadIdx.x;
    if (tok >= N) return;

    int   e_id = expert_ids[tok];
    float w    = weights[tok];

    /* Shared memory: gate_buf[I]  (SwiGLU output) */
    extern __shared__ float smem[];
    float* gate_buf = smem;

    const float* xi = x      + (long long)tok  * H;
    const float* wg = w_gate + (long long)e_id * I * H;
    const float* wu = w_up   + (long long)e_id * I * H;
    const float* wd = w_down + (long long)e_id * H * I;

    /* Step 1: gate + up projections then SwiGLU activation */
    for (int i = tid; i < I; i += blockDim.x) {
        float g = 0.f, u = 0.f;
        const float* wgi = wg + (long long)i * H;
        const float* wui = wu + (long long)i * H;
        for (int h = 0; h < H; h++) {
            g += xi[h] * wgi[h];
            u += xi[h] * wui[h];
        }
        /* SwiGLU: sigmoid(g) * u */
        gate_buf[i] = (1.f / (1.f + expf(-g))) * u;
    }
    __syncthreads();

    /* Step 2: down projection, accumulate into output */
    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        const float* wdh = wd + (long long)h * I;
        for (int i = 0; i < I; i++) acc += gate_buf[i] * wdh[i];
        atomicAdd(&out[(long long)tok * H + h], w * acc);
    }
}

/* Host-callable launcher */
extern "C" void launch_moe_expert_gemm(
    const float* x,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    const int*   expert_ids,
    const float* weights,
    float*       out,
    int N, int H, int I, int E,
    cudaStream_t stream
) {
    int    threads = (H < 256) ? H : 256;
    size_t smem    = I * sizeof(float);
    moe_expert_gemm_kernel<<<N, threads, smem, stream>>>(
        x, w_gate, w_up, w_down, expert_ids, weights, out, N, H, I, E
    );
}
"""

# ── Modal image: CUDA 12.6 devel image (contains nvcc) ───────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("build-essential", "wget")
    .pip_install(
        "torch",
        extra_options="--index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install("numpy")
)

app = modal.App("moe-compile-kernel", image=image)


# ── Remote function: compile + smoke-test the kernel ─────────────────────────
@app.function(gpu="T4", timeout=300)
def compile_and_run(cuda_src: str):
    import subprocess
    import tempfile
    import os
    import ctypes
    import numpy as np
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        cu_path = os.path.join(tmpdir, "moe_expert_gemm.cu")
        so_path = os.path.join(tmpdir, "moe_expert_gemm.so")

        # Write CUDA source to temp file
        with open(cu_path, "w") as f:
            f.write(cuda_src)

        # ── Compile with nvcc ─────────────────────────────────────────────────
        # IMPORTANT: -fPIC must go via --compiler-options, NOT as a bare nvcc flag
        result = subprocess.run(
            [
                "nvcc", "-O3", "-shared",
                "-arch=sm_75",                   # T4=sm_75 | A100=sm_80 | B200=sm_100
                "--compiler-options", "-fPIC",
                "-o", so_path,
                cu_path,
            ],
            capture_output=True,
            text=True,
        )

        print("=== nvcc stdout ===")
        print(result.stdout if result.stdout else "(none)")
        print("=== nvcc stderr ===")
        print(result.stderr if result.stderr else "(none)")

        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed with exit code {result.returncode}")

        so_size = os.path.getsize(so_path)
        print(f"\n✓ Compiled successfully")
        print(f"  Output : {so_path}")
        print(f"  Size   : {so_size:,} bytes")

        # ── Load shared library via ctypes ────────────────────────────────────
        lib = ctypes.CDLL(so_path)
        lib.launch_moe_expert_gemm.restype  = None
        lib.launch_moe_expert_gemm.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # w_gate
            ctypes.c_void_p,  # w_up
            ctypes.c_void_p,  # w_down
            ctypes.c_void_p,  # expert_ids
            ctypes.c_void_p,  # weights
            ctypes.c_void_p,  # out
            ctypes.c_int,     # N
            ctypes.c_int,     # H
            ctypes.c_int,     # I
            ctypes.c_int,     # E
            ctypes.c_void_p,  # stream (NULL = default)
        ]

        # ── Smoke test: N=4 tokens, H=64, I=32, E=4 ──────────────────────────
        N, H, I, E = 4, 64, 32, 4
        rng = np.random.default_rng(42)

        x_np          = rng.standard_normal((N, H)).astype(np.float32)
        w_gate_np     = rng.standard_normal((E, I, H)).astype(np.float32)
        w_up_np       = rng.standard_normal((E, I, H)).astype(np.float32)
        w_down_np     = rng.standard_normal((E, H, I)).astype(np.float32)
        expert_ids_np = np.array([0, 1, 2, 3], dtype=np.int32)
        weights_np    = np.ones(N, dtype=np.float32)

        # Allocate GPU tensors via PyTorch
        x_t          = torch.from_numpy(x_np).cuda()
        w_gate_t     = torch.from_numpy(w_gate_np).cuda()
        w_up_t       = torch.from_numpy(w_up_np).cuda()
        w_down_t     = torch.from_numpy(w_down_np).cuda()
        expert_ids_t = torch.from_numpy(expert_ids_np).cuda()
        weights_t    = torch.from_numpy(weights_np).cuda()
        out_t        = torch.zeros(N, H, dtype=torch.float32).cuda()

        # Launch kernel
        lib.launch_moe_expert_gemm(
            ctypes.c_void_p(x_t.data_ptr()),
            ctypes.c_void_p(w_gate_t.data_ptr()),
            ctypes.c_void_p(w_up_t.data_ptr()),
            ctypes.c_void_p(w_down_t.data_ptr()),
            ctypes.c_void_p(expert_ids_t.data_ptr()),
            ctypes.c_void_p(weights_t.data_ptr()),
            ctypes.c_void_p(out_t.data_ptr()),
            ctypes.c_int(N),
            ctypes.c_int(H),
            ctypes.c_int(I),
            ctypes.c_int(E),
            None,   # default CUDA stream
        )
        torch.cuda.synchronize()

        out_np = out_t.cpu().numpy()

        print(f"\n✓ Kernel executed successfully")
        print(f"  Output shape    : {out_np.shape}")
        print(f"  Output[0, :4]   : {out_np[0, :4]}")
        print(f"  Max abs value   : {np.abs(out_np).max():.6f}")
        print(f"  Non-zero outputs: {np.count_nonzero(out_np)}/{out_np.size}")

        return {
            "status":     "ok",
            "so_size":    so_size,
            "out_shape":  list(out_np.shape),
            "out_sample": out_np[0, :4].tolist(),
            "max_abs":    float(np.abs(out_np).max()),
        }


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("\n=== MoE Expert GEMM CUDA Kernel — Compile & Run on Modal T4 ===\n")
    result = compile_and_run.remote(CUDA_SRC)
    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2))