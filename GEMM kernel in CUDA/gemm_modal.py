import modal

# GEMM kernel (unchanged, but we'll test it)
gemm_kernel_code = """
extern "C" __global__
void gemm_kernel(
    int m, int n, int k,
    float alpha,
    const float* A, int transposeA,
    const float* B, int transposeB,
    float beta,
    float* C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int q = 0; q < k; ++q) {
            float a_val, b_val;

            if (transposeA) {
                a_val = A[q * m + row];      // Aᵀ: (q, row) in original A
            } else {
                a_val = A[row * k + q];      // A: (row, q)
            }

            if (transposeB) {
                b_val = B[col * k + q];      // Bᵀ: (col, q) in original B
            } else {
                b_val = B[q * n + col];      // B: (q, col)
            }

            sum += a_val * b_val;
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
"""

app = modal.App("gemm-debug")

image = modal.Image.from_registry(
    "nvidia/cuda:12.6.0-runtime-ubuntu22.04",
    add_python="3.11"
).pip_install("cupy-cuda12x")

@app.function(image=image, gpu="any")
def test_gemm():
    import cupy as cp
    import numpy as np

    # Fixed tiny matrices (2x2)
    m = n = k = 2
    alpha = 1.0
    beta = 0.0  # simplify

    A_host = np.array([[1.0, 2.0],
                       [3.0, 4.0]], dtype=np.float32)
    B_host = np.array([[5.0, 6.0],
                       [7.0, 8.0]], dtype=np.float32)

    A = cp.asarray(A_host)
    B = cp.asarray(B_host)

    print("A:\n", A_host)
    print("B:\n", B_host)

    gemm_kernel = cp.RawKernel(gemm_kernel_code, "gemm_kernel", options=("-std=c++11",))

    def run_gemm(transA, transB):
        C_out = cp.zeros((m, n), dtype=cp.float32)
        block = (16, 16)
        grid = ((n + block[0] - 1) // block[0], (m + block[1] - 1) // block[1])
        gemm_kernel(
            grid, block,
            args=(m, n, k, alpha, A, int(transA), B, int(transB), beta, C_out)
        )
        cp.cuda.Stream.null.synchronize()
        return C_out.get()

    # ---- Test 1: No transpose (A * B) ----
    expected_AB = np.array([[19.0, 22.0],
                            [43.0, 50.0]], dtype=np.float32)
    C1 = run_gemm(False, False)
    print("\nKernel output (AB):\n", C1)
    print("Expected (AB):\n", expected_AB)
    diff_ab = np.abs(C1 - expected_AB).max()
    print(f"Max diff: {diff_ab:.6f}")

    # ---- Test 2: Transpose A (Aᵀ * B) ----
    # Aᵀ = [[1,3],[2,4]]
    expected_ATB = np.array([[26.0, 30.0],   # [1*5+3*7, 1*6+3*8]
                             [38.0, 44.0]],  # [2*5+4*7, 2*6+4*8]
                            dtype=np.float32)
    C2 = run_gemm(True, False)
    print("\nKernel output (AᵀB):\n", C2)
    print("Expected (AᵀB):\n", expected_ATB)
    diff_atb = np.abs(C2 - expected_ATB).max()
    print(f"Max diff: {diff_atb:.6f}")

    # ---- Test 3: Transpose B (A * Bᵀ) ----
    # Bᵀ = [[5,7],[6,8]]
    expected_ABT = np.array([[17.0, 23.0],   # [1*5+2*6, 1*7+2*8]
                             [39.0, 53.0]],  # [3*5+4*6, 3*7+4*8]
                            dtype=np.float32)
    C3 = run_gemm(False, True)
    print("\nKernel output (ABᵀ):\n", C3)
    print("Expected (ABᵀ):\n", expected_ABT)
    diff_abt = np.abs(C3 - expected_ABT).max()
    print(f"Max diff: {diff_abt:.6f}")

    # ---- Test 4: Both transposed (Aᵀ * Bᵀ) ----
    # Aᵀ * Bᵀ with above values = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]]? Wait careful:
    # Aᵀ (2x2) = [[1,3],[2,4]], Bᵀ (2x2) = [[5,7],[6,8]]
    # Product = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]] = [[5+18, 7+24], [10+24, 14+32]] = [[23,31],[34,46]]
    expected_ATBT = np.array([[23.0, 31.0],
                              [34.0, 46.0]], dtype=np.float32)
    C4 = run_gemm(True, True)
    print("\nKernel output (AᵀBᵀ):\n", C4)
    print("Expected (AᵀBᵀ):\n", expected_ATBT)
    diff_atbt = np.abs(C4 - expected_ATBT).max()
    print(f"Max diff: {diff_atbt:.6f}")

    # Summarize
    print("\n--- Summary ---")
    if all(d < 1e-4 for d in [diff_ab, diff_atb, diff_abt, diff_atbt]):
        print("✅ All small tests passed! The kernel is correct.")
    else:
        print("❌ Some tests failed. The kernel has a bug.")

@app.local_entrypoint()
def main():
    test_gemm.remote()