# GEMM kernel in CUDA (Naive, global-memory only)

Implements:
C <- alpha * op(A) * op(B) + beta * C

Row-major storage.

## Transpose storage conventions (important)
To support transpose flags without doing an actual transpose inside the kernel:

- transposeA=false: A is stored as m x k
- transposeA=true : A is stored as k x m   (op(A)=A^T is m x k)

- transposeB=false: B is stored as k x n
- transposeB=true : B is stored as n x k   (op(B)=B^T is k x n)

So, when you set transposeA=true, you must pass A in k-by-m layout.
When you set transposeB=true, you must pass B in n-by-k layout.

## Build and run (Linux/macOS with CUDA)
```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
./build/naive_gemm
