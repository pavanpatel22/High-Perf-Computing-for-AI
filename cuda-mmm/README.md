# CUDA MMM / SGEMM Worklog (H100-ready)

Implements multiple SGEMM kernels mirroring the worklog progression:
0) cuBLAS baseline
1) Naive
2) Coalesced mapping (1D thread indexing)
3) Shared-memory tiling (32x32)
4) 1D block tiling: multiple columns per thread
5) 2D block tiling: register micro-tile (BM=BN=128, BK=8, TM=TN=8)
6) Vectorized global loads (float4) for BM=BN=128, BK=8

## Build (H100)
```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
