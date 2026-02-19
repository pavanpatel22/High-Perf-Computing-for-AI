#pragma once

// Computes: C <- alpha * op(A) * op(B) + beta * C
// Row-major storage.
//
// Dimensions:
//   op(A): m x k
//   op(B): k x n
//   C:     m x n
//
// Storage conventions (important):
// - transposeA == false: A is stored as m x k
// - transposeA == true : A is stored as k x m  (so op(A)=A^T has shape m x k)
//
// - transposeB == false: B is stored as k x n
// - transposeB == true : B is stored as n x k  (so op(B)=B^T has shape k x n)
void gemm_cuda(
    int m, int n, int k,
    float alpha,
    const float* A, bool transposeA,
    const float* B, bool transposeB,
    float beta,
    float* C);
