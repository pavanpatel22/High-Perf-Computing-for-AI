# CUDA GEMM Implementation Assignment

## Overview
Implementation of a General Matrix Multiplication (GEMM) kernel in CUDA with support for:
- Basic matrix multiplication: D = α * A * B + β * C
- Extended GEMM with optional transpose: C = α * op(A) * op(B) + β * C
- In-place updates (for extended version)



## Requirements
- CUDA Toolkit (10.0 or higher)
- NVIDIA GPU with compute capability 3.5 or higher
- Linux/MacOS with g++ or Windows with Visual Studio

## Build Instructions
```bash
# Compile
make

# Run tests
make run

# Clean build files
make clean