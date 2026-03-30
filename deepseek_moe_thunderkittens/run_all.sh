#!/bin/bash
echo "=== Step 1: ThunderKittens MoE correctness + benchmark ==="
modal run modal_moe.py

echo ""
echo "=== Step 2: WMMA CUDA kernel on B200 ==="
modal run modal_compile_kernel.py