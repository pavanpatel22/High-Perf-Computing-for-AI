#!/usr/bin/env bash
set -euo pipefail

BIN=./build/sgemm_bench
M=4096; N=4096; K=4096
ALGO=${1:-6}

# Example:
#   ./scripts/profile_ncu.sh 6
ncu --set full --target-processes all \
  $BIN --m=$M --n=$N --k=$K --algo=$ALGO --iters=20 --warmup=5 --alpha=1 --beta=0
