#!/usr/bin/env bash
set -euo pipefail

BIN=./build/sgemm_bench
M=4096; N=4096; K=4096

for a in 0 1 2 3 4 5 6; do
  echo "---- algo $a ----"
  $BIN --m=$M --n=$N --k=$K --algo=$a --iters=50 --warmup=10 --alpha=1 --beta=0
done
