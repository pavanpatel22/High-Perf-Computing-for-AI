import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from generate_tests import TinyConfig, TinyMoE, set_deterministic
from moe_ep_distributed import run_case


def bench_baseline(model, x, warmup=5, iters=30):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000


def bench_ep(test_dir, world_size=2, iters=3):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_case(test_dir, world_size=world_size)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def main():
    config = TinyConfig()
    config.hidden_size            = 64
    config.moe_intermediate_size  = 128
    config.n_routed_experts       = 8
    config.n_shared_experts       = 1
    config.num_experts_per_tok    = 2

    set_deterministic(42)
    model = TinyMoE(config).eval()

    print("Baseline (single-process HF-style):")
    print(f"{'Tokens':>8} {'ms':>10} {'tok/s':>12}")
    print("-" * 35)

    baseline_results = []
    for B, S in [(1, 32), (2, 64), (4, 128), (8, 256)]:
        tokens = B * S
        x = torch.randn(B, S, config.hidden_size)
        ms = bench_baseline(model, x)
        tps = tokens / (ms / 1000)
        print(f"{tokens:>8} {ms:>10.3f} {tps:>12.0f}")
        baseline_results.append({
            "tokens": tokens,
            "ms": round(ms, 3),
            "tps": round(tps, 1)
        })

    print("\nEP distributed (2 simulated ranks, case_01):")
    test_dir = os.path.join(
        os.path.dirname(__file__), "..", "tests", "case_01"
    )
    ep_ms  = bench_ep(test_dir, world_size=2, iters=3)
    ep_tps = (2 * 3) / (ep_ms / 1000)
    print(f"  avg latency : {ep_ms:.1f} ms")
    print(f"  throughput  : {ep_tps:.0f} tok/s")

    os.makedirs("benchmarks", exist_ok=True)
    results = {
        "baseline": baseline_results,
        "ep_distributed": {
            "ms": round(ep_ms, 1),
            "tps": round(ep_tps, 1)
        }
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmarks", "comparison_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison_results.json")


if __name__ == "__main__":
    main()