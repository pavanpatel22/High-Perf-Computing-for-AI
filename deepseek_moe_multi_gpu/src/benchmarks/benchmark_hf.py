"""
Benchmark: TinyMoE (PyTorch reference) vs batch sizes.
Measures forward-pass latency and tokens/sec.
"""
import math, time, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from generate_tests import TinyConfig, TinyMoE, set_deterministic


def bench(model, x, warmup=5, iters=50):
    model.eval()
    device = x.device
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000   # ms per call


def main():
    config = TinyConfig()
    # Scale up hidden dim for a more realistic benchmark
    config.hidden_size          = 64
    config.moe_intermediate_size = 128
    config.n_routed_experts     = 8
    config.n_shared_experts     = 1
    config.num_experts_per_tok  = 2

    set_deterministic(42)
    model = TinyMoE(config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"Device: {device}")
    print(f"{'BatchSize':>10} {'SeqLen':>8} {'Tokens':>8} {'ms':>10} {'tok/s':>12}")
    print("-" * 55)

    results = []
    for B in [1, 2, 4, 8, 16]:
        for S in [32, 128, 512]:
            x = torch.randn(B, S, config.hidden_size, device=device)
            ms = bench(model, x)
            tps = (B * S) / (ms / 1000)
            print(f"{B:>10} {S:>8} {B*S:>8} {ms:>10.3f} {tps:>12.0f}")
            results.append({"B": B, "S": S, "tokens": B*S,
                            "ms": round(ms, 4), "tok_per_s": round(tps, 1)})

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/hf_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmarks/hf_results.json")


if __name__ == "__main__":
    main()