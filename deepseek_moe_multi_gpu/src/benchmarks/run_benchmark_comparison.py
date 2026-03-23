"""
Side-by-side: PyTorch (CPU) vs GPU timings from moe_nccl
Reads benchmarks/hf_results.json, also runs GPU timings if CUDA available.
"""
import json, os, torch, time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_tests import TinyConfig, TinyMoE, set_deterministic


def bench_torch(model, x, warmup=5, iters=30):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(x)
        if x.device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): model(x)
        if x.device.type == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def main():
    config = TinyConfig()
    config.hidden_size           = 64
    config.moe_intermediate_size = 128
    config.n_routed_experts      = 8
    config.n_shared_experts      = 1
    config.num_experts_per_tok   = 2

    set_deterministic(42)
    cpu_model = TinyMoE(config).eval()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_model = TinyMoE(config).cuda().eval()

    print(f"\n{'Tokens':>8} {'CPU ms':>10} {'CPU tok/s':>12}", end="")
    if has_cuda:
        print(f" {'GPU ms':>10} {'GPU tok/s':>12} {'Speedup':>8}", end="")
    print()
    print("-" * (45 + (35 if has_cuda else 0)))

    results = []
    for total_tokens in [128, 512, 2048, 8192, 32768]:
        B, S = 4, total_tokens // 4
        x_cpu = torch.randn(B, S, config.hidden_size)
        cpu_ms  = bench_torch(cpu_model, x_cpu)
        cpu_tps = total_tokens / (cpu_ms / 1000)
        row = {"tokens": total_tokens, "cpu_ms": round(cpu_ms, 3),
               "cpu_tps": round(cpu_tps, 1)}

        line = f"{total_tokens:>8} {cpu_ms:>10.3f} {cpu_tps:>12.0f}"

        if has_cuda:
            x_gpu = x_cpu.cuda()
            gpu_ms  = bench_torch(gpu_model, x_gpu)
            gpu_tps = total_tokens / (gpu_ms / 1000)
            speedup = cpu_ms / gpu_ms
            row.update({"gpu_ms": round(gpu_ms, 3), "gpu_tps": round(gpu_tps, 1),
                        "speedup": round(speedup, 2)})
            line += f" {gpu_ms:>10.3f} {gpu_tps:>12.0f} {speedup:>8.2f}x"

        print(line)
        results.append(row)

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmarks/comparison_results.json")


if __name__ == "__main__":
    main()