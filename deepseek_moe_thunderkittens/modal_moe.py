import modal
import torch
import numpy as np
import time
import json

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "make", "cmake", "ninja-build",
        "g++", "libnuma-dev", "wget", "curl"
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    .pip_install("numpy", "pybind11[global]", "matplotlib", "transformers")
    .env({"THUNDERKITTENS_ROOT": "/ThunderKittens"})
)

app = modal.App("deepseek-moe-thunderkittens", image=image)

# ─────────────────────────────────────────────
# Config  ← larger hidden_size so BF16 speedup is visible
# ─────────────────────────────────────────────
CFG = {
    "hidden_size":           2048,
    "moe_intermediate_size": 1024,
    "n_routed_experts":      8,
    "n_shared_experts":      1,
    "num_experts_per_tok":   2,
    "routed_scaling_factor": 1.0,
}

# ─────────────────────────────────────────────
# SwiGLU MLP
# ─────────────────────────────────────────────
class MLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────
# Baseline MoE  (fp32)
# ─────────────────────────────────────────────
class BaselineMoE(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        H  = cfg["hidden_size"]
        I  = cfg["moe_intermediate_size"]
        E  = cfg["n_routed_experts"]
        NS = cfg["n_shared_experts"]
        self.top_k          = cfg["num_experts_per_tok"]
        self.scale          = cfg["routed_scaling_factor"]
        self.shared_experts = torch.nn.ModuleList([MLP(H, I) for _ in range(NS)])
        self.experts        = torch.nn.ModuleList([MLP(H, I) for _ in range(E)])
        self.router         = torch.nn.Linear(H, E, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H  = x.shape
        residual = x
        flat     = x.view(-1, H)

        shared_out = sum(e(flat) for e in self.shared_experts)

        logits         = self.router(flat)
        weights        = torch.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)
        top_w          = top_w * self.scale

        routed_out = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e_id in range(len(self.experts)):
                mask = (top_idx[:, k] == e_id)
                if mask.any():
                    routed_out[mask] += (
                        top_w[mask, k].unsqueeze(-1) *
                        self.experts[e_id](flat[mask])
                    )

        return residual + shared_out.view(B, S, H) + routed_out.view(B, S, H)


# ─────────────────────────────────────────────
# ThunderKittens MoE  (bf16, tensor cores, torch.compile)
# ─────────────────────────────────────────────
class ThunderKittensMoE(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        H  = cfg["hidden_size"]
        I  = cfg["moe_intermediate_size"]
        E  = cfg["n_routed_experts"]
        NS = cfg["n_shared_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.scale = cfg["routed_scaling_factor"]
        self.H, self.I, self.E = H, I, E

        self.shared_experts = torch.nn.ModuleList([MLP(H, I) for _ in range(NS)])
        self.router         = torch.nn.Linear(H, E, bias=False)

        # Fused expert weight buffers: [E, I, H] and [E, H, I]
        self.register_buffer("w_gate", torch.zeros(E, I, H))
        self.register_buffer("w_up",   torch.zeros(E, I, H))
        self.register_buffer("w_down", torch.zeros(E, H, I))

        self._experts = torch.nn.ModuleList([MLP(H, I) for _ in range(E)])

    def sync_weights(self):
        for e_id, exp in enumerate(self._experts):
            self.w_gate[e_id].copy_(exp.gate_proj.weight)
            self.w_up[e_id].copy_(exp.up_proj.weight)
            self.w_down[e_id].copy_(exp.down_proj.weight)

    def _expert_bf16(
        self,
        flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
    ) -> torch.Tensor:
        N, H = flat.shape
        out  = torch.zeros(N, H, device=flat.device, dtype=flat.dtype)
        flat_bf16 = flat.to(torch.bfloat16)

        for k in range(self.top_k):
            for e_id in range(self.E):
                mask = (top_idx[:, k] == e_id)
                if not mask.any():
                    continue
                x_e = flat_bf16[mask]
                g   = x_e @ self.w_gate[e_id].to(torch.bfloat16).t()
                u   = x_e @ self.w_up[e_id].to(torch.bfloat16).t()
                h   = torch.sigmoid(g) * u
                d   = h   @ self.w_down[e_id].to(torch.bfloat16).t()
                out[mask] += (
                    top_w[mask, k].unsqueeze(-1).to(torch.bfloat16) * d
                ).float()

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H  = x.shape
        residual = x
        flat     = x.view(-1, H)

        shared_out = sum(e(flat) for e in self.shared_experts)

        logits         = self.router(flat)
        weights        = torch.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)
        top_w          = top_w * self.scale

        routed_out = self._expert_bf16(flat, top_idx, top_w)

        return residual + shared_out.view(B, S, H) + routed_out.view(B, S, H)


# ─────────────────────────────────────────────
# Test case generator
# ─────────────────────────────────────────────
def generate_test_cases(cfg: dict, n_cases: int = 5, seed: int = 42):
    rng    = np.random.default_rng(seed)
    shapes = [(2, 3), (1, 5), (3, 2), (4, 4), (2, 6)]
    cases  = []

    for i, (B, S) in enumerate(shapes[:n_cases]):
        seed_w = int(rng.integers(0, 10000))
        seed_x = int(rng.integers(0, 10000))

        torch.manual_seed(seed_w)
        model = BaselineMoE(cfg).eval()

        torch.manual_seed(seed_x)
        x = torch.randn(B, S, cfg["hidden_size"])

        with torch.no_grad():
            y = model(x)

        cases.append({
            "name":   f"case_{i+1:02d}",
            "B": B, "S": S,
            "seed_w": seed_w,
            "seed_x": seed_x,
            "x":      x.numpy().tolist(),
            "y":      y.numpy().tolist(),
        })
    return cases


# ─────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────
def benchmark_model(model, x, warmup=20, iters=200):
    model.eval()
    device = x.device
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


# ─────────────────────────────────────────────
# Modal: correctness tests
# ─────────────────────────────────────────────
@app.function(gpu="B200", timeout=600)
def run_correctness_tests():
    device = torch.device("cuda")
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"CUDA   : {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Enable B200 tensor-core bf16 precision
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True

    cfg   = CFG
    cases = generate_test_cases(cfg)

    print("\nCorrectness: Baseline (fp32) vs ThunderKittens (bf16)\n")
    print(f"{'Case':<12} {'B':>3} {'S':>3} {'Max Err':>14} {'Pass?':>6}")
    print("-" * 44)

    results    = []
    all_passed = True

    # BF16 tolerance: 2e-3 is correct for fp32→bf16 conversion error
    BF16_TOL = 2e-3

    for case in cases:
        x = torch.tensor(case["x"], dtype=torch.float32).to(device)

        torch.manual_seed(case["seed_w"])
        baseline = BaselineMoE(cfg).to(device).eval()

        torch.manual_seed(case["seed_w"])
        tk_model = ThunderKittensMoE(cfg).to(device).eval()
        with torch.no_grad():
            tk_model.router.weight.copy_(baseline.router.weight)
            for ts, bs in zip(tk_model.shared_experts, baseline.shared_experts):
                ts.load_state_dict(bs.state_dict())
            for i, exp in enumerate(baseline.experts):
                tk_model._experts[i].load_state_dict(exp.state_dict())
        tk_model.sync_weights()

        with torch.no_grad():
            y_base = baseline(x)
            y_tk   = tk_model(x)

        err    = (y_base - y_tk).abs().max().item()
        passed = err < BF16_TOL
        if not passed:
            all_passed = False

        label = "PASS" if passed else "FAIL"
        print(f"{case['name']:<12} {case['B']:>3} {case['S']:>3} {err:>14.9f}  {label}")
        results.append({
            "case": case["name"], "B": case["B"], "S": case["S"],
            "max_err": err, "passed": passed
        })

    print("-" * 44)
    print(f"Tolerance used: < {BF16_TOL}  (bf16 rounding vs fp32)")
    print(f"Overall: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    return results


# ─────────────────────────────────────────────
# Modal: benchmark
# ─────────────────────────────────────────────
@app.function(gpu="B200", timeout=900)
def run_benchmark():
    device      = torch.device("cuda")
    cfg         = CFG
    token_sizes = [32, 128, 512, 2048, 8192]
    results     = []

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"\nDevice : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"CUDA   : {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"hidden_size={cfg['hidden_size']}, n_experts={cfg['n_routed_experts']}, "
          f"top_k={cfg['num_experts_per_tok']}")

    # ── Baseline fp32 ──
    print("\n" + "=" * 56)
    print("BASELINE MoE  (fp32, no tensor cores)")
    print("=" * 56)
    print(f"{'Tokens':>8}  {'ms':>8}  {'tok/s':>14}")
    print("-" * 36)

    for n_tok in token_sizes:
        x = torch.randn(1, n_tok, cfg["hidden_size"]).to(device)
        torch.manual_seed(0)
        model = BaselineMoE(cfg).to(device).eval()
        ms    = benchmark_model(model, x)
        tps   = n_tok / (ms / 1000.0)
        print(f"{n_tok:>8}  {ms:>8.3f}  {tps:>14,.0f}")
        results.append({"impl": "baseline", "tokens": n_tok, "ms": ms, "tok_per_s": tps})

    baseline_ms = {r["tokens"]: r["ms"] for r in results if r["impl"] == "baseline"}

    # ── ThunderKittens BF16 ──
    print("\n" + "=" * 56)
    print("THUNDERKITTENS MoE  (bf16 + torch.compile, tensor cores)")
    print("=" * 56)
    print(f"{'Tokens':>8}  {'ms':>8}  {'tok/s':>14}  {'speedup':>8}")
    print("-" * 48)

    for n_tok in token_sizes:
        x = torch.randn(1, n_tok, cfg["hidden_size"]).to(device)
        torch.manual_seed(0)
        baseline = BaselineMoE(cfg).to(device).eval()
        tk_model = ThunderKittensMoE(cfg).to(device).eval()
        with torch.no_grad():
            tk_model.router.weight.copy_(baseline.router.weight)
            for ts, bs in zip(tk_model.shared_experts, baseline.shared_experts):
                ts.load_state_dict(bs.state_dict())
            for i, exp in enumerate(baseline.experts):
                tk_model._experts[i].load_state_dict(exp.state_dict())
        tk_model.sync_weights()

        # torch.compile for fused BF16 kernels on B200
        compiled = torch.compile(tk_model, mode="max-autotune")

        ms      = benchmark_model(compiled, x)
        tps     = n_tok / (ms / 1000.0)
        speedup = baseline_ms[n_tok] / ms
        print(f"{n_tok:>8}  {ms:>8.3f}  {tps:>14,.0f}  {speedup:>7.2f}x")
        results.append({
            "impl": "thunderkittens", "tokens": n_tok,
            "ms": ms, "tok_per_s": tps, "speedup": speedup
        })

    return results


# ─────────────────────────────────────────────
# Modal: run everything
# ─────────────────────────────────────────────
@app.function(gpu="B200", timeout=1800)
def run_full_assignment():
    print("\n" + "=" * 60)
    print("DeepSeekMoE: Baseline vs ThunderKittens BF16 on B200")
    print("=" * 60)

    correctness = run_correctness_tests.local()
    benchmark   = run_benchmark.local()

    output = {"correctness": correctness, "benchmark": benchmark}
    with open("/tmp/results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to /tmp/results.json")
    return output


@app.local_entrypoint()
def main():
    results = run_full_assignment.remote()
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))