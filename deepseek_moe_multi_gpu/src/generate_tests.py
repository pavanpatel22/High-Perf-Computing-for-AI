import math
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConfig:
    def __init__(self):
        self.hidden_size = 8
        self.moe_intermediate_size = 12
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.routed_scaling_factor = 2.5
        self.topk_method = "greedy"
        self.n_group = 1
        self.topk_group = 1


def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TinyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class TinyRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k  = config.num_experts_per_tok
        self.scale  = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        B, S, H = x.shape
        flat    = x.reshape(-1, H)
        scores  = F.linear(flat, self.weight).softmax(dim=-1)
        w, idx  = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        return idx.reshape(B, S, self.top_k), (w * self.scale).reshape(B, S, self.top_k)


class TinyMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config         = config
        self.gate           = TinyRouter(config)
        self.shared_experts = nn.ModuleList([TinyMLP(config) for _ in range(config.n_shared_experts)])
        self.experts        = nn.ModuleList([TinyMLP(config) for _ in range(config.n_routed_experts)])

    def forward(self, x):
        B, S, H    = x.shape
        residual   = x
        shared_out = sum(e(x) for e in self.shared_experts)
        idx, w     = self.gate(x)
        flat_x     = x.view(B * S, H)
        all_expert = torch.stack([e(flat_x) for e in self.experts], dim=1)   # [N,E,H]
        idx_exp    = idx.view(-1, self.config.num_experts_per_tok).unsqueeze(-1).expand(-1, -1, H)
        selected   = torch.gather(all_expert, 1, idx_exp)                    # [N,K,H]
        routed     = (selected * w.view(-1, self.config.num_experts_per_tok, 1)).sum(1).view(B, S, H)
        return residual + shared_out + routed


def save_case(out_dir, config, model, hidden_states):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        outputs             = model(hidden_states)
        topk_indices, topk_weights = model.gate(hidden_states)

    def s(name, arr):
        arr.astype(np.float32).tofile(os.path.join(out_dir, name + ".bin"))

    def si(name, arr):
        arr.astype(np.int32).tofile(os.path.join(out_dir, name + ".bin"))

    s("inputs",  hidden_states.detach().cpu().numpy())
    s("outputs", outputs.detach().cpu().numpy())
    si("topk_indices", topk_indices.detach().cpu().numpy())
    s("topk_weights",  topk_weights.detach().cpu().numpy())
    s("router_weight", model.gate.weight.detach().cpu().numpy())

    for i, e in enumerate(model.shared_experts):
        s(f"shared_{i}_gate", e.gate_proj.weight.detach().cpu().numpy())
        s(f"shared_{i}_up",   e.up_proj.weight.detach().cpu().numpy())
        s(f"shared_{i}_down", e.down_proj.weight.detach().cpu().numpy())

    for i, e in enumerate(model.experts):
        s(f"expert_{i}_gate", e.gate_proj.weight.detach().cpu().numpy())
        s(f"expert_{i}_up",   e.up_proj.weight.detach().cpu().numpy())
        s(f"expert_{i}_down", e.down_proj.weight.detach().cpu().numpy())

    meta = {
        "hidden_size":          config.hidden_size,
        "intermediate_size":    config.moe_intermediate_size,
        "n_routed_experts":     config.n_routed_experts,
        "n_shared_experts":     config.n_shared_experts,
        "top_k":                config.num_experts_per_tok,
        "routed_scaling_factor":config.routed_scaling_factor,
        "batch_size":           hidden_states.shape[0],
        "seq_len":              hidden_states.shape[1],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def generate_all():
    base = "tests"
    os.makedirs(base, exist_ok=True)
    config = TinyConfig()

    specs = [
        ("case_01", 2024, 2025, 2, 3),
        ("case_02", 123,  456,  1, 5),
        ("case_03", 999,  1001, 3, 2),
        ("case_04", 7777, 8888, 4, 4),
        ("case_05", 3141, 2718, 2, 6),
    ]

    manifest = []
    for name, sw, sx, B, S in specs:
        print(f"Generating {name}  (B={B}, S={S}, seed_w={sw}, seed_x={sx})")
        set_deterministic(sw)
        model = TinyMoE(config)
        set_deterministic(sx)
        x = torch.randn(B, S, config.hidden_size)
        save_case(os.path.join(base, name), config, model, x)
        manifest.append({"name": name, "batch": B, "seq": S,
                          "seed_w": sw, "seed_x": sx})

    with open(os.path.join(base, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(specs)} test cases to '{base}/'")


if __name__ == "__main__":
    generate_all()