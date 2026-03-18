import math
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Config & deterministic
# -----------------------

class TinyConfig:
    def __init__(
        self,
        hidden_size=8,
        moe_intermediate_size=12,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        topk_method="greedy",
        n_group=1,
        topk_group=1,
    ):
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group


def set_deterministic(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------
# MLP expert block
# -----------------------

class TinyMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        hidden_size = config.hidden_size
        if intermediate_size is None:
            intermediate_size = config.moe_intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden = torch.sigmoid(gate) * up
        out = self.down_proj(hidden)
        return out


# -----------------------
# Top‑K Router
# -----------------------

class TinyTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, config.hidden_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seqlen, hidden = hidden_states.shape
        x = hidden_states.reshape(-1, hidden)  # [B*S, H]
        logits = F.linear(x, self.weight)      # [B*S, E]
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        if self.topk_method != "greedy":
            raise NotImplementedError("Only 'greedy' topk_method implemented")

        topk_weights, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        topk_weights = topk_weights * self.routed_scaling_factor

        topk_indices = topk_indices.reshape(bsz, seqlen, self.top_k)
        topk_weights = topk_weights.reshape(bsz, seqlen, self.top_k)
        return topk_indices, topk_weights


# -----------------------
# MoE block
# -----------------------

class TinyMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.shared_experts = nn.ModuleList(
            TinyMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.n_shared_experts)
        )

        self.experts = nn.ModuleList(
            TinyMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )

        self.gate = TinyTopkRouter(config)

    def forward(self, hidden_states):
        residual = hidden_states
        bsz, seqlen, hidden_size = hidden_states.shape

        # shared
        shared_out = 0.0
        for expert in self.shared_experts:
            shared_out = shared_out + expert(hidden_states)

        # routed
        topk_indices, topk_weights = self.gate(hidden_states)

        x_flat = hidden_states.view(bsz * seqlen, hidden_size)
        k = topk_indices.shape[-1]

        expert_outputs = []
        for expert in self.experts:
            y = expert(x_flat)  # [N, H]
            expert_outputs.append(y)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [N, E, H]

        idx_flat = topk_indices.view(-1, k)          # [N, K]
        w_flat = topk_weights.view(-1, k).unsqueeze(-1)  # [N, K, 1]

        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, hidden_size)  # [N,K,H]
        selected = torch.gather(expert_outputs, 1, idx_expanded)          # [N,K,H]

        routed_out_flat = (selected * w_flat).sum(dim=1)  # [N,H]
        routed_out = routed_out_flat.view(bsz, seqlen, hidden_size)

        hidden_states = residual + shared_out + routed_out
        return hidden_states


# -----------------------
# Helper: save one case
# -----------------------

def save_case_to_bin(case_dir, config, model, hidden_states):
    os.makedirs(case_dir, exist_ok=True)

    # collect weights
    weights = {}
    weights["router_weight"] = model.gate.weight.detach().cpu().numpy()
    for i, expert in enumerate(model.shared_experts):
        weights[f"shared_{i}_gate_proj_weight"] = expert.gate_proj.weight.detach().cpu().numpy()
        weights[f"shared_{i}_up_proj_weight"] = expert.up_proj.weight.detach().cpu().numpy()
        weights[f"shared_{i}_down_proj_weight"] = expert.down_proj.weight.detach().cpu().numpy()
    for i, expert in enumerate(model.experts):
        weights[f"expert_{i}_gate_proj_weight"] = expert.gate_proj.weight.detach().cpu().numpy()
        weights[f"expert_{i}_up_proj_weight"] = expert.up_proj.weight.detach().cpu().numpy()
        weights[f"expert_{i}_down_proj_weight"] = expert.down_proj.weight.detach().cpu().numpy()

    with torch.no_grad():
        outputs = model(hidden_states)
        topk_indices, topk_weights = model.gate(hidden_states)

    inputs_np = hidden_states.detach().cpu().numpy().astype(np.float32)
    outputs_np = outputs.detach().cpu().numpy().astype(np.float32)
    topk_indices_np = topk_indices.detach().cpu().numpy().astype(np.int32)
    topk_weights_np = topk_weights.detach().cpu().numpy().astype(np.float32)

    # save config + meta for C
    meta = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.moe_intermediate_size,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "top_k": config.num_experts_per_tok,
        "routed_scaling_factor": config.routed_scaling_factor,
        "batch_size": int(inputs_np.shape[0]),
        "seq_len": int(inputs_np.shape[1]),
    }
    with open(os.path.join(case_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    def save_bin(name, arr):
        arr.tofile(os.path.join(case_dir, name + ".bin"))

    save_bin("inputs", inputs_np)
    save_bin("outputs", outputs_np)
    save_bin("topk_indices", topk_indices_np)
    save_bin("topk_weights", topk_weights_np)

    for k, v in weights.items():
        save_bin(k, v.astype(np.float32))


# -----------------------
# Multi‑test generator
# -----------------------

def generate_all_cases():
    base_dir = "deepseek_moe_tests_multi"
    os.makedirs(base_dir, exist_ok=True)

    # global config (same dimensions across tests so C is simpler)
    config = TinyConfig(
        hidden_size=8,
        moe_intermediate_size=12,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        topk_method="greedy",
        n_group=1,
        topk_group=1,
    )

    # Define test cases with different seeds and batch/seq sizes
    test_specs = [
        # (name, seed_weights, seed_inputs, batch_size, seq_len)
        ("case_01", 2024, 2025, 2, 3),
        ("case_02", 123, 456, 1, 5),
        ("case_03", 999, 1001, 3, 2),
    ]

    all_meta = []

    for name, seed_w, seed_x, bsz, seqlen in test_specs:
        print("Generating", name)
        set_deterministic(seed_w)
        model = TinyMoE(config)
        model.eval()

        set_deterministic(seed_x)
        hidden_states = torch.randn(bsz, seqlen, config.hidden_size)

        case_dir = os.path.join(base_dir, name)
        save_case_to_bin(case_dir, config, model, hidden_states)

        all_meta.append({
            "name": name,
            "seed_weights": seed_w,
            "seed_inputs": seed_x,
            "batch_size": bsz,
            "seq_len": seqlen,
        })

    # Master list for C test runner
    with open(os.path.join(base_dir, "cases.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    print("Wrote", os.path.join(base_dir, "cases.json"))


if __name__ == "__main__":
    generate_all_cases()