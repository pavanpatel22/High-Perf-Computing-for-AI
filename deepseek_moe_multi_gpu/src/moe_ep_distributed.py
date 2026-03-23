import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


# -----------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))

    def load_weights(self, gate_w, up_w, down_w):
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.from_numpy(gate_w))
            self.up_proj.weight.copy_(torch.from_numpy(up_w))
            self.down_proj.weight.copy_(torch.from_numpy(down_w))


# -----------------------------------------------------------------------
# All-to-all implemented via point-to-point send/recv (gloo compatible)
# send_list[r] is the tensor this rank sends to rank r
# returns recv_list[r] = tensor received from rank r
# -----------------------------------------------------------------------

def all_to_all_p2p(send_list, rank, world_size):
    recv_list = [torch.zeros_like(send_list[r]) for r in range(world_size)]
    reqs = []

    # Post all sends and receives
    for r in range(world_size):
        if r != rank:
            reqs.append(dist.isend(send_list[r].contiguous(), dst=r))
            reqs.append(dist.irecv(recv_list[r], src=r))

    # Copy own data locally (no communication needed)
    recv_list[rank].copy_(send_list[rank])

    # Wait for all transfers to complete
    for req in reqs:
        req.wait()

    return recv_list


# -----------------------------------------------------------------------
# One rank's MoE EP forward pass
# -----------------------------------------------------------------------

def moe_ep_forward_rank(rank, world_size, cfg, test_dir, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    torch.manual_seed(0)

    H     = cfg["hidden_size"]
    I     = cfg["intermediate_size"]
    E     = cfg["n_routed_experts"]
    NS    = cfg["n_shared_experts"]
    K     = cfg["top_k"]
    B     = cfg["batch_size"]
    S     = cfg["seq_len"]
    N     = B * S

    def lf(name, shape):
        path = os.path.join(test_dir, name + ".bin")
        arr  = np.fromfile(path, dtype=np.float32)
        return arr.reshape(shape)

    def li(name, shape):
        path = os.path.join(test_dir, name + ".bin")
        arr  = np.fromfile(path, dtype=np.int32)
        return arr.reshape(shape)

    inputs   = torch.from_numpy(lf("inputs",       (N, H)))
    expected = torch.from_numpy(lf("outputs",      (N, H)))
    topk_idx = torch.from_numpy(li("topk_indices", (N, K)))
    topk_w   = torch.from_numpy(lf("topk_weights", (N, K)))

    # Load shared experts
    shared_experts = []
    for si in range(NS):
        mlp = TinyMLP(H, I)
        mlp.load_weights(
            lf(f"shared_{si}_gate", (I, H)),
            lf(f"shared_{si}_up",   (I, H)),
            lf(f"shared_{si}_down", (H, I))
        )
        mlp.eval()
        shared_experts.append(mlp)

    # Load all routed experts
    all_experts = []
    for e in range(E):
        mlp = TinyMLP(H, I)
        mlp.load_weights(
            lf(f"expert_{e}_gate", (I, H)),
            lf(f"expert_{e}_up",   (I, H)),
            lf(f"expert_{e}_down", (H, I))
        )
        mlp.eval()
        all_experts.append(mlp)

    # ----------------------------------------------------------------
    # Data Parallelism: split tokens across ranks
    # ----------------------------------------------------------------
    tokens_per_rank = (N + world_size - 1) // world_size
    tok_start = rank * tokens_per_rank
    tok_end   = min(tok_start + tokens_per_rank, N)
    N_local   = tok_end - tok_start

    local_inputs   = inputs[tok_start:tok_end]
    local_topk_idx = topk_idx[tok_start:tok_end]
    local_topk_w   = topk_w[tok_start:tok_end]

    # ----------------------------------------------------------------
    # Dispatch: build per-destination send buffers
    # Expert e is owned by rank: e % world_size
    # ----------------------------------------------------------------
    send_tok_ids    = [[] for _ in range(world_size)]
    send_expert_ids = [[] for _ in range(world_size)]
    send_weights    = [[] for _ in range(world_size)]
    send_embeds     = [[] for _ in range(world_size)]

    for t in range(N_local):
        for ki in range(K):
            e_id  = local_topk_idx[t, ki].item()
            owner = e_id % world_size
            send_tok_ids[owner].append(t)
            send_expert_ids[owner].append(e_id)
            send_weights[owner].append(local_topk_w[t, ki].item())
            send_embeds[owner].append(local_inputs[t].clone())

    # Exchange send counts so all ranks agree on buffer sizes
    local_send_counts = torch.tensor(
        [len(send_tok_ids[r]) for r in range(world_size)], dtype=torch.int32
    )
    gathered_counts = [torch.zeros(world_size, dtype=torch.int32)
                       for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_send_counts)
    # gathered_counts[r][s] = how many slots rank r sends to rank s

    max_slots = max(c.max().item() for c in gathered_counts)
    max_slots = max(int(max_slots), 1)

    # Pack into [world_size, max_slots, SLOT] tensor
    SLOT = 3 + H
    send_tensor = torch.zeros(world_size, max_slots, SLOT)
    for r in range(world_size):
        for si in range(len(send_tok_ids[r])):
            send_tensor[r, si, 0] = float(send_tok_ids[r][si])
            send_tensor[r, si, 1] = float(send_expert_ids[r][si])
            send_tensor[r, si, 2] = send_weights[r][si]
            send_tensor[r, si, 3:] = send_embeds[r][si]

    # All-to-all dispatch using p2p (gloo compatible)
    send_list = [send_tensor[r].contiguous() for r in range(world_size)]
    recv_list = all_to_all_p2p(send_list, rank, world_size)

    # ----------------------------------------------------------------
    # Local expert compute on received tokens
    # ----------------------------------------------------------------
    # gathered_counts[src_rank][rank] = how many slots src_rank sent to me
    expert_results = []
    with torch.no_grad():
        for r in range(world_size):
            n_recv = int(gathered_counts[r][rank].item())
            for si in range(n_recv):
                slot        = recv_list[r][si]
                tok_id_orig = int(slot[0].item())
                e_id        = int(slot[1].item())
                w           = slot[2].item()
                embed       = slot[3:].unsqueeze(0)
                out         = all_experts[e_id](embed).squeeze(0)
                expert_results.append((r, tok_id_orig, w, out))

    # ----------------------------------------------------------------
    # Combine: send results back to token owners
    # ----------------------------------------------------------------
    RSLOT = 2 + H
    back_counters    = [0] * world_size
    back_send_tensor = torch.zeros(world_size, max_slots, RSLOT)

    for (src_rank, tok_id_orig, w, out) in expert_results:
        si = back_counters[src_rank]
        if si < max_slots:
            back_send_tensor[src_rank, si, 0] = float(tok_id_orig)
            back_send_tensor[src_rank, si, 1] = w
            back_send_tensor[src_rank, si, 2:] = out.detach()
            back_counters[src_rank] += 1

    back_send_counts_t = torch.tensor(back_counters, dtype=torch.int32)
    all_back_counts    = [torch.zeros(world_size, dtype=torch.int32)
                          for _ in range(world_size)]
    dist.all_gather(all_back_counts, back_send_counts_t)

    back_send_list = [back_send_tensor[r].contiguous() for r in range(world_size)]
    back_recv_list = all_to_all_p2p(back_send_list, rank, world_size)

    # ----------------------------------------------------------------
    # Accumulate routed output for local tokens
    # ----------------------------------------------------------------
    routed_out = torch.zeros(N_local, H)
    for r in range(world_size):
        n_back = int(all_back_counts[r][rank].item())
        for si in range(n_back):
            slot        = back_recv_list[r][si]
            tok_id_orig = int(slot[0].item())
            w           = slot[1].item()
            out         = slot[2:]
            if 0 <= tok_id_orig < N_local:
                routed_out[tok_id_orig] += w * out

    # ----------------------------------------------------------------
    # Shared experts: replicated on all ranks, run on local tokens
    # ----------------------------------------------------------------
    shared_out = torch.zeros(N_local, H)
    with torch.no_grad():
        for se in shared_experts:
            shared_out += se(local_inputs)

    # ----------------------------------------------------------------
    # Final output = residual + shared + routed
    # ----------------------------------------------------------------
    local_final = local_inputs + shared_out + routed_out  # [N_local, H]

    # ----------------------------------------------------------------
    # Gather all ranks' outputs on rank 0 and compare
    # ----------------------------------------------------------------
    padded = torch.zeros(tokens_per_rank, H)
    padded[:N_local] = local_final
    gathered = [torch.zeros(tokens_per_rank, H) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if rank == 0:
        full_out = torch.cat(gathered, dim=0)[:N]
        max_err  = (full_out - expected).abs().max().item()
        result_queue.put(max_err)
    else:
        result_queue.put(None)

    dist.destroy_process_group()


# -----------------------------------------------------------------------
# Public API: run one test case across world_size simulated ranks
# -----------------------------------------------------------------------

def run_case(test_dir, world_size=2):
    with open(os.path.join(test_dir, "meta.json")) as f:
        cfg = json.load(f)

    ctx          = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=moe_ep_forward_rank,
            args=(rank, world_size, cfg, test_dir, result_queue)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = []
    while not result_queue.empty():
        r = result_queue.get()
        if r is not None:
            results.append(r)

    return results[0] if results else float("inf")