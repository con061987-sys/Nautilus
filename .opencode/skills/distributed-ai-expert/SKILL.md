---
name: distributed-ai-expert
description: MUST BE USED when designing model sharding strategies, optimizing communication patterns, or working with multi-device execution. Deep knowledge of tensor parallelism, pipeline parallelism, FSDP, sequence parallelism, collective communication optimization, and overlap strategies. The difference between sharding that works and sharding that performs.
---

# Distributed AI Expert — Parallelism & Sharding

## The Three Parallelism Dimensions

### 1. Tensor Parallelism (Intra-Operator)
Split individual tensor operations across devices.

| Split | When | Communication per Op |
|---|---|---|
| Row-wise (split along input dim) | Large hidden dims | All-reduce after matmul |
| Column-wise (split along output dim) | Large hidden dims | All-reduce before activation |
| Attention head split | Multi-head attention | All-gather after softmax |

**Rules of thumb:**
- Use when model hidden dimension > 4096 and interconnect is fast (NVLink/NvSwitch: 900 GB/s)
- For slow interconnects (Ethernet: 25-100 Gbps), tensor parallelism adds too much overhead
- Best for transformer decoder layers where GEMMs dominate

**Communication volume (for row-wise matmul split):**
```
Input:  [B, M, K]  Output: [B, M, N/P] per P devices
All-reduce of [B, M, N/P]  →  volume = B × M × N/P × 4 bytes × (P-1)/P
```

### 2. Pipeline Parallelism (Inter-Operator)
Split layers across devices. Each device handles a contiguous set of layers.

| Schedule | Bubble Overhead | Memory Savings |
|---|---|---|
| GPipe | O(P-1) bubble | 1/P of single-device memory |
| 1F1B (One-Forward-One-Backward) | O(P-1) bubble | ~2× GPipe memory efficiency |
| Interleaved 1F1B | O((P-1)/M) | Higher throughput, more communication |

**Interleaved 1F1B (modern standard):**
```
Step 1: Device 0 does F(0-3), Device 1 does F(4-7)   # Forward passes
Step 2: Device 0 sends activations to Device 1
Step 3: Device 1 does B(7-4), Device 0 does B(3-0)   # Backward passes
```
M = number of microbatches. P = number of pipeline stages.

**Rules of thumb:**
- Use when model doesn't fit on one device (>80B parameters)
- Pipeline flush overhead must be <5% for good efficiency
- Number of microbatches ≥ 4× number of pipeline stages

### 3. Data Parallelism / FSDP
Replicate model, split data across devices. FSDP shards optimizer states, gradients, and optionally parameters.

| Strategy | Memory per Device | Communication | Compute |
|---|---|---|---|
| DDP (Data Parallel) | Full model + optimizer | All-reduce gradients | All data |
| FSDP (no shard) | Full model + optimizer | All-gather params, reduce-scatter grads | All data |
| FSDP (shard optim) | O(model/P) + full fwd | All-gather params, reduce-scatter grads | All data |
| FSDP (shard optim+param) | O(model/P) fwd O(model) | All-gather params (every fwd/bwd) | All data |

**Hybrid FSDP sharding policy for transformers:**
```
Embedding → FSDP (no shard)    # Small, accessed every step
Self-Attention → FSDP (shard)  # Large, computation-heavy
MLP → FSDP (shard)             # Largest, communication-heavy
LM Head → FSDP (no shard)      # Must gather full logits
```

### Communication Topology-Aware Sharding

The optimal strategy depends on your hardware topology:

```
Nvidia DGX H100 (8× H100, NVLink):
  Intra-node: 900 GB/s per GPU → TP is efficient
  Inter-node: 400 Gbps IB → DP/FSDP is better

AMD MI300X (8× MI300X, Infinity Fabric):
  Intra-node: ~800 GB/s per GCD → TP is efficient  
  Inter-node: 400 Gbps Ethernet → DP/FSDP is better

Mixed Cluster (2× AMD + 2× Nvidia + 2× Intel):
  Stratified sharding: TP within vendor-matched pairs
                      PP across vendors (slower interconnect between pairs)
                      DP across all 6 devices (after topology mapping)
```

## Communication Optimization

### Overlap Strategies
```python
# Pattern: Overlap all-gather with compute
def overlapped_fsdp_forward(param, input):
    # Begin all-gather asynchronously
    handle = all_gather_async(param.full_param, param.shard_group)
    # Compute on local shard's input while waiting
    local_output = compute(input, param.shard)
    # Wait for full parameters
    wait(handle)
    # Complete computation with full params
    full_output = compute_remainder(input, param)
    return full_output
```

### Collective Communication Performance (8× GPU)
| Operation | NVLink | 400 Gbps IB | 100 Gbps Ethernet |
|---|---|---|---|
| All-reduce (1GB) | ~5 ms | ~25 ms | ~100 ms |
| All-gather (1GB) | ~3 ms | ~15 ms | ~60 ms |
| Reduce-scatter (1GB) | ~3 ms | ~15 ms | ~60 ms |
| P2P (1GB) | ~2 ms | ~10 ms | ~40 ms |

### Memory Bandwidth for Sharding
```python
# Communication computation ratio
comm_ratio = (communication_volume_bytes * num_devices) / (compute_time_seconds * bandwidth_bytes_per_sec)
# If comm_ratio > 0.3: communication bound, need overlap
# If comm_ratio < 0.1: compute bound, sharding can be more aggressive
```

## Recommended Strategies by Model Size

| Model Size | Single GPU | ≤ 4 GPUs | 4-16 GPUs | 16+ GPUs |
|---|---|---|---|---|
| < 7B | FSDP only | FSDP | FSDP + TP | FSDP + TP + PP |
| 7B-30B | Not possible | FSDP + TP | FSDP + TP | FSDP + TP + PP |
| 30B-100B | Not possible | Not possible | FSDP + TP + PP | 3D Parallelism |
| 100B+ | Not possible | Not possible | Not possible | 3D + Sequence Parallel |

## When This Skill Triggers

- Designing sharding for the auto-sharding bridge (Phase 3)
- Analyzing communication patterns in distributed execution
- Optimizing the GSPMD cost model for our specific cluster topology
- Debugging distributed training performance issues
- Deciding between tensor vs. pipeline vs. data parallelism for a given model
- Configuring device mesh for OpenXLA
