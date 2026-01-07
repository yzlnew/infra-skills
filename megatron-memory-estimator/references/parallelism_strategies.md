# Parallelism Strategies Guide

Guide for choosing and configuring parallelism strategies for different model sizes and GPU configurations.

## Overview

Megatron supports multiple parallelism dimensions:
- **Data Parallelism (DP)**: Replicate model, split data
- **Tensor Parallelism (TP)**: Split weights within layers
- **Pipeline Parallelism (PP)**: Split layers across stages
- **Expert Parallelism (EP)**: Split experts (MoE only)
- **Context Parallelism (CP)**: Split sequence dimension

## Recommended Strategies by Model Size

### Small Models (< 10B parameters)

**Recommended:**
- TP = 1-2
- PP = 1
- EP = 1 (or 2 for MoE)
- CP = 1

**Rationale:**
- Small models fit in single GPU with optimization
- Avoid communication overhead from excessive parallelism
- Focus on data parallelism for throughput

**Example (7B dense model on 8 GPUs):**
```yaml
parallelism:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  # DP = 4 (computed automatically)
```

### Medium Models (10B-50B parameters)

**Recommended:**
- TP = 4-8
- PP = 2-4
- EP = 2-4 (for MoE)
- CP = 1

**Rationale:**
- Balance memory distribution with communication cost
- Use TP for large hidden dimensions
- Use PP for deep models
- Use EP to split MoE experts efficiently

**Example (Mixtral 8x7B on 16 GPUs):**
```yaml
parallelism:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  expert_model_parallel_size: 2
  # DP = 1 (computed automatically)
```

### Large Models (50B-200B parameters)

**Recommended:**
- TP = 8
- PP = 4-8
- EP = 4-8 (for MoE)
- CP = 1-2

**Rationale:**
- Maximize parallelism to fit model in memory
- TP=8 is sweet spot (diminishing returns beyond)
- Increase PP for very deep models
- High EP for models with many experts

**Example (DeepSeek-V3 lite on 64 GPUs):**
```yaml
parallelism:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 4
  expert_model_parallel_size: 4
  # DP = 1 (computed automatically)
```

### Very Large Models (> 200B parameters)

**Recommended:**
- TP = 8
- PP = 8-16
- EP = 8-16 (for MoE)
- CP = 2-4 (for long sequences)

**Rationale:**
- Maximum parallelism needed
- Consider 3D parallelism (TP + PP + DP)
- High EP critical for large MoE models

## Parallelism Trade-offs

### Tensor Parallelism (TP)

**Advantages:**
- Even memory distribution
- No pipeline bubbles
- Simple to implement

**Disadvantages:**
- All-reduce communication on every layer
- Communication overhead increases with TP size
- Diminishing returns beyond TP=8

**When to use:**
- Large hidden dimensions (hidden_size >= 4096)
- Models that don't fit in single GPU
- Fast interconnect (NVLink, InfiniBand)

**When to avoid:**
- Small models that fit in GPU
- Slow interconnect
- TP > 8 (use PP instead)

### Pipeline Parallelism (PP)

**Advantages:**
- Only communicate at stage boundaries
- Scales to many GPUs
- Works with slower interconnect

**Disadvantages:**
- Pipeline bubbles (unused GPU cycles)
- More complex implementation
- Uneven memory distribution

**When to use:**
- Very deep models (num_layers >= 40)
- Need to scale beyond TP limits
- Slower interconnect available

**When to avoid:**
- Shallow models
- Small batch sizes (more bubbles)
- Need maximum throughput

### Expert Parallelism (EP)

**Advantages:**
- Linear memory reduction for experts
- Minimal communication overhead
- Natural fit for MoE architecture

**Disadvantages:**
- Only applies to MoE models
- Load imbalance if routing is skewed
- num_experts must be divisible by EP

**When to use:**
- MoE models with many experts
- Experts don't fit in GPU memory
- Even expert load distribution

**When to avoid:**
- Dense models
- Models with few experts (< 8)

### Context Parallelism (CP)

**Advantages:**
- Reduces activation memory for long sequences
- Enables training with very long contexts

**Disadvantages:**
- Communication overhead
- Only helps with long sequences
- Requires sequence_parallel: true and TP > 1

**When to use:**
- seq_length >= 16384
- Activation memory dominates
- TP already enabled

**When to avoid:**
- Short sequences (< 8192)
- When activation memory is not bottleneck

## Strategy Selection Workflow

### Step 1: Baseline Configuration

Start with minimal parallelism:
```yaml
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1  # For MoE
```

### Step 2: Check Memory

Run estimation:
```bash
python scripts/estimate.py --config baseline.yaml
```

If memory fits (< 80 GB for A100), done. Otherwise, continue.

### Step 3: Add Tensor Parallelism

Increase TP incrementally:
```bash
python scripts/estimate.py --config baseline.yaml --tp 2
python scripts/estimate.py --config baseline.yaml --tp 4
python scripts/estimate.py --config baseline.yaml --tp 8
```

Stop when memory fits or TP = 8.

### Step 4: Add Pipeline Parallelism

If TP = 8 and still doesn't fit, add PP:
```bash
python scripts/estimate.py --config baseline.yaml --tp 8 --pp 2
python scripts/estimate.py --config baseline.yaml --tp 8 --pp 4
```

### Step 5: Add Expert Parallelism (MoE only)

If MoE model and still doesn't fit, increase EP:
```bash
python scripts/estimate.py --config baseline.yaml --tp 8 --pp 4 --ep 2
python scripts/estimate.py --config baseline.yaml --tp 8 --pp 4 --ep 4
```

### Step 6: Enable Memory Optimizations

If still doesn't fit, enable:
1. Distributed optimizer
2. Activation recomputation
3. Sequence parallelism

See configuration_guide.md for details.

## Communication Topology

### NVLink (within node)

Best for:
- TP (requires high bandwidth)
- Small-scale DP

Typical configuration:
- 8 GPUs per node with NVLink
- TP = 8 within node
- PP/DP across nodes

### InfiniBand (across nodes)

Best for:
- PP (point-to-point)
- Large-scale DP
- EP (all-to-all)

Typical configuration:
- TP within nodes
- PP/EP/DP across nodes

## Example Configurations

### Configuration 1: 7B Dense, 8× A100 40GB

```yaml
parallelism:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
training:
  micro_batch_size: 4
  use_distributed_optimizer: true
  sequence_parallel: true
  recompute_granularity: "full"
```

**Result:** ~30 GB/GPU, DP = 4

### Configuration 2: Mixtral 8x7B, 16× A100 80GB

```yaml
parallelism:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  expert_model_parallel_size: 2
training:
  micro_batch_size: 1
  use_distributed_optimizer: true
  sequence_parallel: true
  recompute_granularity: "full"
```

**Result:** ~60 GB/GPU, DP = 1

### Configuration 3: DeepSeek-V3 lite, 64× A100 80GB

```yaml
parallelism:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 4
  expert_model_parallel_size: 4
training:
  micro_batch_size: 1
  use_distributed_optimizer: true
  sequence_parallel: true
  recompute_granularity: "full"
```

**Result:** ~50 GB/GPU, DP = 1

## Troubleshooting

### "Memory exceeds GPU capacity"

Solutions (in order):
1. Enable distributed optimizer
2. Enable recompute_granularity: "full"
3. Increase TP
4. Increase PP
5. Increase EP (MoE only)
6. Reduce micro_batch_size

### "Training is slow"

Potential causes:
- Too much PP (pipeline bubbles)
- Too much TP (communication overhead)
- micro_batch_size too small
- Excessive recomputation

Solutions:
- Reduce PP, increase TP
- Increase micro_batch_size
- Use "selective" instead of "full" recomputation
- Check interconnect bandwidth

### "Out of memory despite estimation"

Possible reasons:
- Framework overhead (~10-15%)
- Memory fragmentation
- Temporary buffers
- Communication buffers

Solutions:
- Leave 20% memory headroom
- Reduce micro_batch_size by 1
- Enable gradient accumulation
