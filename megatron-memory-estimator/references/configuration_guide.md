# Configuration Guide

Complete reference for all configuration options in Megatron Memory Estimator.

## Model Section

### Architecture Parameters

**num_layers** (required)
- Number of transformer layers
- Example: 32 for most 7B models, 80+ for larger models
- Impact: Linear effect on parameters and memory

**hidden_size** (required)
- Hidden dimension size
- Common values: 2048, 4096, 5120, 8192
- Must be divisible by num_attention_heads
- Impact: Quadratic effect on parameters

**num_attention_heads** (required)
- Number of attention heads
- Common values: 32, 40, 64
- Must divide hidden_size evenly
- Impact: Affects attention computation memory

**num_query_groups** (optional, default: num_attention_heads)
- Number of query groups for Grouped Query Attention (GQA)
- Set to < num_attention_heads for GQA
- Example: num_attention_heads=32, num_query_groups=8 for Mixtral
- Impact: Reduces KV cache size

**ffn_hidden_size** (required)
- Feed-forward network intermediate dimension
- Common formula: 4 × hidden_size or (8/3) × hidden_size for MoE
- Example: 11008 for LLaMA, 14336 for Mixtral
- Impact: Major contributor to parameter count

### MoE Parameters

**num_experts** (optional, default: null)
- Number of experts in MoE layer
- Set to null for dense models
- Common values: 8 (Mixtral), 64 (DeepSeek-V3), 128+
- Must be divisible by expert_model_parallel_size

**moe_router_topk** (optional, default: 2)
- Number of experts activated per token
- Common values: 2 (Mixtral), 6 (DeepSeek-V3)
- Higher values = more computation but potentially better quality

**moe_ffn_hidden_size** (optional, default: ffn_hidden_size)
- FFN dimension for each expert
- Usually same as ffn_hidden_size
- Can be different for specialized architectures

**moe_shared_expert_intermediate_size** (optional, default: null)
- Enable shared experts (always activated)
- Example: 5632 for DeepSeek-V3
- Null = no shared experts

### Vocabulary Parameters

**vocab_size** (required)
- Vocabulary size
- Common values: 32000 (LLaMA), 128256 (Llama 3)
- Rounded to multiples of 128 or 256 for efficiency

**max_position_embeddings** (required)
- Maximum sequence length support
- Common values: 2048, 4096, 8192, 32768
- Must be >= seq_length used in training

### Other Model Parameters

**normalization** (optional, default: "RMSNorm")
- Layer normalization type
- Options: "RMSNorm", "LayerNorm"
- RMSNorm is more efficient

**layernorm_epsilon** (optional, default: 1e-5)
- Epsilon for numerical stability
- Common values: 1e-5, 1e-6

**activation_func** (optional, default: "swiglu")
- Activation function
- Options: "swiglu", "gelu"
- SwiGLU requires gated_linear_unit: true

**gated_linear_unit** (optional, default: true)
- Use gated activation (SwiGLU, GeGLU)
- Set to true for modern architectures

**qk_layernorm** (optional, default: false)
- Apply layernorm to queries and keys
- Used in some architectures for stability

**multi_latent_attention** (optional, default: false)
- Enable Multi-Latent Attention (MLA)
- Advanced feature, rarely used

## Parallelism Section

### Tensor Parallelism (TP)

**tensor_model_parallel_size** (required)
- Split weights across TP GPUs
- Common values: 1, 2, 4, 8
- Best for: Large hidden dimensions
- Trade-off: More communication, better memory distribution

### Pipeline Parallelism (PP)

**pipeline_model_parallel_size** (required)
- Split layers across PP stages
- Common values: 1, 2, 4, 8
- Best for: Deep models with many layers
- Trade-off: Pipeline bubbles, lower GPU utilization

### Expert Parallelism (EP)

**expert_model_parallel_size** (optional, default: 1)
- Split experts across EP GPUs (MoE only)
- Common values: 2, 4, 8, 16
- num_experts must be divisible by this value
- Best for: Models with many experts

**expert_tensor_parallel_size** (optional, default: 1)
- Apply TP within each expert
- Usually 1, only use for very large experts
- Adds communication overhead

### Context Parallelism (CP)

**context_parallel_size** (optional, default: 1)
- Split sequence dimension across CP GPUs
- Common values: 1, 2, 4
- Requires sequence_parallel: true
- Best for: Long sequences

### Virtual Pipeline Parallelism

**virtual_pipeline_model_parallel_size** (optional, default: null)
- Interleaved pipeline schedule
- Reduces pipeline bubbles
- Example: PP=4, VP=2 means 8 model chunks
- Advanced feature, use with caution

### Parallelism Constraints

**Critical constraint:**
```
world_size = tp × pp × ep × etp × cp × dp
```

where dp (data parallelism) is computed automatically.

**Example calculation:**
- world_size = 64
- tp = 4, pp = 2, ep = 4, cp = 1, etp = 1
- dp = 64 / (4 × 2 × 4 × 1 × 1) = 2

## Training Section

### Batch Configuration

**micro_batch_size** (required)
- Batch size per GPU per forward pass
- Common values: 1, 2, 4
- Lower = less memory, slower training
- Higher = more memory, faster training

**seq_length** (required)
- Sequence length for training
- Must be <= max_position_embeddings
- Common values: 2048, 4096, 8192
- Impact: Quadratic memory growth (attention)

### Optimizer Configuration

**use_distributed_optimizer** (optional, default: true)
- Shard optimizer states across data parallel ranks
- Highly recommended: reduces memory by ~6 bytes/param
- Minimal performance impact

**sequence_parallel** (optional, default: false)
- Shard activations along sequence dimension
- Requires tensor_model_parallel_size > 1
- Reduces activation memory by factor of TP

### Activation Recomputation

**recompute_granularity** (optional, default: null)
- Enable activation checkpointing
- Options:
  - null: No recomputation (highest memory)
  - "full": Recompute entire layers (significant memory savings)
  - "selective": Recompute specific modules (balanced)

**recompute_method** (optional, default: null)
- How to select layers for recomputation
- Options:
  - "uniform": Recompute every N layers
  - "block": Recompute first/last N layers
- Only used with recompute_granularity: "full"

**recompute_num_layers** (optional, default: 0)
- Number of layers to NOT recompute
- With "block": keep first N layers in memory
- With "uniform": recompute every N layers

### Infrastructure

**world_size** (required)
- Total number of GPUs
- Must satisfy parallelism constraint
- Example: 8, 16, 32, 64, 128

## Precision Section

**params_dtype** (optional, default: "bf16")
- Parameter data type
- Options: "bf16", "fp16"
- BF16 recommended for modern hardware (better numerical stability)

**fp8** (optional, default: false)
- Enable FP8 computation
- Requires H100 or newer
- Experimental feature

## Memory Impact Summary

**High impact (quadratic or higher):**
- seq_length (quadratic for attention)
- hidden_size (quadratic for weights)
- num_layers (linear but large coefficient)

**Medium impact:**
- micro_batch_size (linear on activation)
- ffn_hidden_size (linear on weights)
- num_experts (linear, but offset by EP)

**Low impact:**
- vocab_size (only affects embedding layer)
- num_attention_heads (minor effect with GQA)

## Memory Optimization Priority

1. **Enable use_distributed_optimizer** (free 6 bytes/param)
2. **Enable recompute_granularity: "full"** (50-70% activation memory reduction)
3. **Enable sequence_parallel** (reduce activation by factor of TP)
4. **Increase EP** for MoE models (linear memory reduction)
5. **Increase PP** (splits model across more GPUs)
6. **Reduce micro_batch_size** (last resort, hurts throughput)
