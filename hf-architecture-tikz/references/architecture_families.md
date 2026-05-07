# Architecture Family Detection

`extract_arch.py` classifies a model into one **attention family** plus a set of orthogonal flags. The classification is purely config-field-based — no weight loading, no model-class inspection.

## Attention families (mutually exclusive)

Detected in this order. The first matching rule wins.

### 1. `dsv4` — DeepSeek-V4-Flash family

Triggers on **either**:
- `model_type == "deepseek_v4"`, **or**
- presence of `hc_mult`, `compress_ratios`, `index_n_heads`, `o_lora_rank` (all four).

Required dimension fields: `hidden_size`, `num_attention_heads`, `head_dim`, `q_lora_rank`, `o_lora_rank`, `o_groups`, `qk_rope_head_dim`, `index_n_heads`, `index_head_dim`, `index_topk`, `hc_mult`, `compress_ratios` (list).

### 2. `mla` — Multi-head Latent Attention (DeepSeek-V2 / V3)

Triggers when **all** of `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` are present and `dsv4` did not match.

### 3. `gqa` — Grouped Query Attention

Triggers when `num_key_value_heads` is present and `num_key_value_heads < num_attention_heads`.

Examples: Llama-3, Qwen3, Mistral.

### 4. `mha` — Multi-Head Attention (default)

Otherwise — typically `num_key_value_heads == num_attention_heads` or absent.

Examples: GPT-2, OPT, early Llama.

## Orthogonal flags

These flags compose freely with the attention family.

| Flag | Detector | Meaning |
|------|----------|---------|
| `moe` | `n_routed_experts > 0` OR `num_local_experts > 0` OR `num_moe_experts > 0` | MoE FFN replaces dense FFN (per-layer; see `first_k_dense_replace`) |
| `shared_experts` | `n_shared_experts > 0` | Always-active shared expert(s) alongside routed experts |
| `hash_routing` | `num_hash_layers > 0` | First `num_hash_layers` layers use a learned `tid2eid` table; rest use score routing |
| `mtp` | `num_nextn_predict_layers > 0` | One or more `MTPBlock`s after the main stack |
| `tied_lm_head` | `tie_word_embeddings == True` | LM head shares weights with token embedding |
| `first_k_dense` | `first_k_dense_replace > 0` | First `k` layers use dense FFN, remaining layers use MoE (V3 pattern) |
| `sliding_window` | `sliding_window` present and finite | Attention attends to a fixed local window in addition to (or instead of) global attention |
| `swiglu_clip` | `swiglu_limit` present | SwiGLU activations are clipped at `±swiglu_limit` (FP4 stability) |

## Per-layer overrides (DSv4 only)

For DSv4 models the `compress_ratios` list (length `num_hidden_layers + num_nextn_predict_layers`) controls per-layer attention behavior:

- `compress_ratios[i] == 0` — full attention (no Compressor, no Indexer).
- `compress_ratios[i] == 4` — Compressor with overlapping windows + Indexer for learned sparse top-K selection.
- `compress_ratios[i] == 128` (or any other > 4) — Compressor with block pooling, no Indexer.

The renderer draws one representative Block plus a small pattern strip showing the full `compress_ratios` array.

## Field name aliasing

The same logical concept can carry different names across config schemas. The detector normalizes:

| Logical name | Aliases tried (in order) |
|--------------|--------------------------|
| `hidden_size` | `hidden_size`, `d_model`, `n_embd` |
| `num_hidden_layers` | `num_hidden_layers`, `n_layer`, `num_layers` |
| `num_attention_heads` | `num_attention_heads`, `n_head`, `num_heads` |
| `num_key_value_heads` | `num_key_value_heads`, `num_kv_heads` |
| `intermediate_size` | `intermediate_size`, `ffn_hidden_size`, `ffn_dim` |
| `vocab_size` | `vocab_size`, `n_vocab` |
| `n_routed_experts` | `n_routed_experts`, `num_local_experts`, `num_moe_experts`, `num_experts` |
| `num_experts_per_tok` | `num_experts_per_tok`, `moe_router_topk`, `moe_topk`, `top_k` |
| `n_shared_experts` | `n_shared_experts`, `num_shared_experts` |
| `moe_intermediate_size` | `moe_intermediate_size`, `moe_ffn_hidden_size` |

## When detection fails

If no rule matches or a required field is missing, `extract_arch.py` exits with a diagnostic listing which family was the closest match and which fields were missing. The user should either pass `--family <name>` to force a family, or open `references/architecture_families.md` and `extract_arch.py` to extend the rules.
