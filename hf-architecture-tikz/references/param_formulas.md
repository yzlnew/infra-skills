# Parameter-Count Formulas

Closed-form expressions for every architectural unit drawn by this skill. The Python script `extract_arch.py` computes the numbers from the loaded HF config; this document is the reference for what those numbers mean and how they were derived.

## Symbols

| Symbol | Config field | Description |
|--------|--------------|-------------|
| `d`  | `hidden_size` | Hidden state dim |
| `v`  | `vocab_size` | Vocabulary size |
| `L`  | `num_hidden_layers` | Number of transformer layers |
| `H`  | `num_attention_heads` | Number of attention heads |
| `Hkv`| `num_key_value_heads` | KV heads (GQA / DSv4) |
| `dh` | `head_dim` | Attention head dim |
| `f`  | `intermediate_size` (or `moe_intermediate_size` for MoE experts) | FFN hidden dim |
| `E`  | `n_routed_experts` | Number of routed experts |
| `Es` | `n_shared_experts` | Number of shared experts |
| `K`  | `num_experts_per_tok` | Top-K experts per token |
| `qr` | `q_lora_rank` | Q LoRA rank (MLA, DSv4) |
| `kr` | `kv_lora_rank` | KV LoRA rank (MLA only) |
| `or` | `o_lora_rank` | O LoRA rank (DSv4) |
| `og` | `o_groups` | O projection grouping (DSv4) |
| `dn` | `qk_nope_head_dim` | MLA non-positional Q/K dim |
| `dr` | `qk_rope_head_dim` | RoPE Q/K dim (MLA, DSv4) |
| `dv` | `v_head_dim` | MLA V head dim |
| `hm` | `hc_mult` | Hyper-Connection multiplier (DSv4) |
| `Ih` | `index_n_heads` | Indexer heads (DSv4) |
| `Id` | `index_head_dim` | Indexer head dim (DSv4) |
| `cr` | `compress_ratios[i]` | Per-layer KV compression ratio (DSv4) |
| `nh` | `num_hash_layers` | First-`nh` layers using hash routing (DSv4) |
| `nm` | `num_nextn_predict_layers` | MTP heads (DSv4) |

All formulas count **trainable parameters** (excludes RoPE caches, attention masks, etc.). Bias terms are included only when the corresponding HF config flag (e.g. `attention_bias`) is true; the formulas below are the bias-free baseline and `extract_arch.py` adds bias terms when set.

## Universal blocks

| Block | Formula | Notes |
|-------|---------|-------|
| Token embedding | `v · d` | Skipped if `tie_word_embeddings`; folded into LM head |
| RMSNorm | `d` | One scalar per hidden dim, no bias |
| Final RMSNorm | `d` | One per model |
| Plain LM head | `v · d` | Untied case |
| Tied LM head | `0` | Reuses embedding |

## Attention families

### MHA (`mha`)

```
QKVO = 4 · d²
```

### GQA (`gqa`)

```
Q proj  = d · (H · dh)        (often = d² when H · dh == d)
K proj  = d · (Hkv · dh)
V proj  = d · (Hkv · dh)
O proj  = (H · dh) · d        (often = d²)
total   = 2·d² + 2·d·Hkv·dh   (when H · dh == d)
```

### MLA (`mla`, DeepSeek-V2/V3)

```
wq_a   : d · qr
wq_b   : qr · H · (dn + dr)
wkv_a  : d · (kr + dr)
wkv_b  : kr · H · (dn + dv)
wo     : H · dv · d
total  : d·qr + qr·H·(dn+dr) + d·(kr+dr) + kr·H·(dn+dv) + H·dv·d
```

### DSv4 (`dsv4`, DeepSeek-V4-Flash)

Verified against `inference/model.py` from `deepseek-ai/DeepSeek-V4-Flash`.

```
wq_a       : d · qr                          (Linear, no bias)
q_norm     : qr                              (RMSNorm)
wq_b       : qr · (H · dh)                   (ColumnParallelLinear)
wkv        : d · dh                          (Hkv = 1 means single shared head)
kv_norm    : dh
wo_a       : (H · dh / og) · (og · or)       (= H · dh · or, grouped einsum)
wo_b       : (og · or) · d
attn_sink  : H                               (per-head learnable bias)
─────────────────────────────────────────
attn_base  = d·qr + qr + qr·H·dh
           + d·dh + dh
           + H·dh·or + og·or·d
           + H
```

Per-layer optional add-ons (depend on `compress_ratios[i]`):

```
Compressor (cr=4, overlap=True):
  wkv      : d · 2·dh         (output dim doubled by overlap_transform)
  wgate    : d · 2·dh
  ape      : 2·cr · 2·dh      (positional embedding for compressed window)
  norm     : dh
  total    = 2·d·2·dh + 2·cr·2·dh + dh
           = 4·d·dh + 4·cr·dh + dh

Compressor (cr ≥ 5, e.g. 128, overlap=False):
  wkv      : d · dh
  wgate    : d · dh
  ape      : cr · dh
  norm     : dh
  total    = 2·d·dh + cr·dh + dh

Indexer (only when cr=4):
  wq_b      : qr · (Ih · Id)
  weights   : d · Ih
  internal Compressor (cr=4, separate instance)
  total     = qr·Ih·Id + d·Ih + Compressor(cr=4)
```

### Hyper-Connection overhead (DSv4 only)

Two `hc_pre / hc_post` pairs per Block (one for the attention sublayer, one for the MoE sublayer). Each pair contributes:

```
hc_*_fn    : (2 + hm) · hm · d
hc_*_base  : (2 + hm) · hm        (bias)
hc_*_scale : 3                    (scalar triple)
total per pair ≈ (2 + hm) · hm · d
```

The Sinkhorn iterations are runtime-only and add zero parameters.

## FFN families

### Dense SwiGLU

```
gate (w1) : d · f
up   (w3) : d · f
down (w2) : f · d
total     = 3 · d · f
```

`swiglu_limit` (when set) clips activations but adds no parameters.

### MoE (score-routed)

```
router          : d · E   (+ E if router has bias)
routed expert i : 3 · d · f       (each)
shared expert   : 3 · d · f       (× Es; always active)
total experts   = E · 3·d·f + Es · 3·d·f
total           = router + total_experts
active per tok  = (K + Es) · 3·d·f + router  (the routed-expert active mass)
```

### MoE (hash-routed, DSv4 first `nh` layers)

```
tid2eid table : v · K     (integer indices, but each entry is a learned int32 — counted as a discrete param)
shared expert : 3 · d · f
routed expert : 3 · d · f       (× E)
total         = v·K + (E + Es) · 3·d·f
```

The `tid2eid` table replaces the score router. There is no `weight` matrix and no `bias` for hash-routing layers.

## MTP head (DSv4)

```
e_proj      : d · d
h_proj      : d · d
e_norm      : d
h_norm      : d
+ a full Block's worth of params (HC overhead + DSv4 attention + MoE)
+ HC head:   v·d + (2 + hm) · hm · d
total = 2·d² + 2·d + Block_params + (v·d + (2+hm)·hm·d)
```

The HC head is shared across the main stack and the MTP head in V4-Flash, so depending on accounting you may want to count it only once. `extract_arch.py` follows the convention used by the safetensors index: count it once with the main stack, and count only the MTP-specific fusion (`e_proj + h_proj + their norms`) plus an additional Block as MTP-specific.

## Whole-model totals

For a given config:

```
embed     = v · d                       (0 if tied)
lm_head   = v · d                       (in the HC-aware case for DSv4 add (2+hm)·hm·d)
final_norm = d
per_layer = HC_overhead + attn + 2·norms + (MoE or dense FFN)
total     = embed + L · per_layer + final_norm + lm_head + nm · MTPBlock
```

`active params per token` = `total − inactive routed-expert mass`. For an MoE layer:

```
inactive_mass = (E − K) · 3·d·f
active_mass   = (K + Es) · 3·d·f + router_params
```

The script reports both.

## Quantization note

DeepSeek-V4-Flash ships in **FP8 weights for most tensors, FP4 for routed experts, BF16 for gates and norms**. Parameter *counts* are quantization-independent; storage size is not. The summary box in the rendered diagram includes a one-line note when `quantization_config` is present in the HF config.

## Verification target for DeepSeek-V4-Flash

The safetensors index lists ~158B total parameters (counting one entry per tensor including FP4 mantissa/exponent shards). The closed-form sum from this formula library should land within ~0.5% of that figure; any larger discrepancy indicates a missing block or a wrong dim alias.
