---
name: hf-architecture-tikz
description: "Draw Sebastian-Raschka-gallery-style TikZ architecture diagrams for any HuggingFace decoder-only LLM, with per-block parameter formulas and concrete numbers. Supports MHA, GQA, MLA, DeepSeek-V4-Flash (Hyper-Connections + Sparse Attention with learned indexer), dense and MoE FFNs (incl. hash routing), and MTP heads. Use when the user asks to visualize / diagram / illustrate a transformer or LLM architecture (DeepSeek, Qwen, Llama, Mistral, gpt-oss, etc.), wants a Raschka-style figure, or wants a TikZ/LaTeX rendering of an HF model."
---

# HF Architecture → TikZ

Generate a publication-quality vertical architecture diagram (in the style of Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)) for any HuggingFace decoder-only LLM. The diagram annotates every sub-block with its parameter-count formula and the concrete number for the loaded config.

## When to use

- "Draw the architecture of `<HF repo>`."
- "Visualize how `<model>` is structured" / "make a diagram of `<model>` like Raschka's gallery."
- "I want a TikZ figure of `<model>` for a paper / blog post."
- The user mentions DeepSeek-V4-Flash, mHC / Hyper-Connections, MLA, MoE, sparse attention, MTP, and asks for a figure.

If the user just wants memory / parallelism numbers, prefer `megatron-memory-estimator` instead.

## Quick start

```bash
cd hf-architecture-tikz/

# 1. Pull config from HF + emit normalized arch.json
uv run python scripts/extract_arch.py deepseek-ai/DeepSeek-V4-Flash \
    --output examples/deepseek-v4-flash/arch.json

# 2. Render TikZ from arch.json
uv run python scripts/render_tikz.py \
    examples/deepseek-v4-flash/arch.json \
    --output examples/deepseek-v4-flash/deepseek-v4-flash.tex

# 3. Compile to PNG
bash scripts/compile.sh examples/deepseek-v4-flash/deepseek-v4-flash.tex
```

For a model with custom code (e.g. brand-new architectures), pass `--trust-remote-code`. For a local config:

```bash
uv run python scripts/extract_arch.py /path/to/config.json --output arch.json
```

## Workflow

1. **Acquire config.** `extract_arch.py` tries `transformers.AutoConfig` first; if the installed `transformers` doesn't recognize the `model_type` (e.g. `deepseek_v4` introduces `hc_mult`, `compress_ratios`), it falls back to raw JSON via `huggingface_hub.hf_hub_download`. Local file paths bypass network.
2. **Detect architecture family.** Pure config-field rules — see `references/architecture_families.md`. The script labels the model with a family tag (`mha`, `gqa`, `mla`, `dsv4`) plus orthogonal flags (MoE, hash routing, shared experts, MTP, tied LM head, first_k_dense_replace).
3. **Compute parameter counts.** Closed-form formulas keyed by family — see `references/param_formulas.md`. The script (not Claude) does the arithmetic and emits `arch.json` with one entry per architectural unit, each carrying `name`, `family`, `shape_in`, `shape_out`, `formula_symbolic`, `formula_concrete`, `param_count`.
4. **Assemble TikZ.** `render_tikz.py` reads `arch.json` plus `templates/anthropic.tex.j2` (Jinja2 template — all block macros are inlined for shared coordinate-space layout). The repeated transformer block is drawn once with a `× N layers` annotation; per-layer-varying behavior (V4-Flash compress_ratios, hash vs score routing) appears as a small pattern strip beneath the block.
5. **Compile.** `bash scripts/compile.sh out.tex` runs `xelatex` ×2 (TikZ `fit`/`positioning` needs a second pass) then `pdftocairo -png -r 300 -singlefile`. Falls back to `pdflatex` if XeTeX is unavailable.

## Architecture family detection

Detection rules live in [`references/architecture_families.md`](references/architecture_families.md). Summary:

| Family | Detector | Examples |
|--------|----------|----------|
| `dsv4` | `model_type == "deepseek_v4"` or presence of `hc_mult`+`compress_ratios`+`index_n_heads` | DeepSeek-V4-Flash |
| `mla`  | `q_lora_rank` + `kv_lora_rank` + `qk_nope_head_dim` + `qk_rope_head_dim` + `v_head_dim` | DeepSeek-V2/V3 |
| `gqa`  | `num_key_value_heads < num_attention_heads` | Llama-3, Qwen3, Mistral |
| `mha`  | otherwise | GPT-2, OPT |

Orthogonal flags: MoE (`n_routed_experts`/`num_local_experts`), hash routing (`num_hash_layers > 0`), shared experts (`n_shared_experts > 0`), MTP head (`num_nextn_predict_layers > 0`), tied LM head (`tie_word_embeddings`), dense-prefix layers (`first_k_dense_replace > 0`).

## Parameter formulas

Full table in [`references/param_formulas.md`](references/param_formulas.md). One-line summary per family attention: MHA `4·d²`; GQA `2·d² + 2·d·Hkv·dh`; MLA six projections; DSv4 `wq_a + q_norm + wq_b + wkv + kv_norm + wo_a + wo_b + attn_sink (+ Compressor + Indexer)`. SwiGLU `3·d·f`. Standard MoE = `E` routed experts (each `3·d·f`) + router `d·E` + `Es` shared. Hash MoE replaces router with a `vocab×topk` token→expert table.

## Worked example: DeepSeek-V4-Flash

The example under `examples/deepseek-v4-flash/` covers the most architecturally novel components in the supported set:

- **Hyper-Connections (mHC):** four parallel hidden-state copies, with Sinkhorn-balanced reduction (`hc_sinkhorn_iters=20`) before each sublayer and weighted expansion + cross-copy mixing after. Drawn as a fan-in / fan-out inside each block.
- **Sparse Attention:** Q-LoRA (`d → q_lora_rank → H·dh`), KV projection (`d → dh`, `Hkv=1`), per-layer Compressor (overlap pooling for `compress_ratio=4`, block pooling for `compress_ratio=128`), learned Indexer for `compress_ratio=4` layers (top-`index_topk=512` selection over compressed KV), sliding window of 128, grouped O-LoRA (`o_groups=8`, `o_lora_rank=1024`).
- **MoE with hash routing:** first 3 layers use a learned `tid2eid` table (`vocab × topk`); remaining 40 layers use `sqrtsoftplus` scoring + top-6 routing.
- **MTP head:** one `MTPBlock` (= `e_proj` + `h_proj` + their RMSNorms + a full Block) for next-token prediction.
- **Compress-ratios pattern strip:** drawn beneath the block to make the per-layer alternation `[0, 0, 4, 128, 4, 128, …, 4, 0]` visible.

## Customization

- **Palette.** Reuses the warm-pastel palette from `tikz-flowchart/themes/anthropic.md` (lavender = attention, mint = norm, teal = projection, cream = router/MoE infra, amber = experts, peach = embedding/output).
- **Detail level.** The default is full expansion (every sub-block separately). To collapse sub-blocks, edit the `dsv4` branch of `templates/anthropic.tex.j2` and replace the inner attention expansion with a single rounded card.
- **Other models.** The non-`dsv4` branch of `templates/anthropic.tex.j2` covers `mha` / `gqa` / `mla` (with optional MoE FFN) as a simpler vertical stack. The renderer dispatches based on the family flag emitted by `extract_arch.py`.

## Troubleshooting

- **`AutoConfig` raises on unknown fields.** Expected for very new model types. The loader catches and falls back to raw JSON automatically. If both fail, pass a local `config.json` path.
- **`mbridge` is unavailable / unsupported model.** Not required — we use `transformers` + raw JSON. `mbridge` is referenced only for cross-checking V3/Qwen counts.
- **`trust_remote_code` warnings.** `extract_arch.py` does not enable this flag silently. Pass `--trust-remote-code` only if the user explicitly requests it.
- **Tied embeddings double-counting.** When `tie_word_embeddings=True`, the embedding-table contribution is folded into the LM head and not counted twice.
- **Tall PNG.** Full expansion + side annotations + MTP branch typically renders to 4–6k pixels tall. Use `--no-mtp` (renderer flag) to suppress the MTP branch if you need a shorter figure.
- **`xelatex` not installed.** The compile script falls back to `pdflatex` automatically. Font macros are guarded with `\IfFontExistsTF`.

## Dependencies

Python: `transformers`, `huggingface_hub`, `jinja2`. Run via `uv run`.
System: `xelatex` (preferred) or `pdflatex`; `pdftocairo` (from `poppler`).
