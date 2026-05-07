#!/usr/bin/env python3
"""Extract a normalized architecture spec from a HuggingFace model config.

Usage:
    extract_arch.py <hf_repo_id_or_local_config_path> [--output PATH] [--family NAME]
                    [--trust-remote-code] [--revision REV]

Loads a config (in this order: local JSON file, transformers AutoConfig, raw HF
download), classifies the attention family + orthogonal flags, computes
closed-form parameter counts for every architectural unit, and emits a JSON
spec consumed by render_tikz.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(source: str, trust_remote_code: bool = False, revision: str | None = None) -> dict[str, Any]:
    """Return the raw HF config dict for `source`.

    `source` may be a path to a local config.json or a HuggingFace repo id.
    """
    if os.path.exists(source):
        with open(source) as f:
            return json.load(f)

    # Try transformers first (handles known model_types cleanly)
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        d = cfg.to_dict()
        # AutoConfig drops unknown fields. For DSv4-class models we need them all,
        # so fall through to raw download if the config looks impoverished.
        if d.get("model_type") in {"deepseek_v4", "deepseek_v3", "deepseek_v2"}:
            raise RuntimeError("Force raw fallback for DeepSeek configs")
        return d
    except Exception as exc_transformers:
        # Fall back to raw download from the Hub
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                repo_id=source,
                filename="config.json",
                revision=revision,
            )
            with open(local) as f:
                return json.load(f)
        except Exception as exc_hub:
            raise RuntimeError(
                f"Could not load config from {source!r}.\n"
                f"  transformers attempt: {exc_transformers}\n"
                f"  huggingface_hub attempt: {exc_hub}"
            )


# ---------------------------------------------------------------------------
# Field aliasing
# ---------------------------------------------------------------------------

ALIASES: dict[str, list[str]] = {
    "hidden_size": ["hidden_size", "d_model", "n_embd"],
    "num_hidden_layers": ["num_hidden_layers", "n_layer", "num_layers"],
    "num_attention_heads": ["num_attention_heads", "n_head", "num_heads"],
    "num_key_value_heads": ["num_key_value_heads", "num_kv_heads"],
    "head_dim": ["head_dim", "kv_channels"],
    "intermediate_size": ["intermediate_size", "ffn_hidden_size", "ffn_dim"],
    "vocab_size": ["vocab_size", "n_vocab"],
    "n_routed_experts": ["n_routed_experts", "num_local_experts", "num_moe_experts", "num_experts"],
    "num_experts_per_tok": ["num_experts_per_tok", "moe_router_topk", "moe_topk", "top_k"],
    "n_shared_experts": ["n_shared_experts", "num_shared_experts"],
    "moe_intermediate_size": ["moe_intermediate_size", "moe_ffn_hidden_size"],
    "tie_word_embeddings": ["tie_word_embeddings"],
    "first_k_dense_replace": ["first_k_dense_replace"],
    "num_nextn_predict_layers": ["num_nextn_predict_layers"],
    "num_hash_layers": ["num_hash_layers"],
    "hc_mult": ["hc_mult"],
    "compress_ratios": ["compress_ratios"],
    "index_n_heads": ["index_n_heads"],
    "index_head_dim": ["index_head_dim"],
    "index_topk": ["index_topk"],
    "q_lora_rank": ["q_lora_rank"],
    "kv_lora_rank": ["kv_lora_rank"],
    "o_lora_rank": ["o_lora_rank"],
    "o_groups": ["o_groups"],
    "qk_nope_head_dim": ["qk_nope_head_dim"],
    "qk_rope_head_dim": ["qk_rope_head_dim"],
    "v_head_dim": ["v_head_dim"],
    "sliding_window": ["sliding_window"],
    "swiglu_limit": ["swiglu_limit"],
    "attention_bias": ["attention_bias"],
}


def get(cfg: dict[str, Any], logical: str, default: Any = None) -> Any:
    """Look up a logical field via its known aliases."""
    for k in ALIASES.get(logical, [logical]):
        if k in cfg:
            return cfg[k]
    return default


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

@dataclass
class Spec:
    family: str
    flags: set[str] = field(default_factory=set)
    dims: dict[str, Any] = field(default_factory=dict)


def detect_family(cfg: dict[str, Any], forced: str | None = None) -> str:
    if forced:
        return forced
    mt = cfg.get("model_type", "")
    if mt == "deepseek_v4":
        return "dsv4"
    if all(get(cfg, k) is not None for k in ["hc_mult", "compress_ratios", "index_n_heads", "o_lora_rank"]):
        return "dsv4"
    mla_fields = ["q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"]
    if all(get(cfg, k) is not None for k in mla_fields):
        return "mla"
    H = get(cfg, "num_attention_heads")
    Hkv = get(cfg, "num_key_value_heads")
    if H is not None and Hkv is not None and Hkv < H:
        return "gqa"
    return "mha"


def detect_flags(cfg: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    if get(cfg, "n_routed_experts", 0) and int(get(cfg, "n_routed_experts", 0)) > 0:
        flags.add("moe")
    if get(cfg, "n_shared_experts", 0) and int(get(cfg, "n_shared_experts", 0)) > 0:
        flags.add("shared_experts")
    if get(cfg, "num_hash_layers", 0) and int(get(cfg, "num_hash_layers", 0)) > 0:
        flags.add("hash_routing")
    if get(cfg, "num_nextn_predict_layers", 0) and int(get(cfg, "num_nextn_predict_layers", 0)) > 0:
        flags.add("mtp")
    if get(cfg, "tie_word_embeddings", False):
        flags.add("tied_lm_head")
    if get(cfg, "first_k_dense_replace", 0) and int(get(cfg, "first_k_dense_replace", 0)) > 0:
        flags.add("first_k_dense")
    sw = get(cfg, "sliding_window")
    if sw is not None and sw not in (False, 0):
        flags.add("sliding_window")
    if get(cfg, "swiglu_limit") is not None:
        flags.add("swiglu_clip")
    return flags


# ---------------------------------------------------------------------------
# Parameter-count formulas
# ---------------------------------------------------------------------------

def fmt_int(n: int) -> str:
    """Format an integer with thousands separators."""
    return f"{n:,}"


def fmt_count(n: int) -> str:
    """Format a parameter count with K / M / B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def block_entry(
    name: str,
    kind: str,
    shape_in: str,
    shape_out: str,
    symbolic: str,
    concrete: str,
    param_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "name": name,
        "kind": kind,
        "shape_in": shape_in,
        "shape_out": shape_out,
        "formula_symbolic": symbolic,
        "formula_concrete": concrete,
        "param_count": param_count,
    }
    if extra:
        out.update(extra)
    return out


# ---- Universal blocks ----

def embedding_block(v: int, d: int, tied: bool) -> dict[str, Any]:
    n = 0 if tied else v * d
    return block_entry(
        "token_embed",
        "embedding",
        "[B, T]",
        f"[B, T, {d}]",
        "v · d" if not tied else "0  (tied to lm_head)",
        f"{v} · {d} = {fmt_count(n)}" if not tied else "0",
        n,
    )


def rmsnorm_block(name: str, d: int, shape: str) -> dict[str, Any]:
    return block_entry(
        name,
        "rmsnorm",
        shape,
        shape,
        "d",
        f"{d} = {fmt_count(d)}",
        d,
    )


def lm_head_block(v: int, d: int, hm: int | None, tied: bool) -> dict[str, Any]:
    head = 0 if tied else v * d
    if hm is not None:
        # HC-aware head: extra hc_head_fn = (2+hm)·hm·d
        hc_extra = (2 + hm) * hm * d
        params = head + hc_extra
        sym = ("0  (tied)" if tied else "v · d") + f"  +  (2+{hm})·{hm}·d"
        conc = (
            f"{0 if tied else f'{v}·{d}'}"
            f" + {(2 + hm) * hm}·{d} = {fmt_count(params)}"
        )
        kind = "hc_lm_head"
    else:
        params = head
        sym = "0  (tied)" if tied else "v · d"
        conc = f"0  (tied)" if tied else f"{v} · {d} = {fmt_count(params)}"
        kind = "lm_head"
    return block_entry(
        "lm_head",
        kind,
        f"[B, T, {hm}, {d}]" if hm is not None else f"[B, T, {d}]",
        f"[B, T, {v}]",
        sym,
        conc,
        params,
    )


# ---- Attention families ----

def attn_mha(d: int) -> tuple[int, dict[str, Any]]:
    n = 4 * d * d
    return n, {
        "symbolic": "4 · d²",
        "concrete": f"4 · {d}² = {fmt_count(n)}",
    }


def attn_gqa(d: int, H: int, Hkv: int, dh: int) -> tuple[int, dict[str, Any]]:
    q = d * H * dh
    k = d * Hkv * dh
    v = d * Hkv * dh
    o = H * dh * d
    n = q + k + v + o
    return n, {
        "symbolic": "d·(H·dh) + 2·d·(Hkv·dh) + (H·dh)·d",
        "concrete": (
            f"{d}·{H * dh} + 2·{d}·{Hkv * dh} + {H * dh}·{d}"
            f" = {fmt_count(n)}"
        ),
    }


def attn_mla(d: int, H: int, qr: int, kr: int, dn: int, dr: int, dv: int) -> tuple[int, dict[str, Any]]:
    wq_a = d * qr
    wq_b = qr * H * (dn + dr)
    wkv_a = d * (kr + dr)
    wkv_b = kr * H * (dn + dv)
    wo = H * dv * d
    n = wq_a + wq_b + wkv_a + wkv_b + wo
    return n, {
        "symbolic": "d·qr + qr·H·(dn+dr) + d·(kr+dr) + kr·H·(dn+dv) + H·dv·d",
        "concrete": (
            f"{d}·{qr} + {qr}·{H}·{dn + dr} + {d}·{kr + dr} + "
            f"{kr}·{H}·{dn + dv} + {H}·{dv}·{d} = {fmt_count(n)}"
        ),
    }


def attn_dsv4(d: int, H: int, dh: int, qr: int, or_: int, og: int) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    """Return (total params, per-sub-block list)."""
    sub: list[dict[str, Any]] = []

    n = d * qr
    sub.append(
        block_entry(
            "wq_a", "linear", f"[B, T, {d}]", f"[B, T, {qr}]",
            "d · qr", f"{d} · {qr} = {fmt_count(n)}", n,
        )
    )

    n_qn = qr
    sub.append(rmsnorm_block("q_norm", qr, f"[B, T, {qr}]"))

    n_qb = qr * H * dh
    sub.append(
        block_entry(
            "wq_b", "linear", f"[B, T, {qr}]", f"[B, T, {H}, {dh}]",
            "qr · H · dh", f"{qr} · {H} · {dh} = {fmt_count(n_qb)}", n_qb,
        )
    )

    n_kv = d * dh
    sub.append(
        block_entry(
            "wkv", "linear", f"[B, T, {d}]", f"[B, T, {dh}]",
            "d · dh", f"{d} · {dh} = {fmt_count(n_kv)}", n_kv,
        )
    )

    sub.append(rmsnorm_block("kv_norm", dh, f"[B, T, {dh}]"))

    n_woa = H * dh * or_
    sub.append(
        block_entry(
            "wo_a", "grouped_linear", f"[B, T, {og}, {H * dh // og}]", f"[B, T, {og}, {or_}]",
            "(H·dh/og) · (og·or)", f"{H * dh // og} · {og * or_} · {og} = {fmt_count(n_woa)}", n_woa,
            extra={"o_groups": og},
        )
    )

    n_wob = og * or_ * d
    sub.append(
        block_entry(
            "wo_b", "linear", f"[B, T, {og * or_}]", f"[B, T, {d}]",
            "(og·or) · d", f"{og * or_} · {d} = {fmt_count(n_wob)}", n_wob,
        )
    )

    n_sink = H
    sub.append(
        block_entry(
            "attn_sink", "param", "—", "—",
            "H", f"{H} = {fmt_count(n_sink)}", n_sink,
        )
    )

    total = n + n_qn + n_qb + n_kv + dh + n_woa + n_wob + n_sink
    return total, {"subblocks": sub}


def compressor_params(d: int, dh: int, cr: int) -> tuple[int, dict[str, Any]]:
    """Return (param count, dict for arch.json)."""
    if cr == 0:
        return 0, {"present": False, "ratio": 0}
    if cr == 4:  # overlap
        # wkv: d·2·dh, wgate: d·2·dh, ape: 2·cr·2·dh, norm: dh
        n = 2 * d * 2 * dh + 2 * cr * 2 * dh + dh
        symbolic = "2·d·2·dh + 2·cr·2·dh + dh"
    else:  # block pooling
        n = 2 * d * dh + cr * dh + dh
        symbolic = "2·d·dh + cr·dh + dh"
    concrete = f"cr={cr}: {fmt_count(n)}"
    return n, {
        "present": True,
        "ratio": cr,
        "overlap": cr == 4,
        "param_count": n,
        "formula_symbolic": symbolic,
        "formula_concrete": concrete,
    }


def indexer_params(d: int, qr: int, Ih: int, Id: int, dh: int) -> tuple[int, dict[str, Any]]:
    n_qb = qr * Ih * Id
    n_w = d * Ih
    # internal Compressor (cr=4)
    n_comp, _ = compressor_params(d, dh, 4)
    n = n_qb + n_w + n_comp
    return n, {
        "present": True,
        "param_count": n,
        "formula_symbolic": "qr·Ih·Id + d·Ih + Compressor(cr=4)",
        "formula_concrete": (
            f"{qr}·{Ih}·{Id} + {d}·{Ih} + Compressor(4)={fmt_count(n_comp)}"
            f" = {fmt_count(n)}"
        ),
    }


def hc_pair_params(d: int, hm: int) -> int:
    # hc_*_fn: (2+hm)·hm·d ; hc_*_base: (2+hm)·hm ; hc_*_scale: 3
    return (2 + hm) * hm * d + (2 + hm) * hm + 3


# ---- FFN families ----

def dense_swiglu(d: int, f: int) -> tuple[int, dict[str, str]]:
    n = 3 * d * f
    return n, {
        "symbolic": "3 · d · f",
        "concrete": f"3 · {d} · {f} = {fmt_count(n)}",
    }


def moe_score_routed(d: int, f: int, E: int, K: int, Es: int) -> tuple[int, dict[str, Any]]:
    router = d * E
    routed = E * 3 * d * f
    shared = Es * 3 * d * f
    n = router + routed + shared
    return n, {
        "router": router,
        "routed_total": routed,
        "shared_total": shared,
        "active_per_token": (K + Es) * 3 * d * f + router,
        "symbolic": "d·E + E·3·d·f + Es·3·d·f",
        "concrete": (
            f"router {fmt_count(router)} + "
            f"{E}·{fmt_count(3 * d * f)} routed + "
            f"{Es}·{fmt_count(3 * d * f)} shared = {fmt_count(n)}"
        ),
        "active_label": (
            f"{K + Es}·{fmt_count(3 * d * f)} + router = {fmt_count((K + Es) * 3 * d * f + router)}"
        ),
    }


def moe_hash_routed(d: int, f: int, E: int, K: int, Es: int, vocab: int) -> tuple[int, dict[str, Any]]:
    table = vocab * K
    routed = E * 3 * d * f
    shared = Es * 3 * d * f
    n = table + routed + shared
    return n, {
        "tid2eid_table": table,
        "routed_total": routed,
        "shared_total": shared,
        "active_per_token": (K + Es) * 3 * d * f,  # no router cost at inference
        "symbolic": "v·K + E·3·d·f + Es·3·d·f",
        "concrete": (
            f"tid2eid {fmt_count(table)} + "
            f"{E}·{fmt_count(3 * d * f)} routed + "
            f"{Es}·{fmt_count(3 * d * f)} shared = {fmt_count(n)}"
        ),
        "active_label": f"{K + Es}·{fmt_count(3 * d * f)} = {fmt_count((K + Es) * 3 * d * f)}",
    }


# ---------------------------------------------------------------------------
# Spec assembly per family
# ---------------------------------------------------------------------------

def build_spec_dsv4(cfg: dict[str, Any]) -> dict[str, Any]:
    d = int(get(cfg, "hidden_size"))
    v = int(get(cfg, "vocab_size"))
    L = int(get(cfg, "num_hidden_layers"))
    H = int(get(cfg, "num_attention_heads"))
    Hkv = int(get(cfg, "num_key_value_heads", 1))
    dh = int(get(cfg, "head_dim"))
    qr = int(get(cfg, "q_lora_rank"))
    or_ = int(get(cfg, "o_lora_rank"))
    og = int(get(cfg, "o_groups"))
    dr = int(get(cfg, "qk_rope_head_dim"))
    hm = int(get(cfg, "hc_mult"))
    Ih = int(get(cfg, "index_n_heads"))
    Id = int(get(cfg, "index_head_dim"))
    Itopk = int(get(cfg, "index_topk"))
    f_moe = int(get(cfg, "moe_intermediate_size"))
    E = int(get(cfg, "n_routed_experts", 0))
    K = int(get(cfg, "num_experts_per_tok", 0))
    Es = int(get(cfg, "n_shared_experts", 0))
    nh = int(get(cfg, "num_hash_layers", 0))
    nm = int(get(cfg, "num_nextn_predict_layers", 0))
    sw = get(cfg, "sliding_window")
    sw_int = int(sw) if sw not in (None, False) else None
    sl = get(cfg, "swiglu_limit")
    tied = bool(get(cfg, "tie_word_embeddings", False))
    compress_ratios = list(get(cfg, "compress_ratios", []))

    # Per-layer attention overhead bins (representative compress_ratios for the main 43 layers)
    main_ratios = compress_ratios[:L] if compress_ratios else [0] * L
    bins: dict[int, int] = {}
    for r in main_ratios:
        bins[int(r)] = bins.get(int(r), 0) + 1

    # ---- Embeddings ----
    blocks: list[dict[str, Any]] = []
    blocks.append(embedding_block(v, d, tied))
    blocks.append(
        block_entry(
            "hc_expand",
            "hc_expand",
            f"[B, T, {d}]",
            f"[B, T, {hm}, {d}]",
            "0  (reshape only)",
            "0",
            0,
            extra={"hc_mult": hm},
        )
    )

    # ---- Single transformer Block (drawn once with × L annotation) ----
    attn_total, attn_pack = attn_dsv4(d, H, dh, qr, or_, og)
    hc_pair = hc_pair_params(d, hm)

    # Compressor / Indexer params (bin-wise totals)
    bin_packs: dict[int, dict[str, Any]] = {}
    bin_attn_extra: dict[int, int] = {}
    for cr in bins:
        n_comp, comp_pack = compressor_params(d, dh, cr)
        if cr == 4:
            n_idx, idx_pack = indexer_params(d, qr, Ih, Id, dh)
        else:
            n_idx, idx_pack = 0, {"present": False}
        bin_packs[cr] = {"compressor": comp_pack, "indexer": idx_pack}
        bin_attn_extra[cr] = n_comp + n_idx

    # MoE / hash split
    moe_score_n, moe_score_pack = moe_score_routed(d, f_moe, E, K, Es)
    moe_hash_n, moe_hash_pack = moe_hash_routed(d, f_moe, E, K, Es, v) if nh > 0 else (moe_score_n, moe_score_pack)

    norm_pair = 2 * d
    layer_attn_avg_extra = sum(bins.get(cr, 0) * bin_attn_extra[cr] for cr in bins) / max(L, 1)

    # Layer-by-layer total
    layers_total = 0
    for i, cr in enumerate(main_ratios):
        per_layer_attn = attn_total + bin_attn_extra.get(int(cr), 0)
        per_layer_hc = 2 * hc_pair  # one for attn, one for ffn
        if i < nh:
            per_layer_ffn = moe_hash_n
        else:
            per_layer_ffn = moe_score_n
        layers_total += per_layer_attn + per_layer_hc + norm_pair + per_layer_ffn

    block_subblocks = (
        [
            block_entry(
                "hc_pre_attn", "hc_pre",
                f"[B, T, {hm}, {d}]", f"[B, T, {d}]",
                "(2+hm)·hm·d", f"{(2 + hm) * hm} · {d} = {fmt_count(hc_pair)}",
                hc_pair,
                extra={"sinkhorn_iters": int(get(cfg, "hc_sinkhorn_iters", 20))},
            ),
            block_entry(
                "rmsnorm_pre_attn", "rmsnorm",
                f"[B, T, {d}]", f"[B, T, {d}]",
                "d", f"{d} = {fmt_count(d)}", d,
            ),
            block_entry(
                "dsv4_attention", "dsv4_attention",
                f"[B, T, {d}]", f"[B, T, {d}]",
                "Σ sub-blocks",
                f"= {fmt_count(attn_total)} (+ Compressor/Indexer per layer)",
                attn_total,
                extra={
                    "subblocks": attn_pack["subblocks"],
                    "compressor_indexer_bins": {
                        str(cr): {
                            "layers": bins[cr],
                            "compressor": bin_packs[cr]["compressor"],
                            "indexer": bin_packs[cr]["indexer"],
                            "extra_params": bin_attn_extra[cr],
                        }
                        for cr in sorted(bins)
                    },
                    "index_topk": Itopk,
                    "sliding_window": sw_int,
                },
            ),
            block_entry(
                "hc_post_attn", "hc_post",
                f"[B, T, {d}]", f"[B, T, {hm}, {d}]",
                "(2+hm)·hm·d", f"{(2 + hm) * hm} · {d} = {fmt_count(hc_pair)}",
                hc_pair,
            ),
            block_entry(
                "hc_pre_ffn", "hc_pre",
                f"[B, T, {hm}, {d}]", f"[B, T, {d}]",
                "(2+hm)·hm·d", f"{(2 + hm) * hm} · {d} = {fmt_count(hc_pair)}",
                hc_pair,
            ),
            block_entry(
                "rmsnorm_pre_ffn", "rmsnorm",
                f"[B, T, {d}]", f"[B, T, {d}]",
                "d", f"{d} = {fmt_count(d)}", d,
            ),
            block_entry(
                "moe", "moe_dsv4",
                f"[B, T, {d}]", f"[B, T, {d}]",
                moe_score_pack["symbolic"],
                moe_score_pack["concrete"],
                moe_score_n,
                extra={
                    "score_routed": moe_score_pack,
                    "hash_routed": moe_hash_pack if nh > 0 else None,
                    "n_hash_layers": nh,
                    "n_score_layers": L - nh,
                    "n_routed_experts": E,
                    "topk": K,
                    "n_shared_experts": Es,
                    "moe_intermediate_size": f_moe,
                    "swiglu_limit": sl,
                },
            ),
            block_entry(
                "hc_post_ffn", "hc_post",
                f"[B, T, {d}]", f"[B, T, {hm}, {d}]",
                "(2+hm)·hm·d", f"{(2 + hm) * hm} · {d} = {fmt_count(hc_pair)}",
                hc_pair,
            ),
        ]
    )

    blocks.append(
        block_entry(
            "transformer_block", "transformer_block",
            f"[B, T, {hm}, {d}]", f"[B, T, {hm}, {d}]",
            "Σ sub-blocks",
            f"avg per layer ≈ {fmt_count(int(layers_total / max(L, 1)))}",
            int(layers_total / max(L, 1)),
            extra={
                "repeat": L,
                "subblocks": block_subblocks,
                "compress_ratios": main_ratios,
                "compress_ratio_bins": dict(bins),
                "n_hash_layers": nh,
            },
        )
    )

    blocks.append(rmsnorm_block("final_norm", d, f"[B, T, {hm}, {d}]"))
    blocks.append(lm_head_block(v, d, hm, tied=tied))

    # ---- MTP head ----
    mtp_total = 0
    if nm > 0:
        # MTPBlock = e_proj + h_proj + 2 norms + a full Block (attention + MoE + HC)
        e_proj = d * d
        h_proj = d * d
        norms = 2 * d
        # Use cr from compress_ratios[L:L+nm], default 0
        mtp_ratios = compress_ratios[L : L + nm] if compress_ratios else [0] * nm
        mtp_block_total = 0
        for cr in mtp_ratios:
            per = attn_total + bin_attn_extra.get(int(cr), compressor_params(d, dh, int(cr))[0])
            per += 2 * hc_pair + norm_pair + moe_score_n
            mtp_block_total += per
        mtp_total = (e_proj + h_proj + norms) * nm + mtp_block_total

        blocks.append(
            block_entry(
                "mtp_head", "mtp_head",
                f"[B, T, {hm}, {d}]", f"[B, T, {v}]",
                "nm·(2·d² + 2·d) + nm·Block",
                (
                    f"{nm}·(2·{d}² + 2·{d}) + {nm}·Block ≈ "
                    f"{fmt_count(mtp_total)}"
                ),
                mtp_total,
                extra={
                    "repeat": nm,
                    "compress_ratios": mtp_ratios,
                    "fusion": {
                        "e_proj": d * d,
                        "h_proj": d * d,
                        "e_norm": d,
                        "h_norm": d,
                    },
                },
            )
        )

    # ---- Totals ----
    embed_n = blocks[0]["param_count"]
    final_norm_n = d
    lm_head_n = blocks[-1 - (1 if nm > 0 else 0)]["param_count"]  # the lm_head entry
    main_total = embed_n + layers_total + final_norm_n + lm_head_n
    total = main_total + mtp_total
    # Active-per-token
    if nh > 0:
        avg_layer_active = (
            (nh * moe_hash_pack["active_per_token"] + (L - nh) * moe_score_pack["active_per_token"]) / L
        )
    else:
        avg_layer_active = moe_score_pack["active_per_token"]
    active_attn_avg = attn_total + sum(
        bins.get(cr, 0) * bin_attn_extra[cr] for cr in bins
    ) / max(L, 1)
    active_per_layer = active_attn_avg + 2 * hc_pair + norm_pair + avg_layer_active
    active_total = embed_n + L * active_per_layer + final_norm_n + lm_head_n  # MTP head not active during forward of main token

    summary = {
        "total_params": int(main_total),
        "total_label": fmt_count(int(main_total)),
        "total_with_mtp_params": int(total),
        "total_with_mtp_label": fmt_count(int(total)),
        "active_params": int(active_total),
        "active_label": fmt_count(int(active_total)),
        "mtp_params": int(mtp_total),
        "mtp_label": fmt_count(int(mtp_total)) if nm > 0 else "—",
        "quantization": _quant_label(cfg),
    }

    return {
        "model": {
            "id": cfg.get("_source_id", ""),
            "model_type": cfg.get("model_type", "deepseek_v4"),
            "family": "dsv4",
            "flags": sorted(detect_flags(cfg)),
            "dims": {
                "d": d, "v": v, "L": L, "H": H, "Hkv": Hkv, "dh": dh,
                "qr": qr, "or": or_, "og": og, "dr": dr,
                "hm": hm, "Ih": Ih, "Id": Id, "index_topk": Itopk,
                "f_moe": f_moe, "E": E, "K": K, "Es": Es, "nh": nh, "nm": nm,
                "sliding_window": sw_int, "swiglu_limit": sl,
            },
            "compress_ratios": main_ratios,
            "compress_ratio_bins": dict(bins),
            "summary": summary,
        },
        "blocks": blocks,
    }


def build_spec_simple(cfg: dict[str, Any], family: str) -> dict[str, Any]:
    """Spec builder for non-DSv4 families (mha / gqa / mla)."""
    d = int(get(cfg, "hidden_size"))
    v = int(get(cfg, "vocab_size"))
    L = int(get(cfg, "num_hidden_layers"))
    H = int(get(cfg, "num_attention_heads"))
    Hkv = int(get(cfg, "num_key_value_heads", H))
    dh = int(get(cfg, "head_dim", d // max(H, 1)))
    f_dense = int(get(cfg, "intermediate_size", 4 * d))
    f_moe = int(get(cfg, "moe_intermediate_size", f_dense))
    E = int(get(cfg, "n_routed_experts", 0))
    K = int(get(cfg, "num_experts_per_tok", 0))
    Es = int(get(cfg, "n_shared_experts", 0))
    fkd = int(get(cfg, "first_k_dense_replace", 0))
    tied = bool(get(cfg, "tie_word_embeddings", False))
    flags = detect_flags(cfg)
    is_moe = "moe" in flags

    blocks: list[dict[str, Any]] = []
    blocks.append(embedding_block(v, d, tied))

    if family == "mha":
        attn_n, attn_pack = attn_mha(d)
    elif family == "gqa":
        attn_n, attn_pack = attn_gqa(d, H, Hkv, dh)
    elif family == "mla":
        qr = int(get(cfg, "q_lora_rank"))
        kr = int(get(cfg, "kv_lora_rank"))
        dn = int(get(cfg, "qk_nope_head_dim"))
        dr = int(get(cfg, "qk_rope_head_dim"))
        dv = int(get(cfg, "v_head_dim"))
        attn_n, attn_pack = attn_mla(d, H, qr, kr, dn, dr, dv)
    else:
        raise ValueError(f"unknown family {family}")

    norm_pair = 2 * d
    if is_moe:
        ffn_n, ffn_pack = moe_score_routed(d, f_moe, E, K, Es)
    else:
        ffn_n, ffn_pack = dense_swiglu(d, f_dense)

    per_layer = attn_n + norm_pair + ffn_n

    block_sub = [
        block_entry(
            "rmsnorm_pre_attn", "rmsnorm",
            f"[B, T, {d}]", f"[B, T, {d}]",
            "d", f"{d}", d,
        ),
        block_entry(
            "attention", f"attn_{family}",
            f"[B, T, {d}]", f"[B, T, {d}]",
            attn_pack["symbolic"], attn_pack["concrete"], attn_n,
        ),
        block_entry(
            "rmsnorm_pre_ffn", "rmsnorm",
            f"[B, T, {d}]", f"[B, T, {d}]",
            "d", f"{d}", d,
        ),
        block_entry(
            "ffn", "moe" if is_moe else "ffn_swiglu",
            f"[B, T, {d}]", f"[B, T, {d}]",
            ffn_pack["symbolic"], ffn_pack["concrete"], ffn_n,
            extra=ffn_pack if is_moe else None,
        ),
    ]

    blocks.append(
        block_entry(
            "transformer_block", "transformer_block",
            f"[B, T, {d}]", f"[B, T, {d}]",
            "Σ sub-blocks",
            f"per layer = {fmt_count(per_layer)}",
            per_layer,
            extra={
                "repeat": L,
                "subblocks": block_sub,
                "first_k_dense_replace": fkd,
            },
        )
    )

    blocks.append(rmsnorm_block("final_norm", d, f"[B, T, {d}]"))
    blocks.append(lm_head_block(v, d, None, tied=tied))

    embed_n = blocks[0]["param_count"]
    lm_head_n = blocks[-1]["param_count"]
    total = embed_n + L * per_layer + d + lm_head_n

    if is_moe:
        active_layer = attn_n + norm_pair + ffn_pack["active_per_token"]
        active_total = embed_n + L * active_layer + d + lm_head_n
    else:
        active_total = total

    summary = {
        "total_params": int(total),
        "total_label": fmt_count(int(total)),
        "active_params": int(active_total),
        "active_label": fmt_count(int(active_total)),
        "mtp_params": 0,
        "mtp_label": "—",
        "quantization": _quant_label(cfg),
    }

    return {
        "model": {
            "id": cfg.get("_source_id", ""),
            "model_type": cfg.get("model_type", family),
            "family": family,
            "flags": sorted(flags),
            "dims": {
                "d": d, "v": v, "L": L, "H": H, "Hkv": Hkv, "dh": dh,
                "f_dense": f_dense, "f_moe": f_moe,
                "E": E, "K": K, "Es": Es, "first_k_dense_replace": fkd,
            },
            "compress_ratios": [],
            "compress_ratio_bins": {},
            "summary": summary,
        },
        "blocks": blocks,
    }


def _quant_label(cfg: dict[str, Any]) -> str:
    qc = cfg.get("quantization_config")
    if not qc:
        return ""
    method = qc.get("quant_method", "")
    fmt = qc.get("fmt") or qc.get("activation_scheme")
    expert_dtype = cfg.get("expert_dtype")
    bits = []
    if method:
        bits.append(method.upper())
    if fmt:
        bits.append(str(fmt))
    if expert_dtype:
        bits.append(f"experts={expert_dtype}")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="HF repo id or local config.json path")
    ap.add_argument("--output", "-o", default="-", help="Output arch.json path (- for stdout)")
    ap.add_argument("--family", default=None, help="Force a family: mha|gqa|mla|dsv4")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    cfg = load_config(args.source, trust_remote_code=args.trust_remote_code, revision=args.revision)
    cfg["_source_id"] = args.source

    family = detect_family(cfg, forced=args.family)
    if family == "dsv4":
        spec = build_spec_dsv4(cfg)
    else:
        spec = build_spec_simple(cfg, family)

    out = json.dumps(spec, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Wrote {args.output} ({spec['model']['family']} family, "
              f"{spec['model']['summary']['total_label']} params)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
