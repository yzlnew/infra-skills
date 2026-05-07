# Raschka-Style Visual Conventions

Distilled from Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/). The point isn't pixel-identical reproduction — it's matching the **information design** so the reader can read the architecture top-to-bottom without rereading the legend.

## Reading direction

- **Top-to-bottom** is the data flow. `input_ids` at the top, `logits` at the bottom.
- The model is unrolled in *depth*, not in *time*. Repeated transformer layers are drawn **once** with a `× L layers` annotation outside the box.
- Auxiliary heads (MTP, value heads) hang off the side, not interleaved.

## Three-column layout

| Left column | Center column | Right column |
|-------------|---------------|--------------|
| Activation shape | Block (the actual diagram element) | Parameter formula + concrete value |

The shape and the formula sit at the *output* of each block (i.e. on the connector arrow leaving the block, not entering).

Shapes are written `[B, T, d]` style, monospace, in a muted color. Formula annotations are right-aligned, two-line: line 1 symbolic (`d · qr`), line 2 substituted-and-evaluated (`4096 · 1024 = 4.19M`).

## Block aesthetics

- Rounded rectangles, ~5pt corner radius, thin (~0.9pt) strokes.
- One soft pastel fill per role (attention / norm / projection / FFN / etc.) — see the Anthropic palette (`tikz-flowchart/themes/anthropic.md`).
- No drop shadows, no gradients, no 3D affordances.
- Two text levels per block: bold primary label (e.g. `wq_b`), light secondary (e.g. `LoRA-up`).

## Sub-block expansion

When a logical block contains internally interesting structure (MLA's six projections, MoE's router + experts, mHC's fan-in/fan-out), expand it **inside a dashed grouping container** rather than collapsing to one box. The container has an italic muted header (e.g. `Sparse Attention`).

For repeated identical sub-blocks (MoE has 256 routed experts), draw two explicitly + an ellipsis node + the last one. This keeps the figure readable while making the count evident.

## Connectors

- Orthogonal: prefer straight horizontal/vertical, with a single bend if needed.
- Thin (0.85pt), arrowheads small.
- Connect to explicit anchors (`.north`, `.south`, etc.) — the segment touching a node should be perpendicular to that node edge.
- Residual streams as a thin track on the right side, re-entering via `+` nodes.

## Annotations

- **Shape annotations** sit on connector arrows, not inside boxes.
- **Formula annotations** sit in the right column, aligned roughly to the block's vertical center. Use `\texttt{}` (monospace).
- **Per-layer pattern annotations** (DSv4 compress_ratios) sit beneath the Block group as a thin horizontal strip — one tiny cell per layer, color-coded by ratio bucket (gray=0, light=4, dark=128). This makes the alternation visible without cluttering the main column.
- **Layer-multiplier annotations** (`× 43 layers`, `× 3 hash-routed layers`) sit to the *right* of the dashed grouping container, vertically centered.

## Bottom summary box

A small rounded card in the lower-right corner with three lines:

```
total params:    158.07 B
active per tok:    8.47 B
MTP head:          5.82 B
weights: FP8 + FP4 (experts) + BF16 (gates/norms)
```

The summary box uses a faint cream fill so it reads as metadata, not as an architectural component.

## Anti-patterns to avoid

- Putting parameter counts *inside* the block — crowds the label, hides the formula.
- Drawing all 256 experts — busy without being informative.
- Mixing diagonal and orthogonal connectors — pick one (orthogonal).
- Color-coding by some semantic the reader can't reconstruct (e.g. "blue means trainable" — everything is trainable; not useful).
- Crossing connectors when re-routing would avoid it.
- Multiple residual streams drawn as one — for HC, draw a thin band of `hc_mult` parallel tracks, even if abstracted.

## When the model has a feature not yet in the gallery

Examples: V4-Flash's mHC, learned Indexer for sparse attention, hash-routed MoE. Treat them as new block types and design a small visual idiom:

- mHC fan-in / fan-out: a thin band of 4 tracks merging into a Sinkhorn-mixer node, then re-expanding.
- Learned Indexer: a small parallelogram (selection node) with `top-K` annotation on the output edge.
- Hash router: a vertical lookup-table glyph (rectangle with horizontal lines) instead of the score-router's circle.

Document the new idiom in this file once it's in use, so future renders stay consistent.
