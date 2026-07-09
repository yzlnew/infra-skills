# OpenAI dotcom chart — design tokens & rules

Extracted from OpenAI's live "dotcom-chart" CSS theme and the published figures.
Use this when you need more than the two-series default, a different hue, an
offline build, or want to theme surrounding elements (page, flow diagrams) to
match the charts.

## Color palette

Each hue has 5 shades. `keyline` == shade 1 (the lightest). Bars use a **fill**
plus a **darker stroke of the same hue** — pick the stroke one or two steps
deeper than the fill.

| Hue | 1 (light) | 2 | 3 | 4 | 5 (dark) |
|---|---|---|---|---|---|
| pink   | `#fcdad6` | `#f5bacc` | `#f390ca` | `#bd569b` | `#8a3a6f` |
| blue   | `#eaf1fe` | `#cedffe` | `#a3befa` | `#5477c4` | `#2e4780` |
| green  | `#d8ecbd` | `#beeb96` | `#a3d576` | `#71b436` | `#386411` |
| orange | `#ffedde` | `#ffbda1` | `#ff9365` | `#cc6f47` | `#804126` |
| yellow | `#fff4c2` | `#ffea8f` | `#ffe15b` | `#b8a037` | `#736422` |
| gray   | `#f0f1f2` | `#dddee1` | `#bcbec4` | `#767881` | `#47484f` |

Ink / neutrals: text and axes `#0d0d0d`; muted text `#6e6e6e`; hairline borders
`#e6e6e6`; page background `#ffffff`.

### Fill → stroke pairing (the rule that makes bars read as "OpenAI")

- Dark bar: fill shade **4**, stroke shade **5**.
- Light bar: fill shade **1**, stroke shade **4**.

The default two-series grouped chart is the pink pair:
`{fill:#bd569b, stroke:#8a3a6f}` and `{fill:#fcdad6, stroke:#bd569b}`.

### Multiple series (>2)

Prefer staying within **one hue** using shades 4 and 1 for two series. For three
or more, either use shades 4 / 3 / 1 of one hue, or move to distinct hues (pink →
blue → green → orange), each with its own fill(4)/stroke(5). The bundled renderer
does the latter automatically via its `PALETTE` array.

## Typography

- Family: **OpenAI Sans** (proprietary but publicly served, CORS-open) from
  `https://cdn.openai.com/common/fonts/openai-sans/OpenAISans-{Regular|Medium|Semibold|Bold}.woff2`
  → weights 400 / 500 / 600 / 700. Fallback: `"Helvetica Neue", Arial, system-ui, sans-serif`.
- Chart title: 700. Legend + axis + value + category labels: 400. Everything near-black.

## Chart construction rules

1. **No gridlines. No bottom axis line.** Only a vertical y-axis line, with short
   ticks pointing *outward* (left, ~6px) at each labeled value.
2. **Y labels** right-aligned just left of the axis; y-axis title rotated -90°.
3. **Bars**: rounded rectangles, `rx ≈ 4` on all corners. Fill + 1.5px same-hue
   darker stroke.
4. **Grouped bars**: group occupies ~66% of its slot (band padding ~0.34); bars
   within a group nearly touch (~4px gap).
5. **Value label** above every bar, centered, near-black.
6. **Category labels** rotated **-45°**, end-anchored, wrapping to 2 lines via `\n`.
7. **Legend** drawn as filled circle + darker ring, left edge aligned to the
   y-axis line (x = left margin), regular weight, above the plot.

## Provenance

- Charts on openai.com render via a React `DotcomChart` component wrapping a
  **Vega-Lite v6** spec, themed with `--dotcom-chart-theme-*` CSS variables (the
  palette above). This skill reproduces the *output* with plain SVG so there is
  no Vega/D3 dependency.
- Flow/process diagrams in the same posts are **not** charting output — they are
  hand-authored SVG illustrations (mono uppercase pill labels, black 1.5px
  keylines, rounded boxes, pink highlight, dashed = negative branch). If theming
  a diagram to match, reuse the ink + pink tokens and OpenAI Sans / a monospace
  for the pill labels.
