---
name: openai-dotcom-viz
description: >-
  Build figures in OpenAI's blog / research / system-card "dotcom" visual style —
  both (a) bar charts (monochrome bars with a darker same-hue stroke, rounded
  corners, a black y-axis with outward ticks and no gridlines, circle legend
  markers left-aligned to the y-axis, value labels above bars, angled category
  labels) and (b) flow / process diagrams (rounded boxes, monospace uppercase
  pills, pink highlights, thin open-chevron "arrow" connectors with rounded
  corners, dashed = negative branch). Everything uses the real OpenAI Sans font
  and is emitted as self-contained HTML/SVG with zero dependencies. Use this
  whenever the user wants a chart, pipeline/flow diagram, or figure that looks
  like OpenAI's website / posts / papers, asks to reproduce an OpenAI figure, or
  is building a deck or page in an OpenAI-flavored style. Prefer this over generic
  charting (matplotlib/Chart.js/Recharts) or generic diagram tools (Mermaid) when
  the aesthetic should match OpenAI.
---

# OpenAI dotcom figures

Reproduce the two figure types OpenAI uses in its posts, as **self-contained
HTML/SVG** (no Vega, D3, Mermaid, or build step):

1. **Bar charts** — grouped or single-series. → see "Bar charts" below.
2. **Flow / process diagrams** — e.g. the "Quality assurance pipeline". → see
   "Flow diagrams" below.

Both share one design system: the real **OpenAI Sans** font (loaded from OpenAI's
public CDN), near-black ink `#0d0d0d`, and a pink accent. The full color palette
(6 hues × 5 shades) and token rules are in `references/design-spec.md` — read it
when you need extra hues, more series, or to theme surrounding page elements so
everything reads as one system.

## Bar charts

Use the bundled renderer `assets/openai-chart.js` — a zero-dependency function
that draws the chart into an `<svg>` from a data spec. It injects the font + the
chart styles itself, so a `<script src>` plus a spec is all a page needs.

```html
<script src="assets/openai-chart.js"></script>
<svg id="chart"></svg>
<script>
renderOpenAIBarChart("#chart", {
  title: "Share of Dataset Flagged by Issue Type",
  yAxisTitle: "Percent of total dataset",
  valueFormat: "percent",            // values are fractions (0.144 -> "14.4%")
  yMax: 0.20, yTickStep: 0.05,
  categories: ["Overly strict\ntests", "Low-coverage\ntests", /* … */],  // \n wraps a label
  series: [
    { name: "Human supervised agent review", values: [0.144, 0.041, /* … */] },
    { name: "Human annotations",             values: [0.178, 0.094, /* … */] }
  ]
});
</script>
```

Colors auto-assign from the palette, so you usually pass only names + values.
`assets/chart-example.html` reproduces the reference figure.

**Spec fields:** `categories` (use `\n` to wrap), `series[]` (`{name, values[], fill?, stroke?}`),
`title`, `yAxisTitle`, `valueFormat` (`"percent"` default | `"number"` | `fn`),
`yMax`, `yTickStep`/`yTicks`, `width`/`height`, `bottomMargin` (raise if labels clip),
`legend:false`, `palette` (override colors). The look — one-hue bars with a darker
same-hue stroke, rounded rects, black y-axis with ticks and no gridlines, circle
legend left-aligned to the axis, value labels above bars, −45° category labels —
is baked in.

## Flow diagrams

These are **hand-arranged**, not generated (branching/alignment are judgement
calls). The skill gives you the primitives and the look; you assemble the layout.
Include `assets/openai-figure.css`, wrap the diagram in `<div class="oai-figure">`,
add the arrow marker once, and compose from `.box` / `.subbox` / `.pill` /
`.conn`. Copy `assets/flow-example.html` (a complete working pipeline) as a start.

Read `references/flow-diagram.md` for: the primitives, the open-chevron arrow
marker, rounded-corner connector recipes, and the column layout approach. Key
style points: rounded boxes + 1.5px black keylines, monospace UPPERCASE pills
(`.pill`, `.pill.accent` = solid pink, `.pill.dotted` = stippled pink), thin
open-chevron connectors (never filled triangles), rounded corners on turns, and a
dashed shaft for a negative/"did-not-happen" branch.

> Gotcha: never put a literal three-dash arrow inside an HTML comment — the `-->`
> inside it closes the comment early and leaks text onto the page.

## Fonts & offline

OpenAI Sans loads over the network from `cdn.openai.com` (CORS-open). For strictly
offline/print output, download the four weights
(`OpenAISans-{Regular,Medium,Semibold,Bold}.woff2` under
`https://cdn.openai.com/common/fonts/openai-sans/`) and inline them as base64
`@font-face src:url(data:font/woff2;base64,…)`, replacing the CDN URLs. Otherwise
it falls back to Helvetica/Arial — close but not exact. (OpenAI's monospace, used
for pill labels, is not on the public CDN; a generic monospace matches the look.)

## Verifying output

Render headless and eyeball it (give the font time to load, or you'll snapshot
the fallback):

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --hide-scrollbars --window-size=1000,760 \
  --screenshot=out.png --virtual-time-budget=3500 "file://$PWD/figure.html"
```
