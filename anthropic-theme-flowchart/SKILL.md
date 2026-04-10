---
name: html-flowchart-anthropic
description: Create and revise pure HTML/CSS flowcharts using an Anthropic-inspired design language. Use when Codex needs to produce process diagrams, decision trees, pipelines, or system flows that should share warm ivory backgrounds, transparent dashed grouping containers, pastel node fills, SF Pro-style sans-serif labels, smaller rounded corners, quiet orthogonal connectors, and theme-tinted text hierarchy in standalone `.html` outputs.
---

# Anthropic Flowcharts In HTML/CSS

## Overview

Build flowcharts as standalone HTML/CSS artifacts using one visual system: warm neutral surfaces, pastel role-based fills, transparent dashed grouping containers, slightly tighter rounded nodes, quiet orthogonal connectors, and concise sans-serif labeling with primary/secondary color hierarchy. Use this skill when the user wants the highest-fidelity result rather than diagram-tool compatibility.

## Workflow

1. Reduce the process to 4-8 nodes before writing syntax.
2. Choose a layout:
   - Use a two-lane grouped layout when the process splits across active and background systems.
   - Use a single-column stack when the process is strictly sequential or mobile-first.
3. Start from the HTML references:
   - Shared tokens: `references/design-language.md`
   - HTML guidance: `references/html-flowchart.md`
   - Generator entrypoint: `src/build.ts`
   - Diagram specs: `src/diagrams/*.ts`
4. Assign 4-6 role colors from the shared pastel palette instead of falling back to monochrome.
5. Keep labels short and use intentional line breaks only when they improve scanning.
6. Split label hierarchy:
   - Primary line should use a near-black version of the node's own theme color.
   - Secondary line should use a deeper version of the node's own theme color.
7. Prefer one coordinate system for nodes and connectors:
   - Use inline SVG with `foreignObject` cards when connector accuracy matters.
   - Route arrows from explicit edge anchors, not approximate floating overlays.
8. Keep connector routing deterministic:
   - Prefer orthogonal elbow connectors over curves.
   - Use hollow `>` arrowheads instead of filled triangle tips.
   - Match connector annotation font size to the node secondary line.
9. When lanes sit in separate SVGs:
   - Match CSS grid column widths to the SVG `viewBox` widths to avoid hidden scaling drift.
   - Add geometry comments that name each node's west/east/north/south center anchors before the connector paths.

## Labeling Rules

- Keep labels to 2 lines whenever possible.
- Use sentence case, not all caps, inside nodes.
- Use one short noun phrase plus one short supporting phrase.
- If a trailing word is semantically weaker, move it into the secondary line.
  - Example: `Observation` + `collector`, `System prompt` + `injection`.
- Prefer verbs that imply progression: define, review, route, validate, deliver.
- Use sans-serif in all flow boxes, lane labels, and connector annotations.
- Reduce corner radii slightly compared with editorial card layouts. The target is diagrammatic, not pill-like.
- Avoid pure black text inside nodes. Primary text should still be theme-aware.
- Make the primary line slightly larger and heavier than the secondary line, but keep lane labels regular weight.

## HTML/CSS Mapping

- Use pure HTML/CSS for every output in this skill.
- Author diagrams in TS specs and generate HTML from them; do not hand-maintain connector coordinates in `assets/*.html`.
- Default to an SF Pro-style system stack:
  - Body and annotations: `"SF Pro Text", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Segoe UI", sans-serif`
  - Node headings: `"SF Pro Display", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Segoe UI", sans-serif`
- Keep the full diagram sans-serif when the output should match modern product diagrams more than editorial pages.
- Use smaller card radii than a landing page UI and keep shadows restrained.
- Use inline SVG when connectors need exact control.
- Keep connector strokes thin and quiet. Arrowheads should feel technical, not illustrative.
- Default dashed grouping frames to transparent fills.
- For grouped layouts, put the secondary lane title inside the dashed frame when that keeps the grouping visually clearer.
- Preserve mobile readability by collapsing multi-column or alternating layouts into a single vertical stack.

## Connector Discipline

- Treat each node as a rectangle with explicit anchor points:
  - `north = (x + w / 2, y)`
  - `south = (x + w / 2, y + h)`
  - `west = (x, y + h / 2)`
  - `east = (x + w, y + h / 2)`
- Write connector coordinates from those anchors, then leave a short HTML comment showing the derivation.
- If a connector crosses between separate SVG columns, document the cross-column transform in a short comment block.
- When the visual needs to bridge from a node in one SVG to a connector in another, either:
  - add a short stub from the node edge to the column boundary, or
  - render the whole connector in the bridge SVG with the correct converted coordinates.

## Resources

- Read `references/design-language.md` for the shared token system.
- Read `references/html-flowchart.md` for the HTML-specific layout and styling rules.
- Start from these HTML artifacts:
  - `src/build.ts`
  - `src/diagrams/review-demo.ts`
  - `src/diagrams/template.ts`
  - Then inspect generated artifacts:
  - `assets/anthropic-flowchart-template.html`
  - `assets/review-demo.html`

## Trigger Examples

- "Draw this as a pure HTML/CSS flowchart."
- "Create a browser-renderable flowchart in Anthropic style."
- "Restyle this diagram as an HTML/CSS grouped flowchart."

## Anti-Patterns

- Do not overload the chart with icons or long paragraphs.
- Do not use saturated rainbow colors or sharp neon contrast.
- Do not simulate a slide deck unless the user explicitly wants a presentation page.
- Do not introduce diagram-tool-specific syntax or export formats in this skill.
