---
name: tikz-flowchart
description: "Creates professional TikZ flowcharts with standardized themes, including Google Material-like and Anthropic-inspired options."
---

# TikZ Flowchart Skill

This skill provides standardized templates/styles for establishing professional technical diagrams using LaTeX TikZ. It is designed for creating flowcharts, architecture diagrams, and process flows.

## Usage

When asked to "create a flowchart" or "draw a diagram" in this project, first choose a single theme, then open only that theme's file as the starting point.

Before writing any `\node` or `\draw` commands, plan the layout first: decide the main flow direction, group nodes into rows/columns or lanes, and make sure related nodes are aligned so the connectors can stay orthogonal.

If you need to judge whether the generated diagram actually satisfies the user's requirements, do not rely only on your own generation pass. Spawn a subagent for an independent review and use that review to verify compliance or identify gaps.

- Use the **default Material-like theme** for conventional engineering diagrams with stronger semantic color coding and distinct data/storage/compute shapes.
- Use the **Anthropic theme** when the user wants warm ivory backgrounds, quiet dashed grouping containers, pastel cards, thin connectors, and a calm product-diagram look.

## Theme Selection

- If the request mentions Anthropic, warm/product-diagram styling, pastel cards, or dashed ownership lanes, use the **Anthropic theme**.
- Otherwise default to the **Material-like theme**.

## Theme Files

| Theme | File |
|-------|------|
| Default Material-Like | [`themes/material-like.md`](themes/material-like.md) |
| Anthropic | [`themes/anthropic.md`](themes/anthropic.md) |

After selecting a theme, read the corresponding file for the full style guide, color definitions, node styles, and LaTeX template.

## Best Practices

1.  **Layout First**: Determine the node layout before drawing edges. Decide the dominant reading direction, assign nodes to rows/columns or swimlanes, and align related nodes so most edges can be drawn with zero or one bend. If the routing looks awkward, fix the layout first instead of forcing the edge.
2.  **Relative Positioning**: Use `right=of Node`, `below=of Node`, or explicit coordinates to keep the layout stable. Adjust spacing via `node distance` or fixed `x`/`y` gaps so nodes have enough room for labels and bends.
3.  **Perpendicular Edge Entry**: Connect to explicit anchors such as `.east`, `.west`, `.north`, and `.south`. The segment that touches a node should be perpendicular to that side: horizontal into `.east`/`.west`, vertical into `.north`/`.south`.
4.  **Orthogonal Edges**: Prefer straight `--` lines for nodes that already share a row or column, and use `-|` / `|-` only when a bend is actually needed. Avoid diagonal lines and avoid lines that hit a node corner or graze an edge at an angle.
5.  **Grouping**: Use the `fit` library and a background-layer group style to draw lane or phase containers. For the Anthropic theme, keep containers transparent and dashed.
6.  **Conciseness**: Keep node text short. Use `\\` for line breaks and `\scriptsize` or a dedicated label macro for secondary details.
7.  **Anthropic Node Shapes**: Prefer rounded rectangles as the default node shape in the Anthropic theme. Use diamonds only for true decisions and avoid heavy cylinders or shadowed compute boxes.
8.  **Anthropic Typography**: Keep the full diagram sans-serif, with tinted primary and secondary lines inside each node instead of pure black text.
9.  **Independent Review**: When evaluating whether the generated flowchart meets the user's constraints, spawn a subagent to review the output against the requirements. Do not treat self-evaluation during generation as sufficient verification.
