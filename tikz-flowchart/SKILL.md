---
name: tikz-flowchart
description: "Creates professional TikZ flowcharts with standardized themes, including Google Material-like and Anthropic-inspired options."
---

# TikZ Flowchart Skill

This skill provides standardized templates/styles for establishing professional technical diagrams using LaTeX TikZ. It is designed for creating flowcharts, architecture diagrams, and process flows.

## Usage

When asked to "create a flowchart" or "draw a diagram" in this project, first choose a single theme, then open only that theme's file as the starting point.

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

1.  **Relative Positioning**: Use `right=of Node`, `below=of Node` for layout stability. Adjust distances via `node distance` in the `tikzpicture` options.
2.  **Grouping**: Use the `fit` library and a background-layer group style to draw lane or phase containers. For the Anthropic theme, keep containers transparent and dashed.
3.  **Orthogonal Edges**: Use `-|` and `|-` path operations for clean, orthogonal lines (e.g., `(A) -| (B)`).
4.  **Conciseness**: Keep node text short. Use `\\` for line breaks and `\scriptsize` or a dedicated label macro for secondary details.
5.  **Anthropic Node Shapes**: Prefer rounded rectangles as the default node shape in the Anthropic theme. Use diamonds only for true decisions and avoid heavy cylinders or shadowed compute boxes.
6.  **Anthropic Typography**: Keep the full diagram sans-serif, with tinted primary and secondary lines inside each node instead of pure black text.
