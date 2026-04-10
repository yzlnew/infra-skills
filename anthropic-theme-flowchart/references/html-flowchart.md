# HTML/CSS Flowchart Guidance

Use HTML/CSS for every output in this skill.

## When To Prefer HTML

- The chart should match a grouped product diagram with dashed containers and pastel fills.
- The output should feel more like a designed artifact than a generic diagram export.
- The chart must adapt cleanly between desktop and mobile.
- The user wants a browser-ready file or a PDF rendered from HTML.

## Structure Options

- Lane grid:
  - Best for phases, owners, or "foreground vs background" systems.
  - Use dashed rounded containers and 2-4 cards per lane.
  - Add a small number of explicit bridge arrows between lanes.
  - Prefer a fixed desktop grid with a dedicated bridge column when cross-lane connectors matter.
- Hub and spoke:
  - Use sparingly and only when a central node is semantically correct.

## Styling Rules

- Use sans-serif for the whole diagram when the reference is a modern systems chart.
- Background uses a warm ivory gradient or subtle neutral field.
- Cards use `14px` to `18px` radius.
- Border is soft, shadow is subtle, and each node role can use its own muted pastel fill.
- Lane containers should use dashed borders and italic or light-emphasis headers.
- Node copy should be split into two levels:
  - Heading in a near-black version of the node hue
  - Supporting line in a darker node-tinted color
- Connector strokes should usually stay around `1.4px` to `1.8px`.
- Arrowheads should be small and should terminate exactly on node edge anchors.

## Interaction Rules

- Animations should be minimal.
- Respect `prefers-reduced-motion`.
- Avoid hover-driven logic for essential meaning.

## Output Pattern

- Prefer a single self-contained HTML file with inline CSS.
- Use inline SVG only when connectors need exact control.
- For the most deterministic layout, put both connectors and node cards inside the same SVG coordinate system via `foreignObject`.
