# Shared Design Language

Use these tokens across standalone HTML/CSS flowcharts so outputs feel like one family.

## Core Tokens

```text
Background: #FAF8F3
Soft background: #F4F0E8
Primary panel: #FFFDFC
Lane border: #D7D0C5
Connector line: #9EA9B1
Text / ink: #31424B
Muted text: #66757E
Lavender fill: #E7E3FF
Lavender stroke: #8D86D8
Lavender primary text: #373457
Lavender secondary text: #5852A7
Mint fill: #DDF4EC
Mint stroke: #78B5A7
Mint primary text: #294B45
Mint secondary text: #3F8F7F
Teal fill: #D9F0F2
Teal stroke: #6CA9B0
Teal primary text: #29464A
Teal secondary text: #457E85
Cream fill: #F4EEE3
Cream stroke: #C1B39E
Cream primary text: #4E4437
Cream secondary text: #8E7A61
Amber fill: #F8E7C9
Amber stroke: #D6A86B
Amber primary text: #57401C
Amber secondary text: #A27431
Peach fill: #F8E0D9
Peach stroke: #CF8E81
Peach primary text: #55342E
Peach secondary text: #A66559
```

## Typography Roles

- Title, labels, chips, lane headers:
  - `Manrope`, `Inter`, `Avenir Next`, `Segoe UI`, `sans-serif`

Use sans-serif for the full diagram when the target look is closer to a modern product or systems diagram than an editorial cover page.
Within nodes, primary text should use a near-black version of the node's own hue, while secondary text should shift further toward the node color.

## Shape Rules

- Radius:
  - HTML/CSS: `14px` to `18px`
- Borders: thin and low contrast
- Shadows: only HTML should use a soft shadow
- Grouping: dashed rounded containers for lanes or ownership scopes
- Color usage: 4-6 pastel role fills are acceptable when they remain soft and semantically consistent
- Text hierarchy: near-black theme tint for the primary line, darker theme tint for the secondary line
- Connectors: use thin strokes and small arrowheads anchored to node edges

## Composition Rules

- Use generous whitespace.
- Keep labels short.
- Prefer grouped lanes over free-floating nodes when the process splits across systems.
- Prefer calm pastel separation over dark emphasis blocks.
- Keep connector paths quiet so color is carried by nodes, not arrows.
- Prefer one shared SVG coordinate system for cards and connectors when precision matters.
