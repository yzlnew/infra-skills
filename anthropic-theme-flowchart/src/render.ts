import {
  formatPoint,
  getAnchor,
  getNode,
  resolveEdgeEndpoint,
  routePath,
} from "./geometry.ts";
import type {
  DiagramSpec,
  EdgeSpec,
  NodeSpec,
  SpaceSpec,
  SvgFrameSpec,
  TextAnchor,
} from "./types.ts";

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function markerId(diagram: DiagramSpec, space: SpaceSpec): string {
  return `${diagram.id}-${space.id}-arrow`;
}

function edgeComment(diagram: DiagramSpec, edge: EdgeSpec): string {
  if (edge.comment) {
    return edge.comment;
  }

  const describeEndpoint = (endpoint: EdgeSpec["from"]): string => {
    if ("point" in endpoint) {
      return endpoint.label;
    }
    return `${endpoint.nodeId} ${endpoint.side}-center`;
  };

  return `${describeEndpoint(edge.from)} -> ${describeEndpoint(edge.to)}`;
}

function renderHeader(diagram: DiagramSpec): string {
  if (!diagram.header) {
    return "";
  }

  const parts: string[] = ['      <header class="header">'];
  if (diagram.header.eyebrow) {
    parts.push(`        <p class="kicker">${escapeHtml(diagram.header.eyebrow)}</p>`);
  }
  if (diagram.header.title) {
    parts.push(`        <h1>${escapeHtml(diagram.header.title)}</h1>`);
  }
  if (diagram.header.description) {
    parts.push(`        <p class="intro-copy">${escapeHtml(diagram.header.description)}</p>`);
  }
  parts.push("      </header>");
  return parts.join("\n");
}

function gridTemplateColumns(diagram: DiagramSpec): string {
  return diagram.columns.map((column) => `${column.width}px`).join(" ");
}

function renderStyles(diagram: DiagramSpec): string {
  return `      :root {
        --canvas: #faf8f3;
        --lane-border: #d7d0c5;
        --line: #9ea9b1;
        --ink: #31424b;
        --muted: #697780;
        --lavender: #e7e3ff;
        --lavender-stroke: #8d86d8;
        --lavender-ink: #373457;
        --lavender-subtle: #5852a7;
        --mint: #ddf4ec;
        --mint-stroke: #78b5a7;
        --mint-ink: #294b45;
        --mint-subtle: #3f8f7f;
        --teal: #d9f0f2;
        --teal-stroke: #6ca9b0;
        --teal-ink: #29464a;
        --teal-subtle: #457e85;
        --cream: #f4eee3;
        --cream-stroke: #c1b39e;
        --cream-ink: #4e4437;
        --cream-subtle: #8e7a61;
        --amber: #f8e7c9;
        --amber-stroke: #d6a86b;
        --amber-ink: #57401c;
        --amber-subtle: #a27431;
        --peach: #f8e0d9;
        --peach-stroke: #cf8e81;
        --peach-ink: #55342e;
        --peach-subtle: #a66559;
        --shadow: 0 18px 36px rgba(73, 84, 92, 0.07);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family:
          "SF Pro Text",
          "SF Pro Display",
          -apple-system,
          BlinkMacSystemFont,
          "Helvetica Neue",
          "Segoe UI",
          sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(141, 134, 216, 0.15), transparent 24%),
          radial-gradient(circle at 78% 16%, rgba(120, 181, 167, 0.12), transparent 22%),
          radial-gradient(circle at 85% 92%, rgba(214, 168, 107, 0.12), transparent 18%),
          linear-gradient(180deg, #fffdf9 0%, var(--canvas) 54%, #f4efe6 100%);
      }

      main {
        max-width: 1260px;
        margin: 0 auto;
        padding: 30px 24px 48px;
      }

      .header {
        margin-bottom: 16px;
      }

      .kicker {
        margin: 0 0 8px;
        color: var(--muted);
        font-size: 12.5px;
        font-weight: 400;
        letter-spacing: 0.18em;
        text-transform: uppercase;
      }

      h1 {
        margin: 0 0 10px;
        font-size: clamp(1.9rem, 3vw, 2.7rem);
        line-height: 1.02;
        letter-spacing: -0.04em;
      }

      .intro-copy {
        margin: 0;
        max-width: 66ch;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.6;
      }

      .diagram {
        display: grid;
        grid-template-columns: ${gridTemplateColumns(diagram)};
        justify-content: center;
        align-items: start;
        gap: 0;
      }

      .lane {
        border: 2px dashed var(--lane-border);
        border-radius: 26px;
        padding: 18px 18px 22px;
        background: transparent;
      }

      .lane-title {
        margin: 0 0 12px;
        padding-left: 10px;
        color: var(--muted);
        font-size: 17px;
        font-style: italic;
        font-weight: 400;
      }

      .lane-canvas,
      .bridge-canvas {
        display: block;
        width: 100%;
        height: auto;
      }

      .bridge-canvas {
        overflow: visible;
      }

      .secondary-panel {
        position: relative;
        z-index: 1;
        padding-top: 18px;
      }

      .bridge-column {
        position: relative;
        z-index: 2;
        padding-top: 56px;
      }

      .secondary-panel .lane-canvas {
        overflow: visible;
      }

      .node {
        display: flex;
        flex-direction: column;
        justify-content: center;
        width: 100%;
        height: 100%;
        gap: 2px;
        padding: 8px 4px;
        border-radius: 16px;
        border: 1.5px solid transparent;
        box-shadow: var(--shadow);
        text-align: center;
        line-height: 1.22;
        --node-ink: var(--ink);
        --node-subtle: var(--muted);
      }

      .node h2 {
        margin: 0;
        font-family:
          "SF Pro Display",
          "SF Pro Text",
          -apple-system,
          BlinkMacSystemFont,
          "Helvetica Neue",
          "Segoe UI",
          sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: var(--node-ink);
      }

      .node p {
        margin: 0;
        color: var(--node-subtle);
        font-size: 0.97rem;
        font-weight: 400;
      }

      .node--lavender {
        background: var(--lavender);
        border-color: var(--lavender-stroke);
        --node-ink: var(--lavender-ink);
        --node-subtle: var(--lavender-subtle);
      }

      .node--mint {
        background: var(--mint);
        border-color: var(--mint-stroke);
        --node-ink: var(--mint-ink);
        --node-subtle: var(--mint-subtle);
      }

      .node--teal {
        background: var(--teal);
        border-color: var(--teal-stroke);
        --node-ink: var(--teal-ink);
        --node-subtle: var(--teal-subtle);
      }

      .node--cream {
        background: var(--cream);
        border-color: var(--cream-stroke);
        --node-ink: var(--cream-ink);
        --node-subtle: var(--cream-subtle);
      }

      .node--amber {
        background: var(--amber);
        border-color: var(--amber-stroke);
        --node-ink: var(--amber-ink);
        --node-subtle: var(--amber-subtle);
      }

      .node--peach {
        background: var(--peach);
        border-color: var(--peach-stroke);
        --node-ink: var(--peach-ink);
        --node-subtle: var(--peach-subtle);
      }

      .svg-note {
        fill: var(--muted);
        font-size: 0.97rem;
        font-weight: 400;
        letter-spacing: -0.01em;
      }

      .svg-lane-title {
        fill: var(--muted);
        font-size: 17px;
        font-style: italic;
        font-weight: 400;
      }

      .connector {
        fill: none;
        stroke: var(--line);
        stroke-width: 1.6;
        stroke-linecap: round;
        stroke-linejoin: round;
      }

      .connector--soft {
        stroke-width: 1.4;
        stroke-dasharray: 5 6;
      }

      @media (max-width: 1180px) {
        .diagram {
          grid-template-columns: 1fr;
          gap: 18px;
        }

        .bridge-column {
          display: none;
        }
      }`;
}

function renderFrame(frame: SvgFrameSpec): string {
  const rx = frame.rx ?? 26;
  const ry = frame.ry ?? rx;
  return `            <rect x="${frame.x}" y="${frame.y}" width="${frame.width}" height="${frame.height}" rx="${rx}" ry="${ry}" fill="none" stroke="#d7d0c5" stroke-width="2" stroke-dasharray="8 6"></rect>`;
}

function renderNode(node: NodeSpec): string {
  const subtitle = node.subtitle ? `\n                <p>${escapeHtml(node.subtitle)}</p>` : "";
  return `            <foreignObject x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}">
              <div xmlns="http://www.w3.org/1999/xhtml" class="node node--${node.tone}">
                <h2>${escapeHtml(node.title)}</h2>${subtitle}
              </div>
            </foreignObject>`;
}

function textAnchorValue(anchor: TextAnchor | undefined): string {
  return anchor ?? "start";
}

function renderLabel(label: NonNullable<EdgeSpec["label"]>): string {
  if (label.lines.length === 1) {
    return `            <text class="svg-note" x="${label.x}" y="${label.y}" text-anchor="${textAnchorValue(label.anchor)}">${escapeHtml(label.lines[0])}</text>`;
  }

  const lineHeight = 16;
  const spans = label.lines
    .map((line, index) => {
      const dy = index === 0 ? 0 : lineHeight;
      return `              <tspan x="${label.x}" dy="${dy}">${escapeHtml(line)}</tspan>`;
    })
    .join("\n");

  return `            <text class="svg-note" x="${label.x}" y="${label.y}" text-anchor="${textAnchorValue(label.anchor)}">
${spans}
            </text>`;
}

function renderEdge(diagram: DiagramSpec, space: SpaceSpec, edge: EdgeSpec): string {
  const start = resolveEdgeEndpoint(diagram, space.id, edge.from);
  const end = resolveEdgeEndpoint(diagram, space.id, edge.to);
  const path = routePath(start, end, edge.route);
  const classes = edge.style === "soft" ? "connector connector--soft" : "connector";
  const marker = markerId(diagram, space);
  const attrs: string[] = [`class="${classes}"`, `d="${path}"`];

  if (edge.arrows === "end" || edge.arrows === "both") {
    attrs.push(`marker-end="url(#${marker})"`);
  }
  if (edge.arrows === "both") {
    attrs.push(`marker-start="url(#${marker})"`);
  }

  return `            <!-- ${escapeHtml(edgeComment(diagram, edge))} -->\n            <path ${attrs.join(" ")}></path>`;
}

function nodeGeometryComment(node: NodeSpec): string[] {
  const north = getAnchor(node, "north");
  const south = getAnchor(node, "south");
  const west = getAnchor(node, "west");
  const east = getAnchor(node, "east");
  const heading = `              ${node.title}: x=${node.x}, y=${node.y}, w=${node.width}, h=${node.height}`;
  const anchors = `                north=${formatPoint(north)} south=${formatPoint(south)} west=${formatPoint(west)} east=${formatPoint(east)}`;
  return [heading, anchors];
}

function renderGeometryComment(diagram: DiagramSpec, space: SpaceSpec): string {
  const lines: string[] = [];

  if (space.commentLines?.length) {
    lines.push(...space.commentLines.map((line) => `              ${line}`));
  }

  const spaceNodes = diagram.nodes.filter((node) => node.spaceId === space.id);
  if (spaceNodes.length > 0) {
    lines.push(`              ${space.title ?? space.id} node geometry`);
    for (const node of spaceNodes) {
      lines.push(...nodeGeometryComment(node));
    }
  }

  if (lines.length === 0) {
    return "";
  }

  return `            <!--\n${lines.join("\n")}\n            -->`;
}

function renderSvg(space: SpaceSpec, diagram: DiagramSpec): string {
  const marker = markerId(diagram, space);
  const edges = diagram.edges.filter((edge) => edge.spaceId === space.id);
  const nodes = diagram.nodes.filter((node) => node.spaceId === space.id);

  const sections: string[] = [
    `          <svg class="${space.panelKind === "bridge" ? "bridge-canvas" : "lane-canvas"}" viewBox="0 0 ${space.width} ${space.height}" aria-hidden="true">`,
    "            <defs>",
    `              <marker id="${marker}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">`,
    '                <path d="M 1 1 L 7 4 L 1 7" fill="none" stroke="#9ea9b1" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"></path>',
    "              </marker>",
    "            </defs>",
  ];

  const geometryComment = renderGeometryComment(diagram, space);
  if (geometryComment) {
    sections.push("", geometryComment);
  }

  if (space.svgFrame) {
    sections.push(renderFrame(space.svgFrame));
  }

  if (space.title && space.titlePlacement === "inside-frame") {
    sections.push(`            <text class="svg-lane-title" x="20" y="28">${escapeHtml(space.title)}</text>`);
  }

  if (edges.length > 0) {
    sections.push(...edges.map((edge) => renderEdge(diagram, space, edge)));
  }

  if (nodes.length > 0) {
    sections.push(...nodes.map((node) => renderNode(node)));
  }

  const labels = edges
    .filter((edge) => edge.label)
    .map((edge) => renderLabel(edge.label!));

  if (labels.length > 0) {
    sections.push(...labels);
  }

  sections.push("          </svg>");
  return sections.join("\n");
}

function renderPanel(space: SpaceSpec, diagram: DiagramSpec): string {
  const svg = renderSvg(space, diagram);
  const titleOutside =
    space.title && space.titlePlacement === "outside"
      ? `\n          <p class="lane-title">${escapeHtml(space.title)}</p>`
      : "";

  if (space.panelKind === "lane") {
    return `        <article class="${space.panelClass}">${titleOutside}
${svg}
        </article>`;
  }

  if (space.panelKind === "bridge") {
    return `        <div class="${space.panelClass}" aria-hidden="true">
${svg}
        </div>`;
  }

  return `        <section class="${space.panelClass}">${titleOutside}
${svg}
        </section>`;
}

export function renderDiagram(diagram: DiagramSpec): string {
  const panels = diagram.spaces.map((space) => renderPanel(space, diagram)).join("\n\n");
  const header = renderHeader(diagram);

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(diagram.pageTitle)}</title>
    <style>
${renderStyles(diagram)}
    </style>
  </head>
  <body>
    <main>
${header}

      <section class="diagram" aria-label="${escapeHtml(diagram.ariaLabel)}">
${panels}
      </section>
    </main>
  </body>
</html>
`;
}
