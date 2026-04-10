import { getAnchor, getNode, routePath, routePoints } from "./geometry.ts";
import type { DiagramSpec, EdgeSpec, NodeSpec, SpaceSpec } from "./types.ts";

const PALETTE: Record<string, { fill: string; stroke: string; ink: string; subtle: string }> = {
  lavender: { fill: "#e7e3ff", stroke: "#8d86d8", ink: "#373457", subtle: "#5852a7" },
  mint: { fill: "#ddf4ec", stroke: "#78b5a7", ink: "#294b45", subtle: "#3f8f7f" },
  teal: { fill: "#d9f0f2", stroke: "#6ca9b0", ink: "#29464a", subtle: "#457e85" },
  cream: { fill: "#f4eee3", stroke: "#c1b39e", ink: "#4e4437", subtle: "#8e7a61" },
  amber: { fill: "#f8e7c9", stroke: "#d6a86b", ink: "#57401c", subtle: "#a27431" },
  peach: { fill: "#f8e0d9", stroke: "#cf8e81", ink: "#55342e", subtle: "#a66559" },
};

function escapeXml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function panelOuterBox(space: SpaceSpec): { x: number; y: number; width: number; height: number } {
  if (space.panelKind === "lane") {
    return { x: 0, y: 0, width: 680, height: 506 };
  }
  if (space.panelKind === "panel") {
    return { x: 812, y: 18, width: 324, height: 448 };
  }
  return { x: 680, y: 56, width: 132, height: 448 };
}

function renderPanelShell(space: SpaceSpec): string {
  if (space.panelKind === "lane") {
    return `<rect x="0" y="0" width="680" height="506" rx="26" ry="26" fill="none" stroke="#d7d0c5" stroke-width="2" stroke-dasharray="8 6"></rect>`;
  }
  return "";
}

function spaceTranslate(space: SpaceSpec): { x: number; y: number } {
  if (space.panelKind === "lane") {
    return { x: 20, y: 40 };
  }
  if (space.panelKind === "bridge") {
    return { x: 680, y: 56 };
  }
  return { x: 812, y: 18 };
}

function renderNode(node: NodeSpec): string {
  const tone = PALETTE[node.tone];
  const titleY = node.subtitle ? node.y + 30 : node.y + node.height / 2 + 6;
  const subtitleY = node.y + 52;
  const rect = `<rect x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}" rx="16" ry="16" fill="${tone.fill}" stroke="${tone.stroke}" stroke-width="1.5"></rect>`;
  const title = `<text x="${node.x + node.width / 2}" y="${titleY}" text-anchor="middle" font-family="'SF Pro Display','SF Pro Text',-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',sans-serif" font-size="17.6" font-weight="600" letter-spacing="-0.02em" fill="${tone.ink}">${escapeXml(node.title)}</text>`;
  if (!node.subtitle) {
    return `${rect}\n${title}`;
  }
  const subtitle = `<text x="${node.x + node.width / 2}" y="${subtitleY}" text-anchor="middle" font-family="'SF Pro Text','SF Pro Display',-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',sans-serif" font-size="15.5" font-weight="400" fill="${tone.subtle}">${escapeXml(node.subtitle)}</text>`;
  return `${rect}\n${title}\n${subtitle}`;
}

function resolveSpaceLocalPoint(diagram: DiagramSpec, edge: EdgeSpec, endpoint: EdgeSpec["from"]): { x: number; y: number } {
  if ("point" in endpoint) {
    if (endpoint.spaceId !== edge.spaceId) {
      throw new Error("SVG export only supports point endpoints in the same space.");
    }
    return endpoint.point;
  }

  const node = getNode(diagram, endpoint.nodeId);
  if (node.spaceId !== edge.spaceId) {
    throw new Error(`SVG export only supports intra-space edges. Edge ${edge.id} crosses spaces.`);
  }
  return getAnchor(node, endpoint.side);
}

function renderArrowTip(tip: { x: number; y: number }, neighbor: { x: number; y: number }): string {
  const dx = tip.x - neighbor.x;
  const dy = tip.y - neighbor.y;
  const length = Math.hypot(dx, dy);
  if (length === 0) {
    return "";
  }

  const ux = dx / length;
  const uy = dy / length;
  const backX = tip.x - ux * 8;
  const backY = tip.y - uy * 8;
  const perpX = -uy * 3.5;
  const perpY = ux * 3.5;

  const leftX = backX + perpX;
  const leftY = backY + perpY;
  const rightX = backX - perpX;
  const rightY = backY - perpY;

  return `<path d="M ${leftX} ${leftY} L ${tip.x} ${tip.y} L ${rightX} ${rightY}" fill="none" stroke="#9ea9b1" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"></path>`;
}

function renderEdge(diagram: DiagramSpec, edge: EdgeSpec): string {
  const start = resolveSpaceLocalPoint(diagram, edge, edge.from);
  const end = resolveSpaceLocalPoint(diagram, edge, edge.to);
  const points = routePoints(start, end, edge.route);
  const path = routePath(start, end, edge.route);
  const dash = edge.style === "soft" ? ` stroke-dasharray="5 6" stroke-width="1.4"` : ` stroke-width="1.6"`;
  const mainPath = `<path d="${path}" fill="none" stroke="#9ea9b1"${dash} stroke-linecap="round" stroke-linejoin="round"></path>`;
  const parts = [mainPath];

  if ((edge.arrows === "end" || edge.arrows === "both") && points.length >= 2) {
    parts.push(renderArrowTip(points[points.length - 1], points[points.length - 2]));
  }
  if (edge.arrows === "both" && points.length >= 2) {
    parts.push(renderArrowTip(points[0], points[1]));
  }

  return parts.join("\n");
}

function renderLabel(edge: EdgeSpec): string {
  if (!edge.label) {
    return "";
  }
  const anchor = edge.label.anchor ?? "start";
  if (edge.label.lines.length === 1) {
    return `<text x="${edge.label.x}" y="${edge.label.y}" text-anchor="${anchor}" font-family="'SF Pro Text','SF Pro Display',-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',sans-serif" font-size="15.5" font-weight="400" fill="#697780">${escapeXml(edge.label.lines[0])}</text>`;
  }
  const spans = edge.label.lines
    .map((line, index) => {
      const dy = index === 0 ? 0 : 16;
      return `<tspan x="${edge.label.x}" dy="${dy}">${escapeXml(line)}</tspan>`;
    })
    .join("");
  return `<text x="${edge.label.x}" y="${edge.label.y}" text-anchor="${anchor}" font-family="'SF Pro Text','SF Pro Display',-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',sans-serif" font-size="15.5" font-weight="400" fill="#697780">${spans}</text>`;
}

function renderSpace(diagram: DiagramSpec, space: SpaceSpec): string {
  const translate = spaceTranslate(space);
  const nodes = diagram.nodes.filter((node) => node.spaceId === space.id);
  const edges = diagram.edges.filter((edge) => edge.spaceId === space.id);
  const parts: string[] = [`<g transform="translate(${translate.x} ${translate.y})">`];

  if (space.svgFrame) {
    parts.push(`<rect x="${space.svgFrame.x}" y="${space.svgFrame.y}" width="${space.svgFrame.width}" height="${space.svgFrame.height}" rx="${space.svgFrame.rx ?? 26}" ry="${space.svgFrame.ry ?? 26}" fill="none" stroke="#d7d0c5" stroke-width="2" stroke-dasharray="8 6"></rect>`);
  }
  if (space.title && space.titlePlacement === "inside-frame") {
    parts.push(`<text x="20" y="28" font-family="'SF Pro Text','SF Pro Display',-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',sans-serif" font-size="17" font-style="italic" font-weight="400" fill="#697780">${escapeXml(space.title)}</text>`);
  }
  parts.push(...edges.map((edge) => renderEdge(diagram, edge)));
  parts.push(...nodes.map((node) => renderNode(node)));
  parts.push(...edges.map((edge) => renderLabel(edge)).filter(Boolean));
  parts.push("</g>");
  return parts.join("\n");
}

export function renderDiagramSvg(diagram: DiagramSpec): string {
  const svgWidth = 1136;
  const svgHeight = 506;
  const primary = diagram.spaces.find((space) => space.id === "primary");
  if (!primary) {
    throw new Error("SVG export expects a primary space.");
  }

  const outsideTitles = diagram.spaces
    .filter((space) => space.title && space.titlePlacement === "outside")
    .map((space) => {
      const box = panelOuterBox(space);
      return `<text x="${box.x + 28}" y="29" font-family="'SF Pro Text','SF Pro Display',-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',sans-serif" font-size="17" font-style="italic" font-weight="400" fill="#697780">${escapeXml(space.title!)}</text>`;
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fffdf9"/>
      <stop offset="54%" stop-color="#faf8f3"/>
      <stop offset="100%" stop-color="#f4efe6"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="${svgWidth}" height="${svgHeight}" fill="url(#bg)"/>
  ${renderPanelShell(primary)}
  ${outsideTitles}
  ${renderSpace(diagram, primary)}
  ${renderSpace(diagram, diagram.spaces.find((space) => space.id === "bridge")!)}
  ${renderSpace(diagram, diagram.spaces.find((space) => space.id === "secondary")!)}
</svg>
`;
}
