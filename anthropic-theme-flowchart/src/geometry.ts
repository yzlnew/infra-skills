import type {
  AnchorSide,
  DiagramSpec,
  EndpointRef,
  EdgeRouteSpec,
  NodeSpec,
  Point,
  SpaceSpec,
} from "./types.ts";

export function getSpace(diagram: DiagramSpec, spaceId: string): SpaceSpec {
  const space = diagram.spaces.find((candidate) => candidate.id === spaceId);
  if (!space) {
    throw new Error(`Unknown space: ${spaceId}`);
  }
  return space;
}

export function getNode(diagram: DiagramSpec, nodeId: string): NodeSpec {
  const node = diagram.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) {
    throw new Error(`Unknown node: ${nodeId}`);
  }
  return node;
}

export function getAnchor(node: NodeSpec, side: AnchorSide): Point {
  switch (side) {
    case "north":
      return { x: node.x + node.width / 2, y: node.y };
    case "south":
      return { x: node.x + node.width / 2, y: node.y + node.height };
    case "west":
      return { x: node.x, y: node.y + node.height / 2 };
    case "east":
      return { x: node.x + node.width, y: node.y + node.height / 2 };
    default:
      throw new Error(`Unsupported side: ${String(side)}`);
  }
}

export function toWorldPoint(space: SpaceSpec, point: Point): Point {
  return {
    x: space.diagramX + point.x,
    y: space.diagramY + point.y,
  };
}

export function convertPoint(
  diagram: DiagramSpec,
  fromSpaceId: string,
  toSpaceId: string,
  point: Point,
): Point {
  const fromSpace = getSpace(diagram, fromSpaceId);
  const toSpace = getSpace(diagram, toSpaceId);
  const world = toWorldPoint(fromSpace, point);
  return {
    x: world.x - toSpace.diagramX,
    y: world.y - toSpace.diagramY,
  };
}

export function resolveEdgeEndpoint(
  diagram: DiagramSpec,
  targetSpaceId: string,
  endpoint: EndpointRef,
): Point {
  if ("point" in endpoint) {
    if (endpoint.spaceId === targetSpaceId) {
      return endpoint.point;
    }
    return convertPoint(diagram, endpoint.spaceId, targetSpaceId, endpoint.point);
  }

  const node = getNode(diagram, endpoint.nodeId);
  const anchor = getAnchor(node, endpoint.side);
  if (node.spaceId === targetSpaceId) {
    return anchor;
  }
  return convertPoint(diagram, node.spaceId, targetSpaceId, anchor);
}

export function routePath(start: Point, end: Point, route: EdgeRouteSpec): string {
  return pointsToPath(routePoints(start, end, route));
}

export function routePoints(start: Point, end: Point, route: EdgeRouteSpec): Point[] {
  switch (route.kind) {
    case "straight":
      return [start, end];
    case "hv":
      return [start, { x: end.x, y: start.y }, end];
    case "vh":
      return [start, { x: start.x, y: end.y }, end];
    case "hvh":
      return [start, { x: route.x, y: start.y }, { x: route.x, y: end.y }, end];
    case "vhv":
      return [start, { x: start.x, y: route.y }, { x: end.x, y: route.y }, end];
    default:
      throw new Error(`Unsupported route: ${String(route)}`);
  }
}

export function pointsToPath(points: Point[]): string {
  if (points.length < 2) {
    throw new Error("Need at least two points to build a path.");
  }
  const [first, ...rest] = points;
  return `M ${first.x} ${first.y} ${rest.map((point) => `L ${point.x} ${point.y}`).join(" ")}`;
}

export function formatPoint(point: Point): string {
  return `(${point.x},${point.y})`;
}
