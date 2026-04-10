export type AnchorSide = "north" | "south" | "east" | "west";
export type ArrowMode = "none" | "end" | "both";
export type EdgeStyle = "solid" | "soft";
export type RouteKind = "straight" | "hv" | "vh" | "hvh" | "vhv";
export type TitlePlacement = "outside" | "inside-frame" | "none";
export type PanelKind = "lane" | "bridge" | "panel";
export type TextAnchor = "start" | "middle" | "end";
export type Tone =
  | "lavender"
  | "mint"
  | "teal"
  | "cream"
  | "amber"
  | "peach";

export type Point = {
  x: number;
  y: number;
};

export type HeaderSpec = {
  eyebrow?: string;
  title?: string;
  description?: string;
};

export type ColumnSpec = {
  id: string;
  width: number;
};

export type SvgFrameSpec = {
  x: number;
  y: number;
  width: number;
  height: number;
  rx?: number;
  ry?: number;
};

export type SpaceSpec = {
  id: string;
  panelKind: PanelKind;
  panelClass: string;
  columnId: string;
  width: number;
  height: number;
  diagramX: number;
  diagramY: number;
  title?: string;
  titlePlacement?: TitlePlacement;
  outerFrame?: boolean;
  svgFrame?: SvgFrameSpec;
  commentLines?: string[];
};

export type NodeSpec = {
  id: string;
  spaceId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  tone: Tone;
  title: string;
  subtitle?: string;
};

export type EdgeRouteSpec =
  | { kind: "straight" }
  | { kind: "hv" }
  | { kind: "vh" }
  | { kind: "hvh"; x: number }
  | { kind: "vhv"; y: number };

export type EdgeLabelSpec = {
  lines: string[];
  x: number;
  y: number;
  anchor?: TextAnchor;
};

export type AnchorRef = {
  nodeId: string;
  side: AnchorSide;
};

export type PointRef = {
  spaceId: string;
  point: Point;
  label: string;
};

export type EndpointRef = AnchorRef | PointRef;

export type EdgeSpec = {
  id: string;
  spaceId: string;
  from: EndpointRef;
  to: EndpointRef;
  route: EdgeRouteSpec;
  style?: EdgeStyle;
  arrows?: ArrowMode;
  comment?: string;
  label?: EdgeLabelSpec;
};

export type DiagramSpec = {
  id: string;
  outputFile: string;
  galleryBaseName?: string;
  pageTitle: string;
  ariaLabel: string;
  header?: HeaderSpec;
  columns: ColumnSpec[];
  spaces: SpaceSpec[];
  nodes: NodeSpec[];
  edges: EdgeSpec[];
};
