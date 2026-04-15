# Default Material-Like Theme

## Style Guide

### Colors
The default template uses a Google Material-like palette:
-   **Green (`greenFill`/`greenStroke`)**: Data, Inputs, Outputs, Tensors.
-   **Orange/Memory (`memFill`/`memStroke`)**: Memory, Weights, Checkpoints.
-   **Blue/Core (`coreFill`/`coreStroke`)**: Operations, Compute Kernels, Processes.
-   **Pink/Optimization (`optFill`/`optStroke`)**: Optimizations, Special Steps.
-   **Yellow/Process (`procFill`/`procStroke`)**: General Processing Steps.

### Standard Nodes
-   `dataNode`: Rectangles for data flow (green).
-   `memNode`: Cylinders for storage/weights (orange).
-   `opNode`: Rectangles for operations (blue).
-   `kernelBox`: Dashed containers for grouping internal kernel logic.
-   `group`: Dashed background containers for logical grouping of phases/stages.

### Layout and Routing Rules

-   Plan the node layout before drawing connectors. Decide the main flow direction first, then place nodes into aligned rows/columns so the routing stays simple.
-   Prefer relative positioning for stable alignment, but adjust spacing early if labels or return paths need more room.
-   Connect with explicit anchors such as `.east`, `.west`, `.north`, and `.south` so the edge meets the node border perpendicularly.
-   Use straight `--` edges when nodes are aligned; use `-|` or `|-` only for deliberate orthogonal bends.

### Template

Use this template as your base for the default theme:

```latex
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usepackage{amssymb}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds, calc, shadows.blur, decorations.pathreplacing}

% --- Color Definitions ---
\definecolor{greenFill}{HTML}{E8F5E9}
\definecolor{greenStroke}{HTML}{43A047}
\definecolor{memFill}{HTML}{FFF3E0}
\definecolor{memStroke}{HTML}{FFB74D}
\definecolor{coreFill}{HTML}{E1F5FE}
\definecolor{coreStroke}{HTML}{0277BD}
\definecolor{optFill}{HTML}{FCE4EC}
\definecolor{optStroke}{HTML}{E91E63}
\definecolor{procFill}{HTML}{FFF9C4}
\definecolor{procStroke}{HTML}{FBC02D}

\begin{document}

\begin{tikzpicture}[
    node distance=1.2cm and 1.8cm, % Vertical and Horizontal spacing
    font=\sffamily\footnotesize,
    >=Stealth,
    % --- Styles ---
    dataNode/.style={
        rectangle, rounded corners=3pt,
        draw=greenStroke, thick,
        fill=greenFill,
        minimum width=2.4cm, minimum height=1.2cm,
        align=center,
        drop shadow
    },
    memNode/.style={
        cylinder, cylinder uses custom fill,
        cylinder body fill=memFill, cylinder end fill=memFill!90!gray,
        shape border rotate=90,
        aspect=0.25,
        draw=memStroke, thick,
        minimum width=1.8cm, minimum height=1.3cm,
        align=center
    },
    opNode/.style={
        rectangle, rounded corners=3pt,
        draw=coreStroke, thick,
        fill=coreFill,
        minimum width=2.6cm, minimum height=1.2cm,
        align=center,
        drop shadow
    },
    kernelBox/.style={
        rectangle, rounded corners=8pt,
        draw=coreStroke, thick, dashed,
        fill=coreFill!20,
        inner sep=12pt,
        align=center
    },
    group/.style={
        draw=gray!30, dashed, rounded corners=8pt, inner sep=12pt, fill=gray!5
    },
    edgeLabel/.style={
        font=\scriptsize,
        text=black!80,
        align=center,
        inner sep=1pt
    }
]

    % --- Nodes ---
    % Plan the layout first, then place aligned nodes:
    % \node[dataNode] (Input) {Input Data};
    % \node[opNode, right=of Input] (Process) {Process};

    % --- Layout Containers (Optional) ---
    % \begin{scope}[on background layer]
    %    \node[group, fit=(Input)(Process)] (MainGroup) {};
    % \end{scope}

    % --- Connections ---
    % Use anchors so edges hit node borders perpendicularly:
    % \draw[->, thick, color=gray!80] (Input.east) -- (Process.west);
    % \draw[->, thick, color=gray!80] (Process.south) |- (Input.east);

\end{tikzpicture}
\end{document}
```
