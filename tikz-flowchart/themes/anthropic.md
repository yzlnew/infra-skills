# Anthropic Theme

## Style Guide

This theme should follow the color and typography direction from `anthropic-theme-flowchart/SKILL.md`, but the node shapes and TikZ styling should be adapted for print-friendly LaTeX diagrams:

- **Background**: warm ivory (`#FAF8F3`) with a softer panel tone (`#FFFDFC`) when needed.
- **Grouping**: transparent dashed rounded containers with italic muted headers.
- **Nodes**: mostly rounded rectangles with smaller radii, thin low-contrast borders, and no drop shadow.
- **Typography**: use sans-serif throughout; prefer an SF Pro-like stack when compiling with XeLaTeX/LuaLaTeX, otherwise use LaTeX `\sffamily`.
- **Text hierarchy**: primary line uses a near-black tint of the node hue; secondary line uses a darker hue-tinted accent.
- **Connectors**: thin, quiet strokes with small arrowheads and mostly orthogonal routing. Plan the node layout first, then route edges so the segment touching a node is perpendicular to that node edge.

### Anthropic Color Tokens

- `bgWarm`: `#FAF8F3`
- `bgPanel`: `#FFFDFC`
- `laneStroke`: `#D7D0C5`
- `connectorStroke`: `#9EA9B1`
- `inkText`: `#31424B`
- `mutedText`: `#66757E`
- `lavFill` / `lavStroke` / `lavText` / `lavSubtext`
- `mintFill` / `mintStroke` / `mintText` / `mintSubtext`
- `tealFill` / `tealStroke` / `tealText` / `tealSubtext`
- `creamFill` / `creamStroke` / `creamText` / `creamSubtext`
- `amberFill` / `amberStroke` / `amberText` / `amberSubtext`
- `peachFill` / `peachStroke` / `peachText` / `peachSubtext`

### Anthropic Standard Nodes

- `anthropicEventNode`: slightly tighter rounded rectangle for session/event entry points.
- `anthropicCardNode`: the default rounded card used for most actions, prompts, feedback, and outputs.
- `anthropicDecisionNode`: a soft diamond for explicit branching or review gates when the flow really requires a decision shape.
- `anthropicGroup`: transparent dashed container for lanes, ownership scopes, or background systems.

### Layout and Routing Rules

- Plan the lane structure and node grid before adding connectors. The Anthropic theme looks best when rows and columns are visibly intentional.
- Align related cards so most connectors can use straight horizontal or vertical segments.
- Connect through explicit anchors such as `.east`, `.west`, `.north`, and `.south` so the edge meets the card border at a right angle.
- If a connector needs multiple bends to avoid collisions, reconsider the layout before adding more routing complexity.

### Anthropic Template

Use this template when the user explicitly asks for Anthropic styling:

```latex
\documentclass[tikz,border=10pt]{standalone}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, fit, backgrounds, calc, shapes.geometric}

\usepackage{fontspec}
\setsansfont{SF Pro Text}[
  BoldFont = SF Pro Text Semibold,
]

% --- Anthropic Color Definitions ---
\definecolor{bgWarm}{HTML}{FAF8F3}
\definecolor{bgPanel}{HTML}{FFFDFC}
\definecolor{laneStroke}{HTML}{D7D0C5}
\definecolor{connectorStroke}{HTML}{9EA9B1}
\definecolor{inkText}{HTML}{31424B}
\definecolor{mutedText}{HTML}{66757E}

\definecolor{lavFill}{HTML}{E7E3FF}
\definecolor{lavStroke}{HTML}{8D86D8}
\definecolor{lavText}{HTML}{373457}
\definecolor{lavSubtext}{HTML}{5852A7}

\definecolor{mintFill}{HTML}{DDF4EC}
\definecolor{mintStroke}{HTML}{78B5A7}
\definecolor{mintText}{HTML}{294B45}
\definecolor{mintSubtext}{HTML}{3F8F7F}

\definecolor{tealFill}{HTML}{D9F0F2}
\definecolor{tealStroke}{HTML}{6CA9B0}
\definecolor{tealText}{HTML}{29464A}
\definecolor{tealSubtext}{HTML}{457E85}

\definecolor{creamFill}{HTML}{F4EEE3}
\definecolor{creamStroke}{HTML}{C1B39E}
\definecolor{creamText}{HTML}{4E4437}
\definecolor{creamSubtext}{HTML}{8E7A61}

\definecolor{amberFill}{HTML}{F8E7C9}
\definecolor{amberStroke}{HTML}{D6A86B}
\definecolor{amberText}{HTML}{57401C}
\definecolor{amberSubtext}{HTML}{A27431}

\definecolor{peachFill}{HTML}{F8E0D9}
\definecolor{peachStroke}{HTML}{CF8E81}
\definecolor{peachText}{HTML}{55342E}
\definecolor{peachSubtext}{HTML}{A66559}

\pagecolor{bgWarm}

\newcommand{\anthropicLabel}[4]{%
  \def\tmp{#4}%
  \ifx\tmp\empty
    {\sffamily\bfseries\small\textcolor{#1}{#3}}%
  \else
    \begin{tabular}{@{}c@{}}
      {\sffamily\bfseries\small\textcolor{#1}{#3}}\\[-1pt]
      {\sffamily\fontsize{8}{9}\selectfont\textcolor{#2}{#4}}
    \end{tabular}%
  \fi
}

\begin{document}

\begin{tikzpicture}[
    x=1cm,
    y=1cm,
    font=\sffamily\footnotesize,
    anthropicEdge/.style={
        -{Straight Barb[angle=60:2pt 3]},
        draw=connectorStroke,
        line width=0.85pt
    },
    anthropicDashedEdge/.style={
        anthropicEdge,
        dashed
    },
    anthropicCardNode/.style={
        rectangle,
        rounded corners=5pt,
        draw,
        line width=0.9pt,
        minimum width=3.2cm,
        minimum height=1.25cm,
        inner sep=6pt,
        align=center,
        fill=bgPanel
    },
    anthropicEventNode/.style={
        anthropicCardNode,
        minimum width=3.3cm,
        minimum height=1.15cm
    },
    anthropicDecisionNode/.style={
        diamond,
        aspect=1.7,
        draw,
        line width=0.9pt,
        inner xsep=0.8em,
        inner ysep=0.55em,
        align=center,
        fill=creamFill
    },
    anthropicGroup/.style={
        rectangle,
        rounded corners=10pt,
        draw=laneStroke,
        dashed,
        line width=0.9pt,
        inner sep=12pt,
        fill opacity=0
    },
    anthropicGroupTitle/.style={
        font=\sffamily\itshape\small,
        text=mutedText,
        fill=bgWarm,
        inner sep=1.5pt
    },
    edgeLabel/.style={
        font=\sffamily\scriptsize,
        text=mutedText,
        fill=none,
        inner sep=1.5pt,
        align=center
    }
]

    % --- Main lane ---
    % Place nodes first so the connector geometry stays orthogonal.
    \node[anthropicEventNode, draw=lavStroke, fill=lavFill] (events) at (0,0)
        {\anthropicLabel{lavText}{lavSubtext}{Session events}{}};
    \node[anthropicCardNode, draw=mintStroke, fill=mintFill] (obs) at (0,-2.2)
        {\anthropicLabel{mintText}{mintSubtext}{Observation}{collector}};
    \node[anthropicCardNode, draw=tealStroke, fill=tealFill] (prompt) at (4.2,-2.2)
        {\anthropicLabel{tealText}{tealSubtext}{System prompt}{injection}};
    \node[anthropicCardNode, draw=lavStroke, fill=lavFill] (loop) at (4.2,-4.2)
        {\anthropicLabel{lavText}{lavSubtext}{Feedback loop}{}};

    % --- Background analyzer lane ---
    \node[anthropicCardNode, draw=creamStroke, fill=creamFill] (reads) at (10.2,-0.05)
        {\anthropicLabel{creamText}{creamSubtext}{Reads observations}{}};
    \node[anthropicCardNode, draw=amberStroke, fill=amberFill] (haiku) at (10.2,-2.1)
        {\anthropicLabel{amberText}{amberSubtext}{Haiku LLM}{analyses patterns}};
    \node[anthropicCardNode, draw=peachStroke, fill=peachFill] (instinct) at (10.2,-4.1)
        {\anthropicLabel{peachText}{peachSubtext}{Instinct files}{.md + YAML frontmatter}};

    % --- Connectors ---
    \draw[anthropicEdge] (events.south) -- (obs.north);
    \draw[anthropicEdge] (obs.east) -- (prompt.west);
    \draw[anthropicEdge] (prompt.south) -- (loop.north);
    \draw[anthropicDashedEdge] (loop.west) -| node[pos=0.35, edgeLabel, below] {records outcomes} (obs.south);

    \draw[anthropicEdge] (reads.south) -- (haiku.north);
    \draw[anthropicEdge] (haiku.south) -- (instinct.north);
    \draw[anthropicEdge] (instinct.west) -- ++(-1.4,0) |- (prompt.east);
    \node[edgeLabel, anchor=south] at ($(instinct.west)+(-1.4,0.15)$) {high-confidence\\instincts injected};

    % --- Groups ---
    \begin{scope}[on background layer]
        \node[
            anthropicGroup,
            fit=(events) (obs) (prompt) (loop),
            label={[anthropicGroupTitle, anchor=south west]north west:{Pi session}}
        ] {};
        \node[
            anthropicGroup,
            fit=(reads) (haiku) (instinct),
            label={[anthropicGroupTitle, anchor=south west]north west:{Background analyzer (cron / launchd)}}
        ] {};
    \end{scope}

\end{tikzpicture}
\end{document}
```
