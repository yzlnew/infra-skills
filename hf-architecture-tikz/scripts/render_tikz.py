#!/usr/bin/env python3
"""Render a TikZ .tex file from arch.json + a Jinja2 template.

Usage:
    render_tikz.py <arch.json> [--template PATH] [--output PATH]
                   [--no-mtp] [--title STR] [--subtitle STR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_TEMPLATE = SKILL_DIR / "templates" / "anthropic.tex.j2"


def fmt_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def escape_tex(s: str) -> str:
    """Escape LaTeX special chars in a literal string."""
    if not isinstance(s, str):
        return str(s)
    table = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(table.get(c, c) for c in s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arch_json", help="arch.json from extract_arch.py")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--output", "-o", default="-", help="Output .tex path (- for stdout)")
    ap.add_argument("--no-mtp", action="store_true", help="Suppress the MTP head branch")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args()

    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        print(
            "jinja2 is required. Install with: uv pip install jinja2  "
            "or run via: uv run --with jinja2 ...",
            file=sys.stderr,
        )
        return 2

    spec = json.loads(Path(args.arch_json).read_text())

    template_path = Path(args.template)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
        undefined=StrictUndefined,
    )
    env.filters["escape_tex"] = escape_tex
    env.globals["fmt_count"] = fmt_count

    template = env.get_template(template_path.name)

    # Default title/subtitle
    model_id = spec["model"].get("id") or spec["model"].get("model_type", "model")
    family = spec["model"]["family"]
    summary = spec["model"]["summary"]
    flags = ", ".join(spec["model"].get("flags", []))
    title = args.title or model_id
    subtitle = args.subtitle or (
        f"{family.upper()} family · {summary['total_label']} total · "
        f"{summary['active_label']} active per token"
        + (f" · {flags}" if flags else "")
    )

    out = template.render(
        spec=spec,
        title=escape_tex(title),
        subtitle=escape_tex(subtitle),
        opts={
            "show_mtp": not args.no_mtp,
            "theme": "anthropic",
        },
    )

    if args.output == "-":
        sys.stdout.write(out)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Wrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
