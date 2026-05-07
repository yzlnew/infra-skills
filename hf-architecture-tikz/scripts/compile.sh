#!/usr/bin/env sh
# Compile a TikZ .tex file produced by render_tikz.py.
# Usage: compile.sh path/to/file.tex [output_dir]
set -eu

if [ $# -lt 1 ]; then
  echo "Usage: $0 path/to/file.tex [output_dir]" >&2
  exit 1
fi

TEX=$1
OUT=${2:-$(dirname "$TEX")}
BASE=$(basename "$TEX" .tex)

mkdir -p "$OUT"

ENGINE=$(command -v xelatex || command -v pdflatex || true)
if [ -z "$ENGINE" ]; then
  echo "Neither xelatex nor pdflatex found in PATH." >&2
  exit 2
fi

# Two passes: TikZ fit/positioning needs to converge.
"$ENGINE" -interaction=nonstopmode -halt-on-error -output-directory "$OUT" "$TEX" >/dev/null
"$ENGINE" -interaction=nonstopmode -halt-on-error -output-directory "$OUT" "$TEX" >/dev/null

PDF="$OUT/$BASE.pdf"
if [ ! -f "$PDF" ]; then
  echo "Compilation produced no PDF at $PDF" >&2
  exit 3
fi

if command -v pdftocairo >/dev/null; then
  pdftocairo -png -r 300 -singlefile "$PDF" "$OUT/$BASE"
  echo "Wrote $OUT/$BASE.png"
else
  echo "pdftocairo not found; PDF only at $PDF" >&2
fi
