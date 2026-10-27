#!/usr/bin/env bash
# Build the manuscript with tectonic (https://tectonic-typesetting.github.io/).
# Lookup order: $TECTONIC, tectonic on PATH, ../tools/tectonic, then xelatex.
# Everything tectonic writes stays inside paper/ (main.tex, .tcache, .texhome).
set -euo pipefail
cd "$(dirname "$0")"
cat _p1.tex _p2.tex _p3.tex > main.tex
BIN="${TECTONIC:-}"
if [ -z "$BIN" ]; then
  if command -v tectonic >/dev/null 2>&1; then BIN=tectonic
  elif [ -x ../tools/tectonic ]; then BIN=../tools/tectonic
  fi
fi
if [ -z "$BIN" ]; then
  echo "no tectonic found; trying xelatex" >&2
  xelatex -interaction=nonstopmode main.tex >/dev/null
  bibtex main >/dev/null || true
  xelatex -interaction=nonstopmode main.tex >/dev/null
  xelatex -interaction=nonstopmode main.tex >/dev/null
  echo "built main.pdf with xelatex"
  exit 0
fi
# tectonic writes its format cache under $HOME/Library on macOS; give it a
# throw-away HOME inside paper/ so a clone is never polluted by build by-products.
mkdir -p .texhome
HOME="$PWD/.texhome" TECTONIC_CACHE_DIR="$PWD/.tcache" "$BIN" -X compile main.tex --keep-logs
echo "built main.pdf"
