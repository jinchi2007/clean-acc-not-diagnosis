#!/usr/bin/env bash
# Build the manuscript. Needs tectonic (https://tectonic-typesetting.github.io/),
# either on PATH or at ../tools/tectonic. Falls back to xelatex if present.
set -euo pipefail
cd "$(dirname "$0")"
cat _p1.tex _p2.tex _p3.tex > main.tex
if command -v tectonic >/dev/null 2>&1; then
  TECTONIC="${TECTONIC:-tectonic}"
elif [ -x ../tools/tectonic ]; then
  TECTONIC=../tools/tectonic
else
  echo "no tectonic found; trying xelatex" >&2
  xelatex -interaction=nonstopmode main.tex >/dev/null
  bibtex main >/dev/null || true
  xelatex -interaction=nonstopmode main.tex >/dev/null
  xelatex -interaction=nonstopmode main.tex >/dev/null
  echo "built main.pdf with xelatex"
  exit 0
fi
HOME="$(cd .. && pwd)" TECTONIC_CACHE_DIR="$PWD/.tcache" "$TECTONIC" -X compile main.tex --keep-logs
echo "built main.pdf"
