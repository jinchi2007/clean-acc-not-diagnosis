#!/bin/bash
# Build a self-contained submission bundle for ScholarOne-style upload:
# assembled LaTeX source, bibliography, figure PDFs, compiled PDF, cover letter.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"
PAPER="$WS/paper"
DIST="$WS/dist"
OUT="$DIST/mst_submission"
rm -rf "$OUT"
mkdir -p "$OUT/figures"

# 1. assembled source (same order as paper/build.sh), with figure paths made bundle-local
cat "$PAPER/_p1.tex" "$PAPER/_p2.tex" "$PAPER/_p3.tex" | sed 's#\.\./figures/#figures/#g' > "$OUT/main.tex"
cp "$PAPER/refs.bib" "$OUT/refs.bib"
cp "$PAPER/main.pdf" "$OUT/main.pdf"
cp "$PAPER/cover_letter.md" "$OUT/cover_letter.md"

# 2. only the figures the manuscript actually includes
grep -o 'figures/[A-Za-z0-9_]*\.pdf' "$OUT/main.tex" | sort -u | while read -r f; do
  cp "$WS/$f" "$OUT/figures/$(basename "$f")"
done

# 3. a short README that says how to rebuild
cat > "$OUT/README.txt" <<'EOF'
Manuscript: "Clean Accuracy Is Not a Diagnosis: A Noise-Stress Re-evaluation of
Lightweight Bearing Fault Diagnosis"

main.tex        assembled LaTeX source (XeLaTeX; needs fontspec, booktabs, natbib)
refs.bib        bibliography (21 entries)
figures/*.pdf   the figure files included by main.tex
main.pdf        compiled manuscript
cover_letter.md cover letter for the Editor

Build:  xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
(tested with tectonic 0.15.0 and TeX Live/XeLaTeX)

Reproduction: code, split files and per-seed results are in the accompanying
repository; every table in the manuscript can be re-checked with
    PYTHONPATH=. python experiments/check_tables.py
EOF

cd "$DIST"
rm -f mst_submission.zip
zip -qr mst_submission.zip mst_submission -x '.*'
echo "--- bundle contents ---"
unzip -l mst_submission.zip | tail -20
