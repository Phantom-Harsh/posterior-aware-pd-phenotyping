#!/bin/bash
# Creates a complete Overleaf-ready submission folder
# Run from anywhere: bash /path/to/this/script.sh

SRC="/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2/manuscript/miccai"
TEMPLATE="$SRC/MICCAI2026-Latex-Template"
OUT="$SRC/overleaf_submission"

mkdir -p "$OUT"

# 1. Template files (required by llncs documentclass)
cp "$TEMPLATE/llncs.cls" "$OUT/"
cp "$TEMPLATE/splncs04.bst" "$OUT/"

# 2. Main paper
cp "$SRC/miccai2026_paper.tex" "$OUT/"

# 3. All figures
cp "$SRC/MICCAI_Pipeline.png" "$OUT/"
cp "$SRC/fig2_explainability.png" "$OUT/"
cp "$SRC/fig2_explainability.pdf" "$OUT/"
cp "$SRC/fig3_chimera_hierarchy.png" "$OUT/"
cp "$SRC/fig3_chimera_hierarchy.pdf" "$OUT/"

echo "=== Overleaf submission folder created ==="
echo "Location: $OUT"
echo ""
echo "Contents:"
ls -la "$OUT/"
echo ""
echo "=== Upload this entire folder to Overleaf ==="
echo "Files needed:"
echo "  1. miccai2026_paper.tex  (main document)"
echo "  2. llncs.cls             (LNCS template class)"
echo "  3. splncs04.bst          (bibliography style)"
echo "  4. MICCAI_Pipeline.png   (Fig 1 - pipeline)"
echo "  5. fig2_explainability.png (Fig 2 - domain profiles + Granger)"
echo "  6. fig3_chimera_hierarchy.png (Fig 3 - chimera + nesting)"
echo ""
echo "In Overleaf: set miccai2026_paper.tex as main document"
echo "Compiler: pdfLaTeX"
