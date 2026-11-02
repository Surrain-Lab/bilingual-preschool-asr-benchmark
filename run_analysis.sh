#!/usr/bin/env bash
# Stage 2 + 3: derived data -> every table and figure in the paper.
# Assumes the venv is active (see README, "Setup").
PYTHON="${PYTHON:-python3}"
set -euo pipefail
cd "$(dirname "$0")"
echo "== verifying the shipped data reproduces the published numbers =="
"$PYTHON" verify.py
echo
echo "== figures =="
"$PYTHON" stage3_tables_figures/make_paper_figures.py --scale 1.35
"$PYTHON" stage3_tables_figures/make_fig_error_counts.py >/dev/null
echo "   wrote figures_out/fig_error_counts_by_speaker_language.png"
"$PYTHON" stage3_tables_figures/make_fig_corr_by_model.py | tail -1
echo
echo "== supplementary statistics =="
"$PYTHON" stage3_tables_figures/make_supp_stats.py
echo
echo "== tables =="
"$PYTHON" stage3_tables_figures/make_tables.py
"$PYTHON" stage3_tables_figures/make_supp_tables.py
echo
echo "done. figures in figures_out/, tables in tables_out/"
