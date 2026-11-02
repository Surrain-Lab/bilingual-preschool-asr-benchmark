#!/usr/bin/env bash
# Creates the analysis venv (stage 2 + 3) and checks the repo reproduces the paper.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
echo "== creating .venv with $PY =="
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo "== installed =="
python -c "import numpy,scipy,matplotlib,openpyxl;print('  ok')"
echo
echo "== verifying against the published numbers =="
python verify.py
echo
echo "Setup complete. Activate with:  . .venv/bin/activate"
echo "Then run:                       ./run_analysis.sh"
