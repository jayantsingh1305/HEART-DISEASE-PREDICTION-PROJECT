#!/usr/bin/env bash
set -euo pipefail

VENV_PY="./.venv/Scripts/python.exe"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: $VENV_PY not found. Create the virtual environment first." >&2
  exit 1
fi

"$VENV_PY" heart_disease_prediction.py
