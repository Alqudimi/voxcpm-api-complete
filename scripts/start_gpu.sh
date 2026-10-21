#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing $VENV_DIR; first run: bash scripts/install_gpu.sh" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python scripts/preflight_gpu.py
exec python -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --workers 1 --loop auto
