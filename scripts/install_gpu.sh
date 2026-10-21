#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found: $PYTHON_BIN" >&2
  exit 2
fi
"$PYTHON_BIN" scripts/prepare_env.py
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("VoxCPM2 requires Python 3.10–3.12; set PYTHON_BIN to a supported interpreter.")
PY
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Install/enable the NVIDIA driver and GPU passthrough first." >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "FFmpeg and ffprobe are required. Install them with your OS package manager before continuing." >&2
  exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel packaging ninja
# CUDA 12.8 wheels support current NVIDIA cards; the host still needs a compatible driver.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
python -m pip install -r requirements.txt
python -m pip install -r requirements-performance.txt
if [[ "${INSTALL_NANOVLLM:-1}" == "1" ]]; then
  if ! python -m pip install -r requirements-gpu.txt; then
    echo "NanoVLLM/FlashAttention install failed. Install a compatible CUDA Toolkit/build toolchain or set TTS_ENGINE=pytorch and rerun with INSTALL_NANOVLLM=0." >&2
    exit 2
  fi
else
  echo "NanoVLLM installation was skipped. Set TTS_ENGINE=pytorch in .env before startup."
fi
python scripts/preflight_gpu.py
cat <<'EOF'
GPU environment is installed and preflight passed.
Before production, set API_KEY to a long random value and verify your model cache/storage paths.
EOF
