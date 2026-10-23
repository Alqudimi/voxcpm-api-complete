#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# VoxCPM API + Audio Mixer
# Google Colab Setup & Startup
#
# - No Google Drive integration
# - Temporary local workspace
# - GPU/CUDA diagnostics
# - FFmpeg + libopus verification
# - FastAPI startup
# - Temporary public URL through Cloudflare Tunnel
# ============================================================

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1

REPO="${REPO_URL:-https://github.com/Alqudimi/voxcpm-api-complete.git}"
PROJECT_NAME="voxcpm-api-complete"

# ============================================================
# 1. Workspace
# ============================================================

BASE_DIR="${WORKSPACE_DIR:-/content/workspace}"
PROJECT="${PROJECT_DIR:-$BASE_DIR/$PROJECT_NAME}"
CACHE_DIR="${CACHE_DIR:-/content/cache}"

mkdir -p "$BASE_DIR" "$CACHE_DIR" "$(dirname "$PROJECT")"

export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME/whisper"

printf '\n%s\n' '============================================================'
printf '%s\n' 'VoxCPM API — Google Colab'
printf '%s\n\n' '============================================================'
printf 'Workspace: %s\nProject:   %s\nCache:     %s\nRepository: %s\n\n' \
    "$BASE_DIR" "$PROJECT" "$CACHE_DIR" "$REPO"

# ============================================================
# 2. Check Colab environment
# ============================================================

if [[ ! -d /content ]]; then
    echo "ERROR: This script is intended for Google Colab." >&2
    exit 1
fi

PYTHON_SYSTEM="${PYTHON_SYSTEM:-python3}"

if ! command -v "$PYTHON_SYSTEM" >/dev/null 2>&1; then
    echo "ERROR: Python is not installed." >&2
    exit 1
fi

"$PYTHON_SYSTEM" - <<'PY'
import sys

print("Python:", sys.version)
if sys.version_info < (3, 11):
    raise SystemExit("ERROR: Python 3.11 or newer is required.")
PY

if ! command -v git >/dev/null 2>&1; then
    apt-get update
    apt-get install -y git
fi

# ============================================================
# 3. Clone or reuse the repository
# ============================================================

if [[ -d "$PROJECT/.git" ]]; then
    echo "Existing Git checkout found: $PROJECT"

    cd "$PROJECT"

    if ! git pull --ff-only; then
        echo "WARNING: Git update failed; using the existing checkout." >&2
    fi
else
    if [[ -e "$PROJECT" ]]; then
        if [[ -n "$(find "$PROJECT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "ERROR: The project directory exists but is not an empty Git checkout." >&2
            echo "Set PROJECT_DIR to another directory or move the existing directory." >&2
            exit 1
        fi

        rmdir "$PROJECT"
    fi

    echo "Cloning project..."
    git clone "$REPO" "$PROJECT"
fi

cd "$PROJECT"

if [[ ! -f requirements.txt ]]; then
    echo "ERROR: requirements.txt is missing." >&2
    exit 1
fi

if [[ ! -f app/audio_mixer/api.py ]] || \
   ! grep -q 'include_audio_mixer' app/main.py; then
    echo "ERROR: Audio Mixer source or FastAPI registration is missing." >&2
    exit 1
fi

# ============================================================
# 4. Install system dependencies
# ============================================================

echo ""
echo "Installing system dependencies..."

apt-get update
apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl \
    git

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: FFmpeg installation failed." >&2
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ERROR: ffprobe is unavailable." >&2
    exit 1
fi

FFMPEG_ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null)"

if ! grep -Eq '[[:space:]]libopus([[:space:]]|$)' <<<"$FFMPEG_ENCODERS"; then
    echo "ERROR: FFmpeg does not provide the libopus encoder." >&2
    exit 1
fi

echo "FFmpeg: $(ffmpeg -version | head -n 1)"
echo "ffprobe: $(ffprobe -version | head -n 1)"
echo "libopus: OK"

# ============================================================
# 5. Python environment
#
# Colab already provides a managed Python environment.
# Reuse it instead of creating a venv that may duplicate
# large CUDA/PyTorch packages.
# ============================================================

PYTHON_BIN="$(command -v "$PYTHON_SYSTEM")"

echo ""
echo "Python executable: $PYTHON_BIN"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# ============================================================
# 6. Check the existing PyTorch/CUDA installation
#
# Do not replace Colab's PyTorch automatically. Installing
# another CUDA wheel can break packages preinstalled by Colab.
# ============================================================

echo ""
echo "Checking PyTorch and GPU..."

if ! "$PYTHON_BIN" -c 'import torch' 2>/dev/null; then
    echo "PyTorch is missing; installing a compatible CUDA wheel."

    "$PYTHON_BIN" -m pip install torch torchvision torchaudio
fi

"$PYTHON_BIN" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA runtime:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))

    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("CUDA computation: OK")
    except Exception as exc:
        raise SystemExit(f"ERROR: CUDA computation failed: {exc}")
else:
    print("WARNING: No CUDA GPU detected.")
    print("In Colab, select Runtime > Change runtime type > GPU.")
PY

# ============================================================
# 7. Install project requirements
#
# Preserve the requirements.txt file.
# Avoid reinstalling pinned torch packages from it.
# ============================================================

FILTERED_REQUIREMENTS="$(mktemp /content/requirements-colab.XXXXXX.txt)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT

sed -E \
    '/^[[:space:]]*torch([[:space:]]|[<=>!~]|$)/d' \
    requirements.txt > "$FILTERED_REQUIREMENTS"

echo ""
echo "Installing project requirements..."

if ! "$PYTHON_BIN" -m pip install -r "$FILTERED_REQUIREMENTS"; then
    echo "ERROR: Project requirements installation failed." >&2
    exit 1
fi

rm -f "$FILTERED_REQUIREMENTS"
trap - EXIT

# ============================================================
# 8. Install Audio Mixer dependencies
# ============================================================

echo ""
echo "Installing Audio Mixer runtime dependencies..."

"$PYTHON_BIN" -m pip install \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.34' \
    'python-multipart>=0.0.20' \
    'pydantic>=2.10' \
    'pydantic-settings>=2.7' \
    'numpy>=1.26' \
    scipy \
    soundfile \
    librosa \
    'psycopg[binary]>=3.2'

# ============================================================
# 9. Create .env without overwriting an existing configuration
# ============================================================

if [[ ! -f "$PROJECT/.env" ]]; then
    echo "Creating .env..."

    cat > "$PROJECT/.env" <<'ENV_EOF'
APP_NAME=VoxCPM2 Professional Voice API
APP_VERSION=1.0.0
API_KEY=CHANGE_ME_TO_A_LONG_RANDOM_SECRET

VOXCPM_MODEL_ID=openbmb/VoxCPM2
WHISPER_MODEL_NAME=large-v3
ECAPA_SOURCE=speechbrain/spkrec-ecapa-voxceleb
ECAPA_SAVEDIR=storage/models/ecapa_voxceleb

DEVICE=cuda
REFERENCE_SAMPLE_RATE=16000
OUTPUT_SAMPLE_RATE=48000
AUDIO_SAMPLE_RATE=24000

MAX_FILE_SIZE_MB=100
MAX_MERGE_FILES=20

DEFAULT_CFG=2.0
DEFAULT_TIMESTEPS=30
DEFAULT_CANDIDATES=3
DEFAULT_RETRY_BADCASE=true
DEFAULT_RETRY_MAX_TIMES=3
DEFAULT_RETRY_RATIO_THRESHOLD=6.0

OUTPUT_BITRATE=96k
OUTPUT_CODEC=libopus

HOST=0.0.0.0
PORT=8000
VM_PRECISION=fp32
PLATFORM_BASE_URL=https://revenues-complement-metal-cardiff.trycloudflare.com
ENV_EOF
fi

# Replace only the known placeholder, not a real existing key.
if grep -Eq '^API_KEY=(CHANGE_ME_TO_A_LONG_RANDOM_SECRET|change-me)?$' "$PROJECT/.env"; then
    GENERATED_API_KEY="$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_urlsafe(32))')"

    API_KEY="$GENERATED_API_KEY" PROJECT_ENV="$PROJECT/.env" \
    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["PROJECT_ENV"])
key = os.environ["API_KEY"]
lines = path.read_text().splitlines()

found = False
for i, line in enumerate(lines):
    if line.startswith("API_KEY="):
        lines[i] = f"API_KEY={key}"
        found = True

if not found:
    lines.append(f"API_KEY={key}")

path.write_text("\n".join(lines) + "\n")
PY

    unset GENERATED_API_KEY
    echo "Generated a random API key in .env."
fi

chmod 600 "$PROJECT/.env"

# ============================================================
# 10. Validate Python dependencies
# ============================================================

echo ""
echo "Checking installed dependencies..."

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

import torch
import fastapi
import uvicorn

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)

for module in (
    "multipart",
    "psycopg",
    "pydantic_settings",
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Missing dependency: {module}")

    print(f"{module}: OK")
PY

# pip check is informative here; Colab may have unrelated
# preinstalled packages with constraints outside this project.
if ! "$PYTHON_BIN" -m pip check; then
    echo "WARNING: pip check reported dependency conflicts." >&2
    echo "Review the conflicts before relying on the environment." >&2
fi

# ============================================================
# 11. Verify FastAPI registration
# ============================================================

echo ""
echo "Checking FastAPI application..."

PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" - <<'PY'
from app.main import app

print("FastAPI import: OK")
print("Registered routes:", len(app.routes))

for route in app.routes:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)

    if path:
        print(f"{','.join(sorted(methods or [])):20} {path}")
PY

# ============================================================
# 12. Install Cloudflare Tunnel (cloudflared)
#
# Uses Cloudflare Quick Tunnel. No Cloudflare account or
# domain is required for a temporary trycloudflare.com URL.
# ============================================================

echo ""
echo "Installing Cloudflare Tunnel..."

ARCH="$(uname -m)"

case "$ARCH" in
    x86_64)
        CLOUDFLARED_ARCH="amd64"
        ;;
    aarch64|arm64)
        CLOUDFLARED_ARCH="arm64"
        ;;
    *)
        echo "ERROR: Unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

CLOUDFLARED_TMP="$(mktemp /content/cloudflared.XXXXXX)"
trap 'rm -f "$CLOUDFLARED_TMP"' EXIT

curl -fsSL \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CLOUDFLARED_ARCH}" \
    -o "$CLOUDFLARED_TMP"

install -m 0755 "$CLOUDFLARED_TMP" /usr/local/bin/cloudflared

rm -f "$CLOUDFLARED_TMP"
trap - EXIT

cloudflared --version

# ============================================================
# 13. Start API and Cloudflare Tunnel
# ============================================================

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

export PORT HOST
export HF_HOME HUGGINGFACE_HUB_CACHE XDG_CACHE_HOME

cd "$PROJECT"

echo ""
echo "============================================================"
echo "Starting VoxCPM API"
echo "============================================================"
echo "Project: $PROJECT"
echo "Host:    $HOST"
echo "Port:    $PORT"

# Launch API in the background so cloudflared can run alongside it.
"$PYTHON_BIN" -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    > /content/voxcpm-api.log 2>&1 &

API_PID=$!

cleanup() {
    echo ""
    echo "Stopping VoxCPM API and Cloudflare Tunnel..."

    if [[ -n "${TUNNEL_PID:-}" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi

    if [[ -n "${API_PID:-}" ]]; then
        kill "$API_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# Wait for FastAPI to start before opening the tunnel.
echo "Waiting for FastAPI..."

API_READY=0

for _ in $(seq 1 120); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "ERROR: FastAPI stopped during startup." >&2
        cat /content/voxcpm-api.log >&2
        exit 1
    fi

    if curl -fsS "http://127.0.0.1:${PORT}/openapi.json" \
        -o /dev/null 2>/dev/null; then
        API_READY=1
        break
    fi

    sleep 2
done

if [[ "$API_READY" != "1" ]]; then
    echo "ERROR: FastAPI did not become ready within 240 seconds." >&2
    cat /content/voxcpm-api.log >&2
    exit 1
fi

echo "FastAPI is responding locally."

# Quick Tunnel outputs its temporary public URL to stderr.
TUNNEL_LOG="/content/cloudflared.log"
: > "$TUNNEL_LOG"

cloudflared tunnel \
    --url "http://127.0.0.1:${PORT}" \
    --no-autoupdate \
    > /content/cloudflared.stdout.log \
    2> "$TUNNEL_LOG" &

TUNNEL_PID=$!

echo ""
echo "Waiting for Cloudflare URL..."

PUBLIC_URL=""

for _ in $(seq 1 60); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "ERROR: Cloudflare Tunnel stopped unexpectedly." >&2
        cat "$TUNNEL_LOG" >&2
        exit 1
    fi

    PUBLIC_URL="$(
        grep -Eo 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" \
            | head -n 1 || true
    )"

    if [[ -n "$PUBLIC_URL" ]]; then
        break
    fi

    sleep 2
done

if [[ -z "$PUBLIC_URL" ]]; then
    echo "ERROR: Could not retrieve a public Cloudflare URL." >&2
    cat "$TUNNEL_LOG" >&2
    exit 1
fi

# Save the public URL to indicator file so FastAPI startup or watcher detects it immediately
echo "$PUBLIC_URL" > /content/tunnel_url.txt
echo "$PUBLIC_URL" > /tmp/tunnel_url.txt

# Proactively notify FastAPI register endpoint with the public Cloudflare URL
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/resources/register-server?server_url=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$PUBLIC_URL'))")" \
    -H "accept: application/json" >/dev/null 2>&1 || true

echo ""
echo "============================================================"
echo "VOXCPM API IS RUNNING"
echo "============================================================"
echo "Public API: $PUBLIC_URL"
echo "Swagger UI: $PUBLIC_URL/docs"
echo "OpenAPI:    $PUBLIC_URL/openapi.json"
echo ""
echo "Local API:  http://127.0.0.1:$PORT"
echo "API logs:   /content/voxcpm-api.log"
echo "Tunnel log: /content/cloudflared.log"
echo ""
echo "IMPORTANT:"
echo "1. Keep this Colab cell running."
echo "2. The URL is temporary and changes when the tunnel restarts."
echo "3. Colab runtime files disappear when the runtime is deleted."
echo "4. Protect your API key; do not publish it."
echo "5. Use the API key required by your project's authentication."
echo "============================================================"

# Keep this cell active and forward logs until interrupted.
tail -n 0 -F /content/voxcpm-api.log "$TUNNEL_LOG" &
TAIL_PID=$!

wait "$API_PID"