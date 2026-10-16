#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# VoxCPM API - RunPod Setup & Startup
# This keeps the original installation flow and adds robust
# workspace selection, FFmpeg, and Audio Mixer dependencies.
# ============================================================

REPO="${REPO_URL:-https://github.com/Alqudimi/voxcpm-api-complete.git}"
PROJECT_NAME="voxcpm-api-complete"

# RunPod normally provides /workspace. If it is absent or not writable,
# use a directory under HOME instead. Override with WORKSPACE_DIR or
# PROJECT_DIR if your environment uses another location.
if [[ -n "${WORKSPACE_DIR:-}" ]]; then
    BASE_DIR="$WORKSPACE_DIR"
elif [[ -n "${RUNPOD_WORKSPACE:-}" ]]; then
    BASE_DIR="$RUNPOD_WORKSPACE"
elif [[ -d /workspace && -w /workspace ]]; then
    BASE_DIR="/workspace"
elif mkdir -p /workspace 2>/dev/null && [[ -w /workspace ]]; then
    BASE_DIR="/workspace"
else
    BASE_DIR="${HOME:-/tmp}/workspace"
fi
mkdir -p "$BASE_DIR"

PROJECT="${PROJECT_DIR:-$BASE_DIR/$PROJECT_NAME}"
if [[ "$PROJECT" != /* ]]; then
    PROJECT="$PWD/$PROJECT"
fi
mkdir -p "$(dirname "$PROJECT" )"

CACHE_DIR="${CACHE_DIR:-$BASE_DIR/.cache}"
if [[ "$CACHE_DIR" != /* ]]; then
    CACHE_DIR="$PWD/$CACHE_DIR"
fi

PYTHON_SYSTEM="${PYTHON_SYSTEM:-python3}"

printf '\n%s\n' '============================================================'
printf '%s\n' 'VoxCPM API - RunPod Setup'
printf '%s\n\n' '============================================================'
printf 'Workspace: %s\nProject:   %s\nRepository: %s\n\n' "$BASE_DIR" "$PROJECT" "$REPO"

# ============================================================
# 1. Check prerequisites and Python version before large installs
# ============================================================

if ! command -v "$PYTHON_SYSTEM" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_SYSTEM" >&2
    exit 1
fi

if ! "$PYTHON_SYSTEM" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: Python 3.11 or newer is required. Found: $("$PYTHON_SYSTEM" --version 2>&1)" >&2
    echo "Set PYTHON_SYSTEM to an installed Python 3.11+ executable and retry." >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required to clone the project." >&2
    exit 1
fi

# ============================================================
# 2. Clone or update project
# ============================================================

if [[ -d "$PROJECT/.git" ]]; then
    echo "Project already exists; trying a fast-forward update..."
    cd "$PROJECT"
    if ! git pull --ff-only; then
        echo "WARNING: git pull failed; continuing with the existing checkout." >&2
    fi
else
    if [[ -e "$PROJECT" ]]; then
        if [[ -n "$(find "$PROJECT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "ERROR: $PROJECT exists but is not a Git checkout and is not empty." >&2
            echo "Set PROJECT_DIR to another location or move that directory first." >&2
            exit 1
        fi
        rmdir "$PROJECT"
    fi
    echo "Cloning repository into the exact project path..."
    git clone "$REPO" "$PROJECT"
fi

cd "$PROJECT"

if [[ ! -f "$PROJECT/requirements.txt" ]]; then
    echo "ERROR: requirements.txt was not found in $PROJECT" >&2
    exit 1
fi

if [[ ! -f "$PROJECT/app/audio_mixer/api.py" ]] || ! grep -q 'include_audio_mixer' "$PROJECT/app/main.py"; then
    echo "ERROR: Audio Mixer source or its FastAPI registration is missing from this checkout." >&2
    echo "Use the Alqudimi/voxcpm-api-complete repository that contains app/audio_mixer." >&2
    exit 1
fi

# ============================================================
# 3. Create and activate virtual environment
# ============================================================

if [[ ! -x "$PROJECT/.venv/bin/python" ]]; then
    echo "Creating Python virtual environment..."
    "$PYTHON_SYSTEM" -m venv "$PROJECT/.venv"
else
    echo "Using existing virtual environment: $PROJECT/.venv"
fi

PYTHON_BIN="$PROJECT/.venv/bin/python"
source "$PROJECT/.venv/bin/activate"
python --version

# ============================================================
# 4. Upgrade packaging tools
# ============================================================

echo "Upgrading pip, setuptools and wheel..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# ============================================================
# 5. Install PyTorch (CUDA 12.8 by default; CPU optional)
# Run with INSTALL_CPU_ONLY=1 to use CPU wheels instead.
# ============================================================

if [[ "${INSTALL_CPU_ONLY:-0}" == "1" ]]; then
    echo "Installing CPU-only PyTorch 2.8.0..."
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    echo "Installing PyTorch 2.8.0 with CUDA 12.8 support..."
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
fi

"$PYTHON_BIN" -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url "$TORCH_INDEX_URL"

# ============================================================
# 6. Install project requirements without editing requirements.txt
# The pinned PyTorch above is excluded only from a temporary copy.
# ============================================================

FILTERED_REQUIREMENTS="$(mktemp "$PROJECT/.requirements-runpod.XXXXXX" )"
sed -E '/^[[:space:]]*torch([[:space:]]|[<=>!~]|$)/d' \
    "$PROJECT/requirements.txt" > "$FILTERED_REQUIREMENTS"

if "$PYTHON_BIN" -m pip install -r "$FILTERED_REQUIREMENTS"; then
    rm -f "$FILTERED_REQUIREMENTS"
else
    rm -f "$FILTERED_REQUIREMENTS"
    echo "ERROR: Installing requirements.txt failed." >&2
    exit 1
fi

# ============================================================
# 7. Install FFmpeg and Audio Mixer runtime requirements
# ============================================================

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "FFmpeg/ffprobe are missing; installing the ffmpeg system package..."

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: apt-get is unavailable. Install ffmpeg and ffprobe manually." >&2
        exit 1
    fi

    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
    else
        echo "ERROR: Installing FFmpeg requires root or sudo privileges." >&2
        exit 1
    fi
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "ERROR: ffmpeg/ffprobe are still unavailable after installation." >&2
    exit 1
fi

echo "FFmpeg: $(ffmpeg -version | head -n 1)"
echo "ffprobe: $(ffprobe -version | head -n 1)"

FFMPEG_ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
if ! grep -q 'libopus' <<<"$FFMPEG_ENCODERS"; then
    echo "ERROR: This FFmpeg build has no libopus encoder, required for OGG/Opus output." >&2
    exit 1
fi
echo "FFmpeg libopus encoder: OK"

# Audio Mixer runtime dependencies. The full VoxCPM requirements are installed above.
"$PYTHON_BIN" -m pip install \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.34' \
    'python-multipart>=0.0.20' \
    'pydantic>=2.10' \
    'pydantic-settings>=2.7' \
    'numpy>=1.26' \
    'scipy' \
    'soundfile' \
    'librosa' \
    'psycopg[binary]>=3.2'

"$PYTHON_BIN" -m pip check

# ============================================================
# 8. Hugging Face / Whisper cache directories and environment
# ============================================================

HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME/whisper"
export HF_HOME HUGGINGFACE_HUB_CACHE XDG_CACHE_HOME

echo "Model cache: $HF_HOME"
echo "Whisper cache: $XDG_CACHE_HOME/whisper"

# ============================================================
# 9. Persist cache environment variables in the user's bashrc
# The generated startup.sh also exports them when launched directly.
# ============================================================

BASHRC="${HOME:-/tmp}/.bashrc"
if touch "$BASHRC" 2>/dev/null; then
    add_bashrc_export() {
        local name="$1"
        local value="$2"
        local escaped
        printf -v escaped '%q' "$value"
        local line="export $name=$escaped"

        if ! grep -qxF "$line" "$BASHRC"; then
            printf '%s\n' "$line" >> "$BASHRC"
        fi
    }

    add_bashrc_export HF_HOME "$HF_HOME"
    add_bashrc_export HUGGINGFACE_HUB_CACHE "$HUGGINGFACE_HUB_CACHE"
    add_bashrc_export XDG_CACHE_HOME "$XDG_CACHE_HOME"
else
    echo "WARNING: Could not update $BASHRC; startup.sh will still set cache paths." >&2
fi

# ============================================================
# 10. Create .env and generate a private API key if needed
# ============================================================

if [[ ! -f "$PROJECT/.env" ]]; then
    cat > "$PROJECT/.env" <<'ENV_EOF'
APP_NAME=VoxCPM2 Professional Voice API
APP_VERSION=1.0.0
API_KEY=8DIEsVE-jyphjyHpTUnBgIgT7dl5b9GA5y5Cz8Cr_AAYyt2ynhED50yE9WGowooF


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
ENV_EOF
    echo "Created $PROJECT/.env"
else
    echo "Keeping existing $PROJECT/.env"
fi

if ! grep -q '^API_KEY=' "$PROJECT/.env" \
    || grep -Eq '^API_KEY=(CHANGE_ME_TO_A_LONG_RANDOM_SECRET|change-me)?$' "$PROJECT/.env"; then
    GENERATED_API_KEY="$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_urlsafe(32))')"

    if grep -q '^API_KEY=' "$PROJECT/.env"; then
        sed -i "s/^API_KEY=.*/API_KEY=$GENERATED_API_KEY/" "$PROJECT/.env"
    else
        printf '\nAPI_KEY=%s\n' "$GENERATED_API_KEY" >> "$PROJECT/.env"
    fi

    unset GENERATED_API_KEY
    echo "Generated a new random API key in $PROJECT/.env"
fi

# ============================================================
# 11. Create startup.sh (preserves original GPU/API startup steps)
# ============================================================

{
    printf '%s\n' '#!/usr/bin/env bash' 'set -Eeuo pipefail'
    printf 'PROJECT=%q\n' "$PROJECT"
    printf 'HF_HOME=%q\n' "$HF_HOME"
    printf 'HUGGINGFACE_HUB_CACHE=%q\n' "$HUGGINGFACE_HUB_CACHE"
    printf 'XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"

    cat <<'STARTUP_EOF'

PYTHON="$PROJECT/.venv/bin/python"
export HF_HOME HUGGINGFACE_HUB_CACHE XDG_CACHE_HOME
cd "$PROJECT"

echo ""
echo "============================================================"
echo "VoxCPM API"
echo "============================================================"
echo "Project: $PROJECT"
echo "Python:  $PYTHON"
echo "GPU:     ${RUNPOD_GPU_NAME:-Unknown}"
echo "Host:    ${HOST:-0.0.0.0}"
echo "Port:    ${PORT:-8000}"

echo ""
echo "============================================================"
echo "GPU / PyTorch CHECK"
echo "============================================================"

"$PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
else:
    print("WARNING: CUDA is unavailable; inference will depend on CPU support and memory.")
PY

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    PUBLIC_URL="https://${RUNPOD_POD_ID}-${PORT:-8000}.proxy.runpod.net"
    echo ""
    echo "RunPod URL: $PUBLIC_URL"
    echo "Swagger:    $PUBLIC_URL/docs"
    echo "OpenAPI:    $PUBLIC_URL/openapi.json"
fi

echo ""
echo "Starting API..."
exec "$PYTHON" -m uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers 1
STARTUP_EOF
} > "$PROJECT/startup.sh"

chmod +x "$PROJECT/startup.sh"

# ============================================================
# 12. Installation checks (Python packages and FFmpeg tools )
# ============================================================

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

import fastapi
import torch
import uvicorn

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
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
        raise SystemExit(f"Required module is missing: {module}")
    print(f"{module}: OK")
PY

PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
from app.main import app

print("FastAPI app import: OK")
print("Registered routes:", len(app.routes))
PY

ffmpeg -version | head -n 1
ffprobe -version | head -n 1
"$PYTHON_BIN" -m pip check

echo ""
echo "============================================================"
echo "Installation checks passed."
echo "Workspace: $BASE_DIR"
echo "Project:   $PROJECT"
echo "Cache:     $CACHE_DIR"
echo "To run later: $PROJECT/startup.sh"
echo "============================================================"
echo ""

# ============================================================
# 13. Start API (same final behavior as the original script)
# ============================================================

exec "$PROJECT/startup.sh"
