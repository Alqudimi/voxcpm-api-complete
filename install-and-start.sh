#!/usr/bin/env bash
set -e

# ============================================================
# VoxCPM API - RunPod Setup & Startup
# ============================================================

PROJECT="/workspace/voxcpm-api-complete2"
REPO="https://github.com/Alqudimi/voxcpm-api-complete.git"

PYTHON_BIN="$PROJECT/.venv/bin/python"

echo ""
echo "============================================================"
echo "04 VoxCPM API - RunPod Setup"
echo "============================================================"
echo ""

# ============================================================
# 1. Clone Project
# ============================================================

if [ ! -d "$PROJECT/.git" ]; then

    echo "9à3 Cloning repository..."

    cd /workspace
    git clone "$REPO"
    

else

    echo "9à4 Project already exists"
    cd "$PROJECT"

    git pull --ff-only || true

fi

cd "$PROJECT"

# ============================================================
# 2. Create Virtual Environment
# ============================================================

if [ ! -d "$PROJECT/.venv" ]; then

    echo "9Ê7 Creating Python virtual environment..."

    python3 -m venv .venv

else

    echo "7½7 Virtual environment already exists"

fi

# ============================================================
# 3. Activate Virtual Environment
# ============================================================

source "$PROJECT/.venv/bin/activate"

echo ""
echo "Python:"
python --version

# ============================================================
# 4. Upgrade pip
# ============================================================

echo ""
echo "8¤815 Upgrading pip..."

python -m pip install --upgrade pip setuptools wheel

# ============================================================
# 5. Install PyTorch CUDA 12.8
# ============================================================

echo ""
echo "9æ7 Installing PyTorch CUDA 12.8..."

python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# 6. Remove torch from requirements.txt
# ============================================================

if [ -f "$PROJECT/requirements.txt" ]; then

    echo ""
    echo "0ä3 Removing torch from requirements.txt..."

    sed -i '/^[[:space:]]*torch[[:space:]]*$/d' \
        "$PROJECT/requirements.txt"

fi

# ============================================================
# 7. Install Requirements
# ============================================================

if [ -f "$PROJECT/requirements.txt" ]; then

    echo ""
    echo "9à4 Installing project requirements..."

    python -m pip install -r "$PROJECT/requirements.txt"

fi

# ============================================================
# 7A. Install Audio Mixer runtime requirements
# ============================================================

echo ""
echo "9¿215 Installing Audio Mixer runtime requirements..."

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "❌ The combined project requires Python 3.11 or newer. Current interpreter: $($PYTHON_BIN --version 2>&1)" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "7Ã4 ffmpeg/ffprobe are missing and apt-get is unavailable." >&2
        exit 1
    fi

    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
    else
        echo "7Ã4 Install FFmpeg as root (or provide sudo) before continuing." >&2
        exit 1
    fi
fi

"$PYTHON_BIN" -m pip install \
    'fastapi>=0.110' \
    'uvicorn>=0.29' \
    'python-multipart>=0.0.9' \
    'psycopg[binary]>=3.2'

"$PYTHON_BIN" -m pip check

echo "7½7 Audio Mixer runtime requirements installed"

# ============================================================
# 8. HuggingFace / Whisper Cache
# ============================================================

echo ""
echo "9Ü7 Creating cache directories..."

mkdir -p \
    /workspace/.cache/huggingface \
    /workspace/.cache/huggingface/hub \
    /workspace/.cache/whisper

# ============================================================
# 9. Environment Variables
# ============================================================

export HF_HOME="/workspace/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/workspace/.cache/huggingface/hub"
export XDG_CACHE_HOME="/workspace/.cache"

# ============================================================
# 10. Save Environment Variables
# ============================================================

echo ""
echo "7±515 Configuring environment variables..."

touch /workspace/.bashrc

grep -qxF \
    'export HF_HOME="/workspace/.cache/huggingface"' \
    /workspace/.bashrc \
    || echo 'export HF_HOME="/workspace/.cache/huggingface"' \
    >> /workspace/.bashrc

grep -qxF \
    'export HUGGINGFACE_HUB_CACHE="/workspace/.cache/huggingface/hub"' \
    /workspace/.bashrc \
    || echo 'export HUGGINGFACE_HUB_CACHE="/workspace/.cache/huggingface/hub"' \
    >> /workspace/.bashrc

grep -qxF \
    'export XDG_CACHE_HOME="/workspace/.cache"' \
    /workspace/.bashrc \
    || echo 'export XDG_CACHE_HOME="/workspace/.cache"' \
    >> /workspace/.bashrc

# ============================================================
# 11. Create .env
# ============================================================

# ============================================================
# 11. Create .env
# ============================================================

echo ""
echo "9ä6 Configuring .env..."

if [ ! -f "$PROJECT/.env" ]; then

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

    echo "7½7 .env created"

else

    echo "✓ .env already exists - keeping existing configuration"

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
    echo "✓ Replaced the default API key with a random secret in .env"

fi

# ============================================================
# 12. Create startup.sh
# ============================================================

echo ""
echo "9ß5 Creating startup.sh..."

cat > "$PROJECT/startup.sh" <<'STARTUP_EOF'
#!/usr/bin/env bash
set -e

PROJECT="/workspace/voxcpm-api-complete2"
PYTHON="$PROJECT/.venv/bin/python"

export HF_HOME="/workspace/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/workspace/.cache/huggingface/hub"
export XDG_CACHE_HOME="/workspace/.cache"

cd "$PROJECT"

echo ""
echo "============================================================"
echo "04 VoxCPM API"
echo "============================================================"

echo "9Ü8 Project: $PROJECT"
echo "9Ê7 Python:  $PYTHON"
echo "9Á2 GPU:     ${RUNPOD_GPU_NAME:-Unknown}"
echo "9ß9 Host:    ${HOST:-0.0.0.0}"
echo "9ä2 Port:    ${PORT:-8000}"

echo ""

# ============================================================
# GPU CHECK
# ============================================================

echo "============================================================"
echo "9Á2 GPU CHECK"
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

    print("7²2 WARNING: CUDA is not available")

PY

echo ""

# ============================================================
# RunPod Public URL
# ============================================================

if [ -n "${RUNPOD_POD_ID:-}" ]; then

    PUBLIC_URL="https://${RUNPOD_POD_ID}-${PORT:-8000}.proxy.runpod.net"

    echo "============================================================"
    echo "9±1 RUNPOD"
    echo "============================================================"

    echo "Public URL:"
    echo "$PUBLIC_URL"

    echo ""
    echo "Swagger:"
    echo "$PUBLIC_URL/docs"

    echo ""
    echo "OpenAPI:"
    echo "$PUBLIC_URL/openapi.json"

fi

echo ""
echo "============================================================"
echo "04 STARTING API"
echo "============================================================"
echo ""

exec "$PYTHON" -m uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers 1

STARTUP_EOF

chmod +x "$PROJECT/startup.sh"

# ============================================================
# 13. Installation Check
# ============================================================

echo ""
echo "============================================================"
echo "9ä3 INSTALLATION CHECK"
echo "============================================================"

"$PYTHON_BIN" - <<'PY'
import sys

print("Python:", sys.version )

try:

    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))

except Exception as e:

    print("PyTorch error:", e)


try:

    import fastapi

    print("FastAPI:", fastapi.__version__)

except Exception:

    print("FastAPI: NOT FOUND")


try:

    import uvicorn

    print("Uvicorn:", uvicorn.__version__)

except Exception:

    print("Uvicorn: NOT FOUND")

PY

# ============================================================
# 14. Cache Configuration
# ============================================================

echo ""
echo "============================================================"
echo "7±515 CACHE CONFIGURATION"
echo "============================================================"

echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"

# ============================================================
# 15. Start API
# ============================================================

echo ""a
echo "============================================================"
echo "04 STARTING VOXCPM API"
echo "============================================================"
echo ""

exec "$PROJECT/startup.sh"
