%%bash
set -Eeuo pipefail

# ============================================================
# VoxCPM API + Cloudflare Quick Tunnel
# Google Colab — Defensive Startup Script
#
# - Reuses an existing project and API when possible
# - Preserves existing .env and API_KEY
# - Bounded startup and validation retries
# - Detects local API and tunnel failures
# - Does not claim success without HTTP validation
# - Cleans up only processes started by this script
# ============================================================

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------
# Configuration
# -----------------------------

REPO_URL="${REPO_URL:-https://github.com/Alqudimi/voxcpm-api-complete.git}"
PROJECT="${PROJECT_DIR:-/content/workspace/voxcpm-api-complete}"
CACHE_DIR="${CACHE_DIR:-/content/cache}"

PORT="${PORT:-8000}"
HOST="0.0.0.0"

API_LOG="/content/voxcpm-api.log"
TUNNEL_LOG="/content/cloudflared.log"
TUNNEL_STDOUT="/content/cloudflared.stdout.log"

LOCAL_READY_ATTEMPTS=45
TUNNEL_START_ATTEMPTS=3
TUNNEL_URL_ATTEMPTS=20
PUBLIC_CHECK_ATTEMPTS=6

API_PID=""
TUNNEL_PID=""
TAIL_PID=""
PUBLIC_URL=""

mkdir -p "$(dirname "$PROJECT")" "$CACHE_DIR"

export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME"

# -----------------------------
# Logging
# -----------------------------

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

# -----------------------------
# Cleanup
# -----------------------------

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    log "Cleaning up processes started by this script..."

    if [[ -n "${TAIL_PID:-}" ]]; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
    fi

    if [[ -n "${TUNNEL_PID:-}" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi

    if [[ -n "${API_PID:-}" ]]; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi

    if [[ "$exit_code" -ne 0 ]]; then
        log "Script exited with code $exit_code."
        log "API log: $API_LOG"
        log "Tunnel log: $TUNNEL_LOG"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# -----------------------------
# 1. Validate Colab
# -----------------------------

log "VoxCPM API — Google Colab + Cloudflare Tunnel"

[[ -d /content ]] || fail "This script requires Google Colab."

command -v python3 >/dev/null 2>&1 ||
    fail "python3 is not installed."

command -v curl >/dev/null 2>&1 ||
    fail "curl is not installed."

command -v git >/dev/null 2>&1 ||
    fail "git is not installed."

PYTHON_BIN="$(command -v python3)"

"$PYTHON_BIN" - <<'PY'
import sys

print("Python:", sys.version)
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required.")
PY

log "Project: $PROJECT"
log "Python:  $PYTHON_BIN"
log "Port:    $PORT"

# -----------------------------
# 2. Reuse or clone project
# -----------------------------

if [[ -d "$PROJECT/.git" ]]; then
    log "Existing Git checkout found."

elif [[ -f "$PROJECT/app/main.py" &&
        -f "$PROJECT/requirements.txt" ]]; then
    log "Existing project files found without Git metadata."
    log "Using the existing project without overwriting it."

elif [[ ! -e "$PROJECT" ]]; then
    log "Cloning repository..."

    git clone --depth 1 "$REPO_URL" "$PROJECT" ||
        fail "Could not clone the repository."

else
    fail "Project directory exists but is not a valid project. Set PROJECT_DIR to another directory."
fi

[[ -f "$PROJECT/app/main.py" ]] ||
    fail "Missing app/main.py."

[[ -f "$PROJECT/requirements.txt" ]] ||
    fail "Missing requirements.txt."

cd "$PROJECT"

# -----------------------------
# 3. Verify system tools
# -----------------------------

log "Checking FFmpeg..."

if ! command -v ffmpeg >/dev/null 2>&1 ||
   ! command -v ffprobe >/dev/null 2>&1; then

    apt-get update
    apt-get install -y --no-install-recommends ffmpeg ca-certificates curl git
fi

command -v ffmpeg >/dev/null 2>&1 ||
    fail "FFmpeg installation failed."

command -v ffprobe >/dev/null 2>&1 ||
    fail "ffprobe installation failed."

ffmpeg -version | head -n 1

FFMPEG_ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"

grep -Eq '[[:space:]]libopus([[:space:]]|$)' <<<"$FFMPEG_ENCODERS" ||
    fail "FFmpeg does not provide the libopus encoder."

log "FFmpeg and libopus are available."

# -----------------------------
# 4. Validate Python environment
# -----------------------------

log "Checking Python dependencies..."

"$PYTHON_BIN" - <<'PY'
import importlib.util

required = [
    "torch",
    "fastapi",
    "uvicorn",
    "pydantic",
]

missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    print("Missing Python modules:", ", ".join(missing))
    raise SystemExit(1)

import torch
import fastapi
import uvicorn

print("PyTorch:", torch.__version__)
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# -----------------------------
# 5. Validate existing .env
# -----------------------------

if [[ ! -f "$PROJECT/.env" ]]; then
    log "No .env found. Creating a minimal configuration."

    cat > "$PROJECT/.env" <<'ENV_EOF'
APP_NAME=VoxCPM2 Professional Voice API
APP_VERSION=1.0.0
API_KEY=CHANGE_ME_TO_A_LONG_RANDOM_SECRET
VOXCPM_MODEL_ID=openbmb/VoxCPM2
DEVICE=cuda
HOST=0.0.0.0
PORT=8000
ENV_EOF

    chmod 600 "$PROJECT/.env"
fi

if grep -Eq '^API_KEY=(CHANGE_ME_TO_A_LONG_RANDOM_SECRET|change-me)?$' "$PROJECT/.env"; then
    log "Replacing the placeholder API key."

    API_KEY_VALUE="$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export API_KEY_VALUE

    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["PROJECT_DIR"]) / ".env"
key = os.environ["API_KEY_VALUE"]

lines = path.read_text().splitlines()
found = False

for i, line in enumerate(lines):
    if line.startswith("API_KEY="):
        lines[i] = "API_KEY=" + key
        found = True

if not found:
    lines.append("API_KEY=" + key)

path.write_text("\n".join(lines) + "\n")
PY
fi

chmod 600 "$PROJECT/.env"

# -----------------------------
# 6. Install cloudflared
# -----------------------------

log "Checking cloudflared..."

if ! command -v cloudflared >/dev/null 2>&1; then
    ARCH="$(uname -m)"

    case "$ARCH" in
        x86_64)
            CLOUDFLARED_ARCH="amd64"
            ;;
        aarch64|arm64)
            CLOUDFLARED_ARCH="arm64"
            ;;
        *)
            fail "Unsupported architecture: $ARCH"
            ;;
    esac

    TMP_BIN="$(mktemp /content/cloudflared.XXXXXX)"

    if ! curl -fLsS --retry 3 --connect-timeout 15 \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CLOUDFLARED_ARCH}" \
        -o "$TMP_BIN"; then
        rm -f "$TMP_BIN"
        fail "Could not download cloudflared."
    fi

    install -m 0755 "$TMP_BIN" /usr/local/bin/cloudflared
    rm -f "$TMP_BIN"
fi

cloudflared --version

# -----------------------------
# 7. Detect existing local API
# -----------------------------

log "Checking local API..."

LOCAL_API_URL="http://127.0.0.1:${PORT}"
OPENAPI_URL="${LOCAL_API_URL}/openapi.json"

API_ALREADY_RUNNING=0

if curl -fsS --connect-timeout 3 --max-time 10 \
    "$OPENAPI_URL" -o /dev/null 2>/dev/null; then

    API_ALREADY_RUNNING=1
    log "An API is already responding on port $PORT."
    log "Reusing the existing API."

else
    log "Starting FastAPI..."

    "$PYTHON_BIN" -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers 1 \
        > "$API_LOG" 2>&1 &

    API_PID=$!
fi

# -----------------------------
# 8. Wait for local API
# -----------------------------

log "Waiting for local API..."

LOCAL_READY=0

for ((i=1; i<=LOCAL_READY_ATTEMPTS; i++)); do

    if curl -fsS --connect-timeout 3 --max-time 10 \
        "$OPENAPI_URL" -o /dev/null 2>/dev/null; then

        LOCAL_READY=1
        break
    fi

    if [[ -n "${API_PID:-}" ]] &&
       ! kill -0 "$API_PID" 2>/dev/null; then

        log "FastAPI process exited unexpectedly."
        tail -n 100 "$API_LOG" >&2 || true
        fail "FastAPI failed during startup."
    fi

    sleep 2
done

[[ "$LOCAL_READY" -eq 1 ]] ||
    fail "Local API did not respond within the startup deadline."

log "Local API validated successfully."

# -----------------------------
# 9. Start and validate tunnel
# -----------------------------

for ((attempt=1; attempt<=TUNNEL_START_ATTEMPTS; attempt++)); do

    log "Starting Cloudflare Quick Tunnel: $attempt/$TUNNEL_START_ATTEMPTS"

    : > "$TUNNEL_LOG"
    : > "$TUNNEL_STDOUT"

    # Use HTTP/2 to avoid depending on UDP/QUIC connectivity.
    cloudflared tunnel \
        --url "$LOCAL_API_URL" \
        --protocol http2 \
        --no-autoupdate \
        > "$TUNNEL_STDOUT" \
        2> "$TUNNEL_LOG" &

    TUNNEL_PID=$!
    CANDIDATE_URL=""

    for ((i=1; i<=TUNNEL_URL_ATTEMPTS; i++)); do

        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            log "Tunnel process exited before publishing a URL."
            break
        fi

        CANDIDATE_URL="$(
            grep -Eho \
                'https://[a-zA-Z0-9-]+\.trycloudflare\.com' \
                "$TUNNEL_LOG" "$TUNNEL_STDOUT" 2>/dev/null \
                | head -n 1 || true
        )"

        [[ -n "$CANDIDATE_URL" ]] && break

        sleep 2
    done

    if [[ -z "$CANDIDATE_URL" ]]; then
        tail -n 60 "$TUNNEL_LOG" >&2 || true

        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
        TUNNEL_PID=""

        sleep 3
        continue
    fi

    log "Cloudflare published a URL. Validating the public API..."

    PUBLIC_READY=0

    for ((i=1; i<=PUBLIC_CHECK_ATTEMPTS; i++)); do

        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            log "Tunnel stopped during public validation."
            break
        fi

        HTTP_STATUS="$(
            curl -sS \
                --connect-timeout 5 \
                --max-time 12 \
                -o /dev/null \
                -w '%{http_code}' \
                "${CANDIDATE_URL}/openapi.json" 2>/dev/null \
                || true
        )"

        log "Public check $i/$PUBLIC_CHECK_ATTEMPTS: HTTP ${HTTP_STATUS:-connection failed}"

        if [[ "$HTTP_STATUS" == "200" ]]; then
            PUBLIC_READY=1
            break
        fi

        sleep 3
    done

    if [[ "$PUBLIC_READY" -eq 1 ]]; then
        PUBLIC_URL="$CANDIDATE_URL"
        break
    fi

    log "Public validation failed. Inspecting tunnel logs..."
    tail -n 50 "$TUNNEL_LOG" >&2 || true

    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""

    sleep 3
done

# -----------------------------
# 10. Report failure honestly
# -----------------------------

if [[ -z "$PUBLIC_URL" ]]; then
    echo
    echo "============================================================"
    echo "PUBLIC TUNNEL FAILED"
    echo "============================================================"
    echo "Local API: $LOCAL_API_URL"
    echo "API log: $API_LOG"
    echo "Tunnel log: $TUNNEL_LOG"
    echo
    echo "The local API may still be running."
    echo "No unverified public URL will be reported as working."
    echo "============================================================"

    exit 1
fi

# -----------------------------
# 11. Display validated URL
# -----------------------------

echo
echo "============================================================"
echo "VOXCPM API — PUBLIC ENDPOINT VERIFIED"
echo "============================================================"
echo "Public API: $PUBLIC_URL"
echo "Swagger UI: $PUBLIC_URL/docs"
echo "OpenAPI: $PUBLIC_URL/openapi.json"
echo "Local API: $LOCAL_API_URL"
echo "API log: $API_LOG"
echo "Tunnel log: $TUNNEL_LOG"
echo
echo "Public OpenAPI returned HTTP 200."
echo "Keep this cell running."
echo "Quick Tunnel URLs are temporary."
echo "Do not share your API key."
echo "============================================================"

# Keep the cell alive while preserving the tunnel.
tail -n 0 -F "$TUNNEL_LOG" &
TAIL_PID=$!

wait "$TUNNEL_PID"