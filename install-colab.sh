#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# VoxCPM API + Audio Mixer — Google Colab setup and startup
# - No Google Drive integration
# - Workspace and model cache are local to the Colab runtime
# - Preserves an existing project and any real API key
# - Keeps Colab's installed PyTorch/CUDA instead of replacing it
# - Installs project requirements and checks FFmpeg/libopus
# - Starts FastAPI, creates a Cloudflare Quick Tunnel, and
#   validates the public OpenAPI endpoint before reporting success
# ============================================================

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_URL="${REPO_URL:-https://github.com/Alqudimi/voxcpm-api-complete.git}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/content/workspace}"
PROJECT_NAME="voxcpm-api-complete"
PROJECT="${PROJECT_DIR:-$WORKSPACE_DIR/$PROJECT_NAME}"
CACHE_DIR="${CACHE_DIR:-/content/cache}"
PORT="${PORT:-8000}"
HOST="0.0.0.0"
UPDATE_REPO="${UPDATE_REPO:-1}"
API_READY_TIMEOUT="${API_READY_TIMEOUT:-1800}"
TUNNEL_START_ATTEMPTS="${TUNNEL_START_ATTEMPTS:-3}"
TUNNEL_URL_TIMEOUT="${TUNNEL_URL_TIMEOUT:-90}"
PUBLIC_CHECK_ATTEMPTS="${PUBLIC_CHECK_ATTEMPTS:-12}"

API_LOG="/content/voxcpm-api.log"
TUNNEL_LOG="/content/cloudflared.log"
TUNNEL_STDOUT="/content/cloudflared.stdout.log"
LOCAL_OPENAPI_FILE="/content/voxcpm-openapi-local.json"
PUBLIC_CHECK_FILE="/content/voxcpm-openapi-public.tmp"

PYTHON_SYSTEM="${PYTHON_SYSTEM:-python3}"
PYTHON_BIN=""
API_PID=""
TUNNEL_PID=""
TAIL_PID=""
FILTERED_REQUIREMENTS=""
API_STARTED_BY_SCRIPT=0
PUBLIC_URL=""

# Export this explicitly. The earlier version referenced PROJECT_DIR
# from Python without exporting it, causing KeyError: PROJECT_DIR.
export PROJECT_DIR="$PROJECT"

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

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

    # Only stop FastAPI when this script launched it. An API reused from
    # an earlier cell/session is deliberately left running.
    if [[ "${API_STARTED_BY_SCRIPT:-0}" == "1" && -n "${API_PID:-}" ]]; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi

    if [[ -n "${FILTERED_REQUIREMENTS:-}" ]]; then
        rm -f "$FILTERED_REQUIREMENTS" 2>/dev/null || true
    fi

    rm -f "$PUBLIC_CHECK_FILE" 2>/dev/null || true

    if [[ "$exit_code" -ne 0 ]]; then
        log "Script stopped with exit code $exit_code."
        log "API log: $API_LOG"
        log "Tunnel stderr log: $TUNNEL_LOG"
        log "Tunnel stdout log: $TUNNEL_STDOUT"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ============================================================
# 1. Check runtime and create local workspace/cache
# ============================================================

log "VoxCPM API + Audio Mixer — Google Colab + Cloudflare Tunnel"

[[ -d /content ]] || fail "This script is intended for Google Colab (/content is missing)."
command -v "$PYTHON_SYSTEM" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_SYSTEM"
command -v curl >/dev/null 2>&1 || fail "curl is unavailable."
command -v git >/dev/null 2>&1 || fail "git is unavailable."

PYTHON_BIN="$(command -v "$PYTHON_SYSTEM")"
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number."
(( PORT >= 1 && PORT <= 65535 )) || fail "PORT must be between 1 and 65535."
[[ "$API_READY_TIMEOUT" =~ ^[0-9]+$ ]] || fail "API_READY_TIMEOUT must be a number of seconds."

mkdir -p "$WORKSPACE_DIR" "$(dirname "$PROJECT")" "$CACHE_DIR"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_DIR/torch}"
export HOST PORT
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME/whisper" "$TORCH_HOME"

"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.version.replace("\n", " "))
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required.")
PY

log "Workspace: $WORKSPACE_DIR"
log "Project:   $PROJECT"
log "Cache:     $CACHE_DIR"
log "Python:    $PYTHON_BIN"
log "Port:      $PORT"

# ============================================================
# 2. Reuse or clone project; update only a clean Git checkout
# ============================================================

if [[ -d "$PROJECT/.git" ]]; then
    log "Existing Git checkout found."
    if [[ "$UPDATE_REPO" == "1" ]]; then
        if [[ -z "$(git -C "$PROJECT" status --porcelain 2>/dev/null || true)" ]]; then
            log "Updating the clean checkout with git pull --ff-only..."
            if ! git -C "$PROJECT" pull --ff-only; then
                log "WARNING: Git update failed; continuing with the existing checkout."
            fi
        else
            log "Working-tree changes detected; preserving them and skipping git pull."
        fi
    fi
elif [[ -f "$PROJECT/app/main.py" && -f "$PROJECT/requirements.txt" ]]; then
    log "Existing project files found without Git metadata; preserving them."
elif [[ ! -e "$PROJECT" ]]; then
    log "Cloning repository..."
    git clone --depth 1 "$REPO_URL" "$PROJECT" || fail "Could not clone $REPO_URL"
else
    fail "Project directory exists but is not a valid project: $PROJECT. Set PROJECT_DIR to another path."
fi

[[ -f "$PROJECT/app/main.py" ]] || fail "Missing $PROJECT/app/main.py"
[[ -f "$PROJECT/requirements.txt" ]] || fail "Missing $PROJECT/requirements.txt"
[[ -f "$PROJECT/app/audio_mixer/api.py" ]] || fail "Missing Audio Mixer source: app/audio_mixer/api.py"
grep -q 'include_audio_mixer' "$PROJECT/app/main.py" || fail "app/main.py does not register include_audio_mixer."

cd "$PROJECT"

# ============================================================
# 3. Install and verify FFmpeg/libopus
# ============================================================

log "Checking FFmpeg and libopus..."
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg ca-certificates curl git
fi

command -v ffmpeg >/dev/null 2>&1 || fail "FFmpeg installation failed."
command -v ffprobe >/dev/null 2>&1 || fail "ffprobe is unavailable."
FFMPEG_ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
grep -Eq '[[:space:]]libopus([[:space:]]|$)' <<<"$FFMPEG_ENCODERS" || fail "FFmpeg does not provide the libopus encoder."
log "$(ffmpeg -version 2>&1 | sed -n '1p')"
log "FFmpeg, ffprobe, and libopus are available."

# ============================================================
# 4. Validate Colab's existing PyTorch/CUDA before package installs
# ============================================================

log "Checking Colab's existing PyTorch and GPU..."
"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    raise SystemExit(
        "PyTorch is missing from this Colab runtime. Reconnect/select a GPU runtime; "
        "this script will not replace Colab's managed CUDA/PyTorch installation automatically."
    )

import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA GPU is available. In Colab select Runtime > Change runtime type > GPU, "
        "then rerun this cell."
    )

print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))

try:
    test = torch.randn((512, 512), device="cuda")
    _ = test @ test
    torch.cuda.synchronize()
    print("CUDA matrix-multiplication test: OK")
del test
except Exception as exc:
    raise SystemExit(f"CUDA computation test failed: {type(exc).__name__}: {exc}")
PY

# ============================================================
# 5. Create .env only when absent; preserve a real API key/config
# ============================================================

if [[ ! -f "$PROJECT/.env" ]]; then
    log "Creating a new .env file."
    cat > "$PROJECT/.env" <<'ENV_EOF'
APP_NAME=VoxCPM2 Professional Voice API
APP_VERSION=3.0.0
API_KEY=CHANGE_ME_TO_A_LONG_RANDOM_SECRET
VOXCPM_MODEL_ID=openbmb/VoxCPM2
WHISPER_MODEL_NAME=large-v3
ECAPA_SOURCE=speechbrain/spkrec-ecapa-voxceleb
ECAPA_SAVEDIR=storage/models/ecapa_voxceleb
DEVICE=cuda
REFERENCE_SAMPLE_RATE=16000
OUTPUT_SAMPLE_RATE=48000
AUDIO_SAMPLE_RATE=24000
AUDIO_CHANNELS=1
MAX_FILE_SIZE_MB=100
MAX_MERGE_FILES=20
DEFAULT_CFG=2.0
DEFAULT_TIMESTEPS=30
DEFAULT_CANDIDATES=1
DEFAULT_RETRY_BADCASE=true
DEFAULT_RETRY_MAX_TIMES=3
DEFAULT_RETRY_RATIO_THRESHOLD=6.0
OUTPUT_BITRATE=96k
OUTPUT_CODEC=libopus
HOST=0.0.0.0
PORT=8000
ENV_EOF
fi

log "Validating API_KEY in .env without printing it..."
export PROJECT_ENV="$PROJECT/.env"
"$PYTHON_BIN" - <<'PY'
import os
import secrets
from pathlib import Path

path = Path(os.environ["PROJECT_ENV"])
lines = path.read_text(encoding="utf-8").splitlines()
key_index = None
key_value = None

for index, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        continue
    name, value = line.split("=", 1)
    if name.strip() == "API_KEY":
        key_index = index
        key_value = value.strip().strip('"').strip("'")
        break

placeholders = {None, "", "change-me", "CHANGE_ME_TO_A_LONG_RANDOM_SECRET"}
if key_value in placeholders:
    new_key = secrets.token_urlsafe(32)
    entry = f"API_KEY={new_key}"
    if key_index is None:
        lines.append(entry)
    else:
        lines[key_index] = entry
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("A fresh random API key has been written to .env.")
else:
    print("Existing non-placeholder API key preserved.")
PY
chmod 600 "$PROJECT/.env"
unset PROJECT_ENV

# ============================================================
# 6. Install requirements without explicitly replacing torch stack
# ============================================================

log "Preparing requirements while preserving Colab's PyTorch stack..."
FILTERED_REQUIREMENTS="$(mktemp /content/requirements-colab.XXXXXX.txt)"
export FILTERED_REQUIREMENTS
"$PYTHON_BIN" - "$PROJECT/requirements.txt" "$FILTERED_REQUIREMENTS" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
# Do not directly ask pip to replace Colab-managed CUDA packages.
core = re.compile(r"^\s*(torch|torchvision|torchaudio|torchao|triton)(?=\s*(?:\[|[<>=!~@;#]|$))", re.I)
lines = source.read_text(encoding="utf-8").splitlines()
kept = []
skipped = []
for line in lines:
    if core.match(line):
        skipped.append(line.strip())
    else:
        kept.append(line)
target.write_text("\n".join(kept) + "\n", encoding="utf-8")
print("Filtered core package requirements:", ", ".join(skipped) if skipped else "none")
print("Remaining requirement lines:", sum(1 for line in kept if line.strip() and not line.lstrip().startswith("#")))
PY

log "Installing project and Audio Mixer dependencies. This can take several minutes..."
if ! "$PYTHON_BIN" -m pip install -r "$FILTERED_REQUIREMENTS"; then
    fail "pip could not install requirements.txt. Review the pip output above."
fi
rm -f "$FILTERED_REQUIREMENTS"
FILTERED_REQUIREMENTS=""
unset FILTERED_REQUIREMENTS

# ============================================================
# 7. Validate installed requirements, GPU remains functional
# ============================================================

log "Validating Python dependencies..."
"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = {
    "torch": "torch",
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "Pydantic": "pydantic",
    "Pydantic Settings": "pydantic_settings",
    "multipart": "multipart",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "SoundFile": "soundfile",
    "librosa": "librosa",
    "soxr": "soxr",
    "psycopg": "psycopg",
    "VoxCPM": "voxcpm",
    "SpeechBrain": "speechbrain",
    "Transformers": "transformers",
    "Accelerate": "accelerate",
    "faster-whisper": "faster_whisper",
    "openai-whisper": "whisper",
}
missing = [label for label, module in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit("Missing Python dependencies after installation: " + ", ".join(missing))

import torch, fastapi, uvicorn
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
print("CUDA available after installation:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA became unavailable after dependency installation.")
try:
    value = torch.ones((128, 128), device="cuda")
    _ = value @ value
    torch.cuda.synchronize()
    print("Post-install CUDA test: OK")
except Exception as exc:
    raise SystemExit(f"Post-install CUDA test failed: {type(exc).__name__}: {exc}")
PY

if ! "$PYTHON_BIN" -m pip check; then
    log "WARNING: pip check reported package conflicts in the environment. Startup/import checks will determine whether they are fatal."
fi

# ============================================================
# 8. Import the real FastAPI app and list API routes
# ============================================================

log "Checking FastAPI application and Audio Mixer registration..."
PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
from app.main import app

print("FastAPI import: OK")
print("Application title:", app.title)
print("Registered routes:", len(app.routes))
paths = []
for route in app.routes:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if path:
        paths.append(path)
        print(f"{','.join(sorted(methods or [])):20} {path}")
if not any("clone-and-mix" in path for path in paths):
    raise SystemExit("The clone-and-mix route is not registered; check the repository version.")
PY

# ============================================================
# 9. Install cloudflared
# ============================================================

log "Checking cloudflared..."
if ! command -v cloudflared >/dev/null 2>&1; then
    case "$(uname -m)" in
        x86_64) CLOUDFLARED_ARCH="amd64" ;;
        aarch64|arm64) CLOUDFLARED_ARCH="arm64" ;;
        *) fail "Unsupported CPU architecture: $(uname -m)" ;;
    esac

    CLOUDFLARED_TMP="$(mktemp /content/cloudflared.XXXXXX)"
    if ! curl -fLsS --retry 3 --connect-timeout 15 \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CLOUDFLARED_ARCH}" \
        -o "$CLOUDFLARED_TMP"; then
        rm -f "$CLOUDFLARED_TMP"
        fail "Could not download cloudflared from GitHub Releases."
    fi
    install -m 0755 "$CLOUDFLARED_TMP" /usr/local/bin/cloudflared
    rm -f "$CLOUDFLARED_TMP"
fi
cloudflared --version

# ============================================================
# 10. Helpers to validate the expected local/public API
# ============================================================

LOCAL_API_URL="http://127.0.0.1:${PORT}"
OPENAPI_URL="${LOCAL_API_URL}/openapi.json"

openapi_is_voxcpm() {
    local file="$1"
    "$PYTHON_BIN" - "$file" <<'PY'
import json
import sys
from pathlib import Path
try:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    title = str(data.get("info", {}).get("title", "")).lower()
    paths = data.get("paths", {})
    expected = ("voxcpm" in title) or any("clone-and-mix" in p for p in paths)
    raise SystemExit(0 if expected and paths else 1)
except Exception:
    raise SystemExit(1)
PY
}

local_api_ready() {
    curl -fsS --connect-timeout 3 --max-time 10 "$OPENAPI_URL" \
        -o "$LOCAL_OPENAPI_FILE" 2>/dev/null && openapi_is_voxcpm "$LOCAL_OPENAPI_FILE"
}

# ============================================================
# 11. Reuse an existing valid API or start FastAPI
# ============================================================

log "Checking local FastAPI at $OPENAPI_URL..."
if local_api_ready; then
    log "Existing VoxCPM API is responding; reusing it."
else
    log "Starting FastAPI. Model loading/warm-up may take several minutes..."
    "$PYTHON_BIN" -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers 1 \
        > "$API_LOG" 2>&1 &
    API_PID=$!
    API_STARTED_BY_SCRIPT=1
fi

log "Waiting for the local API to finish model loading and warm-up..."
API_READY=0
START_SECONDS=$SECONDS
NEXT_STATUS=$((SECONDS + 30))
API_DEADLINE=$((SECONDS + API_READY_TIMEOUT))

while (( SECONDS < API_DEADLINE )); do
    if local_api_ready; then
        API_READY=1
        break
    fi

    if [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
        log "FastAPI process exited unexpectedly. Last API log lines:"
        tail -n 120 "$API_LOG" >&2 || true
        fail "FastAPI exited before OpenAPI became available."
    fi

    if (( SECONDS >= NEXT_STATUS )); then
        log "Still waiting for model loading... $((SECONDS - START_SECONDS)) seconds elapsed."
        tail -n 5 "$API_LOG" 2>/dev/null || true
        NEXT_STATUS=$((SECONDS + 30))
    fi
    sleep 5
done

if [[ "$API_READY" != "1" ]]; then
    log "FastAPI did not become ready before the ${API_READY_TIMEOUT}s deadline. Last API log lines:"
    tail -n 150 "$API_LOG" >&2 || true
    fail "Local API readiness timeout."
fi
log "Local VoxCPM API validated successfully."

# ============================================================
# 12. Start Cloudflare Quick Tunnel and validate public endpoint
# ============================================================

for ((attempt=1; attempt<=TUNNEL_START_ATTEMPTS; attempt++)); do
    log "Starting Cloudflare Quick Tunnel ($attempt/$TUNNEL_START_ATTEMPTS)..."
    : > "$TUNNEL_LOG"
    : > "$TUNNEL_STDOUT"

    cloudflared tunnel \
        --url "$LOCAL_API_URL" \
        --protocol http2 \
        --no-autoupdate \
        > "$TUNNEL_STDOUT" 2> "$TUNNEL_LOG" &
    TUNNEL_PID=$!

    CANDIDATE_URL=""
    URL_DEADLINE=$((SECONDS + TUNNEL_URL_TIMEOUT))
    while (( SECONDS < URL_DEADLINE )); do
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            log "cloudflared exited before publishing a URL."
            break
        fi
        CANDIDATE_URL="$(grep -Eho 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" "$TUNNEL_STDOUT" 2>/dev/null | head -n 1 || true)"
        if [[ -n "$CANDIDATE_URL" ]]; then
            break
        fi
        sleep 2
    done

    if [[ -z "$CANDIDATE_URL" ]]; then
        log "No public URL was published. cloudflared log:"
        tail -n 80 "$TUNNEL_LOG" >&2 || true
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
        TUNNEL_PID=""
        sleep 3
        continue
    fi

    log "Cloudflare URL discovered: $CANDIDATE_URL"
    log "Checking the public OpenAPI endpoint (a new tunnel may need time to activate)..."
    PUBLIC_READY=0

    for ((check=1; check<=PUBLIC_CHECK_ATTEMPTS; check++)); do
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            log "cloudflared stopped during public validation."
            break
        fi

        HTTP_STATUS="$(curl -sS -L --connect-timeout 8 --max-time 15 \
            -o "$PUBLIC_CHECK_FILE" -w '%{http_code}' \
            "${CANDIDATE_URL}/openapi.json" 2>/dev/null || true)"
        log "Public validation $check/$PUBLIC_CHECK_ATTEMPTS: HTTP ${HTTP_STATUS:-connection failed}"

        if [[ "$HTTP_STATUS" == "200" ]] && openapi_is_voxcpm "$PUBLIC_CHECK_FILE"; then
            PUBLIC_READY=1
            break
        fi
        sleep 4
    done

    if [[ "$PUBLIC_READY" == "1" ]]; then
        PUBLIC_URL="$CANDIDATE_URL"
        break
    fi

    log "Public API validation failed. Last response (first 12 lines):"
    head -n 12 "$PUBLIC_CHECK_FILE" 2>/dev/null || true
    log "Cloudflare tunnel log (last 80 lines):"
    tail -n 80 "$TUNNEL_LOG" >&2 || true
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""
    sleep 3
done

if [[ -z "$PUBLIC_URL" ]]; then
    log "Public tunnel failed validation. The script will not label an unverified URL as working."
    log "Local API: $LOCAL_API_URL"
    log "API log: $API_LOG"
    log "Tunnel stderr log: $TUNNEL_LOG"
    log "Tunnel stdout log: $TUNNEL_STDOUT"
    fail "Cloudflare Quick Tunnel did not serve the expected OpenAPI JSON with HTTP 200."
fi

# ============================================================
# 13. Print only a URL that passed public HTTP/OpenAPI validation
# ============================================================

printf '\n============================================================\n'
printf 'VOXCPM API + AUDIO MIXER — PUBLIC ENDPOINT VERIFIED\n'
printf '============================================================\n'
printf 'Public API: %s\n' "$PUBLIC_URL"
printf 'Swagger UI: %s/docs\n' "$PUBLIC_URL"
printf 'OpenAPI:    %s/openapi.json\n' "$PUBLIC_URL"
printf 'Health:     %s/api/v1/health\n' "$PUBLIC_URL"
printf 'Clone+Mix:  %s/api/v1/voice/clone-and-mix\n' "$PUBLIC_URL"
printf 'Local API:  %s\n' "$LOCAL_API_URL"
printf 'API log:    %s\n' "$API_LOG"
printf 'Tunnel log: %s\n' "$TUNNEL_LOG"
printf '\nPublic OpenAPI returned HTTP 200 and matched the VoxCPM API.\n'
printf 'Keep this cell running; the Quick Tunnel URL is temporary.\n'
printf 'The API key is stored in .env and was not printed.\n'
printf '============================================================\n\n'

# Keep the Colab cell alive while the tunnel runs and stream useful logs.
tail -n 0 -F "$API_LOG" "$TUNNEL_LOG" "$TUNNEL_STDOUT" &
TAIL_PID=$!
wait "$TUNNEL_PID"
