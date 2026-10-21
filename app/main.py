import os
from importlib.util import find_spec

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, ORJSONResponse

load_dotenv()

from app.core.config import settings

if settings.API_KEY in {"change-me", "replace-with-a-long-random-secret"} or len(settings.API_KEY) < 32:
    raise RuntimeError("Set a unique API_KEY of at least 32 characters in .env before starting the service")

from app.api.routes import audio, health, transcription, voice
from app.audio_mixer.api import include_audio_mixer
from app.core.lifecycle import lifespan
from app.core.logging import configure_logging

os.environ.setdefault("AUDIO_MIXER_API_KEY", settings.API_KEY)

configure_logging()
settings.ensure_directories()
response_class = ORJSONResponse if find_spec("orjson") else JSONResponse

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Professional API for VoxCPM2 voice cloning, Arabic transcription, "
        "speaker similarity scoring, audio merging and OGG/Opus output."
    ),
    default_response_class=response_class,
    lifespan=lifespan,
)
include_audio_mixer(app)
app.include_router(health.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(transcription.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
