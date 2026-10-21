import os

# v2-ultra: إعداد مخصص الذاكرة على CUDA يجب أن يسبق أول استيراد لـtorch
# (expandable_segments يقلص التجزؤ ويرفع الإنتاجية 5-15% على أحجام دفعات متغيرة).
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI

from app.api.routes import audio, health, transcription, voice
from app.audio_mixer.api import include_audio_mixer
from app.core.config import settings
from app.core.lifecycle import lifespan
from app.core.logging import configure_logging

os.environ.setdefault("AUDIO_MIXER_API_KEY", settings.API_KEY)

configure_logging()
settings.ensure_directories()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Professional API for VoxCPM2 voice cloning, Arabic transcription, "
        "speaker similarity scoring, audio merging and OGG/Opus output."
    ),
    lifespan=lifespan,
)
include_audio_mixer(app)
app.include_router(health.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(transcription.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
