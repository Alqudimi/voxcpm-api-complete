import asyncio
import logging
import threading

import torch
import whisper
from voxcpm import VoxCPM
from speechbrain.inference.speaker import SpeakerRecognition

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.voxcpm = None
        self.whisper = None
        self.whisper_backend = settings.WHISPER_BACKEND.lower()
        self.speaker_verification = None
        self.ready = False
        self._whisper_lock = threading.Lock()

    async def load_all(self):
        if self.ready:
            return

        settings.ensure_directories()
        logger.info("Loading VoxCPM2: %s", settings.VOXCPM_MODEL_ID)
        self.voxcpm = VoxCPM.from_pretrained(
            settings.VOXCPM_MODEL_ID,
            load_denoiser=False,
            device=settings.DEVICE,
            optimize=settings.VOXCPM_OPTIMIZE,
        )
        logger.info(
            "VoxCPM2 loaded device=%s optimize=%s",
            settings.DEVICE,
            settings.VOXCPM_OPTIMIZE,
        )

        if settings.VOXCPM_WARMUP:
            logger.info("Warming VoxCPM2 with configured CFG/timesteps")
            with torch.inference_mode():
                warmup_audio = self.voxcpm.generate(
                    text=settings.VOXCPM_WARMUP_TEXT,
                    cfg_value=settings.DEFAULT_CFG,
                    inference_timesteps=settings.DEFAULT_TIMESTEPS,
                    retry_badcase=True,
                    retry_badcase_max_times=settings.DEFAULT_RETRY_MAX_TIMES,
                    retry_badcase_ratio_threshold=settings.DEFAULT_RETRY_RATIO_THRESHOLD,
                )
            del warmup_audio
            logger.info("VoxCPM2 warm-up complete")

        if settings.WHISPER_PRELOAD:
            self._load_whisper_sync()
        else:
            logger.info("Whisper lazy loading enabled; it will load only for transcription requests")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading ECAPA speaker verification on %s", device)
        self.speaker_verification = SpeakerRecognition.from_hparams(
            source=settings.ECAPA_SOURCE,
            savedir=settings.ECAPA_SAVEDIR,
            run_opts={"device": device},
        )
        logger.info("ECAPA loaded")

        self.ready = True
        logger.info("All required AI models are ready")

    def _load_whisper_sync(self):
        if self.whisper is not None:
            return self.whisper
        with self._whisper_lock:
            if self.whisper is not None:
                return self.whisper
            logger.info("Loading Whisper: %s", settings.WHISPER_MODEL_NAME)
            if self.whisper_backend == "faster-whisper":
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "WHISPER_BACKEND=faster-whisper requires the optional faster-whisper dependency"
                    ) from exc
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = settings.WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"
                self.whisper = WhisperModel(
                    settings.WHISPER_MODEL_NAME,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info(
                    "faster-whisper loaded: device=%s compute_type=%s vad=%s",
                    device,
                    compute_type,
                    settings.WHISPER_VAD,
                )
            elif self.whisper_backend == "openai-whisper":
                self.whisper = whisper.load_model(settings.WHISPER_MODEL_NAME)
            else:
                raise ValueError("WHISPER_BACKEND must be openai-whisper or faster-whisper")
        return self.whisper

    async def ensure_whisper_loaded(self):
        return await asyncio.to_thread(self._load_whisper_sync)

    async def unload_all(self):
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self.ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()
