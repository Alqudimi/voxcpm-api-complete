import logging
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
        self.speaker_verification = None
        self.ready = False

    async def load_all(self):
        if self.ready:
            return

        settings.ensure_directories()

        logger.info("Loading VoxCPM2: %s", settings.VOXCPM_MODEL_ID)
        self.voxcpm = VoxCPM.from_pretrained(
            settings.VOXCPM_MODEL_ID,
            load_denoiser=False,
        )
        logger.info("VoxCPM2 loaded")

        logger.info("Loading Whisper: %s", settings.WHISPER_MODEL_NAME)
        self.whisper = whisper.load_model(settings.WHISPER_MODEL_NAME)
        logger.info("Whisper loaded")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading ECAPA speaker verification on %s", device)
        self.speaker_verification = SpeakerRecognition.from_hparams(
            source=settings.ECAPA_SOURCE,
            savedir=settings.ECAPA_SAVEDIR,
            run_opts={"device": device},
        )
        logger.info("ECAPA loaded")

        self.ready = True
        logger.info("All AI models are ready")

    async def unload_all(self):
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self.ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

model_manager = ModelManager()
