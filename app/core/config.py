from pathlib import Path
import shutil
import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "VoxCPM2 Professional Voice API"
    APP_VERSION: str = "1.0.0"
    API_KEY: str = "change-me"

    VOXCPM_MODEL_ID: str = "openbmb/VoxCPM2"
    TTS_ENGINE: str = "nanovllm"
    NANO_DEVICE_INDEX: int = Field(default=0, ge=0)
    NANO_MAX_NUM_SEQS: int = Field(default=0, ge=0, le=64)
    NANO_MAX_NUM_BATCHED_TOKENS: int = Field(default=8192, ge=256, le=32768)
    NANO_MAX_MODEL_LEN: int = Field(default=4096, ge=512, le=16384)
    NANO_GPU_MEMORY_UTILIZATION: float = Field(default=0.0, ge=0.0, le=0.98)
    NANO_MAX_GENERATE_LENGTH: int = Field(default=2000, ge=16, le=8192)
    MAX_TARGET_TEXT_CHARS: int = Field(default=1600, ge=1, le=16000)
    MAX_TARGET_CHUNK_CHARS: int = Field(default=350, ge=64, le=1000)
    MAX_PROMPT_TEXT_CHARS: int = Field(default=500, ge=1, le=4000)
    MAX_REFERENCE_DURATION_SECONDS: int = Field(default=20, ge=1, le=60)
    WHISPER_MODEL_NAME: str = "large-v3"
    WHISPER_BACKEND: str = "openai-whisper"
    WHISPER_PRELOAD: bool = False
    WHISPER_COMPUTE_TYPE: str = "int8_float16"
    WHISPER_VAD: bool = False
    REFERENCE_MODE: str = "auto"
    ECAPA_SOURCE: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_SAVEDIR: str = "storage/models/ecapa_voxceleb"

    DEVICE: str = "cuda"
    VOXCPM_OPTIMIZE: bool = True
    VOXCPM_WARMUP: bool = True
    VOXCPM_WARMUP_TEXT: str = "مرحباً بك في الخدمة."
    REFERENCE_SAMPLE_RATE: int = 16000
    OUTPUT_SAMPLE_RATE: int = 48000
    AUDIO_SAMPLE_RATE: int = 24000
    AUDIO_CHANNELS: int = 1

    MAX_FILE_SIZE_MB: int = 100
    MAX_MERGE_FILES: int = 20

    DEFAULT_CFG: float = 2.0
    DEFAULT_TIMESTEPS: int = 30
    DEFAULT_CANDIDATES: int = 1
    DEFAULT_RETRY_BADCASE: bool = True
    DEFAULT_RETRY_MAX_TIMES: int = 3
    DEFAULT_RETRY_RATIO_THRESHOLD: float = 6.0

    OUTPUT_BITRATE: str = "96k"
    OUTPUT_CODEC: str = "libopus"

    STORAGE_ROOT: str = "storage"
    UPLOAD_DIR: str = "storage/uploads"
    OUTPUT_DIR: str = "storage/outputs"
    TEMP_DIR: str = "storage/temp"
    AUDIO_TMPFS_DIR: str = "/dev/shm/voxcpm-tmp"
    AUDIO_TMPFS_FALLBACK_DIR: str = "storage/temp"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_file_size(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def ensure_directories(self):
        requested_tmpfs = Path(self.AUDIO_TMPFS_DIR) if self.AUDIO_TMPFS_DIR else None
        if requested_tmpfs:
            try:
                requested_tmpfs.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(requested_tmpfs).free
                if free_bytes >= max(256 * 1024**2, min(self.max_file_size, 1024**3)):
                    self.TEMP_DIR = str(requested_tmpfs)
                else:
                    logging.getLogger(__name__).warning(
                        "tmpfs has insufficient free space; using disk fallback"
                    )
                    self.TEMP_DIR = self.AUDIO_TMPFS_FALLBACK_DIR
            except OSError:
                logging.getLogger(__name__).warning(
                    "tmpfs is unavailable; using disk fallback"
                )
                self.TEMP_DIR = self.AUDIO_TMPFS_FALLBACK_DIR
        for item in (
            self.STORAGE_ROOT,
            self.UPLOAD_DIR,
            self.OUTPUT_DIR,
            self.TEMP_DIR,
            self.ECAPA_SAVEDIR,
        ):
            Path(item).mkdir(parents=True, exist_ok=True)

settings = Settings()
