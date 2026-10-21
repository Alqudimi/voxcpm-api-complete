import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 10000
    database_url: str = "sqlite:///./audio_mixer.db"
    storage_path: Path = Path("./storage")
    temp_path: Path = Path("./storage/temp")
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    max_upload_size: int = 256 * 1024 * 1024
    max_audio_duration: float = 1800.0
    max_concurrent_jobs: int = 2
    max_queue_size: int = 8
    memory_job_inputs: bool = False
    min_noise_margin_seconds: float = 2.0
    dataset_path: Path = Path("./data/mixkit_sfx_scraped_raw.json")
    auto_import_mixkit: bool = False
    import_workers: int = 8
    import_timeout: int = 45
    retention_hours: int = 24
    cleanup_interval_seconds: int = 3600
    app_version: str = "3.3.0"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("AUDIO_MIXER_HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", os.getenv("AUDIO_MIXER_PORT", "10000"))),
            database_url=os.getenv(
                "AUDIO_MIXER_DATABASE_URL",
                os.getenv("DATABASE_URL", "sqlite:///./audio_mixer.db"),
            ),
            storage_path=Path(os.getenv("AUDIO_MIXER_STORAGE_PATH", "./storage")),
            temp_path=Path(os.getenv("AUDIO_MIXER_TEMP_PATH", "./storage/temp")),
            ffmpeg_path=os.getenv("AUDIO_MIXER_FFMPEG_PATH", "ffmpeg"),
            ffprobe_path=os.getenv("AUDIO_MIXER_FFPROBE_PATH", "ffprobe"),
            max_upload_size=int(
                os.getenv("AUDIO_MIXER_MAX_UPLOAD_SIZE", str(256 * 1024 * 1024))
            ),
            max_audio_duration=float(
                os.getenv("AUDIO_MIXER_MAX_AUDIO_DURATION", "1800")
            ),
            max_concurrent_jobs=int(os.getenv("AUDIO_MIXER_MAX_CONCURRENT_JOBS", "2")),
            max_queue_size=int(os.getenv("AUDIO_MIXER_MAX_QUEUE_SIZE", "8")),
            memory_job_inputs=os.getenv("AUDIO_MIXER_MEMORY_JOB_INPUTS", "false").lower()
            in {"1", "true", "yes", "on"},
            dataset_path=Path(
                os.getenv(
                    "AUDIO_MIXER_DATASET_PATH", "./data/mixkit_sfx_scraped_raw.json"
                )
            ),
            auto_import_mixkit=os.getenv(
                "AUDIO_MIXER_AUTO_IMPORT_MIXKIT", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            import_workers=int(os.getenv("AUDIO_MIXER_IMPORT_WORKERS", "8")),
            import_timeout=int(os.getenv("AUDIO_MIXER_IMPORT_TIMEOUT", "45")),
            retention_hours=max(1, int(os.getenv("AUDIO_MIXER_RETENTION_HOURS", "24"))),
            cleanup_interval_seconds=max(
                60, int(os.getenv("AUDIO_MIXER_CLEANUP_INTERVAL_SECONDS", "3600"))
            ),
            app_version=os.getenv("AUDIO_MIXER_APP_VERSION", "3.3.0"),
        )

    def prepare(self) -> None:
        self.temp_path.mkdir(parents=True, exist_ok=True)
