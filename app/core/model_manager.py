import asyncio
import hashlib
import logging
import threading
from collections import OrderedDict

import torch
import whisper
from voxcpm import VoxCPM
from speechbrain.inference.speaker import SpeakerRecognition

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.voxcpm = None
        self.tts_backend = settings.TTS_ENGINE.lower()
        self.sample_rate = settings.OUTPUT_SAMPLE_RATE
        self.nano_max_num_seqs = None
        self.nano_gpu_memory_utilization = None
        self.nano_generation_semaphore = None
        self.whisper = None
        self.whisper_backend = settings.WHISPER_BACKEND.lower()
        self.speaker_verification = None
        self.ready = False
        self._whisper_lock = threading.Lock()
        self._latent_lock = threading.Lock()
        self._reference_latent_cache = OrderedDict()

    async def load_all(self):
        if self.ready:
            return
        settings.ensure_directories()
        if self.tts_backend == "nanovllm":
            if not torch.cuda.is_available():
                raise RuntimeError("NanoVLLM requires an available NVIDIA CUDA GPU; select TTS_ENGINE=pytorch to use the fallback backend")
            if settings.NANO_DEVICE_INDEX >= torch.cuda.device_count():
                raise RuntimeError(
                    f"NANO_DEVICE_INDEX={settings.NANO_DEVICE_INDEX} is outside the visible CUDA device range"
                )
            free_bytes, total_bytes = torch.cuda.mem_get_info(settings.NANO_DEVICE_INDEX)
            total_gib = total_bytes / 1024**3
            free_gib = free_bytes / 1024**3
            if free_gib < 6:
                raise RuntimeError(
                    f"NanoVLLM needs more free GPU memory to start; only {free_gib:.2f} GiB is available. "
                    "Close other GPU workloads or use a GPU with more VRAM."
                )
            if total_gib <= 12 or free_gib < 9:
                auto_seqs, auto_util = 4, 0.70
            elif total_gib <= 24 or free_gib < 18:
                auto_seqs, auto_util = 8, 0.78
            else:
                auto_seqs, auto_util = 16, 0.85
            max_seqs = settings.NANO_MAX_NUM_SEQS or auto_seqs
            safe_fraction = max(0.20, min(0.90, (free_gib - 1.5) / total_gib))
            memory_utilization = settings.NANO_GPU_MEMORY_UTILIZATION or min(auto_util, safe_fraction)
            self.nano_max_num_seqs = max_seqs
            self.nano_gpu_memory_utilization = memory_utilization
            self.nano_generation_semaphore = asyncio.Semaphore(max_seqs)
            try:
                from nanovllm_voxcpm import VoxCPM as NanoVoxCPM
            except ImportError as exc:
                raise RuntimeError(
                    "TTS_ENGINE=nanovllm requires nano-vllm-voxcpm and its CUDA/FlashAttention dependencies; "
                    "use TTS_ENGINE=pytorch to fall back to standard VoxCPM"
                ) from exc
            self.voxcpm = NanoVoxCPM.from_pretrained(
                model=settings.VOXCPM_MODEL_ID,
                devices=[settings.NANO_DEVICE_INDEX],
                inference_timesteps=settings.DEFAULT_TIMESTEPS,
                max_num_batched_tokens=settings.NANO_MAX_NUM_BATCHED_TOKENS,
                max_num_seqs=max_seqs,
                max_model_len=settings.NANO_MAX_MODEL_LEN,
                gpu_memory_utilization=memory_utilization,
            )
            await self.voxcpm.wait_for_ready()
            model_info = await self.voxcpm.get_model_info()
            self.sample_rate = int(model_info["sample_rate"])
            logger.info(
                "NanoVLLM VoxCPM2 ready device=%d sample_rate=%d max_seqs=%d gpu_memory_utilization=%.2f",
                settings.NANO_DEVICE_INDEX,
                self.sample_rate,
                max_seqs,
                memory_utilization,
            )
            if settings.VOXCPM_WARMUP:
                logger.info("Warming NanoVLLM VoxCPM2 request scheduler")
                async for _chunk in self.voxcpm.generate(
                    target_text=settings.VOXCPM_WARMUP_TEXT,
                    cfg_value=settings.DEFAULT_CFG,
                    max_generate_length=min(128, settings.NANO_MAX_GENERATE_LENGTH),
                    seed=0,
                ):
                    pass
        elif self.tts_backend == "pytorch":
            logger.info("Loading VoxCPM2: %s", settings.VOXCPM_MODEL_ID)
            self.voxcpm = VoxCPM.from_pretrained(
                settings.VOXCPM_MODEL_ID,
                load_denoiser=False,
                device=settings.DEVICE,
                optimize=settings.VOXCPM_OPTIMIZE,
            )
            self.sample_rate = int(
                getattr(self.voxcpm.tts_model, "sample_rate", settings.OUTPUT_SAMPLE_RATE)
            )
            logger.info(
                "VoxCPM2 loaded device=%s optimize=%s dtype=%s",
                settings.DEVICE,
                settings.VOXCPM_OPTIMIZE,
                getattr(getattr(self.voxcpm.tts_model, "config", None), "dtype", "model-default"),
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
        else:
            raise ValueError("TTS_ENGINE must be nanovllm or pytorch")

        if settings.WHISPER_PRELOAD:
            self._load_whisper_sync()
        else:
            logger.info("Whisper lazy loading enabled; it loads only for transcription requests")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading ECAPA speaker verification on %s", device)
        self.speaker_verification = SpeakerRecognition.from_hparams(
            source=settings.ECAPA_SOURCE,
            savedir=settings.ECAPA_SAVEDIR,
            run_opts={"device": device},
        )
        self.ready = True
        logger.info("Required models ready; tts_backend=%s", self.tts_backend)

    async def encode_reference_latents(self, wav_bytes: bytes) -> bytes:
        if self.tts_backend != "nanovllm":
            raise RuntimeError("Reference latent encoding is only available in NanoVLLM mode")
        key = hashlib.sha256(wav_bytes).digest()
        with self._latent_lock:
            cached = self._reference_latent_cache.get(key)
            if cached is not None:
                self._reference_latent_cache.move_to_end(key)
                return cached
        latent_bytes = await self.voxcpm.encode_latents(wav_bytes, "wav")
        with self._latent_lock:
            self._reference_latent_cache[key] = latent_bytes
            self._reference_latent_cache.move_to_end(key)
            while len(self._reference_latent_cache) > 32:
                self._reference_latent_cache.popitem(last=False)
        return latent_bytes

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
                        "WHISPER_BACKEND=faster-whisper requires faster-whisper dependencies"
                    ) from exc
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = settings.WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"
                self.whisper = WhisperModel(
                    settings.WHISPER_MODEL_NAME,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("faster-whisper loaded device=%s compute_type=%s", device, compute_type)
            elif self.whisper_backend == "openai-whisper":
                self.whisper = whisper.load_model(settings.WHISPER_MODEL_NAME)
            else:
                raise ValueError("WHISPER_BACKEND must be openai-whisper or faster-whisper")
        return self.whisper

    async def ensure_whisper_loaded(self):
        return await asyncio.to_thread(self._load_whisper_sync)

    async def unload_all(self):
        if self.tts_backend == "nanovllm" and self.voxcpm is not None:
            await self.voxcpm.stop()
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self._reference_latent_cache.clear()
        self.nano_generation_semaphore = None
        self.ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()
