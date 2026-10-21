import logging

from app.core.config import settings
from app.core.model_manager import model_manager
from app.utils.audio import convert_to_wav

logger = logging.getLogger(__name__)


class TranscriptionService:
    """خدمة النسخ — محرك مزدوج مع تفضيل faster-whisper int8 (أسرع 3-5×)
    وقص الصمت عبر VAD، والتنفيذ دائماً في خيط منفصل حتى لا يحجب event loop."""

    async def transcribe(self, upload_file):
        from pathlib import Path
        import asyncio
        import uuid

        raw = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}_input"
        wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
        data = await upload_file.read()
        if not data:
            raise ValueError("Audio file is empty.")
        if len(data) > settings.max_file_size:
            raise ValueError("Audio file exceeds the maximum allowed size.")
        raw.write_bytes(data)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: convert_to_wav(raw, wav, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS)
        )
        return await self.transcribe_file(wav)

    def _transcribe_sync(self, wav_path) -> str:
        backend = getattr(model_manager, "whisper_backend", "openai-whisper")
        # v3: حجم دفعة مضبوط تلقائياً من VRAM + فك حتمي greedy (~1.5-2× على المقاطع القصيرة)
        try:
            batch_size = int(settings.whisper_batch_size)
        except Exception:
            batch_size = 8
        greedy = bool(settings.whisper_greedy)
        greedy_kwargs = (
            {"beam_size": 1, "condition_on_previous_text": False, "temperature": 0.0}
            if greedy
            else {}
        )
        if backend == "faster-whisper" and model_manager.whisper is not None:
            # v2: BatchedInferencePipeline — ~3.7× موثق على الملفات الطويلة
            batched = getattr(model_manager, "whisper_batched", None)
            if batched is not None:
                try:
                    segments, _info = batched.transcribe(
                        str(wav_path),
                        language="ar",
                        task="transcribe",
                        vad_filter=settings.whisper_vad,
                        vad_parameters={"min_silence_duration_ms": 300},
                        batch_size=batch_size,
                        **greedy_kwargs,
                    )
                    return "".join(segment.text for segment in segments).strip()
                except Exception as exc:
                    logger.warning("Batched whisper failed (single mode): %s", exc)
            segments, _info = model_manager.whisper.transcribe(
                str(wav_path),
                language="ar",
                task="transcribe",
                vad_filter=settings.whisper_vad,
                vad_parameters={"min_silence_duration_ms": 300},
                **greedy_kwargs,
            )
            return "".join(segment.text for segment in segments).strip()
        # openai-whisper (المسار الأصلي)
        import torch

        result = model_manager.whisper.transcribe(
            str(wav_path),
            language="ar",
            task="transcribe",
            fp16=torch.cuda.is_available(),
            temperature=0,
            verbose=False,
            condition_on_previous_text=False,  # المقاطع القصيرة لا تحتاج حمل السياق (أسرع)
            **({"beam_size": 1} if greedy else {}),
        )
        return result.get("text", "").strip()

    async def transcribe_file(self, wav_path):
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._transcribe_sync(wav_path))
