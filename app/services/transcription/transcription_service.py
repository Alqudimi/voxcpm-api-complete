import asyncio
import torch
from app.core.model_manager import model_manager
from app.core.config import settings
from app.utils.audio import convert_to_wav

class TranscriptionService:
    async def transcribe(self, upload_file):
        from pathlib import Path
        import uuid
        raw = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}_input"
        wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
        data = await upload_file.read()
        if not data:
            raise ValueError("Audio file is empty.")
        if len(data) > settings.max_file_size:
            raise ValueError("Audio file exceeds the maximum allowed size.")
        try:
            await asyncio.to_thread(raw.write_bytes, data)
            await asyncio.to_thread(
                convert_to_wav, raw, wav, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS
            )
            return await self.transcribe_file(wav)
        finally:
            raw.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)

    async def transcribe_file(self, wav_path):
        await model_manager.ensure_whisper_loaded()
        return await asyncio.to_thread(self._transcribe_file_sync, str(wav_path))

    @staticmethod
    def _transcribe_file_sync(wav_path):
        if model_manager.whisper_backend == "faster-whisper":
            segments, _info = model_manager.whisper.transcribe(
                wav_path,
                language="ar",
                task="transcribe",
                vad_filter=settings.WHISPER_VAD,
                temperature=0,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        result = model_manager.whisper.transcribe(
            wav_path,
            language="ar",
            task="transcribe",
            fp16=torch.cuda.is_available(),
            temperature=0,
            verbose=False,
        )
        return result.get("text", "").strip()
