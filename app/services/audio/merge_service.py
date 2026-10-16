from pathlib import Path
from app.core.config import settings
from app.utils.audio import convert_to_wav, merge_audio_files, convert_to_ogg

class AudioMergeService:
    async def merge(self, files):
        if len(files) < 2:
            raise ValueError("At least two audio files are required.")
        if len(files) > settings.MAX_MERGE_FILES:
            raise ValueError(f"Maximum is {settings.MAX_MERGE_FILES} files.")

        normalized = []
        for upload in files:
            data = await upload.read()
            if not data:
                raise ValueError(f"Empty file: {upload.filename}")
            if len(data) > settings.max_file_size:
                raise ValueError(f"File too large: {upload.filename}")
            raw = Path(settings.TEMP_DIR) / f"{Path(upload.filename or 'audio').stem}_{id(upload)}"
            raw.write_bytes(data)
            wav = raw.with_suffix(".wav")
            convert_to_wav(raw, wav, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS)
            normalized.append(wav)

        merged_wav = Path(settings.TEMP_DIR) / f"merged_{id(files)}.wav"
        merge_audio_files(normalized, merged_wav, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS)

        output = Path(settings.OUTPUT_DIR) / f"merged_{id(files)}.ogg"
        convert_to_ogg(merged_wav, output, settings.OUTPUT_BITRATE, settings.OUTPUT_SAMPLE_RATE)
        return output
