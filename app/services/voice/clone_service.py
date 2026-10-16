import inspect
import uuid
import numpy as np
import soundfile as sf
from app.core.config import settings
from app.core.model_manager import model_manager
from app.utils.audio import (
    normalize_peak, prepare_reference, reference_quality_report,
    normalize_iraqi_target, convert_to_ogg,
    split_text_at_question_marks, concatenate_with_silence,
)

class VoiceCloneService:
    def _speaker_score(self, reference_path, generated_path):
        score, prediction = model_manager.speaker_verification.verify_files(
            str(reference_path), str(generated_path)
        )
        return {
            "score": float(score.squeeze().cpu().item()),
            "same_speaker": int(prediction.squeeze().cpu().item()) == 1,
        }

    def _generate_one(self, target_text, prompt_text, reference_path, cfg, timesteps,
                      seed, retry_badcase, retry_max_times, retry_ratio_threshold):
        model = model_manager.voxcpm
        signature = inspect.signature(model.generate)
        supported = set(signature.parameters.keys())

        kwargs = {
            "text": target_text,
            "prompt_wav_path": str(reference_path),
            "prompt_text": prompt_text,
            "reference_wav_path": str(reference_path),
            "cfg_value": float(cfg),
            "inference_timesteps": int(timesteps),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_max_times),
            "retry_badcase_ratio_threshold": float(retry_ratio_threshold),
        }

        if "seed" in supported:
            kwargs["seed"] = int(seed)
        else:
            import torch
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if "kwargs" not in signature.parameters:
            kwargs = {k: v for k, v in kwargs.items() if k in supported}

        if "text" not in kwargs:
            raise RuntimeError("CRITICAL: text was removed before model.generate().")

        wav = model.generate(**kwargs)
        if wav is None:
            raise RuntimeError("VoxCPM2 returned None.")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            raise RuntimeError("VoxCPM2 returned empty audio.")
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_peak(wav)

    async def clone(self, reference_audio, prompt_text, target_text, **options):
        if not model_manager.ready:
            raise RuntimeError("AI models are not ready.")

        reference_clean = settings.TEMP_DIR
        import os
        from pathlib import Path
        Path(reference_clean).mkdir(parents=True, exist_ok=True)
        raw = Path(reference_clean) / f"{uuid.uuid4().hex}_raw"
        clean = Path(reference_clean) / f"{uuid.uuid4().hex}_reference.wav"
        raw.write_bytes(await reference_audio.read())

        prepare_reference(raw, clean, settings.REFERENCE_SAMPLE_RATE)
        qc = reference_quality_report(clean)

        if options.get("auto_transcribe"):
            from app.services.transcription.transcription_service import TranscriptionService
            prompt_text = await TranscriptionService().transcribe_file(clean)
        if not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_text is required unless auto_transcribe=true")
        if not target_text or not target_text.strip():
            raise ValueError("target_text is required")

        target_text = normalize_iraqi_target(target_text)
        base_seed = int(options.get("seed") if options.get("seed") is not None else 12345)
        count = 1

        ranked = []
        model_sr = int(getattr(model_manager.voxcpm.tts_model, "sample_rate", settings.OUTPUT_SAMPLE_RATE))
        chunk_long_text = bool(options.get("chunk_long_text", False))
        silence_ms = int(options.get("silence_ms", 350))

        # Determine segments: split at ؟ when feature is enabled
        if chunk_long_text:
            chunks = split_text_at_question_marks(target_text)
            if not chunks:
                chunks = [target_text]
        else:
            chunks = [target_text]

        import logging
        logger = logging.getLogger(__name__)
        logger.info("Text chunks (%d): %s", len(chunks), chunks)

        for index in range(count):
            candidate_seed = base_seed + index * 1009

            # Generate each segment independently, then join
            segment_wavs = []
            for seg_index, chunk in enumerate(chunks):
                seg_seed = candidate_seed + seg_index
                wav_seg = self._generate_one(
                    chunk, prompt_text.strip(), clean,
                    options.get("cfg_value", settings.DEFAULT_CFG),
                    options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS),
                    seg_seed,
                    options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE),
                    options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES),
                    options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD),
                )
                segment_wavs.append(wav_seg)

            wav = concatenate_with_silence(segment_wavs, model_sr, silence_ms=silence_ms)
            wav = normalize_peak(wav)
            candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
            sf.write(candidate_wav, wav, model_sr, subtype="PCM_24")
            similarity = self._speaker_score(clean, candidate_wav)
            ranked.append({
                "path": candidate_wav,
                "score": similarity["score"],
                "same_speaker": similarity["same_speaker"],
                "seed": candidate_seed,
                "duration": len(wav) / model_sr,
            })

        best = max(ranked, key=lambda x: x["score"])
        output = Path(settings.OUTPUT_DIR) / f"{uuid.uuid4().hex}.ogg"
        convert_to_ogg(best["path"], output, settings.OUTPUT_BITRATE, settings.OUTPUT_SAMPLE_RATE)

        return {
            "request_id": output.stem,
            "output": output,
            "prompt_text": prompt_text.strip(),
            "target_text": target_text,
            "seed": best["seed"],
            "duration": best["duration"],
            "speaker_score": best["score"],
            "same_speaker": best["same_speaker"],
            "reference_qc": qc,
            "chunks_count": len(chunks),
            "ranking": [
                {
                    "rank": i + 1,
                    "seed": x["seed"],
                    "score": x["score"],
                    "same_speaker": x["same_speaker"],
                    "duration": x["duration"],
                }
                for i, x in enumerate(sorted(ranked, key=lambda x: x["score"], reverse=True))
            ],
        }
