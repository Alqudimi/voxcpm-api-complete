import inspect
import asyncio
import logging
import threading
import time
import uuid
import numpy as np
import soundfile as sf
from app.core.config import settings
from app.core.model_manager import model_manager
from app.utils.audio import (
    normalize_peak, prepare_reference, reference_quality_report,
    normalize_iraqi_target, convert_to_ogg,
    split_text_for_generation, concatenate_with_silence,
)

logger = logging.getLogger(__name__)
_generation_lock = threading.Lock()
_generate_parameters = None

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
                      seed, retry_badcase, retry_max_times, retry_ratio_threshold,
                      reference_mode="hifi"):
        model = model_manager.voxcpm
        global _generate_parameters
        if _generate_parameters is None:
            _generate_parameters = inspect.signature(model.generate).parameters
        supported = set(_generate_parameters.keys())

        kwargs = {
            "text": target_text,
            "reference_wav_path": str(reference_path),
            "cfg_value": float(cfg),
            "inference_timesteps": int(timesteps),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_max_times),
            "retry_badcase_ratio_threshold": float(retry_ratio_threshold),
        }
        if reference_mode == "hifi":
            kwargs["prompt_wav_path"] = str(reference_path)
            kwargs["prompt_text"] = prompt_text
        elif reference_mode != "reference":
            raise ValueError("reference_mode must be hifi or reference")

        if "seed" in supported:
            kwargs["seed"] = int(seed)
        else:
            import torch
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if "kwargs" not in _generate_parameters:
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

    async def _generate_nano(self, target_text, prompt_text, reference_path, cfg,
                             seed, retry_badcase, retry_max_times,
                             retry_ratio_threshold, reference_mode):
        if reference_mode not in {"hifi", "reference"}:
            raise ValueError("reference_mode must be hifi or reference")
        reference_bytes = await asyncio.to_thread(reference_path.read_bytes)
        latent_bytes = await model_manager.encode_reference_latents(reference_bytes)
        kwargs = {
            "target_text": target_text,
            "prompt_latents": latent_bytes if reference_mode == "hifi" else None,
            "prompt_text": prompt_text or "",
            "ref_audio_latents": latent_bytes,
            "cfg_value": float(cfg),
        }
        # The upstream model caps generation at target_token_count * ratio + 10.
        # Character count is a conservative upper bound for tokenizer length; the
        # hard cap is configurable to prevent runaway decoding and context OOM.
        max_length = min(
            settings.NANO_MAX_GENERATE_LENGTH,
            max(16, int(len(target_text) * float(retry_ratio_threshold) + 10)),
        )
        kwargs["max_generate_length"] = max_length
        audio = None
        attempts = max(1, int(retry_max_times) + 1) if retry_badcase else 1
        for attempt in range(attempts):
            chunks = []
            async for chunk in model_manager.voxcpm.generate(
                **kwargs, seed=int(seed) + attempt
            ):
                value = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if value.size:
                    chunks.append(value)
            if not chunks:
                raise RuntimeError("NanoVLLM returned no audio chunks")
            audio = np.concatenate(chunks)
            # NanoVLLM yields one waveform chunk per generated latent token. Hitting
            # the same text-relative ceiling used by VoxCPM is the observable sign
            # of a likely length badcase; retry with a deterministic new seed.
            badcase = len(chunks) >= int(len(target_text) * float(retry_ratio_threshold))
            if not badcase or attempt + 1 >= attempts:
                if badcase:
                    logger.warning(
                        "nano_badcase_retry_exhausted text_chars=%d output_tokens=%d attempts=%d",
                        len(target_text), len(chunks), attempts,
                    )
                break
            logger.warning(
                "nano_badcase_retry text_chars=%d output_tokens=%d attempt=%d",
                len(target_text), len(chunks), attempt + 1,
            )
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_peak(audio)

    async def clone(self, reference_audio, prompt_text, target_text, **options):
        request_started = time.perf_counter()
        if not model_manager.ready:
            raise RuntimeError("AI models are not ready.")

        reference_clean = settings.TEMP_DIR
        import os
        from pathlib import Path
        Path(reference_clean).mkdir(parents=True, exist_ok=True)
        raw = Path(reference_clean) / f"{uuid.uuid4().hex}_raw"
        clean = Path(reference_clean) / f"{uuid.uuid4().hex}_reference.wav"
        reference_bytes = await reference_audio.read(settings.max_file_size + 1)
        if not reference_bytes:
            raise ValueError("reference_audio is empty")
        if len(reference_bytes) > settings.max_file_size:
            raise ValueError("reference_audio exceeds the configured upload limit")
        await asyncio.to_thread(raw.write_bytes, reference_bytes)

        prep_started = time.perf_counter()
        def prepare_and_report():
            prepare_reference(raw, clean, settings.REFERENCE_SAMPLE_RATE)
            reference_duration = float(sf.info(clean).duration)
            if reference_duration > settings.MAX_REFERENCE_DURATION_SECONDS:
                raw.unlink(missing_ok=True)
                clean.unlink(missing_ok=True)
                raise ValueError(
                    f"reference_audio must be <= {settings.MAX_REFERENCE_DURATION_SECONDS} seconds"
                )
            reference_pcm, reference_sr = sf.read(clean, dtype="float32", always_2d=False)
            return reference_quality_report(clean, reference_pcm, reference_sr)
        qc = await asyncio.to_thread(prepare_and_report)
        logger.info("voice_stage request_id=%s stage=reference_prep seconds=%.4f", raw.stem, time.perf_counter() - prep_started)

        reference_mode = str(options.get("reference_mode") or settings.REFERENCE_MODE).lower()
        if reference_mode not in {"auto", "hifi", "reference"}:
            raise ValueError("reference_mode must be auto, hifi, or reference")
        if reference_mode == "auto":
            reference_mode = "hifi" if (prompt_text and prompt_text.strip()) or options.get("auto_transcribe") else "reference"
        if reference_mode == "hifi" and (not prompt_text or not prompt_text.strip()) and options.get("auto_transcribe"):
            from app.services.transcription.transcription_service import TranscriptionService
            prompt_text = await TranscriptionService().transcribe_file(clean)
        if reference_mode == "hifi" and (not prompt_text or not prompt_text.strip()):
            raise ValueError("prompt_text is required unless auto_transcribe=true")
        if reference_mode == "reference":
            prompt_text = None
        elif prompt_text:
            prompt_text = prompt_text.strip()
        if not target_text or not target_text.strip():
            raise ValueError("target_text is required")

        target_text = normalize_iraqi_target(target_text)
        if len(target_text) > settings.MAX_TARGET_TEXT_CHARS:
            raise ValueError(
                f"target_text exceeds {settings.MAX_TARGET_TEXT_CHARS} characters"
            )
        if prompt_text and len(prompt_text) > settings.MAX_PROMPT_TEXT_CHARS:
            raise ValueError(
                f"prompt_text exceeds {settings.MAX_PROMPT_TEXT_CHARS} characters"
            )
        base_seed = int(options.get("seed") if options.get("seed") is not None else 12345)
        count = int(options.get("candidates", settings.DEFAULT_CANDIDATES))
        if not 1 <= count <= 8:
            raise ValueError("candidates must be between 1 and 8")

        ranked = []
        early_mix_job_id = None
        model_sr = int(
            model_manager.sample_rate
            or getattr(getattr(model_manager.voxcpm, "tts_model", None), "sample_rate", settings.OUTPUT_SAMPLE_RATE)
        )
        if model_manager.tts_backend == "nanovllm":
            request_timesteps = int(options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS))
            if request_timesteps != settings.DEFAULT_TIMESTEPS:
                raise ValueError(
                    "NanoVLLM uses the startup DEFAULT_TIMESTEPS setting; change it in .env and restart"
                )
        chunk_long_text = bool(options.get("chunk_long_text", False))
        silence_ms = int(options.get("silence_ms", 350))

        # Determine segments: split at ؟ when feature is enabled
        if chunk_long_text or len(target_text) > settings.MAX_TARGET_CHUNK_CHARS:
            chunks = split_text_for_generation(
                target_text, settings.MAX_TARGET_CHUNK_CHARS
            )
            if not chunks:
                chunks = [target_text]
        else:
            chunks = [target_text]

        logger.info("Text chunks (%d): %s", len(chunks), chunks)

        generation_started = time.perf_counter()
        for index in range(count):
            candidate_seed = base_seed + index * 1009

            # Generate each segment independently, then join
            segment_wavs = []
            for seg_index, chunk in enumerate(chunks):
                seg_seed = candidate_seed + seg_index
                if model_manager.tts_backend == "nanovllm":
                    wav_seg = await self._generate_nano(
                        chunk,
                        prompt_text,
                        clean,
                        options.get("cfg_value", settings.DEFAULT_CFG),
                        seg_seed,
                        options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE),
                        options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES),
                        options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD),
                        reference_mode,
                    )
                else:
                    def generate_segment():
                        import torch
                        with _generation_lock, torch.inference_mode():
                            return self._generate_one(
                                chunk, prompt_text, clean,
                                options.get("cfg_value", settings.DEFAULT_CFG),
                                options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS),
                                seg_seed,
                                options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE),
                                options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES),
                                options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD),
                                reference_mode,
                            )
                    wav_seg = await asyncio.to_thread(generate_segment)
                segment_wavs.append(wav_seg)

            wav = concatenate_with_silence(segment_wavs, model_sr, silence_ms=silence_ms)
            wav = normalize_peak(wav)
            candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
            await asyncio.to_thread(sf.write, candidate_wav, wav, model_sr, subtype="PCM_24")
            callback = options.get("on_candidate_ready")
            if callback is not None and count == 1:
                # Launch mixing from lossless PCM before ECAPA scoring completes.
                try:
                    early_mix_job_id = await callback(candidate_wav)
                except Exception:
                    candidate_wav.unlink(missing_ok=True)
                    raise
            similarity = await asyncio.to_thread(self._speaker_score, clean, candidate_wav)
            ranked.append({
                "path": candidate_wav,
                "score": similarity["score"],
                "same_speaker": similarity["same_speaker"],
                "seed": candidate_seed,
                "duration": len(wav) / model_sr,
            })
        logger.info("voice_stage request_id=%s stage=generation seconds=%.4f candidates=%d", raw.stem, time.perf_counter() - generation_started, count)

        best = max(ranked, key=lambda x: x["score"])
        output = best["path"] if early_mix_job_id else Path(settings.OUTPUT_DIR) / f"{uuid.uuid4().hex}.{'wav' if options.get('preserve_pcm', False) else 'ogg'}"
        try:
            if early_mix_job_id:
                pass  # Callback already submitted the lossless candidate to the mixer.
            elif options.get("preserve_pcm", False):
                audio_best, sr_best = sf.read(best["path"], dtype="float32", always_2d=False)
                sf.write(output, audio_best, sr_best, subtype="PCM_24")
            else:
                convert_to_ogg(best["path"], output, settings.OUTPUT_BITRATE, settings.OUTPUT_SAMPLE_RATE)
        finally:
            for candidate in ranked:
                candidate["path"].unlink(missing_ok=True)

        logger.info("voice_stage request_id=%s stage=clone_total seconds=%.4f", raw.stem, time.perf_counter() - request_started)
        raw.unlink(missing_ok=True)
        clean.unlink(missing_ok=True)
        return {
            "request_id": output.stem,
            "output": output,
            "mix_job_id": early_mix_job_id,
            "prompt_text": prompt_text,
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
