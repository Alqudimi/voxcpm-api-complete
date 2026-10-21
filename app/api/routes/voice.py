import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.audio_mixer.api import service as audio_mixer_service
from app.core.config import settings
from app.core.security import require_api_key
from app.services.voice.clone_service import VoiceCloneService

router = APIRouter(prefix="/voice", tags=["Voice Cloning"])
logger = logging.getLogger(__name__)


def _validate_request_text(
    target_text: str,
    prompt_text: str | None,
    retry_badcase: bool,
    reference_mode: str | None = None,
    auto_transcribe: bool = False,
    inference_timesteps: int | None = None,
):
    if not target_text.strip() or len(target_text) > settings.MAX_TARGET_TEXT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"target_text must contain 1..{settings.MAX_TARGET_TEXT_CHARS} characters",
        )
    if prompt_text and len(prompt_text) > settings.MAX_PROMPT_TEXT_CHARS:
        raise HTTPException(status_code=422, detail="prompt_text exceeds the configured limit")
    if not retry_badcase:
        raise HTTPException(status_code=422, detail="retry_badcase must remain enabled for output quality")
    effective_mode = (reference_mode or settings.REFERENCE_MODE).lower()
    if effective_mode not in {"auto", "hifi", "reference"}:
        raise HTTPException(status_code=422, detail="reference_mode must be auto, hifi, or reference")
    if effective_mode == "hifi" and not (prompt_text and prompt_text.strip()) and not auto_transcribe:
        raise HTTPException(status_code=422, detail="hifi mode requires prompt_text or auto_transcribe=true")
    if (
        settings.TTS_ENGINE.lower() == "nanovllm"
        and inference_timesteps is not None
        and inference_timesteps != settings.DEFAULT_TIMESTEPS
    ):
        raise HTTPException(status_code=422, detail="For NanoVLLM, inference_timesteps must equal DEFAULT_TIMESTEPS in .env")


@router.post("/clone", dependencies=[Depends(require_api_key)])
async def clone_voice(
    request: Request,
    reference_audio: UploadFile = File(...),
    target_text: str = Form(...),
    prompt_text: str | None = Form(None),
    reference_mode: str | None = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(30),
    candidates: int = Form(1, ge=1, le=8),
    seed: int | None = Form(None),
    retry_badcase: bool = Form(True),
    retry_max_times: int = Form(3, ge=1, le=3),
    retry_ratio_threshold: float = Form(6.0, gt=0, le=20),
    auto_transcribe: bool = Form(False),
    silence_ms: int = Form(350, ge=0, le=2000),
):
    _validate_request_text(
        target_text, prompt_text, retry_badcase, reference_mode, auto_transcribe, inference_timesteps
    )
    result = await VoiceCloneService().clone(
        reference_audio,
        prompt_text,
        target_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        candidates=candidates,
        seed=seed,
        retry_badcase=retry_badcase,
        retry_max_times=retry_max_times,
        retry_ratio_threshold=retry_ratio_threshold,
        auto_transcribe=auto_transcribe,
        reference_mode=reference_mode,
        chunk_long_text=any(mark in target_text for mark in "؟?!"),
        silence_ms=silence_ms,
    )
    base = str(request.base_url).rstrip("/")
    return {
        "success": True,
        "request_id": result["request_id"],
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_url": f"{base}/api/v1/voice/audio/{Path(result['output']).name}",
        "prompt_text": result["prompt_text"],
        "seed": result["seed"],
        "duration": result["duration"],
        "speaker_score": result["speaker_score"],
        "same_speaker": result["same_speaker"],
        "reference_qc": result["reference_qc"],
        "chunks_count": result["chunks_count"],
        "ranking": result["ranking"],
    }


@router.post("/clone-and-mix", status_code=202, dependencies=[Depends(require_api_key)])
async def clone_voice_and_mix(
    request: Request,
    reference_audio: UploadFile = File(...),
    target_text: str = Form(...),
    noise_audio: UploadFile = File(...),
    prompt_text: str | None = Form(None),
    reference_mode: str | None = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(30),
    candidates: int = Form(1, ge=1, le=8),
    seed: int | None = Form(None),
    retry_badcase: bool = Form(True),
    retry_max_times: int = Form(3, ge=1, le=3),
    retry_ratio_threshold: float = Form(6.0, gt=0, le=20),
    auto_transcribe: bool = Form(False),
    silence_ms: int = Form(350, ge=0, le=2000),
    source_gain_db: float = Form(0),
    noise_gain_db: float = Form(-18),
    variation_seed: int | None = Form(None),
    output_format: str | None = Form(None),
):
    request_started = time.perf_counter()
    _validate_request_text(
        target_text, prompt_text, retry_badcase, reference_mode, auto_transcribe, inference_timesteps
    )
    if not -60 <= source_gain_db <= 6 or not -60 <= noise_gain_db <= 6:
        raise HTTPException(status_code=422, detail="Gain must be between -60 and +6 dB")
    audio_mixer_service.resolve_output_format(output_format)
    if not audio_mixer_service.queue_capacity_available():
        raise HTTPException(
            status_code=429,
            detail="Audio mixing queue is full; retry shortly",
            headers={"Retry-After": "2"},
        )
    noise_task = asyncio.create_task(asyncio.to_thread(audio_mixer_service._read_upload, noise_audio))

    async def submit_candidate(candidate_path):
        noise_bytes = await noise_task
        source_bytes = await asyncio.to_thread(candidate_path.read_bytes)
        return await asyncio.to_thread(
            audio_mixer_service.submit_mix_bytes,
            source_bytes=source_bytes,
            source_reference=candidate_path.name,
            noise_bytes=noise_bytes,
            noise_reference=noise_audio.filename or "noise.audio",
            source_gain=source_gain_db,
            noise_gain=noise_gain_db,
            variation_seed=variation_seed,
            output_format=output_format,
            persist_inputs=not audio_mixer_service.settings.memory_job_inputs,
        )

    try:
        result = await VoiceCloneService().clone(
            reference_audio,
            prompt_text,
            target_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            candidates=candidates,
            seed=seed,
            retry_badcase=retry_badcase,
            retry_max_times=retry_max_times,
            retry_ratio_threshold=retry_ratio_threshold,
            auto_transcribe=auto_transcribe,
            reference_mode=reference_mode,
            preserve_pcm=True,
            on_candidate_ready=submit_candidate if candidates == 1 else None,
            chunk_long_text=any(mark in target_text for mark in "؟?!"),
            silence_ms=silence_ms,
        )
    except Exception:
        if not noise_task.done():
            noise_task.cancel()
        else:
            try:
                noise_task.exception()
            except Exception:
                pass
        if "result" in locals() and result.get("mix_job_id"):
            pass
        elif "result" in locals():
            Path(result["output"]).unlink(missing_ok=True)
        raise

    logger.info(
        "voice_stage request_id=%s stage=noise_ready_and_clone seconds=%.4f",
        result["request_id"],
        time.perf_counter() - request_started,
    )
    job_id = result.get("mix_job_id")
    if not job_id:
        cloned_path = Path(result["output"])
        noise_bytes = await noise_task
        source_bytes = await asyncio.to_thread(cloned_path.read_bytes)
        try:
            job_id = await asyncio.to_thread(
                audio_mixer_service.submit_mix_bytes,
                source_bytes=source_bytes,
                source_reference=cloned_path.name,
                noise_bytes=noise_bytes,
                noise_reference=noise_audio.filename or "noise.audio",
                source_gain=source_gain_db,
                noise_gain=noise_gain_db,
                variation_seed=variation_seed,
                output_format=output_format,
                persist_inputs=not audio_mixer_service.settings.memory_job_inputs,
            )
        finally:
            cloned_path.unlink(missing_ok=True)

    base = str(request.base_url).rstrip("/")
    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "status_url": f"{base}/api/v1/audio/jobs/{job_id}/status",
        "events_url": f"{base}/api/v1/audio/jobs/{job_id}/events",
        "result_url": f"{base}/api/v1/audio/jobs/{job_id}/result",
        "clone_info": {
            "request_id": result["request_id"],
            "prompt_text": result["prompt_text"],
            "seed": result["seed"],
            "duration": result["duration"],
            "speaker_score": result["speaker_score"],
            "same_speaker": result["same_speaker"],
            "reference_qc": result["reference_qc"],
            "chunks_count": result["chunks_count"],
            "ranking": result["ranking"],
        },
    }


@router.get("/audio/{filename}", dependencies=[Depends(require_api_key)])
async def get_voice_audio(filename: str):
    path = Path("storage/outputs") / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/ogg", filename=path.name)
