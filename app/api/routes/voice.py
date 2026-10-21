import asyncio
import time
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import FileResponse
from app.audio_mixer.api import service as audio_mixer_service
from app.core.security import require_api_key
from app.services.voice.clone_service import VoiceCloneService

router = APIRouter(prefix="/voice", tags=["Voice Cloning"])

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
    retry_max_times: int = Form(3),
    retry_ratio_threshold: float = Form(6.0),
    auto_transcribe: bool = Form(False),
    silence_ms: int = Form(350),
):
    result = await VoiceCloneService().clone(
        reference_audio, prompt_text, target_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        candidates=candidates,
        seed=seed,
        retry_badcase=retry_badcase,
        retry_max_times=retry_max_times,
        retry_ratio_threshold=retry_ratio_threshold,
        auto_transcribe=auto_transcribe,
        reference_mode=reference_mode,
        chunk_long_text="؟" in (target_text or ""),
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
    retry_max_times: int = Form(3),
    retry_ratio_threshold: float = Form(6.0),
    auto_transcribe: bool = Form(False),
    silence_ms: int = Form(350),
    source_gain_db: float = Form(0),
    noise_gain_db: float = Form(-18),
    variation_seed: int | None = Form(None),
    output_format: str | None = Form(None),
):
    request_started = time.perf_counter()
    if not -60 <= source_gain_db <= 6 or not -60 <= noise_gain_db <= 6:
        raise HTTPException(status_code=422, detail="Gain must be between -60 and +6 dB")
    # Reject invalid output settings and oversized/empty noise before spending GPU time.
    audio_mixer_service.resolve_output_format(output_format)
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
            chunk_long_text="؟" in (target_text or ""),
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
            pass  # The background job already owns its input bytes.
        elif "result" in locals():
            Path(result["output"]).unlink(missing_ok=True)
        raise
    logger = __import__("logging").getLogger(__name__)
    logger.info("voice_stage request_id=%s stage=noise_ready_and_clone seconds=%.4f", result["request_id"], time.perf_counter() - request_started)

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
