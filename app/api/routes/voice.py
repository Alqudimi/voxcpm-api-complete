from pathlib import Path
import asyncio

from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
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
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(30),
    candidates: int = Form(1),
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
        "performance": result.get("performance", {}),
    }

@router.post("/clone-and-mix", status_code=202, dependencies=[Depends(require_api_key)])
async def clone_voice_and_mix(
    request: Request,
    reference_audio: UploadFile = File(...),
    target_text: str = Form(...),
    noise_audio: UploadFile = File(...),
    prompt_text: str | None = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(30),
    candidates: int = Form(1),
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
        chunk_long_text="؟" in (target_text or ""),
        silence_ms=silence_ms,
    )

    cloned_path = Path(result["output"])
    # فك الحجب: قراءة الملفات من القرص/الرفع في خيط منفصل حتى لا يتجمد event loop
    source_bytes = await asyncio.to_thread(cloned_path.read_bytes)
    noise_bytes = await asyncio.to_thread(
        audio_mixer_service._read_upload, noise_audio
    )

    job_id = audio_mixer_service.submit_mix_bytes(
        source_bytes=source_bytes,
        source_reference=cloned_path.name,
        noise_bytes=noise_bytes,
        noise_reference=noise_audio.filename or "noise.audio",
        source_gain=source_gain_db,
        noise_gain=noise_gain_db,
        variation_seed=variation_seed,
        output_format=output_format,
    )

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
            "performance": result.get("performance", {}),
        },
    }

@router.get("/audio/{filename}", dependencies=[Depends(require_api_key)])
async def get_voice_audio(filename: str):
    path = Path("storage/outputs") / filename
    if not path.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/ogg", filename=path.name)

