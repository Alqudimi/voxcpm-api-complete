from pathlib import Path
import asyncio
from typing import Any, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.audio_mixer.api import service as audio_mixer_service
from app.core.security import require_api_key
from app.services.platform.fast_resource_cache import fast_resource_cache
from app.services.platform.platform_sync_service import platform_sync_service
from app.services.voice.clone_service import VoiceCloneService

router = APIRouter(prefix="/voice", tags=["Voice Cloning"])


class VoiceAudioInMemoryWrapper:
    """غلاف فائق السرعة لقراءة بايتات الصوت المرجعي مباشرة من ذاكرة RAM (Zero Disk I/O)."""
    __slots__ = ("_data", "_pos", "filename")

    def __init__(self, data: bytes, filename: str = "reference.wav"):
        self._data = data
        self._pos = 0
        self.filename = filename

    async def read(self, size: int = -1) -> bytes:
        if size == -1 or size is None:
            return self._data
        res = self._data[self._pos: self._pos + size]
        self._pos += len(res)
        return res

    async def seek(self, offset: int) -> int:
        self._pos = offset
        return self._pos


class CloneVoicePayload(BaseModel):
    voice_sample_id: int | str = Field(..., description="معرّف عينة الصوت رقمياً (ID) أو نصياً (UUID) المخزنة في النظام")
    target_text: str = Field(..., description="النص المطلوب تحويله إلى كلام بالصوت المستنسخ")
    prompt_text: Optional[str] = Field(None, description="النص المرجعي (اختياري، يملأ تلقائياً من العينة المخزنة)")
    cfg_value: float = Field(2.0, description="معامل التوجيه")
    inference_timesteps: int = Field(30, description="خطوات الاستدلال")
    candidates: int = Field(1, description="عدد المرشحين")
    seed: Optional[int] = Field(None, description="البذرة العشوائية")
    retry_badcase: bool = Field(True, description="إعادة المحاولة في الحالات الضعيفة")
    retry_max_times: int = Field(3, description="أقصى عدد محاولات")
    retry_ratio_threshold: float = Field(6.0, description="عتبة النسبة")
    auto_transcribe: bool = Field(False, description="تفريغ تلقائي")
    silence_ms: int = Field(350, description="الصمت بين المقاطع")


class CloneAndMixPayload(BaseModel):
    voice_sample_id: int | str = Field(..., description="معرّف عينة الصوت رقمياً (ID) أو نصياً (UUID) المخزنة في النظام")
    noise_sample_id: int | str = Field(..., description="معرّف عينة الضوضاء رقمياً (ID) أو نصياً (UUID) المخزنة في النظام")
    target_text: str = Field(..., description="النص المطلوب تحويله إلى كلام بالصوت المستنسخ")
    prompt_text: Optional[str] = Field(None, description="النص المرجعي (اختياري، يملأ تلقائياً من العينة المخزنة)")
    cfg_value: float = Field(2.0, description="معامل التوجيه")
    inference_timesteps: int = Field(30, description="خطوات الاستدلال")
    candidates: int = Field(1, description="عدد المرشحين")
    seed: Optional[int] = Field(None, description="البذرة العشوائية")
    retry_badcase: bool = Field(True, description="إعادة المحاولة في الحالات الضعيفة")
    retry_max_times: int = Field(3, description="أقصى عدد محاولات")
    retry_ratio_threshold: float = Field(6.0, description="عتبة النسبة")
    auto_transcribe: bool = Field(False, description="تفريغ تلقائي")
    silence_ms: int = Field(350, description="الصمت بين المقاطع")
    source_gain_db: float = Field(0.0, description="مستوى صوت الكلام بالديسيبل")
    noise_gain_db: float = Field(-18.0, description="مستوى صوت الضوضاء بالديسيبل")
    variation_seed: Optional[int] = Field(None, description="بذرة موضع الضوضاء")
    output_format: Optional[str] = Field(None, description="صيغة الإخراج النهائية (wav, mp3, ogg)")


def _resolve_reference_audio(
    uploaded_file: Any | None,
    identifier: Any | None,
    prompt_text: str | None,
) -> Tuple[Any, str | None]:
    """استرجاع فوري لعينة الصوت المرجعية من ذاكرة RAM في زمن شبه صفري (<1 microsecond)."""
    # 1. إذا تم رفع ملف صوتي يدوي
    if uploaded_file is not None and getattr(uploaded_file, "filename", None):
        return uploaded_file, prompt_text

    # 2. البحث الفوري في كاش الذاكرة RAM (Zero-Disk I/O)
    if identifier is not None and str(identifier).strip():
        val = str(identifier).strip()
        item = fast_resource_cache.get_voice(int(val) if val.isdigit() else val)
        if item is not None:
            prompt = prompt_text or item.reference_text
            return VoiceAudioInMemoryWrapper(item.raw_audio_bytes, item.filename), prompt

        # 3. Fallback: قاعدة بيانات SQLite والقرص
        sample = platform_sync_service.db.get_voice_sample(val)
        if sample is not None:
            local_path = sample.get("local_cleaned_audio_path") or sample.get("local_audio_path")
            if local_path and Path(local_path).is_file():
                data = Path(local_path).read_bytes()
                prompt = prompt_text or sample.get("reference_text") or sample.get("sample_text")
                return VoiceAudioInMemoryWrapper(data, Path(local_path).name), prompt
            raise HTTPException(404, f"الملف الصوتي للعينة {identifier} غير متوفر على القرص")
        raise HTTPException(404, f"عينة الصوت غير موجودة: {identifier}")

    raise HTTPException(
        400,
        "يجب تحديد معرّف عينة الصوت voice_sample_id (أو رفع ملف reference_audio)",
    )


def _resolve_noise_bytes(
    uploaded_file: Any | None,
    identifier: Any | None,
) -> Tuple[bytes, str]:
    """استرجاع فوري لملف الضوضاء من ذاكرة RAM في زمن شبه صفري (<1 microsecond)."""
    # 1. إذا تم رفع ملف ضوضاء يدوي
    if uploaded_file is not None and getattr(uploaded_file, "filename", None):
        noise_bytes = audio_mixer_service._read_upload(uploaded_file)
        return noise_bytes, (uploaded_file.filename or "noise.audio")

    # 2. البحث الفوري في كاش الذاكرة RAM (Zero-Disk I/O)
    if identifier is not None and str(identifier).strip():
        val = str(identifier).strip()
        item = fast_resource_cache.get_noise(int(val) if val.isdigit() else val)
        if item is not None:
            return item.audio_bytes, item.filename

        # 3. Fallback: قاعدة بيانات SQLite والقرص
        sample = platform_sync_service.db.get_noise_sample(val)
        if sample is not None:
            local_path = sample.get("local_audio_path")
            if local_path and Path(local_path).is_file():
                data = Path(local_path).read_bytes()
                return data, Path(local_path).name
            raise HTTPException(404, f"الملف الصوتي لعينة الضوضاء {identifier} غير متوفر على القرص")
        raise HTTPException(404, f"عينة الضوضاء غير موجودة: {identifier}")

    raise HTTPException(
        400,
        "يجب تحديد معرّف عينة الضوضاء noise_sample_id (أو رفع ملف noise_audio)",
    )


async def _extract_request_data(request: Request) -> Tuple[dict, Any, Any]:
    """استخراج المعاملات بمرونة سواء كانت مرسلة كـ JSON أو كـ Multipart/Form-data."""
    ct = request.headers.get("content-type", "").lower()
    if "application/json" in ct:
        try:
            body = await request.json()
            return body if isinstance(body, dict) else {}, None, None
        except Exception:
            return {}, None, None

    # قراءة الفورم والملفات إن وجدت
    form = await request.form()
    data = dict(form)
    ref_file = form.get("reference_audio")
    noise_file = form.get("noise_audio")
    return data, ref_file, noise_file


@router.post("/clone", dependencies=[Depends(require_api_key)])
async def clone_voice(
    request: Request,
    reference_audio: UploadFile | None = File(None),
    voice_sample_id: str | None = Form(None),
    reference_sample_uuid: str | None = Form(None),
    target_text: str | None = Form(None),
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
    """استنساخ الصوت: يقبل معرّف العينة المخزنة في RAM (JSON أو Form) بدون رفع أي ملفات، أو ملفاً مرفوعاً."""
    payload, ref_file_req, _ = await _extract_request_data(request)

    # دمج المعاملات
    final_text = payload.get("target_text") or target_text
    if not final_text or not str(final_text).strip():
        raise HTTPException(422, "target_text مطلوب ولا يمكن أن يكون فارغاً")

    sample_id = (
        payload.get("voice_sample_id")
        or payload.get("sample_id")
        or payload.get("voice_id")
        or payload.get("reference_sample_uuid")
        or voice_sample_id
        or reference_sample_uuid
    )
    prompt_val = payload.get("prompt_text") or prompt_text

    ref_audio, prompt = _resolve_reference_audio(
        ref_file_req or reference_audio, sample_id, prompt_val
    )

    cfg = float(payload.get("cfg_value", cfg_value))
    timesteps = int(payload.get("inference_timesteps", inference_timesteps))
    cands = int(payload.get("candidates", candidates))
    seed_raw = payload.get("seed", seed)
    s_val = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
    retry_bc = str(payload.get("retry_badcase", retry_badcase)).lower() in ("true", "1", "yes")
    retry_mt = int(payload.get("retry_max_times", retry_max_times))
    retry_rt = float(payload.get("retry_ratio_threshold", retry_ratio_threshold))
    auto_tr = str(payload.get("auto_transcribe", auto_transcribe)).lower() in ("true", "1", "yes")
    sil_ms = int(payload.get("silence_ms", silence_ms))

    result = await VoiceCloneService().clone(
        ref_audio, prompt, str(final_text),
        cfg_value=cfg,
        inference_timesteps=timesteps,
        candidates=cands,
        seed=s_val,
        retry_badcase=retry_bc,
        retry_max_times=retry_mt,
        retry_ratio_threshold=retry_rt,
        auto_transcribe=auto_tr,
        chunk_long_text="؟" in (str(final_text) or ""),
        silence_ms=sil_ms,
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
    reference_audio: UploadFile | None = File(None),
    voice_sample_id: str | None = Form(None),
    reference_sample_uuid: str | None = Form(None),
    target_text: str | None = Form(None),
    noise_audio: UploadFile | None = File(None),
    noise_sample_id: str | None = Form(None),
    noise_sample_uuid: str | None = Form(None),
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
    """توليد الصوت المستنسخ ودمجه مع الضوضاء فورياً بواسطة IDs المحملة في RAM (JSON أو Form)."""
    payload, ref_file_req, noise_file_req = await _extract_request_data(request)

    final_text = payload.get("target_text") or target_text
    if not final_text or not str(final_text).strip():
        raise HTTPException(422, "target_text مطلوب ولا يمكن أن يكون فارغاً")

    sample_id = (
        payload.get("voice_sample_id")
        or payload.get("sample_id")
        or payload.get("voice_id")
        or payload.get("reference_sample_uuid")
        or voice_sample_id
        or reference_sample_uuid
    )
    noise_id = (
        payload.get("noise_sample_id")
        or payload.get("noise_id")
        or payload.get("noise_sample_uuid")
        or noise_sample_id
        or noise_sample_uuid
    )
    prompt_val = payload.get("prompt_text") or prompt_text

    # 1. استرجاع فوري من RAM للعينات في زمن <1 microsecond
    ref_audio, prompt = _resolve_reference_audio(
        ref_file_req or reference_audio, sample_id, prompt_val
    )
    noise_bytes, noise_ref = _resolve_noise_bytes(
        noise_file_req or noise_audio, noise_id
    )

    cfg = float(payload.get("cfg_value", cfg_value))
    timesteps = int(payload.get("inference_timesteps", inference_timesteps))
    cands = int(payload.get("candidates", candidates))
    seed_raw = payload.get("seed", seed)
    s_val = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
    retry_bc = str(payload.get("retry_badcase", retry_badcase)).lower() in ("true", "1", "yes")
    retry_mt = int(payload.get("retry_max_times", retry_max_times))
    retry_rt = float(payload.get("retry_ratio_threshold", retry_ratio_threshold))
    auto_tr = str(payload.get("auto_transcribe", auto_transcribe)).lower() in ("true", "1", "yes")
    sil_ms = int(payload.get("silence_ms", silence_ms))

    src_gain = float(payload.get("source_gain_db", source_gain_db))
    ns_gain = float(payload.get("noise_gain_db", noise_gain_db))
    var_seed_raw = payload.get("variation_seed", variation_seed)
    var_seed = int(var_seed_raw) if var_seed_raw is not None and str(var_seed_raw).strip() != "" else None
    out_fmt = payload.get("output_format") or output_format

    # 2. تنفيذ التوليد
    result = await VoiceCloneService().clone(
        ref_audio,
        prompt,
        str(final_text),
        cfg_value=cfg,
        inference_timesteps=timesteps,
        candidates=cands,
        seed=s_val,
        retry_badcase=retry_bc,
        retry_max_times=retry_mt,
        retry_ratio_threshold=retry_rt,
        auto_transcribe=auto_tr,
        chunk_long_text="؟" in (str(final_text) or ""),
        silence_ms=sil_ms,
    )

    cloned_path = Path(result["output"])
    source_bytes = await asyncio.to_thread(cloned_path.read_bytes)

    # 3. إرسال مهمة الدمج ببيانات الضوضاء المسترجعة من RAM مباشرة
    job_id = audio_mixer_service.submit_mix_bytes(
        source_bytes=source_bytes,
        source_reference=cloned_path.name,
        noise_bytes=noise_bytes,
        noise_reference=noise_ref,
        source_gain=src_gain,
        noise_gain=ns_gain,
        variation_seed=var_seed,
        output_format=out_fmt,
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
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/ogg", filename=path.name)


