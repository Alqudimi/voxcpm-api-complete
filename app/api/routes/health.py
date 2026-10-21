from fastapi import APIRouter
from app.core.model_manager import model_manager

router = APIRouter(tags=["System"])

@router.get("/health")
async def health():
    return {
        "success": True,
        "status": "healthy" if model_manager.ready else "starting",
        "models": {
            "voxcpm2": model_manager.voxcpm is not None,
            "whisper": model_manager.whisper is not None,
            "ecapa": model_manager.speaker_verification is not None,
        },
    }

@router.get("/")
async def root():
    return {
        "service": "VoxCPM2 Professional Voice API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "autotune": "/api/v1/autotune",
    }

@router.get("/autotune")
async def autotune_report():
    """تقرير المضبوط التلقائي الكامل: العتاد + كل القيم المحسوبة + المعايرة + الكاشات."""
    from app.core.autotune import tuner
    from app.services.voice.result_cache import result_cache
    from app.services.voice.speaker_cache import speaker_state_cache

    report = tuner.report()
    try:
        from app.core.gpu_scheduler import gpu_scheduler

        report["scheduler"] = gpu_scheduler.stats()
    except Exception:
        report["scheduler"] = {}
    try:
        report["caches"] = {
            "result_cache": result_cache.stats(),
            "speaker_cache": speaker_state_cache.stats(),
        }
    except Exception:
        report["caches"] = {}
    report["runtime"] = {
        "precision": getattr(model_manager, "precision_applied", None),
        "compile_mode": getattr(model_manager, "compile_mode", None),
        "quantization": getattr(model_manager, "quantized", None),
        "whisper_backend": getattr(model_manager, "whisper_backend", None),
        "whisper_model": getattr(model_manager, "whisper_model_used", None),
        "models_ready": bool(model_manager.ready),
    }
    return report


@router.post("/autotune/retune")
async def autotune_retune():
    """إعادة فحص العتاد وإعادة حل كل المتغيرات الآن (يحفظ بروفايل جديداً).

    ملاحظة: قرارات التحميل الثقيلة (dtype/compile/quant) تُطبق عند الإقلاع
    التالي — قيم التزامن والكاش تتحدث فوراً."""
    from app.core.autotune import tuner

    tuner.ensure(force_probe=True)
    return {"success": True, "message": "re-probed and re-solved; heavy load decisions apply on restart", "report": tuner.report()}
