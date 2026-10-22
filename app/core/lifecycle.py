from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.model_manager import model_manager
from app.core.tts_engine import ensure_engine_loaded, reset_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load_all()
    # إحماء عند الإقلاع (VM_WARMUP_ON_BOOT): تمريرة توليد قصيرة تسخّن النوى
    # وتُفعّل التجميع — أول طلب حقيقي يقاس بزمن الطلب لا بزمن بارد.
    # v3: كل تمريرة تُقاس RTF وتُعايَر في بروفايل المضبوط التلقائي.
    if settings.warmup_on_boot:
        await model_manager.warmup()
    # v3-max: ضمان تحميل المحرك البديل المختار تلقائياً (NanoVLLM) —
    # مع إصلاح فجوة v2: أي فشل تحميل = إعادة بناء فورية على الرسمي.
    try:
        await ensure_engine_loaded()
    except Exception as exc:
        # السقوط الأخير (غير متوقع): واصل بخدمة الرسمي إن أمكن
        import logging

        logging.getLogger(__name__).warning("ensure_engine_loaded fallback: %s", exc)

    # مزامنة المنصة الخارجية: تسجيل إعدادات الخادم + تحميل وحفظ العينات والضوضاء في SQLite
    try:
        import asyncio
        from app.services.platform.platform_sync_service import platform_sync_service

        asyncio.create_task(platform_sync_service.run_startup_workflow())
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("platform_sync_service startup failed: %s", exc)

    yield
    await model_manager.unload_all()
    reset_engine()
