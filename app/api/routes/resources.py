from fastapi import APIRouter, Depends, Query
from app.core.security import require_api_key
from app.services.platform.platform_sync_service import platform_sync_service

router = APIRouter(prefix="/resources", tags=["Platform Resources"])


@router.get("/summary", dependencies=[Depends(require_api_key)])
async def get_resources_summary():
    """عرض إحصائيات الموارد المخزنة محلياً في SQLite وحالة الاتصال بالمنصة والذاكرة RAM."""
    from app.services.platform.fast_resource_cache import fast_resource_cache
    summary = platform_sync_service.db.get_summary()
    return {
        "success": True,
        "platform_base_url": platform_sync_service.get_platform_base_url(),
        "detected_server_url": platform_sync_service.detect_server_url(),
        "summary": summary,
        "ram_cache_stats": fast_resource_cache.stats(),
    }


@router.get("/voices", dependencies=[Depends(require_api_key)])
async def get_local_voices():
    """استعراض قائمة كافة الأصوات والعينات الصوتية المخزنة محلياً في SQLite."""
    voices = platform_sync_service.db.list_voices()
    return {
        "success": True,
        "count": len(voices),
        "voices": voices,
    }


@router.get("/noise", dependencies=[Depends(require_api_key)])
async def get_local_noise_profiles():
    """استعراض قائمة كافة ملفات وعينات الضوضاء المخزنة محلياً في SQLite."""
    profiles = platform_sync_service.db.list_noise_profiles()
    return {
        "success": True,
        "count": len(profiles),
        "noise_profiles": profiles,
    }


@router.post("/sync", dependencies=[Depends(require_api_key)])
async def trigger_platform_sync(force: bool = Query(False, description="إعادة تنزيل وفحص كافة الملفات قسراً")):
    """إعادة تشغيل مزامنة الأصوات والضوضاء وتحميل الملفات الناقصة من المنصة يدوياً."""
    result = await platform_sync_service.sync_voices_and_noise(force=force)
    return result


@router.post("/register-server", dependencies=[Depends(require_api_key)])
async def trigger_server_registration():
    """إعادة إرسال دومين هذا الخادم الحالي ومفتاح API إلى نقطة نهاية المنصة الخارجية."""
    result = await platform_sync_service.register_server_settings()
    return result
