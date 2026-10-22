import asyncio
import json
import logging
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from .database import PlatformDatabase

logger = logging.getLogger("voxcpm.platform_sync")


class PlatformSyncService:
    """خدمة المزامنة الذكية مع المنصة الخارجية:
    1. تسجيل دومين الخادم الحالي ومفتاح API في إعدادات المنصة برمجياً.
    2. تحميل كافة أصوات العينات وأصوات الضوضاء مع نصوصها وتخزينها في SQLite محلياً.
    """

    def __init__(self, db: Optional[PlatformDatabase] = None):
        self.db = db or PlatformDatabase(settings.PLATFORM_DB_PATH)
        self.voice_samples_dir = Path(settings.VOICE_SAMPLES_DIR)
        self.noise_samples_dir = Path(settings.NOISE_SAMPLES_DIR)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.noise_samples_dir.mkdir(parents=True, exist_ok=True)

    def detect_server_url(self) -> str:
        """اكتشاف رابط الخادم العام الذي يعمل منه التطبيق تلقائياً وبأعلى دقة."""
        # 1. إعداد صريح في الإعدادات أو متغيرات البيئة
        explicit_url = (
            getattr(settings, "SERVER_PUBLIC_URL", "")
            or os.getenv("SERVER_PUBLIC_URL")
            or os.getenv("PUBLIC_URL")
            or os.getenv("SERVER_URL")
            or os.getenv("APP_URL")
        )
        if explicit_url and str(explicit_url).strip():
            return str(explicit_url).strip().rstrip("/")

        # 2. بيئة RunPod السحابية (معرّف Pod)
        pod_id = os.getenv("RUNPOD_POD_ID")
        if pod_id:
            # منفذ 8000 الافتراضي لخدمة VoxCPM على بروكسي RunPod
            return f"https://{pod_id.strip()}-8000.proxy.runpod.net"

        # 3. روابط الأنفاق المؤقتة (Cloudflare Tunnel / ngrok) إن وجدت
        tunnel_url = os.getenv("TUNNEL_URL") or os.getenv("CLOUDFLARE_URL")
        if tunnel_url:
            return str(tunnel_url).strip().rstrip("/")

        # 4. الرابط الافتراضي المحلي
        port = os.getenv("PORT", "8000")
        return f"http://localhost:{port}"

    def get_platform_base_url(self) -> str:
        """جلب الرابط الأساسي للمنصة الخارجية."""
        url = (
            getattr(settings, "PLATFORM_BASE_URL", "")
            or os.getenv("PLATFORM_BASE_URL")
            or os.getenv("VOICE_CLONING_PLATFORM_URL")
            or os.getenv("MAIN_SITE_URL")
            or ""
        )
        return str(url).strip().rstrip("/")

    # ─────────────────────────────────────────────────────────────
    # أولاً: تسجيل رابط الخادم والمفتاح في المنصة (Register Server Settings)
    # ─────────────────────────────────────────────────────────────

    def register_server_settings_sync(self) -> Dict[str, Any]:
        """إرسال الدومين ومفتاح API إلى نقطة نهاية المنصة:
        POST /api/voice-cloning/server-settings
        """
        base_url = self.get_platform_base_url()
        if not base_url:
            logger.info("تخطي تسجيل إعدادات الخادم: لم يتم تعيين PLATFORM_BASE_URL في .env")
            return {
                "success": False,
                "message": "PLATFORM_BASE_URL is not configured",
                "skipped": True,
            }

        server_url = self.detect_server_url()
        api_key = settings.API_KEY
        timeout_seconds = int(getattr(settings, "PLATFORM_TIMEOUT", 600))

        endpoint_url = f"{base_url}/api/voice-cloning/server-settings"
        payload = {
            "server_url": server_url,
            "api_key": api_key,
            "timeout": timeout_seconds,
        }

        logger.info(
            "إرسال إعدادات الخادم للمنصة: %s | Server URL: %s",
            endpoint_url,
            server_url,
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"VoxCPM-API/{settings.APP_VERSION}",
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            endpoint_url,
            data=data_bytes,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                logger.info("تم تسجيل إعدادات الخادم بنجاح في المنصة: %s", resp_data.get("message"))
                now_iso = datetime.now(timezone.utc).isoformat()
                self.db.set_meta("server_registered_at", now_iso)
                self.db.set_meta("last_registered_server_url", server_url)
                return {"success": True, "data": resp_data, "server_url": server_url}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            logger.warning("خطأ HTTP أثناء تسجيل الخادم (%s): %s", e.code, err_body)
            return {"success": False, "status_code": e.code, "error": err_body}
        except Exception as e:
            logger.warning("تعذر الاتصال بالمنصة لتسجيل الخادم: %s", e)
            return {"success": False, "error": str(e)}

    async def register_server_settings(self) -> Dict[str, Any]:
        """الغلاف اللاتزامني لتسجيل إعدادات الخادم."""
        return await asyncio.to_thread(self.register_server_settings_sync)

    # ─────────────────────────────────────────────────────────────
    # ثانياً: تحميل وتخزين الأصوات وعينات الضوضاء في SQLite
    # ─────────────────────────────────────────────────────────────

    def _download_file(self, url: str, target_path: Path) -> Optional[Path]:
        """تحميل ملف صوتي من رابط وحفظه محلياً مع تخطي التحميل إذا كان موجوداً مسبقاً."""
        if not url:
            return None

        # إذا كان الرابط نسبياً، نضيف الرابط الأساسي للمنصة
        if url.startswith("/"):
            base_url = self.get_platform_base_url()
            if base_url:
                url = f"{base_url}{url}"

        # إذا كان الملف موجوداً مسبقاً وبحجم سليم (> 1KB) نتخطى إعادة تنزيله
        if target_path.exists() and target_path.stat().st_size > 1024:
            return target_path

        temp_target = target_path.with_suffix(f"{target_path.suffix}.tmp")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": f"VoxCPM-API/{settings.APP_VERSION}",
                "Accept": "*/*",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60.0) as response:
                with open(temp_target, "wb") as f:
                    shutil.copyfileobj(response, f)
            temp_target.replace(target_path)
            return target_path
        except Exception as exc:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            logger.warning("فشل تنزيل الملف الصوتي من (%s): %s", url, exc)
            return None

    def _determine_extension(self, sample_dict: Dict[str, Any], default: str = ".wav") -> str:
        orig = sample_dict.get("original_filename") or ""
        suffix = Path(orig).suffix.lower()
        if suffix in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
            return suffix
        mime = sample_dict.get("mime_type") or ""
        if "mpeg" in mime or "mp3" in mime:
            return ".mp3"
        if "ogg" in mime:
            return ".ogg"
        if "wav" in mime:
            return ".wav"
        return default

    def sync_voices_and_noise_sync(self, force: bool = False) -> Dict[str, Any]:
        """جلب شجرة الأصوات والضوضاء، تحميل ملفاتها وحفظها في SQLite محلياً."""
        base_url = self.get_platform_base_url()
        if not base_url:
            logger.info("تخطي مزامنة الأصوات والضوضاء: لم يتم تعيين PLATFORM_BASE_URL في .env")
            return {
                "success": False,
                "message": "PLATFORM_BASE_URL is not configured",
                "skipped": True,
            }

        # فحص إن كانت البيانات موجودة مسبقاً ولم يُطلب فرض التحديث
        if not force and self.db.has_data():
            logger.info("البيانات مخزنة مسبقاً في SQLite، سيتم التحقق من اكتمال الملفات الصوتية...")

        # مسارات محتملة لجلب الأصوات بحسب توثيق المنصة
        candidate_paths = [
            "/api/voice-cloning/voices-and-noise",
            "/api/voices-and-noise",
            "/api/voice-cloning/resources",
        ]

        data = None
        last_error = None

        for path in candidate_paths:
            endpoint_url = f"{base_url}{path}"
            logger.info("محاولة جلب الموارد من: %s", endpoint_url)
            req = urllib.request.Request(
                endpoint_url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": f"VoxCPM-API/{settings.APP_VERSION}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    if resp_json.get("success") or "data" in resp_json:
                        data = resp_json.get("data", resp_json)
                        break
            except Exception as e:
                last_error = str(e)
                logger.debug("فشل المسار %s: %s", path, e)
                continue

        if not data:
            logger.warning("تعذر جلب الأصوات والضوضاء من المنصة: %s", last_error)
            return {"success": False, "error": last_error}

        voices = data.get("voices", [])
        noise_profiles = data.get("noise_profiles", [])
        local_files_map = {}

        logger.info(
            "تم استلام %d صوت و %d ملف ضوضاء من المنصة. بدء فحص وتحميل الملفات الصوتية...",
            len(voices),
            len(noise_profiles),
        )

        # 1. فحص وتحميل عينات الأصوات
        downloaded_voices = 0
        for voice in voices:
            for sample in voice.get("samples", []):
                s_uuid = str(sample.get("uuid"))
                ext = self._determine_extension(sample, default=".wav")

                local_audio_path = None
                local_cleaned_audio_path = None

                # تنزيل الصوت الأساسي
                audio_url = sample.get("audio_url")
                if audio_url:
                    target = self.voice_samples_dir / f"{s_uuid}{ext}"
                    res = self._download_file(audio_url, target)
                    if res:
                        local_audio_path = str(res)
                        downloaded_voices += 1

                # تنزيل الصوت المنقى إن وجد
                cleaned_url = sample.get("cleaned_audio_url")
                if cleaned_url and cleaned_url != audio_url:
                    target_cleaned = self.voice_samples_dir / f"{s_uuid}_cleaned{ext}"
                    res_c = self._download_file(cleaned_url, target_cleaned)
                    if res_c:
                        local_cleaned_audio_path = str(res_c)

                local_files_map[s_uuid] = {
                    "local_audio_path": local_audio_path,
                    "local_cleaned_audio_path": local_cleaned_audio_path,
                }

        # 2. فحص وتحميل عينات الضوضاء
        downloaded_noise = 0
        for profile in noise_profiles:
            for sample in profile.get("samples", []):
                s_uuid = str(sample.get("uuid"))
                ext = self._determine_extension(sample, default=".mp3")

                local_audio_path = None
                audio_url = sample.get("audio_url")
                if audio_url:
                    target = self.noise_samples_dir / f"{s_uuid}{ext}"
                    res = self._download_file(audio_url, target)
                    if res:
                        local_audio_path = str(res)
                        downloaded_noise += 1

                local_files_map[s_uuid] = {
                    "local_audio_path": local_audio_path,
                }

        # 3. حفظ البيانات في SQLite
        self.db.save_voices_and_noise(voices, noise_profiles, local_files_map)

        summary = self.db.get_summary()
        logger.info(
            "اكتملت المزامنة بنجاح! تم حفظ البيانات في SQLite (%s). إجمالي الأصوات: %d، إجمالي عينات الضوضاء: %d",
            summary.get("database_path"),
            summary.get("total_voices"),
            summary.get("total_noise_samples"),
        )

        return {
            "success": True,
            "message": "تمت مزامنة وحفظ جميع الموارد محلياً بنجاح.",
            "summary": summary,
        }

    async def sync_voices_and_noise(self, force: bool = False) -> Dict[str, Any]:
        """الغلاف اللاتزامني لمزامنة الأصوات والضوضاء."""
        return await asyncio.to_thread(self.sync_voices_and_noise_sync, force)

    async def run_startup_workflow(self) -> None:
        """تشغيل دورة المزامنة الكاملة عند إقلاع الخادم لأول مرة."""
        try:
            # 1. إرسال الدومين ومفتاح API للمنصة
            if getattr(settings, "PLATFORM_REGISTER_ON_STARTUP", True):
                await self.register_server_settings()

            # 2. مزامنة وتحميل عينات الأصوات والضوضاء
            if getattr(settings, "PLATFORM_SYNC_ON_STARTUP", True):
                await self.sync_voices_and_noise()
        except Exception as exc:
            logger.exception("حدث استثناء أثناء إجراء المزامنة الأولية عند الإقلاع: %s", exc)


platform_sync_service = PlatformSyncService()
