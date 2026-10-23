import asyncio
import json
import logging
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.config import settings
from .database import PlatformDatabase
from .fast_resource_cache import FastNoiseSample, FastVoiceSample, fast_resource_cache

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
        # تحميل الموارد مسبقاً في ذاكرة RAM الفورية (Fast Cache) إن وُجدت
        try:
            from .fast_resource_cache import fast_resource_cache
            if self.db.has_data() and not fast_resource_cache.is_loaded:
                fast_resource_cache.warm_up(self.db.db_path)
        except Exception as exc:
            logger.warning("تعذر تسخين كاش الذاكرة السريعة في __init__: %s", exc)

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

    def normalize_platform_url(self, url: str) -> str:
        """تصحيح الرابط وتحويله ليعتمد على رابط المنصة الأساسي (تجاوز الدومينات الافتراضية كـ example.com)."""
        if not url:
            return ""
        url = str(url).strip()
        base_url = self.get_platform_base_url()
        if not base_url:
            return url

        base_url = base_url.rstrip("/")
        if url.startswith("/"):
            return f"{base_url}{url}"

        parsed = urllib.parse.urlparse(url)
        base_parsed = urllib.parse.urlparse(base_url)

        # إذا كان الرابط يحتوي على دومين مختلف أو افتراضي مثل example.com أو localhost،
        # نوجهه دائماً إلى رابط المنصة الفعلي في PLATFORM_BASE_URL
        if parsed.netloc != base_parsed.netloc:
            return urllib.parse.urlunparse((
                base_parsed.scheme,
                base_parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ))
        return url

    def _download_file(self, url: str, target_path: Path) -> Optional[Path]:
        """تحميل ملف صوتي من رابط وحفظه محلياً مع تخطي التحميل إذا كان موجوداً مسبقاً."""
        if not url:
            return None

        # 1. إذا كان الملف موجوداً مسبقاً وبحجم سليم (> 1KB) نتخطى إعادة تنزيله
        if target_path.exists() and target_path.stat().st_size > 1024:
            return target_path

        # 2. تطبيع الرابط
        norm_url = self.normalize_platform_url(url)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target_path.with_suffix(f"{target_path.suffix}.tmp")

        headers = {
            "User-Agent": f"VoxCPM-API/{settings.APP_VERSION}",
            "Accept": "*/*",
        }

        # محاولة التنزيل
        for attempt_url in [norm_url, url]:
            if not attempt_url:
                continue
            req = urllib.request.Request(attempt_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30.0) as response:
                    with open(temp_target, "wb") as f:
                        shutil.copyfileobj(response, f)
                if temp_target.exists() and temp_target.stat().st_size > 512:
                    temp_target.replace(target_path)
                    return target_path
            except Exception as exc:
                if temp_target.exists():
                    temp_target.unlink(missing_ok=True)
                logger.debug("فشلت محاولة تنزيل من (%s): %s", attempt_url, exc)

        logger.warning("فشل تنزيل الملف الصوتي نهائياً من (%s)", norm_url)
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
        """جلب شجرة الأصوات والضوضاء، حفظ البيانات الوصفية فوراً في SQLite، وتحميل ملفاتها بالتوازي."""
        base_url = self.get_platform_base_url()
        if not base_url:
            logger.info("تخطي مزامنة الأصوات والضوضاء: لم يتم تعيين PLATFORM_BASE_URL في .env")
            return {
                "success": False,
                "message": "PLATFORM_BASE_URL is not configured",
                "skipped": True,
            }

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

        # الخطوة 1: الحفظ الفوري للبيانات الوصفية في SQLite لتكون المعرفات جاهزة لحظياً
        self.db.save_voices_and_noise(voices, noise_profiles, {})
        logger.info(
            "⚡ تم حفظ البيانات الوصفية لـ %d صوت و %d ملف ضوضاء في SQLite فورياً. بدء التنزيل المتوازي...",
            len(voices),
            len(noise_profiles),
        )

        # الخطوة 2: جمع كافة مهام التنزيل لتنفيذها بالتوازي
        def _download_voice_task(voice_dict: dict, sample_dict: dict) -> Tuple[str, Optional[str], Optional[str]]:
            s_uuid = str(sample_dict.get("uuid"))
            ext = self._determine_extension(sample_dict, default=".wav")
            target = self.voice_samples_dir / f"{s_uuid}{ext}"
            audio_url = sample_dict.get("audio_url") or f"{base_url}/api/voice-cloning/audio/voice-sample/{s_uuid}"

            local_audio = None
            res = self._download_file(audio_url, target)
            if res and res.exists():
                local_audio = str(res)
                # تسجيل فوري في RAM
                try:
                    s_copy = dict(sample_dict)
                    s_copy.setdefault("person_name", voice_dict.get("name", ""))
                    s_copy.setdefault("person_id", voice_dict.get("id"))
                    s_copy.setdefault("person_uuid", voice_dict.get("uuid"))
                    data_bytes = res.read_bytes()
                    fast_resource_cache.register_voice_data(s_copy, data_bytes, local_audio)
                    self.db.update_voice_sample_path(s_uuid, local_audio, is_cleaned=False)
                except Exception as ex:
                    logger.debug("خطأ أثناء تسجيل العينة %s في RAM: %s", s_uuid, ex)

            # تنزيل النسخة المنقاة إن وجدت
            cleaned_url = sample_dict.get("cleaned_audio_url")
            local_cleaned = None
            if cleaned_url and cleaned_url != audio_url:
                target_cleaned = self.voice_samples_dir / f"{s_uuid}_cleaned{ext}"
                res_c = self._download_file(cleaned_url, target_cleaned)
                if res_c and res_c.exists():
                    local_cleaned = str(res_c)
                    self.db.update_voice_sample_path(s_uuid, local_cleaned, is_cleaned=True)

            return s_uuid, local_audio, local_cleaned

        def _download_noise_task(profile_dict: dict, sample_dict: dict) -> Tuple[str, Optional[str]]:
            s_uuid = str(sample_dict.get("uuid"))
            ext = self._determine_extension(sample_dict, default=".mp3")
            target = self.noise_samples_dir / f"{s_uuid}{ext}"
            audio_url = sample_dict.get("audio_url") or f"{base_url}/api/voice-cloning/audio/noise-sample/{s_uuid}"

            local_audio = None
            res = self._download_file(audio_url, target)
            if res and res.exists():
                local_audio = str(res)
                try:
                    s_copy = dict(sample_dict)
                    s_copy.setdefault("noise_profile_name", profile_dict.get("name", ""))
                    s_copy.setdefault("noise_profile_id", profile_dict.get("id"))
                    s_copy.setdefault("noise_profile_uuid", profile_dict.get("uuid"))
                    data_bytes = res.read_bytes()
                    fast_resource_cache.register_noise_data(s_copy, data_bytes, local_audio)
                    self.db.update_noise_sample_path(s_uuid, local_audio)
                except Exception as ex:
                    logger.debug("خطأ أثناء تسجيل الضوضاء %s في RAM: %s", s_uuid, ex)

            return s_uuid, local_audio

        # تنفيذ التنزيل بالتوازي بواسطة ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            v_futures = []
            for voice in voices:
                for sample in voice.get("samples", []):
                    v_futures.append(executor.submit(_download_voice_task, voice, sample))

            n_futures = []
            for profile in noise_profiles:
                for sample in profile.get("samples", []):
                    n_futures.append(executor.submit(_download_noise_task, profile, sample))

            for f in as_completed(v_futures + n_futures):
                try:
                    f.result()
                except Exception as exc:
                    logger.warning("استثناء أثناء مهمة تنزيل وسائط: %s", exc)

        # الخطوة 3: تسخين نهائي للذاكرة السريعة
        try:
            fast_resource_cache.warm_up(self.db.db_path)
        except Exception as exc:
            logger.warning("تعذر تسخين كاش الذاكرة السريعة بعد المزامنة: %s", exc)

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

    def ensure_voice_sample_audio(self, identifier: Any) -> Optional[FastVoiceSample]:
        """ضمان توفر الملف الصوتي لعينة الصوت في الذاكرة الحية فورياً (On-Demand Fetch)."""
        if identifier is None or not str(identifier).strip():
            return None

        val = str(identifier).strip()
        # 1. البحث في RAM أولاً
        item = fast_resource_cache.get_voice(int(val) if val.isdigit() else val)
        if item is not None and item.raw_audio_bytes:
            return item

        # 2. البحث في SQLite
        sample = self.db.get_voice_sample(val)
        if not sample:
            # إذا لم تكن موجودة، نجري مزامنة فورية ونعيد الاستعلام
            self.sync_voices_and_noise_sync(force=False)
            sample = self.db.get_voice_sample(val)
            if not sample:
                return None

        s_uuid = str(sample["uuid"])
        ext = self._determine_extension(sample, default=".wav")
        target_path = self.voice_samples_dir / f"{s_uuid}{ext}"

        # 3. إذا كان الملف موجوداً بالفعل على القرص
        local_path_str = sample.get("local_cleaned_audio_path") or sample.get("local_audio_path")
        disk_file = Path(local_path_str) if local_path_str and Path(local_path_str).is_file() else None
        if not disk_file and target_path.is_file() and target_path.stat().st_size > 512:
            disk_file = target_path

        if disk_file:
            data = disk_file.read_bytes()
            return fast_resource_cache.register_voice_data(sample, data, str(disk_file))

        # 4. تنزيل الملف لحظياً من المنصة
        base_url = self.get_platform_base_url()
        candidates = []
        if sample.get("cleaned_audio_url"):
            candidates.append(sample["cleaned_audio_url"])
        if sample.get("audio_url"):
            candidates.append(sample["audio_url"])
        if base_url:
            candidates.append(f"{base_url}/api/voice-cloning/audio/cleaned-sample/{s_uuid}")
            candidates.append(f"{base_url}/api/voice-cloning/audio/voice-sample/{s_uuid}")

        for u in candidates:
            res = self._download_file(u, target_path)
            if res and res.is_file() and res.stat().st_size > 512:
                data = res.read_bytes()
                self.db.update_voice_sample_path(s_uuid, str(res))
                return fast_resource_cache.register_voice_data(sample, data, str(res))

        return None

    def ensure_noise_sample_audio(self, identifier: Any) -> Optional[FastNoiseSample]:
        """ضمان توفر الملف الصوتي لعينة الضوضاء في الذاكرة الحية فورياً (On-Demand Fetch)."""
        if identifier is None or not str(identifier).strip():
            return None

        val = str(identifier).strip()
        item = fast_resource_cache.get_noise(int(val) if val.isdigit() else val)
        if item is not None and item.audio_bytes:
            return item

        sample = self.db.get_noise_sample(val)
        if not sample:
            self.sync_voices_and_noise_sync(force=False)
            sample = self.db.get_noise_sample(val)
            if not sample:
                return None

        s_uuid = str(sample["uuid"])
        ext = self._determine_extension(sample, default=".mp3")
        target_path = self.noise_samples_dir / f"{s_uuid}{ext}"

        local_path_str = sample.get("local_audio_path")
        disk_file = Path(local_path_str) if local_path_str and Path(local_path_str).is_file() else None
        if not disk_file and target_path.is_file() and target_path.stat().st_size > 512:
            disk_file = target_path

        if disk_file:
            data = disk_file.read_bytes()
            return fast_resource_cache.register_noise_data(sample, data, str(disk_file))

        base_url = self.get_platform_base_url()
        candidates = []
        if sample.get("audio_url"):
            candidates.append(sample["audio_url"])
        if base_url:
            candidates.append(f"{base_url}/api/voice-cloning/audio/noise-sample/{s_uuid}")

        for u in candidates:
            res = self._download_file(u, target_path)
            if res and res.is_file() and res.stat().st_size > 512:
                data = res.read_bytes()
                self.db.update_noise_sample_path(s_uuid, str(res))
                return fast_resource_cache.register_noise_data(sample, data, str(res))

        return None

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
