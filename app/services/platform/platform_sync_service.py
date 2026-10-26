import asyncio
import json
import logging
import os
import re
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
        """اكتشاف رابط الخادم العام الذي يعمل منه التطبيق تلقائياً وبأعلى دقة:
        1. الإعدادات الصريحة في .env أو متغيرات البيئة (مثل SERVER_PUBLIC_URL).
        2. بيئة RunPod السحابية (https://{RUNPOD_POD_ID}-{PORT}.proxy.runpod.net).
        3. بيئة Google Colab عبر نفق Cloudflare Tunnel (فحص /content/cloudflared.log واستخراج رابط trycloudflare.com).
        4. ملفات مؤشرات الأنفاق (مثل /content/tunnel_url.txt).
        5. خدمة ngrok المحلية إن وُجدت.
        6. الرابط الافتراضي المحلي (http://localhost:8000).
        """
        # 1. فحص المتغيرات البيئية ذات الأولوية الصريحة (إذا لم تكن localhost)
        for env_key in [
            "SERVER_PUBLIC_URL",
            "PUBLIC_URL",
            "TUNNEL_URL",
            "CLOUDFLARE_URL",
            "SERVER_URL",
            "APP_URL",
        ]:
            val = getattr(settings, env_key, None) or os.getenv(env_key)
            if val and str(val).strip():
                clean = str(val).strip().rstrip("/")
                if "localhost" not in clean and "127.0.0.1" not in clean:
                    return clean

        # 2. بيئة RunPod السحابية (معرّف Pod)
        pod_id = os.getenv("RUNPOD_POD_ID")
        if pod_id and str(pod_id).strip():
            port = os.getenv("PORT", "8000")
            runpod_url = f"https://{pod_id.strip()}-{port}.proxy.runpod.net"
            logger.info("تم اكتشاف بيئة RunPod السحابية: %s", runpod_url)
            return runpod_url

        # 3. فحص بيئة Google Colab / نفق Cloudflare Tunnel من ملفات السجلات
        # فحص ملفات الإشارة السريعة المباشرة (tunnel_url.txt)
        tunnel_txt_candidates = [
            Path("/content/tunnel_url.txt"),
            Path("/tmp/tunnel_url.txt"),
            Path.cwd() / "tunnel_url.txt",
        ]
        for t_path in tunnel_txt_candidates:
            if t_path.is_file():
                try:
                    txt = t_path.read_text(encoding="utf-8").strip()
                    if txt.startswith("http") and ("localhost" not in txt and "127.0.0.1" not in txt):
                        logger.info("تم اكتشاف رابط النفق من ملف الإشارة %s: %s", t_path, txt)
                        return txt.rstrip("/")
                except Exception:
                    pass

        # فحص ملفات سجلات Cloudflare Tunnel بحثاً عن رابط trycloudflare.com
        cloudflare_log_candidates = [
            Path("/content/cloudflared.log"),
            Path("/content/cloudflared.stdout.log"),
            Path("/tmp/cloudflared.log"),
            Path.cwd() / "cloudflared.log",
            Path.cwd() / "tunnel.log",
        ]
        cf_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        for log_path in cloudflare_log_candidates:
            if log_path.is_file():
                try:
                    content = log_path.read_text(encoding="utf-8", errors="ignore")
                    matches = cf_pattern.findall(content)
                    if matches:
                        detected_cf = matches[-1].strip().rstrip("/")
                        logger.info("تم اكتشاف رابط Cloudflare Tunnel من سجلات %s: %s", log_path, detected_cf)
                        return detected_cf
                except Exception:
                    pass

        # 4. محاولة استعلام ngrok API المحلي إذا كان قيد التشغيل
        try:
            req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for tun in data.get("tunnels", []):
                    p_url = tun.get("public_url")
                    if p_url and p_url.startswith("https"):
                        logger.info("تم اكتشاف رابط ngrok العام: %s", p_url)
                        return p_url.rstrip("/")
        except Exception:
            pass

        # 5. إذا تم تحديد إعداد صريح ولكنه localhost
        explicit_url = getattr(settings, "SERVER_PUBLIC_URL", "") or os.getenv("SERVER_PUBLIC_URL")
        if explicit_url and str(explicit_url).strip():
            return str(explicit_url).strip().rstrip("/")

        # 6. الرابط الافتراضي المحلي
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

    def register_server_settings_sync(self, explicit_url: Optional[str] = None) -> Dict[str, Any]:
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

        server_url = (explicit_url and str(explicit_url).strip().rstrip("/")) or self.detect_server_url()
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

    async def register_server_settings(self, explicit_url: Optional[str] = None) -> Dict[str, Any]:
        """الغلاف اللاتزامني لتسجيل إعدادات الخادم."""
        return await asyncio.to_thread(self.register_server_settings_sync, explicit_url)

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
            headers = {
                "Accept": "application/json",
                "User-Agent": f"VoxCPM-API/{settings.APP_VERSION}",
            }
            if settings.API_KEY:
                headers["X-API-Key"] = settings.API_KEY
                headers["Authorization"] = f"Bearer {settings.API_KEY}"

            req = urllib.request.Request(
                endpoint_url,
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    if resp_json.get("success") or "data" in resp_json:
                        data = resp_json.get("data", resp_json)
                        logger.info("تم جلب الموارد بنجاح من: %s", endpoint_url)
                        break
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="ignore")
                last_error = f"HTTP Error {he.code}: {err_body[:300] if err_body else he.reason}"
                logger.warning("فشل المسار %s (%s): %s", path, he.code, err_body[:200])
                continue
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

    async def _poll_and_register_tunnel(self, max_wait_seconds: int = 90, interval: float = 2.0) -> None:
        """مراقبة دورية في الخلفية لاكتشاف رابط كلاود فلير (أو الرابط العام) وإرساله للمنصة فور تولده."""
        elapsed = 0.0
        while elapsed < max_wait_seconds:
            await asyncio.sleep(interval)
            elapsed += interval
            detected = self.detect_server_url()
            if detected and "localhost" not in detected and "127.0.0.1" not in detected:
                last_reg = self.db.get_meta("last_registered_server_url")
                if last_reg != detected:
                    logger.info("⚡ تم اكتشاف الرابط العام للمنصة تلقائياً: %s (جاري التسجيل...)", detected)
                    await self.register_server_settings(explicit_url=detected)
                break

    async def run_startup_workflow(self) -> None:
        """تشغيل دورة المزامنة الكاملة عند إقلاع الخادم لأول مرة."""
        try:
            # 1. إرسال الدومين ومفتاح API للمنصة
            if getattr(settings, "PLATFORM_REGISTER_ON_STARTUP", True):
                initial_url = self.detect_server_url()
                await self.register_server_settings(explicit_url=initial_url)
                # إذا كان الرابط لا يزال محلياً (كما في كولاب قبل اكتمال إقلاع نفق كلاودفلير)، نشغل مراقباً تلقائياً
                if "localhost" in initial_url or "127.0.0.1" in initial_url:
                    asyncio.create_task(self._poll_and_register_tunnel(max_wait_seconds=90, interval=2.0))

            # 2. مزامنة وتحميل عينات الأصوات والضوضاء
            if getattr(settings, "PLATFORM_SYNC_ON_STARTUP", True):
                await self.sync_voices_and_noise()
        except Exception as exc:
            logger.exception("حدث استثناء أثناء إجراء المزامنة الأولية عند الإقلاع: %s", exc)


platform_sync_service = PlatformSyncService()
