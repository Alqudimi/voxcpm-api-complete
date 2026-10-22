from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

def _env_bool(value: str, default: bool) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if value is not None else default

class Settings(BaseSettings):
    APP_NAME: str = "VoxCPM2 Professional Voice API"
    APP_VERSION: str = "3.0.0"
    API_KEY: str = "change-me"

    VOXCPM_MODEL_ID: str = "openbmb/VoxCPM2"
    WHISPER_MODEL_NAME: str = "large-v3"
    ECAPA_SOURCE: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_SAVEDIR: str = "storage/models/ecapa_voxceleb"

    DEVICE: str = "cuda"
    REFERENCE_SAMPLE_RATE: int = 16000
    OUTPUT_SAMPLE_RATE: int = 48000
    AUDIO_SAMPLE_RATE: int = 24000
    AUDIO_CHANNELS: int = 1

    MAX_FILE_SIZE_MB: int = 100
    MAX_MERGE_FILES: int = 20

    DEFAULT_CFG: float = 2.0
    DEFAULT_TIMESTEPS: int = 30
    DEFAULT_CANDIDATES: int = 1
    DEFAULT_RETRY_BADCASE: bool = True
    DEFAULT_RETRY_MAX_TIMES: int = 3
    DEFAULT_RETRY_RATIO_THRESHOLD: float = 6.0

    OUTPUT_BITRATE: str = "96k"
    OUTPUT_CODEC: str = "libopus"

    STORAGE_ROOT: str = "storage"
    UPLOAD_DIR: str = "storage/uploads"
    OUTPUT_DIR: str = "storage/outputs"
    TEMP_DIR: str = "storage/temp"

    # ─── إعدادات منصة استنساخ الصوت ومزامنة الموارد الخارجية ───
    PLATFORM_BASE_URL: str = os.getenv("AUDIO_MIXER_HOST", "http://127.0.0.1:8000"),                # رابط المنصة الخارجية الرئيسي (مثال: http://localhost:8000)
    SERVER_PUBLIC_URL: str = ""                # رابط الخادم الحالي لإرساله للمنصة (يحسب تلقائياً إن ترك فارغاً)
    PLATFORM_SYNC_ON_STARTUP: bool = True      # تحميل وتخزين أصوات العينات والضوضاء عند الإقلاع
    PLATFORM_REGISTER_ON_STARTUP: bool = True  # إرسال الدومين ومفتاح API للمنصة عند الإقلاع
    PLATFORM_TIMEOUT: int = 600                # مهلة الاتصال بالثواني
    PLATFORM_DB_PATH: str = "storage/platform_resources.db"
    VOICE_SAMPLES_DIR: str = "storage/voice_samples"
    NOISE_SAMPLES_DIR: str = "storage/noise_samples"

    # ─── طبقة الأداء v3-max — المضبوط التلقائي: كل متغير يقبل "auto" ───
    # القيمة "auto" (أو 0 للأعداد) = AutoTuner يحسبها من عتادك ويعيّنها.
    # أي قيمة صريحة تتفوق على المضبوط دائماً (قانون الأولوية).
    VM_AUTOTUNE: str = "on"              # المفتاح الرئيسي للمضبوط التلقائي
    VM_AUTOTUNE_CALIBRATE: str = "on"    # معايرة RTF فعلية أثناء الإحماء
    VM_AUTOTUNE_FILE: str = "storage/autotune_profile.json"  # بروفايل محفوظ (إقلاع سريع)

    # عقد الجودة: هاتان القيمتان للتوثيق فقط ولا يجوز للكود/المضبوط تغييرهما عن الافتراضي.
    VOXCPM_TIMESTEPS: int = 30
    VOXCPM_CFG: float = 2.0

    VM_LOG_STAGES: str = "on"          # سطر JSON بتوقيتات كل مرحلة لكل طلب
    VM_TMPFS_DIR: str = "auto"         # "auto" = /dev/shm/voxmixer إن توفر وإلا القرص
    VM_RESAMPLER: str = "soxr_hq"      # بديل kaiser_best البطيء (نفس الجودة عالية)
    VM_DB_WAL: str = "on"              # SQLite WAL + synchronous=NORMAL

    VM_OFFLOAD_EXECUTOR: str = "thread"  # فك حجب event loop
    VM_EXECUTOR_WORKERS: int = 0         # 0 = auto (AutoTuner يحسبها من الكروت/النوى)
    VM_PARALLEL_PREP: str = "on"         # توازي تحضير المرجع/Whisper/الضجيج
    VM_ECAPA_ASYNC: str = "on"           # ECAPA متداخل مع ترميز OGG خارج المسار الحرج
    VM_ECAPA_DEVICE: str = "auto"        # auto: الكرت إن وُجد وإلا CPU

    VM_SPEAKER_CACHE: str = "on"         # كاش حالة المتحدث — مطابقة بتّية
    VM_SPEAKER_CACHE_TTL: int = 86400    # ثواني
    VM_SPEAKER_CACHE_MAX: int = 0        # 0 = auto (من حجم RAM)

    VM_PRECISION: str = "auto"           # auto: bf16 على Ampere+ وإلا fp16
    VM_TF32: str = "auto"                # auto: Ampere+
    VM_CUDNN_BENCHMARK: str = "on"
    VM_COMPILE: str = "auto"             # auto: reduce-overhead (CUDA Graphs) على كروت قادرة
    VM_COMPILE_WARMUP: int = 0           # 0 = auto (3 مع compile، 1 بدونه)

    VM_ENGINE: str = "auto"              # auto: NanoVLLM إن كان مثبتاً وقابلاً وإلا الرسمي
    VM_WHISPER_BACKEND: str = "auto"     # auto: faster-whisper إن توفر وإلا openai-whisper
    VM_WHISPER_MODEL: str = "auto"       # auto: حسب VRAM (keep/small/distil-large-v3)
    VM_WHISPER_BATCH_SIZE: int = 0       # 0 = auto
    VM_WHISPER_GREEDY: str = "auto"      # auto: فك حتمي beam=1 (~2× على المقاطع القصيرة)
    VM_WHISPER_VAD: str = "on"           # قص الصمت قبل النسخ
    VM_WHISPER_CACHE: str = "on"         # نص مرجعي مخزّن بمعرف hash

    VM_WARMUP_ON_BOOT: str = "on"        # إحماء النموذج عند الإقلاع قبل فتح الاستقبال
    VM_UVLOOP: str = "on"                # فعّل loop="uvloop" عند تشغيل uvicorn (توثيقي)

    # ─── طبقة الأداء v2-ultra — أضعاف مضاعفة، كل تحسين خلف مفتاح مستقل ───
    VM_RESULT_CACHE: str = "on"          # كاش النتائج الكاملة: تكرار نفس الطلب < 100ms (حتى ~100×)
    VM_RESULT_CACHE_TTL: int = 86400     # ثواني بقاء النتيجة المخزّنة
    VM_RESULT_CACHE_MAX: int = 0         # 0 = auto (من حجم RAM)
    VM_RESULT_CACHE_MAX_MB: int = 0      # 0 = auto (من حجم RAM)

    VM_RAM_PIPELINE: str = "on"          # خط أنابيب صفراوي: WAV/OGG وسيط في الذاكرة عبر ffmpeg pipe
    VM_OPUS_FASTENC: str = "on"          # libopus compression_level=4 بدل 10 (~2× ترميز أسرع)
    VM_EMBED_CACHE: str = "on"           # كاش embedding ECAPA المرجعي — نصف عمل التقييم

    VM_PARALLEL_CHUNKS: str = "on"       # توليد مقاطع النص الطويل بالتوازي (شرط: seed أصلي)
    VM_GPU_IDS: str = "auto"             # auto: كل الكروت | "0,1,2" → بركة محركات (إنتاجية N×)
    VM_GPU_PERMITS: int = 0              # 0 = auto: توليدات متزامنة لكل كرت (2 على ≥16GB)
    VM_CHUNK_PARALLEL_MAX: int = 0       # 0 = auto: سقف توازي المقاطع
    VM_AUTO_CHUNK: str = "on"            # تقسيم تلقائي للنص الطويل على فواصل طبيعية
    VM_AUTO_CHUNK_CHARS: int = 0         # 0 = auto (600 حرفاً)
    VM_QUANT: str = "auto"               # auto: fp8 على Ada+ / int8 على ≥7.5 / off (VAE مستثنى دائماً)

    VM_WHISPER_BATCHED: str = "on"       # BatchedInferencePipeline(batch=8) — ~3.7× موثق
    VM_SKIP_OUTPUT_PROBE: str = "on"     # تجاوز ffprobe على خرج الخلط (يُشتق حسابياً)
    VM_JOB_RAM_STAGING: str = "on"       # مدخلات الوظيفة في RAM بدل BLOB مزدوج في SQLite
    VM_RAM_STAGING_MAX_MB: int = 0       # 0 = auto (من حجم RAM)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_file_size(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # — مساعدات الأعلام المنطقية —
    @property
    def log_stages(self) -> bool: return _env_bool(self.VM_LOG_STAGES, True)
    @property
    def db_wal(self) -> bool: return _env_bool(self.VM_DB_WAL, True)
    @property
    def parallel_prep(self) -> bool: return _env_bool(self.VM_PARALLEL_PREP, True)
    @property
    def ecapa_async(self) -> bool: return _env_bool(self.VM_ECAPA_ASYNC, True)
    @property
    def speaker_cache_enabled(self) -> bool: return _env_bool(self.VM_SPEAKER_CACHE, True)
    @property
    def tf32_enabled(self) -> bool:
        try:
            from app.core.autotune import tuner

            return tuner.resolve_bool(self.VM_TF32, "tf32", True)
        except Exception:
            return _env_bool(self.VM_TF32, True)
    @property
    def cudnn_benchmark(self) -> bool: return _env_bool(self.VM_CUDNN_BENCHMARK, True)
    @property
    def compile_enabled(self) -> bool:
        return self.compile_mode != "off"

    @property
    def compile_mode(self) -> str:
        """وضع التجميع المضبوط (reduce-overhead = CUDA Graphs / default / off)."""
        try:
            from app.core.autotune import tuner

            return tuner.resolve_str(self.VM_COMPILE, "compile_mode", "off")
        except Exception:
            return self.VM_COMPILE if self.VM_COMPILE not in ("auto", "") else "off"

    @property
    def compile_warmup_passes(self) -> int:
        if self.VM_COMPILE_WARMUP:
            return max(1, int(self.VM_COMPILE_WARMUP))
        try:
            from app.core.autotune import tuner

            return tuner.resolve_int(0, "compile_warmup", 1)
        except Exception:
            return 3 if self.compile_mode != "off" else 1
    @property
    def whisper_vad(self) -> bool: return _env_bool(self.VM_WHISPER_VAD, True)
    @property
    def whisper_cache(self) -> bool: return _env_bool(self.VM_WHISPER_CACHE, True)
    @property
    def warmup_on_boot(self) -> bool: return _env_bool(self.VM_WARMUP_ON_BOOT, True)

    # — أعلام v2-ultra —
    @property
    def result_cache_enabled(self) -> bool: return _env_bool(self.VM_RESULT_CACHE, True)
    @property
    def ram_pipeline_enabled(self) -> bool: return _env_bool(self.VM_RAM_PIPELINE, True)
    @property
    def opus_fastenc(self) -> bool: return _env_bool(self.VM_OPUS_FASTENC, True)
    @property
    def embed_cache_enabled(self) -> bool: return _env_bool(self.VM_EMBED_CACHE, True)
    @property
    def parallel_chunks_enabled(self) -> bool: return _env_bool(self.VM_PARALLEL_CHUNKS, True)
    @property
    def whisper_batched(self) -> bool: return _env_bool(self.VM_WHISPER_BATCHED, True)
    @property
    def skip_output_probe(self) -> bool: return _env_bool(self.VM_SKIP_OUTPUT_PROBE, True)
    @property
    def job_ram_staging(self) -> bool: return _env_bool(self.VM_JOB_RAM_STAGING, True)

    # — خصائص v3-max: كلها تمر عبر AutoTuner (صريح > مضبوط > افتراضي) —
    @property
    def executor_workers(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_EXECUTOR_WORKERS, "executor_workers", 2)
        except Exception:
            return max(2, int(self.VM_EXECUTOR_WORKERS or 2))

    @property
    def speaker_cache_max(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_SPEAKER_CACHE_MAX, "speaker_cache_max", 12)
        except Exception:
            return int(self.VM_SPEAKER_CACHE_MAX or 12)

    @property
    def result_cache_max(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_RESULT_CACHE_MAX, "result_cache_max", 64)
        except Exception:
            return int(self.VM_RESULT_CACHE_MAX or 64)

    @property
    def result_cache_max_mb(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_RESULT_CACHE_MAX_MB, "result_cache_max_mb", 256)
        except Exception:
            return int(self.VM_RESULT_CACHE_MAX_MB or 256)

    @property
    def ram_staging_max_mb(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_RAM_STAGING_MAX_MB, "ram_staging_max_mb", 512)
        except Exception:
            return int(self.VM_RAM_STAGING_MAX_MB or 512)

    @property
    def ecapa_device(self) -> str:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_str(self.VM_ECAPA_DEVICE, "ecapa_device", "cuda")
        except Exception:
            return self.VM_ECAPA_DEVICE if self.VM_ECAPA_DEVICE not in ("auto", "") else "cuda"

    @property
    def gpu_permits(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_GPU_PERMITS, "gpu_permits", 1)
        except Exception:
            return int(self.VM_GPU_PERMITS or 1)

    @property
    def chunk_parallel_max(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_CHUNK_PARALLEL_MAX, "chunk_parallel_max", 4)
        except Exception:
            return int(self.VM_CHUNK_PARALLEL_MAX or 4)

    @property
    def whisper_batch_size(self) -> int:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_WHISPER_BATCH_SIZE, "whisper_batch_size", 8)
        except Exception:
            return int(self.VM_WHISPER_BATCH_SIZE or 8)

    @property
    def whisper_greedy(self) -> bool:
        try:
            from app.core.autotune import tuner
            return tuner.resolve_bool(self.VM_WHISPER_GREEDY, "whisper_greedy", True)
        except Exception:
            return True

    @property
    def auto_chunk_chars(self) -> int:
        if not _env_bool(self.VM_AUTO_CHUNK, True):
            return 0  # 0 = التقسيم التلقائي معطّل
        try:
            from app.core.autotune import tuner
            return tuner.resolve_int(self.VM_AUTO_CHUNK_CHARS, "auto_chunk_chars", 600)
        except Exception:
            return int(self.VM_AUTO_CHUNK_CHARS or 600)

    @property
    def autotune_enabled(self) -> bool: return _env_bool(self.VM_AUTOTUNE, True)
    @property
    def autotune_calibrate(self) -> bool: return _env_bool(self.VM_AUTOTUNE_CALIBRATE, True)

    @property
    def tmpfs_dir(self) -> Path:
        """مسار وسيط سريع: RAM-disk إن توفر (/dev/shm) وإلا TEMP_DIR العادي.
        قاعدة الاحترام: إذا أعاد المستخدم تعريف TEMP_DIR صراحة (بيئات
        الاختبار/Docker) فذلك يتقدم على tmpfs التلقائي."""
        if self.VM_TMPFS_DIR and self.VM_TMPFS_DIR != "auto":
            return Path(self.VM_TMPFS_DIR)
        if self.TEMP_DIR != "storage/temp":  # تعريف صريح للمستخدم = أولوية
            return Path(self.TEMP_DIR)
        shm = Path("/dev/shm/voxmixer")
        try:
            if shm.parent.exists():
                import shutil
                if shutil.disk_usage(str(shm.parent)).free > 1 << 30:  # ≥1GB حرة
                    return shm
        except OSError:
            pass
        return Path(self.TEMP_DIR)

    def resolve_temp_dir(self) -> Path:
        """TEMP_DIR الفعلي للاستخدام (tmpfs إن أمكن) مع إنشاء المجلد."""
        base = self.tmpfs_dir
        if str(base) != self.TEMP_DIR:
            try:
                base.mkdir(parents=True, exist_ok=True)
            except OSError:
                base = Path(self.TEMP_DIR)
                base.mkdir(parents=True, exist_ok=True)
        else:
            base.mkdir(parents=True, exist_ok=True)
        return base

    def ensure_directories(self):
        for item in (
            self.STORAGE_ROOT,
            self.UPLOAD_DIR,
            self.OUTPUT_DIR,
            self.TEMP_DIR,
            self.ECAPA_SAVEDIR,
            self.VOICE_SAMPLES_DIR,
            self.NOISE_SAMPLES_DIR,
        ):
            Path(item).mkdir(parents=True, exist_ok=True)
        # المجلد الوسيط السريع يُنشأ هنا أيضاً (فشله غير قاتل)
        try:
            self.resolve_temp_dir()
        except OSError:
            pass

settings = Settings()
