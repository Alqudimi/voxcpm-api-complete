"""AutoTuner v3-max — العقل المدبّر: يحسب ويعيّن كل متغيرات الأداء تلقائياً.

الفكرة التي طلبها المستخدم حرفياً: «اجعله يحسب ويعيّن جميع المتغيرات الذي
يحتاجها بشكل آلي تلقائي». هذا الملف يفحص العتاد مرة واحدة عند الإقلاع،
يحلّ كل متغير قابل للضبط بالقيمة المثلى لهذا العتاد بالذات، يطبّقها،
يقيس أداء فعلياً (معايرة RTF)، ويحفظ البروفايل في JSON ليعاد استخدامه.

قانون الأولوية الصارم (في كل الوحدات):
  1. قيمة صريحة من المستخدم (متغير بيئة ≠ "auto" / ≠ 0)  ← تتفوق دائماً
  2. القيمة المحسوبة من بروفايل العتاد (هذا الملف)
  3. الافتراضيات القديمة (v1/v2) كشبكة أمان أخيرة

العقد الصلبة غير القابلة للعَبث (محفوظة في كل الحالات):
  - inference_timesteps=30 وcfg_value=2.0 لا يلمسهما المضبوط أبداً.
  - AudioVAE يُستثنى من كل تكميم دائماً.
  - أي فشل في أي قرار مضبوط = سقوط آمن إلى سلوك v1/v2.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROFILE_SCHEMA = "at1"

AUTO_SENTINELS = {"", "auto", "none", "null"}


def _is_auto(raw: Any) -> bool:
    """هل هذه القيمة تُترك للمضبوط التلقائي؟ (auto / فراغ / None / 0 للأعداد)"""
    if raw is None:
        return True
    if isinstance(raw, bool):
        return False
    if isinstance(raw, (int, float)):
        return float(raw) == 0.0
    return str(raw).strip().lower() in AUTO_SENTINELS


def _flag(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────
# 1) فحص العتاد — بلا torch يعمل أيضاً (بيئات CPU/الاختبار)
# ─────────────────────────────────────────────────────────────────
def _read_ram_mb() -> int:
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(int(line.split()[1]) / 1024)
    except OSError:
        pass
    try:
        import psutil  # اختياري

        return int(psutil.virtual_memory().total / (1 << 20))
    except Exception:
        return 8192  # تقدير آمن 8GB


def _read_shm_free_mb() -> int:
    try:
        import shutil

        if Path("/dev/shm").exists():
            return int(shutil.disk_usage("/dev/shm").free / (1 << 20))
    except OSError:
        pass
    return 0


def _package_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def probe_hardware() -> Dict[str, Any]:
    """فحص شامل للعتاد — يعمل بلا GPU وبلا torch (لا يفشل أبداً)."""
    hw: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count() or 2,
        "ram_mb": _read_ram_mb(),
        "shm_free_mb": _read_shm_free_mb(),
        "cuda": False,
        "gpu_count": 0,
        "gpus": [],
        "torch": None,
        "cuda_version": None,
        "packages": {
            "torchao": _package_available("torchao"),
            "faster_whisper": _package_available("faster_whisper"),
            "nanovllm_voxcpm": _package_available("nanovllm_voxcpm"),
            "uvloop": _package_available("uvloop"),
            "soxr": _package_available("soxr"),
        },
    }
    try:
        import torch

        hw["torch"] = torch.__version__
        hw["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            hw["cuda"] = True
            hw["gpu_count"] = int(torch.cuda.device_count())
            for idx in range(hw["gpu_count"]):
                try:
                    cap = torch.cuda.get_device_capability(idx)
                    props = torch.cuda.get_device_properties(idx)
                    hw["gpus"].append(
                        {
                            "index": idx,
                            "name": props.name,
                            "vram_mb": int(props.total_memory / (1 << 20)),
                            "cc_major": int(cap[0]),
                            "cc_minor": int(cap[1]),
                            "sm_count": int(getattr(props, "multi_processor_count", 0)),
                        }
                    )
                except Exception as exc:  # كرت مكسور جزئياً — نتجاهله
                    logger.warning("GPU probe failed for cuda:%s: %s", idx, exc)
    except Exception:
        pass
    return hw


# ─────────────────────────────────────────────────────────────────
# 2) الحلّال — قواعد خالصة (عتاد → قيم مثلى) قابلة للاختبار الوحدوي
# ─────────────────────────────────────────────────────────────────
def solve(hw: Dict[str, Any]) -> Dict[str, Any]:
    """يحل كل متغيرات الأداء من بروفايل العتاد. دالة نقية — بلا آثار جانبية."""
    gpus: List[Dict[str, Any]] = hw.get("gpus", []) or []
    gpu_count = len(gpus)
    cpu = max(2, int(hw.get("cpu_logical") or 2))
    ram_mb = max(1024, int(hw.get("ram_mb") or 8192))
    pkgs = hw.get("packages", {}) or {}

    if gpus:
        min_cc = min(g["cc_major"] + g["cc_minor"] / 10.0 for g in gpus)
        max_cc = max(g["cc_major"] + g["cc_minor"] / 10.0 for g in gpus)
        max_vram = max(g["vram_mb"] for g in gpus)
        min_vram = min(g["vram_mb"] for g in gpus)
    else:
        min_cc = max_cc = 0.0
        max_vram = min_vram = 0

    tuned: Dict[str, Any] = {}

    # ── الدقة: bf16 على Ampere+ (cc≥8.0)، fp16 على Turing، fp32 أقدم ──
    if not gpus:
        tuned["dtype"] = "fp32"
        tuned["tf32"] = False
    elif min_cc >= 8.0:
        tuned["dtype"] = "bf16"
        tuned["tf32"] = True
    elif min_cc >= 7.5:
        tuned["dtype"] = "fp16"
        tuned["tf32"] = True
    elif min_cc >= 7.0:
        tuned["dtype"] = "fp16"
        tuned["tf32"] = False
    else:
        tuned["dtype"] = "fp32"
        tuned["tf32"] = False
    tuned["matmul_precision"] = "high" if gpus else "highest"
    tuned["cudnn_benchmark"] = bool(gpus)

    # ── المحرك: NanoVLLM تلقائياً إن كان مثبتاً وكابتياً (RTF 0.13 الموثق ≈ 2.3×) ──
    # العقد: المحرك البديل يجب أن يقبل timesteps=30/cfg=2.0 كما هما وإلا سقط للرسمي.
    if (
        gpus
        and pkgs.get("nanovllm_voxcpm")
        and max_cc >= 8.0
        and max_vram >= 8000
        and gpu_count == 1  # محرك nanovllm يدير أجهزته بنفسه — مسار كرت واحد
    ):
        tuned["engine"] = "nanovllm"
    else:
        tuned["engine"] = "official"

    # ── التجميع: CUDA Graphs تلقائياً (reduce-overhead) على كروت قادرة ──
    # شروط محافظة: cc≥7.5 + VRAM كافية + ليس مع محرك nanovllm (يدير نفسه).
    if not gpus or tuned["engine"] == "nanovllm" or min_cc < 7.5:
        tuned["compile_mode"] = "off"
    elif min_vram >= 16000:
        tuned["compile_mode"] = "reduce-overhead"  # CUDA Graphs — ~1.1-1.65×
    elif min_vram >= 10000:
        tuned["compile_mode"] = "default"
    else:
        tuned["compile_mode"] = "off"
    tuned["compile_warmup"] = 3 if tuned["compile_mode"] != "off" else 1

    # ── التكميم weight-only (VAE مستثنى دائماً — العقد الصلب) ──
    # fp8 يحتاج Ada+ (cc≥8.9)؛ int8 يعمل من cc≥7.5؛ يتطلب torchao مثبتاً.
    if not gpus or not pkgs.get("torchao") or tuned["engine"] == "nanovllm":
        tuned["quant"] = "off"
    elif max_cc >= 8.9:
        tuned["quant"] = "fp8"
    elif min_cc >= 7.5:
        tuned["quant"] = "int8"
    else:
        tuned["quant"] = "off"

    # ── بركة GPUs: كل الكروت تلقائياً (إنتاجية N×) ──
    tuned["gpu_ids"] = [g["index"] for g in gpus]

    # ── تصاريح التزامن لكل كرت: 2 على ≥16GB (تراكب kernel للمقاطع)، 1 تحت ──
    tuned["gpu_permits"] = 2 if (gpus and min_vram >= 16000) else (1 if gpus else 1)

    # ── عمال المنفّذ: تغطية كل تصاريح كل الكروت + عمال CPU للترميز/التقييم ──
    cpu_workers = max(2, min(8, cpu // 4))
    tuned["executor_workers"] = min(24, gpu_count * tuned["gpu_permits"] + cpu_workers)

    # ── سقف توازي المقاطع: تصاريح × كروت (سقف 8 لتجنب إجهاد VRAM) ──
    tuned["chunk_parallel_max"] = max(1, min(8, tuned["gpu_permits"] * max(1, gpu_count)))

    # ── Whisper: نموذج/دقة/دفعة حسب VRAM — المسار الساخن (مرجع ≤35s) ──
    if not gpus:
        tuned["whisper_model"] = "small"
        tuned["whisper_compute"] = "int8"
        tuned["whisper_batch_size"] = 4
    elif min_vram >= 11000:
        tuned["whisper_model"] = "keep"  # يحافظ على WHISPER_MODEL_NAME (large-v3)
        tuned["whisper_compute"] = "int8"
        tuned["whisper_batch_size"] = 8
    elif min_vram >= 6000:
        tuned["whisper_model"] = "distil-large-v3"
        tuned["whisper_compute"] = "int8"
        tuned["whisper_batch_size"] = 4
    else:
        tuned["whisper_model"] = "small"
        tuned["whisper_compute"] = "int8"
        tuned["whisper_batch_size"] = 2
    # فك حتمي greedy: ~1.5-2× على المقاطع القصيرة، فرق جودة هامشي على مراجع نظيفة
    tuned["whisper_greedy"] = True

    # ── ECAPA على الكرت إن وُجد ──
    tuned["ecapa_device"] = "cuda" if gpus else "cpu"

    # ── ميزانيات الكاش من حجم RAM ──
    tuned["result_cache_max_mb"] = int(max(64, min(512, ram_mb * 0.02)))
    tuned["result_cache_max"] = int(max(16, min(128, ram_mb // 64)))
    tuned["speaker_cache_max"] = int(max(8, min(48, ram_mb // 512)))
    tuned["ram_staging_max_mb"] = int(max(128, min(1024, ram_mb * 0.04)))

    # ── تقسيم النص الطويل تلقائياً فوق هذا الطول (فواصل طبيعية عند علامات الاستفهام) ──
    tuned["auto_chunk_chars"] = 600

    # ── إسبات tmpfs إن كان صغيراً جداً (أقل من 1GB حرة → قرص) ──
    tuned["tmpfs_usable"] = bool(hw.get("shm_free_mb", 0) >= 1024)

    return tuned


# ─────────────────────────────────────────────────────────────────
# 3) المضبوط — دورة حياة: فحص → حل → حفظ → معايرة → عرض
# ─────────────────────────────────────────────────────────────────
class AutoTuner:
    def __init__(self):
        self._lock = threading.Lock()
        self._hw: Optional[Dict[str, Any]] = None
        self._tuned: Optional[Dict[str, Any]] = None
        self._calibration: Dict[str, Any] = {}
        self._profile_path: Optional[Path] = None
        self._boot_ts: float = time.time()

    # ── التهيئة (تُستدعى من lifecycle قبل تحميل النماذج، وتعمل كسلوك كسول أيضاً) ──
    def ensure(self, force_probe: bool = False) -> Dict[str, Any]:
        with self._lock:
            if self._tuned is not None and not force_probe:
                return self._tuned
            from app.core.config import settings

            master = _flag(getattr(settings, "VM_AUTOTUNE", "on"), True)
            self._profile_path = self._resolve_profile_path()
            if master and not force_probe and self._load_cached_profile():
                logger.info(
                    "AutoTuner: cached profile loaded (%s) — boot fast path",
                    self._profile_path,
                )
                return self._tuned
            self._hw = probe_hardware()
            self._tuned = solve(self._hw)
            if master:
                self._persist()
            self._log_table()
            return self._tuned

    def _resolve_profile_path(self) -> Path:
        from app.core.config import settings

        raw = str(getattr(settings, "VM_AUTOTUNE_FILE", "") or "storage/autotune_profile.json")
        try:
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            return Path("storage/autotune_profile.json")

    def _fingerprint(self, hw: Dict[str, Any]) -> str:
        material = json.dumps(
            {
                "gpus": [
                    [g.get("name"), g.get("vram_mb"), g.get("cc_major"), g.get("cc_minor")]
                    for g in hw.get("gpus", [])
                ],
                "cpu": hw.get("cpu_logical"),
                "ram": hw.get("ram_mb"),
                "machine": hw.get("machine"),
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]

    # ── الكاش: بروفايل بنفس بصمة العتاد = قفز فوق الفحص والمعايرة ──
    def _load_cached_profile(self) -> bool:
        try:
            if self._profile_path is None or not self._profile_path.exists():
                return False
            data = json.loads(self._profile_path.read_text("utf-8"))
            if data.get("schema") != PROFILE_SCHEMA:
                return False
            hw = data.get("hardware", {})
            if data.get("fingerprint") != self._fingerprint(hw):
                return False
            self._hw = hw
            self._tuned = data.get("tuned", {}) or None
            self._calibration = data.get("calibration", {}) or {}
            return self._tuned is not None
        except Exception as exc:
            logger.debug("autotune profile load failed: %s", exc)
            return False

    def _persist(self):
        if self._profile_path is None or self._tuned is None:
            return
        try:
            payload = {
                "schema": PROFILE_SCHEMA,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fingerprint": self._fingerprint(self._hw or {}),
                "hardware": self._hw,
                "tuned": self._tuned,
                "calibration": self._calibration,
            }
            tmp = self._profile_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
            os.replace(tmp, self._profile_path)
        except Exception as exc:
            logger.debug("autotune profile persist failed: %s", exc)

    # ── المعايرة: قياس RTF فعلي من تمريرات الإحماء (يحفظ في البروفايل) ──
    def calibrate(self, rtf: float, gen_seconds: float, audio_seconds: float, engine: str):
        if audio_seconds <= 0:
            return
        with self._lock:
            prev = self._calibration.get("rtf")
            self._calibration = {
                "rtf": min(rtf, prev) if prev else rtf,  # نحتفظ بأفضل قياس
                "last_rtf": rtf,
                "gen_seconds": round(gen_seconds, 3),
                "audio_seconds": round(audio_seconds, 3),
                "engine": engine,
                "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._persist()

    # ── واجهة الحلّ الأولوية: صريح > بروفايل > افتراضي ──
    def resolve_str(self, raw: Any, key: str, default: str) -> str:
        if not _is_auto(raw):
            return str(raw).strip()
        tuned = self._tuned or self.ensure()
        value = tuned.get(key, default)
        return str(value) if value is not None else default

    def resolve_int(self, raw: Any, key: str, default: int) -> int:
        if not _is_auto(raw):
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default
        tuned = self._tuned or self.ensure()
        try:
            return int(tuned.get(key, default))
        except (TypeError, ValueError):
            return default

    def resolve_bool(self, raw: Any, key: str, default: bool) -> bool:
        if not _is_auto(raw):
            return _flag(raw, default)
        tuned = self._tuned or self.ensure()
        return bool(tuned.get(key, default))

    def resolve_gpu_ids(self, raw: str) -> List[int]:
        if not _is_auto(raw):
            from app.core.tts_engine import parse_gpu_ids

            return parse_gpu_ids(raw)
        tuned = self._tuned or self.ensure()
        return [int(i) for i in tuned.get("gpu_ids", [])]

    # ── نموذج whisper: "keep" يعني احترام WHISPER_MODEL_NAME كما هو ──
    def resolve_whisper_model(self, raw: str, fallback: str) -> str:
        if not _is_auto(raw):
            return str(raw).strip()
        tuned = self._tuned or self.ensure()
        model = tuned.get("whisper_model", "keep")
        return fallback if model in {"keep", ""} else str(model)

    # ── تقرير الإقلاع: جدول واحد مقروء ──
    def _log_table(self):
        if self._tuned is None:
            return
        t = self._tuned
        hw = self._hw or {}
        gpus = hw.get("gpus", [])
        gpu_line = (
            ", ".join(f"cuda:{g['index']} {g['name']} {g['vram_mb']}MB cc{g['cc_major']}.{g['cc_minor']}" for g in gpus)
            or "CPU only"
        )
        logger.info("=" * 78)
        logger.info("AutoTuner v3 — تم حساب وتعيين كل متغيرات الأداء تلقائياً")
        logger.info("  العتاد: %s | CPU=%s | RAM=%sMB", gpu_line, hw.get("cpu_logical"), hw.get("ram_mb"))
        logger.info(
            "  المحرك=%s | dtype=%s | compile=%s | quant=%s (VAE مستثنى)",
            t.get("engine"), t.get("dtype"), t.get("compile_mode"), t.get("quant"),
        )
        logger.info(
            "  GPUs=%s | permits/كرت=%s | عمال=%s | توازي مقاطع=%s",
            t.get("gpu_ids"), t.get("gpu_permits"), t.get("executor_workers"), t.get("chunk_parallel_max"),
        )
        logger.info(
            "  Whisper=%s/%s batch=%s greedy=%s | ECAPA=%s",
            t.get("whisper_model"), t.get("whisper_compute"), t.get("whisper_batch_size"),
            t.get("whisper_greedy"), t.get("ecapa_device"),
        )
        logger.info(
            "  كاش النتائج=%sMB/%d | كاش المتحدث=%d | RAM staging=%sMB | auto-chunk>%d حرفاً",
            t.get("result_cache_max_mb"), t.get("result_cache_max"), t.get("speaker_cache_max"),
            t.get("ram_staging_max_mb"), t.get("auto_chunk_chars"),
        )
        logger.info("  الأولوية: قيمة صريحة في البيئة تتفوق على هذا الجدول دائماً")
        logger.info("=" * 78)

    # ── تقرير كامل لنقطة النهاية ──
    def report(self) -> Dict[str, Any]:
        tuned = self._tuned or self.ensure()
        from app.core.config import settings

        return {
            "enabled": _flag(getattr(settings, "VM_AUTOTUNE", "on"), True),
            "profile_file": str(self._profile_path) if self._profile_path else None,
            "hardware": self._hw,
            "tuned": tuned,
            "calibration": self._calibration,
            "boot_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._boot_ts)),
            "precedence": "explicit env var > tuned profile > legacy default",
            "contracts": {
                "timesteps": 30,
                "cfg": 2.0,
                "audiovae_quantization": "forbidden",
            },
        }

    @property
    def tuned(self) -> Dict[str, Any]:
        return self._tuned or self.ensure()


tuner = AutoTuner()
