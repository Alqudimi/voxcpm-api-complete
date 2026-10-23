"""طبقة تجريد محرك التوليد — v3-max (اختيار تلقائي + بركة واعية بالحمل).

- OfficialVoxCPMEngine: المسار الرسمي (خط الأساس للتراجع). ربط بجهاز CUDA
  محدد + كشف دعم seed الأصلي (شرط التوازي الآمن) + torch.inference_mode.
- EnginePool: بركة على عدة GPUs — v3: اختيار **الأقل تحميلاً** عبر
  GPUScheduler (بدل الدوران الأعمى round-robin) + تصاريح تزامن لكل كرت
  تمنع تدافع VRAM.
- NanoVLLMEngine: المحرك السريع (RTF ≈ 0.13 الموثق ≈ 2.3× مقابل الرسمي)
  — v3: يُختار **تلقائياً** عند VM_ENGINE=auto إذا كان مثبتاً والعتاد
  كابياً (cc≥8 وVRAM≥8GB وكرت واحد).
  شرط العقد الصارم: المحرك البديل **يجب** أن يقبل inference_timesteps
  وcfg_value كما هما (30/2.0) وإلا يُرفض التحميل ويسقط النظام تلقائياً
  إلى الرسمي.
- ensure_engine_loaded(): v3 — إصلاح فجوة v2 (pending_load لم يكن
  مستهلكاً في أي مكان): تحميل فعلي في الإقلاع مع سقوط آمن للرسمي.
"""
from __future__ import annotations

import inspect
import logging
import threading
from typing import List, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def _cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def parse_gpu_ids(raw: str) -> List[int]:
    """VM_GPU_IDS="0,1" → [0, 1] — مفلترة على الأجهزة الموجودة فعلاً."""
    if not raw or str(raw).strip().lower() in {"", "auto", "0"}:
        return []
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            continue
        if 0 <= idx:
            out.append(idx)
    available = _cuda_device_count()
    return [i for i in out if i < available] if available else []


class OfficialVoxCPMEngine:
    """المسار الرسمي PyTorch — نفس سلوك الكود الأصلي حرفياً، مربوط بجهاز واحد."""

    is_async = False
    name = "official"

    def __init__(self, model, device_index: int = 0):
        self.model = model
        self.device_index = int(device_index)
        self.supports_native_seed = self._detect_native_seed()
        self.supports_inference_mode = self._detect_inference_mode()

    def _detect_native_seed(self) -> bool:
        """seed الأصلي في generate() = شرط الأمان للتوازي (كل خيط يضمن بذرته الخاصة)."""
        try:
            return "seed" in inspect.signature(self.model.generate).parameters
        except Exception:
            return False

    @staticmethod
    def _detect_inference_mode() -> bool:
        try:
            import torch

            return hasattr(torch, "inference_mode")
        except Exception:
            return False

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.model.tts_model, "sample_rate", settings.OUTPUT_SAMPLE_RATE))

    def generate_sync(
        self,
        target_text: str,
        prompt_text: str,
        reference_path,
        cfg_value: float,
        inference_timesteps: int,
        seed: int,
        retry_badcase: bool,
        retry_max_times: int,
        retry_ratio_threshold: float,
    ) -> np.ndarray:
        model = self.model
        signature = inspect.signature(model.generate)
        supported = set(signature.parameters.keys())

        kwargs = {
            "text": target_text,
            "prompt_wav_path": str(reference_path),
            "prompt_text": prompt_text,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_max_times),
            "retry_badcase_ratio_threshold": float(retry_ratio_threshold),
        }

        if "seed" in supported:
            kwargs["seed"] = int(seed)
        else:
            import torch

            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if "kwargs" not in signature.parameters:
            kwargs = {k: v for k, v in kwargs.items() if k in supported}

        if "text" not in kwargs:
            raise RuntimeError("CRITICAL: text was removed before model.generate().")

        if self.device_index:
            try:
                import torch

                with torch.cuda.device(self.device_index):
                    wav = self._invoke(model, kwargs)
            except (AttributeError, RuntimeError):
                wav = self._invoke(model, kwargs)
        else:
            wav = self._invoke(model, kwargs)

        if wav is None:
            raise RuntimeError("VoxCPM2 returned None.")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            raise RuntimeError("VoxCPM2 returned empty audio.")
        return wav

    def _invoke(self, model, kwargs):
        """الاستدعاء مع inference_mode عند توفره (يقطع تتبع autograd — أسرع من no_grad)."""
        if not self.supports_inference_mode:
            return model.generate(**kwargs)
        try:
            import torch

            with torch.inference_mode():
                return model.generate(**kwargs)
        except RuntimeError as exc:
            logger.debug("inference_mode rejected by model, retrying without: %s", exc)
            self.supports_inference_mode = False
            return model.generate(**kwargs)


class EnginePool:
    """بركة واعية بالحمل على عدة محركات/GPUs.

    v3: كل استدعاء يذهب للكرت **الأقل تحميلاً** (أكبر تصاريح حرة عبر
    GPUScheduler) وكل توليد محجوز بتصريح يمنع تدافع VRAM — الطلبات
    المتزامنة والمقاطع تتناثر تلقائياً على الكروت (إنتاجية ≈ N×).
    """

    is_async = False
    name = "official-pool"

    def __init__(self, engines: List[OfficialVoxCPMEngine]):
        if not engines:
            raise ValueError("EnginePool requires at least one engine.")
        self.engines = engines
        self._rr = 0
        self._lock = threading.Lock()

    @property
    def supports_native_seed(self) -> bool:
        return all(e.supports_native_seed for e in self.engines)

    @property
    def sample_rate(self) -> int:
        return self.engines[0].sample_rate

    @property
    def device_indices(self) -> List[int]:
        return [e.device_index for e in self.engines]

    def pick(self) -> OfficialVoxCPMEngine:
        """الأقل تحميلاً عبر المجدول (v3)؛ كسقوط آمن: دوران بسيط."""
        try:
            from app.core.gpu_scheduler import gpu_scheduler

            if gpu_scheduler.configured():
                dev = gpu_scheduler.least_loaded(self.device_indices)
                for e in self.engines:
                    if e.device_index == dev:
                        return e
        except Exception:
            pass
        with self._lock:
            engine = self.engines[self._rr % len(self.engines)]
            self._rr += 1
            return engine

    def next_engine(self) -> OfficialVoxCPMEngine:
        return self.pick()

    def generate_sync(self, *args, **kwargs) -> np.ndarray:
        engine = self.pick()
        try:
            from app.core.gpu_scheduler import gpu_scheduler

            if gpu_scheduler.configured():
                with gpu_scheduler.slot(engine.device_index):
                    return engine.generate_sync(*args, **kwargs)
        except Exception:
            pass
        return engine.generate_sync(*args, **kwargs)


class NanoVLLMEngine:
    """محرك nano-vllm-voxcpm — تثبيت اختياري: pip install nano-vllm-voxcpm.

    يتطلب نقاط أوزان بصيغة safetensors + config.json + audiovae.pth.
    يعمل بوضع البركة غير المتزامن داخل تطبيق async.
    """

    is_async = True
    name = "nanovllm"

    def __init__(self, model_path: str):
        from nanovllm_voxcpm import VoxCPM as NVEngine  # ImportError → يُلتقط في get_engine

        self._NV = NVEngine
        self._model_path = model_path
        self.server = None
        self._sample_rate = settings.OUTPUT_SAMPLE_RATE

    async def load(self):
        self.server = self._NV.from_pretrained(
            model=self._model_path,
            devices=[0],
        )
        await self.server.wait_for_ready()
        logger.info("NanoVLLM engine loaded from %s", self._model_path)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def generate_async(
        self,
        target_text: str,
        prompt_text: str,
        reference_path,
        cfg_value: float,
        inference_timesteps: int,
        **_ignored,
    ) -> np.ndarray:
        if self.server is None:
            raise RuntimeError("NanoVLLM engine is not loaded.")
        chunks = []
        async for chunk in self.server.generate(
            target_text=target_text,
            prompt_wav=str(reference_path),
            prompt_text=prompt_text,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
        ):
            chunks.append(np.asarray(chunk, dtype=np.float32))
        if not chunks:
            raise RuntimeError("NanoVLLM returned no audio chunks.")
        return np.concatenate(chunks, axis=0)


_engine_singleton = None


def _build_official_pool():
    """يبني البركة: نسخة VoxCPM لكل جهاز مضبوط (auto = كل الكروت).
    v3: يضبط GPUScheduler بتصاريح لكل كرت بعد البناء."""
    from app.core.model_manager import model_manager

    if not model_manager.ready or model_manager.voxcpm is None:
        raise RuntimeError("AI models are not ready.")

    from app.core.autotune import tuner

    gpu_ids = tuner.resolve_gpu_ids(getattr(settings, "VM_GPU_IDS", "auto"))
    if len(gpu_ids) <= 1:
        single = OfficialVoxCPMEngine(
            model_manager.voxcpm, device_index=(gpu_ids[0] if gpu_ids else 0)
        )
        _configure_scheduler([single.device_index])
        return single

    # متعدد GPUs: النسخة الأولى محملة مسبقاً؛ النسخ الإضافية على أجهزتها.
    engines: List[OfficialVoxCPMEngine] = [
        OfficialVoxCPMEngine(model_manager.voxcpm, device_index=gpu_ids[0])
    ]
    extras = model_manager.load_device_instances(gpu_ids[1:])
    for dev_idx, extra_model in extras:
        engines.append(OfficialVoxCPMEngine(extra_model, device_index=dev_idx))
    logger.info(
        "EnginePool active: %d engines on devices %s (throughput ~= %dx)",
        len(engines),
        [e.device_index for e in engines],
        len(engines),
    )
    _configure_scheduler([e.device_index for e in engines])
    return EnginePool(engines)


def _configure_scheduler(devices: List[int]):
    """ضبط مجدول GPU بلا تكرار (فشله غير قاتل — سقوط دوران بسيط)."""
    try:
        from app.core.gpu_scheduler import gpu_scheduler

        gpu_scheduler.configure(devices, settings.gpu_permits)
    except Exception as exc:
        logger.warning("GPUScheduler not configured (round-robin fallback): %s", exc)


def _nanovllm_available() -> bool:
    """فحص توفر الحزمة دون استيرادها الفعلي (رخيص وآمن)."""
    try:
        import importlib.util

        return importlib.util.find_spec("nanovllm_voxcpm") is not None
    except Exception:
        return False


def _requested_engine() -> str:
    """قرار المحرك: صريح > مضبوط (auto = قرار AutoTuner من بروفايل العتاد)."""
    raw = (settings.VM_ENGINE or "auto").strip().lower()
    if raw not in {"", "auto"}:
        return raw
    try:
        from app.core.autotune import tuner

        return str(tuner.tuned.get("engine", "official")).strip().lower()
    except Exception:
        return "official"


def get_engine():
    """يعيد المحرك المطلوب مع سقوط آمن إلى الرسمي عند أي فشل في البديل."""
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton

    requested = _requested_engine()

    if requested == "nanovllm" and getattr(settings, "DEVICE", "cuda") == "cuda":
        if not _nanovllm_available():
            logger.warning("Engine 'nanovllm' selected but package missing; official fallback")
        else:
            try:
                engine = NanoVLLMEngine(settings.VOXCPM_MODEL_ID)
                engine.pending_load = True  # type: ignore[attr-defined]
                _engine_singleton = engine
                return _engine_singleton
            except Exception as exc:
                logger.warning("NanoVLLM unavailable (%s); falling back to official engine", exc)

    _engine_singleton = _build_official_pool()
    return _engine_singleton


async def ensure_engine_loaded():
    """v3 — الضمانة: المحرك البديل المختار تلقائياً يُحمّل فعلياً هنا؛
    أي فشل = إعادة بناء فورية على المحرك الرسمي (الإقلاع لا يفشل أبداً)."""
    engine = get_engine()
    if not getattr(engine, "pending_load", False):
        return engine
    try:
        await engine.load()
        engine.pending_load = False  # type: ignore[attr-defined]
        logger.info("Alternative engine '%s' loaded successfully", getattr(engine, "name", "?"))
        return engine
    except Exception as exc:
        logger.warning("Alternative engine load failed (%s) — rebuilding on official engine", exc)
        global _engine_singleton
        _engine_singleton = None
        return _build_official_pool()


def reset_engine():
    """يُستدعى عند unload_all لإعادة بناء المحرك في التشغيل التالي."""
    global _engine_singleton
    _engine_singleton = None