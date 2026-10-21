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
