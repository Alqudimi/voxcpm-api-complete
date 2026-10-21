"""مجدول GPU (v3-max) — منع التدافع وتوزيع الحمل الذكي على الكروت.

المشكلة التي يحلها: في v2 كان التوازي «أعمى» — كل مقاطع النص تُرمى على
نفس الكرت معاً (تدافع VRAM وإشغال kernels)، والطلبات المتزامنة تتكدس
على الكرت الأول دون وعي بحمل الكروت الأخرى.

الحل:
  - تصاريح (permits) لكل كرت: حد أقصى للتوليدات المتزامنة على كل جهاز —
    يحسبها AutoTuner حسب VRAM (2 على ≥16GB، 1 تحت).
  - اختيار «الأقل تحميلاً»: عند توفر عدة كروت، كل مهمة تذهب للكرت الذي
    فيه أكبر عدد تصاريح حرة — بدل الدوران الأعمى round-robin.
  - إحصاءات حية لأغراض المراقبة (/autotune).

لا يغيّر الحتمية: بذور المقاطع/المرشحين ثابتة لكل مهمة، فالترتيب لا يهم.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GPUScheduler:
    def __init__(self):
        self._lock = threading.Lock()
        self._permits: Dict[int, threading.BoundedSemaphore] = {}
        self._capacity: Dict[int, int] = {}
        self._active: Dict[int, int] = {}
        self._queued_total = 0
        self._rr = 0

    def configure(self, devices: List[int], permits_per_gpu: int):
        """إعادة ضبط التصاريح (تُستدعى من model_manager بعد بناء البركة)."""
        permits_per_gpu = max(1, int(permits_per_gpu))
        with self._lock:
            self._permits = {}
            self._capacity = {}
            self._active = {int(d): 0 for d in devices}
            for d in devices:
                self._permits[int(d)] = threading.BoundedSemaphore(permits_per_gpu)
                self._capacity[int(d)] = permits_per_gpu
            logger.info(
                "GPUScheduler configured: devices=%s permits/gpu=%d", devices, permits_per_gpu
            )

    def configured(self) -> bool:
        return bool(self._permits)

    @contextmanager
    def slot(self, device: int):
        """حجز تصريح توليد على كرت — يمنع التدافع (الطلب ينتظر دوره)."""
        sem = self._permits.get(int(device))
        if sem is None:
            # جهاز غير مضبوط (كسقوط آمن): بلا حدّ
            yield
            return
        acquired = False
        try:
            with self._lock:
                self._queued_total += 1
            acquired = sem.acquire(timeout=120)
            with self._lock:
                self._queued_total = max(0, self._queued_total - 1)
                self._active[int(device)] = self._active.get(int(device), 0) + 1
            yield
        finally:
            if acquired:
                sem.release()
                with self._lock:
                    self._active[int(device)] = max(0, self._active.get(int(device), 0) - 1)

    def least_loaded(self, devices: Optional[List[int]] = None) -> int:
        """الكرت ذو أكبر تصاريح حرة (تعادل → دوران عادل). بلا تصاريح → دوران بسيط."""
        with self._lock:
            devices = list(self._permits.keys()) if devices is None else list(devices)
            if not devices:
                return 0
            if not self._permits:
                idx = devices[self._rr % len(devices)]
                self._rr += 1
                return int(idx)
            best, best_free = None, -1
            for d in devices:
                if int(d) not in self._permits:
                    continue
                free = self._capacity.get(int(d), 1) - self._active.get(int(d), 0)
                if free > best_free or (free == best_free and best is not None and int(d) < best):
                    best, best_free = int(d), free
            return best if best is not None else int(devices[0])

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "devices": len(self._permits),
                "active_total": sum(self._active.values()),
                "queued_total": self._queued_total,
                **{f"active_cuda_{d}": n for d, n in sorted(self._active.items())},
            }


gpu_scheduler = GPUScheduler()
