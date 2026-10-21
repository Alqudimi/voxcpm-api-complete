"""توقيت المراحل لكل طلب — أساس منظومة القياس (المرحلة 0 من خطة التسريع).

يوفر سياقاً متزامناً/غير متزامن واحد لتجميع زمن كل مرحلة داخل الطلب،
ويطبع سطر JSON واحداً عند النهاية عندما VM_LOG_STAGES=on. آمن الخيوط.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

logger = logging.getLogger("voxmixer.stages")

_LOCAL = threading.local()


class StageTimer:
    """مجمّع توقيتات مرحلي لطلب واحد (thread-local لكل مسار تنفيذ)."""

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or ""
        self.stages: Dict[str, float] = {}
        self.extras: Dict[str, str] = {}
        self._t0 = time.perf_counter()
        self._finished = False

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stages[name] = self.stages.get(name, 0.0) + elapsed_ms

    def note(self, key: str, value: str):
        self.extras[key] = str(value)

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def finish_and_log(self):
        """يطبع سطر JSON واحداً بمجموع كل المراحل (مجموعها ≈ الجداري ±3%)."""
        if self._finished:
            return
        self._finished = True
        payload = {
            "request_id": self.request_id,
            "total_ms": round(self.total_ms, 1),
            "stages_ms": {k: round(v, 1) for k, v in self.stages.items()},
        }
        if self.extras:
            payload["extras"] = self.extras
        logger.info("STAGES %s", json.dumps(payload, ensure_ascii=False))

    def snapshot(self) -> Dict[str, float]:
        return {k: round(v, 1) for k, v in self.stages.items()}


def current_timer() -> Optional[StageTimer]:
    return getattr(_LOCAL, "timer", None)


def start_timer(request_id: Optional[str] = None) -> StageTimer:
    timer = StageTimer(request_id)
    _LOCAL.timer = timer
    return timer


def clear_timer():
    _LOCAL.timer = None
