"""كاش حالة المتحدث — أعظم مكسب بلا مخاطرة (المرحلة 3 من خطة التسريع).

المبدأ: التحضير/تقرير الجودة/النسخ الثلاث دوال حتمية صرفة في مدخلاتها،
لذا نتائجها تُخزَّن وتُعاد **مطابقة بتّياً** (نفس sha256 لمخرجات PCM).

الطبقات:
  L1: RAM LRU داخل العملية (أسرع: <1ms)
  L2: ملفات على tmpfs (تنجو من إعادة تشغيل العملية، 2-8ms)
المفتاح يتضمن: بايتات المرجع + المعاملات + معرف نسخة الكاش (إبطال تلقائي عند الترقية).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "v2"

try:
    from app.core.config import settings as _settings
except Exception:  # بيئة اختبار بدون إعدادات كاملة
    _settings = None


def _payload_ok(entry) -> bool:
    return isinstance(entry, tuple) and len(entry) == 2


class SpeakerStateCache:
    def __init__(self, max_entries: int = 12, ttl: int = 86400, disk_dir: Optional[Path] = None):
        if _settings is not None:
            try:
                # v3: الحجم يُضبط تلقائياً من حجم RAM عبر AutoTuner (صريح يتفوق)
                max_entries = int(_settings.speaker_cache_max)
                ttl = int(_settings.VM_SPEAKER_CACHE_TTL)
            except Exception:
                pass
        self._max = max(1, max_entries)
        self._ttl = max(60, ttl)
        self._l1: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()
        self._disk_dir: Optional[Path] = None
        if disk_dir is not None:
            self._disk_dir = disk_dir
        elif _settings is not None:
            try:
                self._disk_dir = _settings.resolve_temp_dir() / "speaker_states"
            except Exception:
                self._disk_dir = None
        if self._disk_dir is not None:
            try:
                self._disk_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._disk_dir = None

    # ─── المفاتيح ───
    def key_prepare(self, raw_bytes: bytes, target_sr: int, top_db: int) -> str:
        h = hashlib.sha256()
        h.update(b"prepare|"); h.update(CACHE_SCHEMA_VERSION.encode())
        h.update(b"|"); h.update(raw_bytes)
        h.update(f"|{target_sr}|{top_db}".encode())
        return h.hexdigest()

    def key_qc(self, clean_bytes: bytes) -> str:
        h = hashlib.sha256()
        h.update(b"qc|"); h.update(CACHE_SCHEMA_VERSION.encode()); h.update(b"|"); h.update(clean_bytes)
        return h.hexdigest()

    def key_transcript(self, clean_bytes: bytes, model_name: str, vad: bool) -> str:
        h = hashlib.sha256()
        h.update(b"transcript|"); h.update(CACHE_SCHEMA_VERSION.encode())
        h.update(b"|"); h.update(clean_bytes)
        h.update(f"|{model_name}|{int(bool(vad))}".encode())
        return h.hexdigest()

    def key_embedding(self, clean_bytes: bytes) -> str:
        """مفتاح embedding ECAPA للمرجع (v2-ultra) — نفس البايتات = نفس المتجه bit-exact."""
        h = hashlib.sha256()
        h.update(b"ecapa_emb|"); h.update(CACHE_SCHEMA_VERSION.encode())
        h.update(b"|"); h.update(clean_bytes)
        return h.hexdigest()

    def get_ecapa_embedding(self, clean_bytes: bytes):
        if not self._enabled():
            return None
        return self.get(self.key_embedding(clean_bytes))

    def put_ecapa_embedding(self, clean_bytes: bytes, emb_np):
        """emb_np: مصفوفة float32 CPU (192-d) — تُحفظ عبر pickle في L1/L2."""
        if self._enabled():
            self.put(self.key_embedding(clean_bytes), {"emb": emb_np})

    # ─── القراءة/الكتابة العامة ───
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            entry = self._l1.get(key)
            if entry is not None:
                ts, payload = entry
                if now - ts <= self._ttl:
                    self._l1.move_to_end(key)
                    return payload
                self._l1.pop(key, None)
        # L2 (tmpfs)
        if self._disk_dir is not None:
            try:
                path = self._disk_dir / f"{key}.pkl"
                if path.exists():
                    with open(path, "rb") as fh:
                        ts, payload = pickle.load(fh)
                    if now - ts <= self._ttl:
                        self._l1_put(key, payload)
                        return payload
                    path.unlink(missing_ok=True)
            except Exception as exc:  # تلف ملف الكاش لا يكسر الطلب أبداً
                logger.debug("speaker-cache L2 read failed: %s", exc)
        return None

    def put(self, key: str, payload: Dict[str, Any]):
        self._l1_put(key, payload)
        if self._disk_dir is not None:
            try:
                path = self._disk_dir / f"{key}.pkl"
                tmp = path.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump((time.time(), payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception as exc:
                logger.debug("speaker-cache L2 write failed: %s", exc)

    def _l1_put(self, key: str, payload):
        with self._lock:
            self._l1[key] = (time.time(), payload)
            self._l1.move_to_end(key)
            while len(self._l1) > self._max:
                self._l1.popitem(last=False)

    def clear(self):
        with self._lock:
            self._l1.clear()
        if self._disk_dir is not None:
            try:
                for f in self._disk_dir.glob("*.pkl"):
                    f.unlink(missing_ok=True)
            except OSError:
                pass

    # ─── واجهات عالية المستوى لدوال الخدمة الثلاث ───
    def get_prepare(self, raw_bytes: bytes, target_sr: int, top_db: int) -> Optional[Dict[str, Any]]:
        if not self._enabled():
            return None
        return self.get(self.key_prepare(raw_bytes, target_sr, top_db))

    def put_prepare(self, raw_bytes: bytes, target_sr: int, top_db: int, clean_wav_bytes: bytes, duration: float):
        if self._enabled():
            self.put(self.key_prepare(raw_bytes, target_sr, top_db),
                     {"clean_wav_bytes": clean_wav_bytes, "duration": duration})

    def get_qc(self, clean_bytes: bytes) -> Optional[str]:
        if not self._enabled():
            return None
        payload = self.get(self.key_qc(clean_bytes))
        return payload.get("qc_text") if payload else None

    def put_qc(self, clean_bytes: bytes, qc_text: str):
        if self._enabled():
            self.put(self.key_qc(clean_bytes), {"qc_text": qc_text})

    def get_transcript(self, clean_bytes: bytes, model_name: str, vad: bool) -> Optional[str]:
        if not self._enabled():
            return None
        payload = self.get(self.key_transcript(clean_bytes, model_name, vad))
        return payload.get("text") if payload else None

    def put_transcript(self, clean_bytes: bytes, model_name: str, vad: bool, text: str):
        if self._enabled():
            self.put(self.key_transcript(clean_bytes, model_name, vad), {"text": text})

    def _enabled(self) -> bool:
        if _settings is None:
            return True
        try:
            return _settings.speaker_cache_enabled
        except Exception:
            return True


# نسخة عملية واحدة مشتركة (LRU عبر الطلبات)
speaker_state_cache = SpeakerStateCache()
