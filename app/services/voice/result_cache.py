"""كاش النتائج الكاملة (v2-ultra) — أكبر معامل مضاعف في النظام.

المبدأ: الطلب الكامل (نص + متحدث + معاملات) حتمي إلى حد نموذج الاستدلال،
لذا عندما يتكرر **نفس الطلب بالضبط** نعيد OGG النهائي والبيانات المرافقة
مباشرة من الذاكرة — التوليد + ECAPA + الترميز كلها تُتخطى.

الاستخدام الحقيقي: عبارات IVR، إشعارات متكررة، إعلانات، اختبارات A/B،
إعادة المحاولة من العميل — نمط شائع جداً في الإنتاج.

المعامل الفعلي: أول طلب ~زمنه الطبيعي، كل تكرار لاحق < 100ms
(حتى 100× على الطلبات المتكررة).

الطبقات:
  L1: RAM LRU بميزانية بايتات (< 1ms)
  L2: ملفات .ogg + .meta.json على tmpfs (تنجو من إعادة التشغيل، 2-8ms)

المفتاح يشمل: النص المطبَّع + نص البرومبت + sha256 لبايتات المرجع النظيف
+ كل معاملات التوليد والترميز + نسخة مخطط الكاش (إبطال تلقائي عند الترقية).
العقد: تعطيل بمتغير واحد VM_RESULT_CACHE=off يعيد السلوك القديم حرفياً.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RESULT_CACHE_SCHEMA = "rc1"

try:
    from app.core.config import settings as _settings
except Exception:  # بيئة اختبار بدون إعدادات كاملة
    _settings = None


class ResultCache:
    """كاش نتائج clone الكاملة: (OGG bytes + metadata) بمفتاح حتمي."""

    def __init__(
        self,
        max_entries: int = 64,
        max_bytes: int = 256 << 20,
        ttl: int = 86400,
        disk_dir: Optional[Path] = None,
    ):
        if _settings is not None:
            try:
                # v3: الميزانيتان تُضبطان تلقائياً من حجم RAM عبر AutoTuner (صريح يتفوق)
                max_entries = int(_settings.result_cache_max)
                max_bytes = int(_settings.result_cache_max_mb) << 20
                ttl = int(_settings.VM_RESULT_CACHE_TTL)
            except Exception:
                pass
        self._max = max(1, max_entries)
        self._max_bytes = max(4 << 20, max_bytes)
        self._ttl = max(60, ttl)
        self._bytes = 0
        self._l1: OrderedDict[str, tuple] = OrderedDict()  # key -> (ts, ogg, meta)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._disk_dir: Optional[Path] = None
        if disk_dir is not None:
            self._disk_dir = disk_dir
        elif _settings is not None:
            try:
                self._disk_dir = _settings.resolve_temp_dir() / "result_cache"
            except Exception:
                self._disk_dir = None
        if self._disk_dir is not None:
            try:
                self._disk_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._disk_dir = None

    # ─── المفتاح الحتمي ───
    @staticmethod
    def build_key(
        target_text: str,
        prompt_text: str,
        clean_wav_bytes: bytes,
        cfg: float,
        timesteps: int,
        seed: int,
        retry_badcase: bool,
        retry_max_times: int,
        retry_ratio_threshold: float,
        silence_ms: int,
        bitrate: str,
        sample_rate: int,
        chunks_count: int = 1,
        engine_fingerprint: str = "",
    ) -> str:
        h = hashlib.sha256()
        h.update(RESULT_CACHE_SCHEMA.encode())
        h.update(b"|t|"); h.update(target_text.encode("utf-8"))
        h.update(b"|p|"); h.update(prompt_text.encode("utf-8"))
        h.update(b"|r|"); h.update(clean_wav_bytes)
        h.update(
            f"|{float(cfg):.6f}|{int(timesteps)}|{int(seed)}|{int(bool(retry_badcase))}"
            f"|{int(retry_max_times)}|{float(retry_ratio_threshold):.4f}"
            f"|{int(silence_ms)}|{bitrate}|{int(sample_rate)}"
            f"|{int(chunks_count)}|{engine_fingerprint}".encode()
        )
        return h.hexdigest()

    # ─── القراءة ───
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._enabled():
            return None
        now = time.time()
        with self._lock:
            entry = self._l1.get(key)
            if entry is not None:
                ts, ogg, meta = entry
                if now - ts <= self._ttl:
                    self._l1.move_to_end(key)
                    self.hits += 1
                    return {"ogg": ogg, "meta": meta}
                self._drop(key)
        if self._disk_dir is not None:
            try:
                ogg_path = self._disk_dir / f"{key}.ogg"
                meta_path = self._disk_dir / f"{key}.json"
                if ogg_path.exists() and meta_path.exists():
                    ts = ogg_path.stat().st_mtime
                    if now - ts <= self._ttl:
                        ogg = ogg_path.read_bytes()
                        meta = json.loads(meta_path.read_text("utf-8"))
                        self._l1_put(key, ogg, meta)
                        self.hits += 1
                        return {"ogg": ogg, "meta": meta}
                    ogg_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("result-cache L2 read failed: %s", exc)
        self.misses += 1
        return None

    # ─── الكتابة ───
    def put(self, key: str, ogg: bytes, meta: Dict[str, Any]):
        if not self._enabled() or not ogg:
            return
        self._l1_put(key, ogg, meta)
        if self._disk_dir is not None:
            try:
                ogg_path = self._disk_dir / f"{key}.ogg"
                meta_path = self._disk_dir / f"{key}.json"
                tmp = ogg_path.with_suffix(".tmp")
                tmp.write_bytes(ogg)
                os.replace(tmp, ogg_path)
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as exc:
                logger.debug("result-cache L2 write failed: %s", exc)

    # ─── إدارة L1 بميزانية بايتات ───
    def _l1_put(self, key: str, ogg: bytes, meta: Dict[str, Any]):
        with self._lock:
            if key in self._l1:
                self._drop(key)
            self._l1[key] = (time.time(), ogg, meta)
            self._bytes += len(ogg)
            while len(self._l1) > self._max or self._bytes > self._max_bytes:
                self._drop_last()

    def _drop(self, key: str):
        entry = self._l1.pop(key, None)
        if entry is not None:
            self._bytes -= len(entry[1])

    def _drop_last(self):
        if self._l1:
            _, entry = self._l1.popitem(last=False)
            self._bytes -= len(entry[1])

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._l1),
                "ram_mb": round(self._bytes / (1 << 20), 1),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(self.hits / max(1, self.hits + self.misses), 3),
            }

    def clear(self):
        with self._lock:
            self._l1.clear()
            self._bytes = 0
        if self._disk_dir is not None:
            try:
                for f in self._disk_dir.glob("*.ogg"):
                    f.unlink(missing_ok=True)
                for f in self._disk_dir.glob("*.json"):
                    f.unlink(missing_ok=True)
            except OSError:
                pass

    def _enabled(self) -> bool:
        if _settings is None:
            return True
        try:
            return _settings.result_cache_enabled
        except Exception:
            return True


result_cache = ResultCache()
