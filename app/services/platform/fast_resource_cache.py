from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("voxcpm.fast_cache")


@dataclass(slots=True)
class FastVoiceSample:
    """كائن عينة الصوت المخزن في RAM للوصول بزمن شبه صفري (Sub-microsecond O(1))."""
    id: int
    uuid: str
    person_name: str
    sample_name: str
    reference_text: str
    raw_audio_bytes: bytes
    audio_path: str
    duration: float
    mime_type: str
    filename: str


@dataclass(slots=True)
class FastNoiseSample:
    """كائن عينة الضوضاء المخزن في RAM للوصول بزمن شبه صفري (Sub-microsecond O(1))."""
    id: int
    uuid: str
    profile_name: str
    sample_name: str
    audio_bytes: bytes
    audio_path: str
    duration: float
    mime_type: str
    filename: str


class FastResourceCache:
    """مخزن فائق السرعة في الذاكرة العشوائية RAM (In-Memory Zero-Disk Zero-SQL Cache):
    يُحمّل كافة عينات الأصوات وعينات الضوضاء وبياناتها الصوتية في RAM عند الإقلاع لمرة واحدة،
    لتتم عمليات البحث والجلب خلال أقل من جزء من المليون من الثانية بدون أي استعلامات SQLite
    وبدون أي قراءة من القرص الصلب أثناء معالجة الطلبات.
    """

    def __init__(self):
        # فهارس O(1) بالـ ID الرقمي والـ UUID النصي
        self._voices_by_id: Dict[int, FastVoiceSample] = {}
        self._voices_by_uuid: Dict[str, FastVoiceSample] = {}
        self._noise_by_id: Dict[int, FastNoiseSample] = {}
        self._noise_by_uuid: Dict[str, FastNoiseSample] = {}
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def warm_up(self, db_path: Optional[str | Path] = None) -> int:
        """تحميل كافة الملفات الصوتية والنصوص والبيانات من SQLite والقرص إلى RAM."""
        from .database import PlatformDatabase
        from app.core.config import settings

        db = PlatformDatabase(db_path or settings.PLATFORM_DB_PATH)
        voices = db.list_voices()
        noise_profiles = db.list_noise_profiles()

        loaded_voices = 0
        loaded_noise = 0

        # 1. تحميل عينات الأصوات في الذاكرة RAM
        for voice in voices:
            for sample in voice.get("samples", []):
                s_id = sample.get("id")
                s_uuid = str(sample.get("uuid"))
                local_path_str = sample.get("local_cleaned_audio_path") or sample.get("local_audio_path")
                if not local_path_str:
                    continue

                local_path = Path(local_path_str)
                if not local_path.is_file():
                    continue

                try:
                    audio_data = local_path.read_bytes()
                    if not audio_data:
                        continue

                    ref_text = (sample.get("reference_text") or sample.get("sample_text") or "").strip()
                    item = FastVoiceSample(
                        id=int(s_id) if s_id is not None else -1,
                        uuid=s_uuid,
                        person_name=voice.get("name", ""),
                        sample_name=sample.get("name", ""),
                        reference_text=ref_text,
                        raw_audio_bytes=audio_data,
                        audio_path=str(local_path),
                        duration=float(sample.get("duration") or 0.0),
                        mime_type=sample.get("mime_type") or "audio/wav",
                        filename=local_path.name,
                    )

                    if s_id is not None:
                        self._voices_by_id[int(s_id)] = item
                    self._voices_by_uuid[s_uuid] = item
                    self._voices_by_uuid[str(s_id)] = item  # بحث بالـ ID كسلسلة نصية أيضاً
                    loaded_voices += 1
                except Exception as exc:
                    logger.warning("فشل تحميل عينة الصوت %s في RAM: %s", s_uuid, exc)

        # 2. تحميل عينات الضوضاء في الذاكرة RAM
        for profile in noise_profiles:
            for sample in profile.get("samples", []):
                s_id = sample.get("id")
                s_uuid = str(sample.get("uuid"))
                local_path_str = sample.get("local_audio_path")
                if not local_path_str:
                    continue

                local_path = Path(local_path_str)
                if not local_path.is_file():
                    continue

                try:
                    audio_data = local_path.read_bytes()
                    if not audio_data:
                        continue

                    item = FastNoiseSample(
                        id=int(s_id) if s_id is not None else -1,
                        uuid=s_uuid,
                        profile_name=profile.get("name", ""),
                        sample_name=sample.get("name", ""),
                        audio_bytes=audio_data,
                        audio_path=str(local_path),
                        duration=float(sample.get("duration") or 0.0),
                        mime_type=sample.get("mime_type") or "audio/mpeg",
                        filename=local_path.name,
                    )

                    if s_id is not None:
                        self._noise_by_id[int(s_id)] = item
                    self._noise_by_uuid[s_uuid] = item
                    self._noise_by_uuid[str(s_id)] = item  # بحث بالـ ID كسلسلة نصية أيضاً
                    loaded_noise += 1
                except Exception as exc:
                    logger.warning("فشل تحميل عينة الضوضاء %s في RAM: %s", s_uuid, exc)

        self._loaded = True
        logger.info(
            "⚡ تم تحميل الذاكرة الفائقة RAM بنجاح: %d عينة صوتية و %d عينة ضوضاء جاهزة للاسترجاع الفوري (0ms I/O).",
            loaded_voices,
            loaded_noise,
        )
        return loaded_voices + loaded_noise

    def get_voice(self, identifier: Union[int, str]) -> Optional[FastVoiceSample]:
        """استرجاع فوري لعينة الصوت من RAM بواسطة ID رقمي أو UUID."""
        if identifier is None:
            return None
        # محاولة البحث كـ int
        if isinstance(identifier, int):
            item = self._voices_by_id.get(identifier)
            if item:
                return item
        # محاولة البحث كـ str
        s_val = str(identifier).strip()
        item = self._voices_by_uuid.get(s_val)
        if item:
            return item
        if s_val.isdigit():
            return self._voices_by_id.get(int(s_val))
        return None

    def get_noise(self, identifier: Union[int, str]) -> Optional[FastNoiseSample]:
        """استرجاع فوري لعينة الضوضاء من RAM بواسطة ID رقمي أو UUID."""
        if identifier is None:
            return None
        # محاولة البحث كـ int
        if isinstance(identifier, int):
            item = self._noise_by_id.get(identifier)
            if item:
                return item
        # محاولة البحث كـ str
        s_val = str(identifier).strip()
        item = self._noise_by_uuid.get(s_val)
        if item:
            return item
        if s_val.isdigit():
            return self._noise_by_id.get(int(s_val))
        return None

    def stats(self) -> Dict[str, Any]:
        """إحصائيات الذاكرة السريعة."""
        return {
            "is_loaded": self._loaded,
            "cached_voice_samples_count": len(self._voices_by_id),
            "cached_noise_samples_count": len(self._noise_by_id),
        }


# نسخة عامة وحيدة (Singleton)
fast_resource_cache = FastResourceCache()
