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
        # فهارس O(1) بالـ ID الرقمي والـ UUID النصي للعينات
        self._voices_by_id: Dict[int, FastVoiceSample] = {}
        self._voices_by_uuid: Dict[str, FastVoiceSample] = {}
        # فهارس للشخصيات والأصوات الرئيسية (Voice/Person ID & UUID)
        self._voices_by_person_id: Dict[int, FastVoiceSample] = {}
        self._voices_by_person_uuid: Dict[str, FastVoiceSample] = {}

        # فهارس عينات الضوضاء
        self._noise_by_id: Dict[int, FastNoiseSample] = {}
        self._noise_by_uuid: Dict[str, FastNoiseSample] = {}
        # فهارس لملفات تعريف الضوضاء (Noise Profile ID & UUID)
        self._noise_by_profile_id: Dict[int, FastNoiseSample] = {}
        self._noise_by_profile_uuid: Dict[str, FastNoiseSample] = {}

        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def register_voice_data(
        self,
        sample: Dict[str, Any],
        raw_audio_bytes: bytes,
        local_path_str: str,
    ) -> FastVoiceSample:
        """تسجيل عينة صوت في الذاكرة الحية RAM فورياً (لحظياً بدون إعادة تحميل كاملة)."""
        s_id = sample.get("id")
        s_uuid = str(sample.get("uuid"))
        local_path = Path(local_path_str)
        ref_text = (sample.get("reference_text") or sample.get("sample_text") or "").strip()

        item = FastVoiceSample(
            id=int(s_id) if s_id is not None else -1,
            uuid=s_uuid,
            person_name=sample.get("person_name") or sample.get("name", ""),
            sample_name=sample.get("name", ""),
            reference_text=ref_text,
            raw_audio_bytes=raw_audio_bytes,
            audio_path=str(local_path),
            duration=float(sample.get("duration") or 0.0),
            mime_type=sample.get("mime_type") or "audio/wav",
            filename=local_path.name,
        )

        if s_id is not None:
            self._voices_by_id[int(s_id)] = item
        self._voices_by_uuid[s_uuid] = item
        self._voices_by_uuid[str(s_id)] = item

        p_id = sample.get("person_id")
        p_uuid = sample.get("person_uuid")
        if p_id is not None and (int(p_id) not in self._voices_by_person_id or sample.get("is_default")):
            self._voices_by_person_id[int(p_id)] = item
        if p_uuid and (str(p_uuid) not in self._voices_by_person_uuid or sample.get("is_default")):
            self._voices_by_person_uuid[str(p_uuid)] = item

        return item

    def register_noise_data(
        self,
        sample: Dict[str, Any],
        audio_bytes: bytes,
        local_path_str: str,
    ) -> FastNoiseSample:
        """تسجيل عينة ضوضاء في الذاكرة الحية RAM فورياً (لحظياً بدون إعادة تحميل كاملة)."""
        s_id = sample.get("id")
        s_uuid = str(sample.get("uuid"))
        local_path = Path(local_path_str)

        item = FastNoiseSample(
            id=int(s_id) if s_id is not None else -1,
            uuid=s_uuid,
            profile_name=sample.get("noise_profile_name") or sample.get("name", ""),
            sample_name=sample.get("name", ""),
            audio_bytes=audio_bytes,
            audio_path=str(local_path),
            duration=float(sample.get("duration") or 0.0),
            mime_type=sample.get("mime_type") or "audio/mpeg",
            filename=local_path.name,
        )

        if s_id is not None:
            self._noise_by_id[int(s_id)] = item
        self._noise_by_uuid[s_uuid] = item
        self._noise_by_uuid[str(s_id)] = item

        np_id = sample.get("noise_profile_id")
        np_uuid = sample.get("noise_profile_uuid")
        if np_id is not None and (int(np_id) not in self._noise_by_profile_id or sample.get("is_default")):
            self._noise_by_profile_id[int(np_id)] = item
        if np_uuid and (str(np_uuid) not in self._noise_by_profile_uuid or sample.get("is_default")):
            self._noise_by_profile_uuid[str(np_uuid)] = item

        return item

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

                    # دمج معلومات الشخص في العينة إن لم تكن متوفرة
                    if "person_name" not in sample:
                        sample["person_name"] = voice.get("name", "")
                    if "person_id" not in sample:
                        sample["person_id"] = voice.get("id")
                    if "person_uuid" not in sample:
                        sample["person_uuid"] = voice.get("uuid")

                    self.register_voice_data(sample, audio_data, str(local_path))
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

                    if "noise_profile_name" not in sample:
                        sample["noise_profile_name"] = profile.get("name", "")
                    if "noise_profile_id" not in sample:
                        sample["noise_profile_id"] = profile.get("id")
                    if "noise_profile_uuid" not in sample:
                        sample["noise_profile_uuid"] = profile.get("uuid")

                    self.register_noise_data(sample, audio_data, str(local_path))
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
        """استرجاع فوري لعينة الصوت من RAM بواسطة ID رقمي أو UUID (عينة أو شخص/صوت)."""
        if identifier is None:
            return None
        # محاولة البحث كـ int
        if isinstance(identifier, int):
            item = self._voices_by_id.get(identifier) or self._voices_by_person_id.get(identifier)
            if item:
                return item
        # محاولة البحث كـ str
        s_val = str(identifier).strip()
        item = self._voices_by_uuid.get(s_val) or self._voices_by_person_uuid.get(s_val)
        if item:
            return item
        if s_val.isdigit():
            i_val = int(s_val)
            return self._voices_by_id.get(i_val) or self._voices_by_person_id.get(i_val)
        return None

    def get_noise(self, identifier: Union[int, str]) -> Optional[FastNoiseSample]:
        """استرجاع فوري لعينة الضوضاء من RAM بواسطة ID رقمي أو UUID (عينة أو ملف تعريف)."""
        if identifier is None:
            return None
        # محاولة البحث كـ int
        if isinstance(identifier, int):
            item = self._noise_by_id.get(identifier) or self._noise_by_profile_id.get(identifier)
            if item:
                return item
        # محاولة البحث كـ str
        s_val = str(identifier).strip()
        item = self._noise_by_uuid.get(s_val) or self._noise_by_profile_uuid.get(s_val)
        if item:
            return item
        if s_val.isdigit():
            i_val = int(s_val)
            return self._noise_by_id.get(i_val) or self._noise_by_profile_id.get(i_val)
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
