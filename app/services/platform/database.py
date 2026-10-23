import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class PlatformDatabase:
    """إدارة قاعدة بيانات SQLite المحلية لتخزين عينات الأصوات والضوضاء وبيانات المنصة."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS voices (
                    id INTEGER PRIMARY KEY,
                    uuid TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    photo_url TEXT,
                    language TEXT,
                    dialect TEXT,
                    gender TEXT,
                    status TEXT,
                    samples_count INTEGER DEFAULT 0,
                    raw_json TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS voice_samples (
                    id INTEGER PRIMARY KEY,
                    uuid TEXT UNIQUE NOT NULL,
                    person_id INTEGER,
                    person_uuid TEXT,
                    person_name TEXT,
                    name TEXT,
                    sample_text TEXT,
                    reference_text TEXT,
                    has_reference_text INTEGER DEFAULT 0,
                    duration REAL,
                    file_size INTEGER,
                    mime_type TEXT,
                    original_filename TEXT,
                    audio_url TEXT,
                    cleaned_audio_url TEXT,
                    is_cleaned INTEGER DEFAULT 0,
                    is_default INTEGER DEFAULT 0,
                    status TEXT,
                    local_audio_path TEXT,
                    local_cleaned_audio_path TEXT,
                    raw_json TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS noise_profiles (
                    id INTEGER PRIMARY KEY,
                    uuid TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    photo_url TEXT,
                    category TEXT,
                    description TEXT,
                    status TEXT,
                    samples_count INTEGER DEFAULT 0,
                    raw_json TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS noise_samples (
                    id INTEGER PRIMARY KEY,
                    uuid TEXT UNIQUE NOT NULL,
                    noise_profile_id INTEGER,
                    noise_profile_uuid TEXT,
                    noise_profile_name TEXT,
                    name TEXT,
                    description TEXT,
                    duration REAL,
                    file_size INTEGER,
                    mime_type TEXT,
                    original_filename TEXT,
                    audio_url TEXT,
                    is_default INTEGER DEFAULT 0,
                    status TEXT,
                    local_audio_path TEXT,
                    raw_json TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_voice_uuid ON voices(uuid);
                CREATE INDEX IF NOT EXISTS idx_voice_sample_uuid ON voice_samples(uuid);
                CREATE INDEX IF NOT EXISTS idx_voice_sample_person ON voice_samples(person_uuid);
                CREATE INDEX IF NOT EXISTS idx_noise_uuid ON noise_profiles(uuid);
                CREATE INDEX IF NOT EXISTS idx_noise_sample_uuid ON noise_samples(uuid);
                CREATE INDEX IF NOT EXISTS idx_noise_sample_profile ON noise_samples(noise_profile_uuid);
                """
            )

    def set_meta(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at;
                """,
                (key, value, now),
            )

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def has_data(self) -> bool:
        """فحص ما إذا كانت قاعدة البيانات تحتوي على أصوات وعينات مسجلة بالفعل."""
        with self._get_connection() as conn:
            v_count = conn.execute("SELECT COUNT(*) AS c FROM voices").fetchone()["c"]
            s_count = conn.execute("SELECT COUNT(*) AS c FROM voice_samples").fetchone()["c"]
            n_count = conn.execute("SELECT COUNT(*) AS c FROM noise_samples").fetchone()["c"]
            return (v_count > 0 or n_count > 0) and s_count > 0

    def save_voices_and_noise(
        self,
        voices: List[Dict[str, Any]],
        noise_profiles: List[Dict[str, Any]],
        local_files_map: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """حفظ أو تحديث بيانات الأصوات والضوضاء والمسارات المحلية في عملية واحدة."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            # 1. حفظ الأصوات وعيناتها
            for voice in voices:
                v_id = voice.get("id")
                v_uuid = str(voice.get("uuid"))
                conn.execute(
                    """
                    INSERT INTO voices (id, uuid, name, photo_url, language, dialect, gender, status, samples_count, raw_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(uuid) DO UPDATE SET
                        name=excluded.name,
                        photo_url=excluded.photo_url,
                        language=excluded.language,
                        dialect=excluded.dialect,
                        gender=excluded.gender,
                        status=excluded.status,
                        samples_count=excluded.samples_count,
                        raw_json=excluded.raw_json,
                        updated_at=excluded.updated_at;
                    """,
                    (
                        v_id,
                        v_uuid,
                        voice.get("name", ""),
                        voice.get("photo_url"),
                        voice.get("language"),
                        voice.get("dialect"),
                        voice.get("gender"),
                        voice.get("status"),
                        voice.get("samples_count", 0),
                        json.dumps(voice, ensure_ascii=False),
                        now,
                    ),
                )

                for sample in voice.get("samples", []):
                    s_id = sample.get("id")
                    s_uuid = str(sample.get("uuid"))
                    file_info = local_files_map.get(s_uuid, {})
                    local_audio = file_info.get("local_audio_path")
                    local_cleaned = file_info.get("local_cleaned_audio_path")

                    conn.execute(
                        """
                        INSERT INTO voice_samples (
                            id, uuid, person_id, person_uuid, person_name, name,
                            sample_text, reference_text, has_reference_text, duration,
                            file_size, mime_type, original_filename, audio_url,
                            cleaned_audio_url, is_cleaned, is_default, status,
                            local_audio_path, local_cleaned_audio_path, raw_json, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(uuid) DO UPDATE SET
                            person_id=excluded.person_id,
                            person_uuid=excluded.person_uuid,
                            person_name=excluded.person_name,
                            name=excluded.name,
                            sample_text=excluded.sample_text,
                            reference_text=excluded.reference_text,
                            has_reference_text=excluded.has_reference_text,
                            duration=excluded.duration,
                            file_size=excluded.file_size,
                            mime_type=excluded.mime_type,
                            original_filename=excluded.original_filename,
                            audio_url=excluded.audio_url,
                            cleaned_audio_url=excluded.cleaned_audio_url,
                            is_cleaned=excluded.is_cleaned,
                            is_default=excluded.is_default,
                            status=excluded.status,
                            local_audio_path=COALESCE(excluded.local_audio_path, voice_samples.local_audio_path),
                            local_cleaned_audio_path=COALESCE(excluded.local_cleaned_audio_path, voice_samples.local_cleaned_audio_path),
                            raw_json=excluded.raw_json,
                            updated_at=excluded.updated_at;
                        """,
                        (
                            s_id,
                            s_uuid,
                            sample.get("person_id") or v_id,
                            sample.get("person_uuid") or v_uuid,
                            sample.get("person_name") or voice.get("name"),
                            sample.get("name", ""),
                            sample.get("sample_text", ""),
                            sample.get("reference_text") or sample.get("sample_text", ""),
                            1 if sample.get("has_reference_text") else 0,
                            sample.get("duration"),
                            sample.get("file_size"),
                            sample.get("mime_type"),
                            sample.get("original_filename"),
                            sample.get("audio_url"),
                            sample.get("cleaned_audio_url"),
                            1 if sample.get("is_cleaned") else 0,
                            1 if sample.get("is_default") else 0,
                            sample.get("status"),
                            local_audio,
                            local_cleaned,
                            json.dumps(sample, ensure_ascii=False),
                            now,
                        ),
                    )

            # 2. حفظ ملفات الضوضاء وعيناتها
            for profile in noise_profiles:
                p_id = profile.get("id")
                p_uuid = str(profile.get("uuid"))
                conn.execute(
                    """
                    INSERT INTO noise_profiles (id, uuid, name, photo_url, category, description, status, samples_count, raw_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(uuid) DO UPDATE SET
                        name=excluded.name,
                        photo_url=excluded.photo_url,
                        category=excluded.category,
                        description=excluded.description,
                        status=excluded.status,
                        samples_count=excluded.samples_count,
                        raw_json=excluded.raw_json,
                        updated_at=excluded.updated_at;
                    """,
                    (
                        p_id,
                        p_uuid,
                        profile.get("name", ""),
                        profile.get("photo_url"),
                        profile.get("category"),
                        profile.get("description"),
                        profile.get("status"),
                        profile.get("samples_count", 0),
                        json.dumps(profile, ensure_ascii=False),
                        now,
                    ),
                )

                for sample in profile.get("samples", []):
                    s_id = sample.get("id")
                    s_uuid = str(sample.get("uuid"))
                    file_info = local_files_map.get(s_uuid, {})
                    local_audio = file_info.get("local_audio_path")

                    conn.execute(
                        """
                        INSERT INTO noise_samples (
                            id, uuid, noise_profile_id, noise_profile_uuid, noise_profile_name,
                            name, description, duration, file_size, mime_type,
                            original_filename, audio_url, is_default, status,
                            local_audio_path, raw_json, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(uuid) DO UPDATE SET
                            noise_profile_id=excluded.noise_profile_id,
                            noise_profile_uuid=excluded.noise_profile_uuid,
                            noise_profile_name=excluded.noise_profile_name,
                            name=excluded.name,
                            description=excluded.description,
                            duration=excluded.duration,
                            file_size=excluded.file_size,
                            mime_type=excluded.mime_type,
                            original_filename=excluded.original_filename,
                            audio_url=excluded.audio_url,
                            is_default=excluded.is_default,
                            status=excluded.status,
                            local_audio_path=COALESCE(excluded.local_audio_path, noise_samples.local_audio_path),
                            raw_json=excluded.raw_json,
                            updated_at=excluded.updated_at;
                        """,
                        (
                            s_id,
                            s_uuid,
                            sample.get("noise_profile_id") or p_id,
                            sample.get("noise_profile_uuid") or p_uuid,
                            sample.get("noise_profile_name") or profile.get("name"),
                            sample.get("name", ""),
                            sample.get("description", ""),
                            sample.get("duration"),
                            sample.get("file_size"),
                            sample.get("mime_type"),
                            sample.get("original_filename"),
                            sample.get("audio_url"),
                            1 if sample.get("is_default") else 0,
                            sample.get("status"),
                            local_audio,
                            json.dumps(sample, ensure_ascii=False),
                            now,
                        ),
                    )

            conn.execute(
                """
                INSERT INTO meta (key, value, updated_at) VALUES ('last_sync_at', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at;
                """,
                (now, now),
            )

    def get_voice_sample(self, uuid_or_id: str | int) -> Optional[Dict[str, Any]]:
        """البحث عن عينة صوت بواسطة sample uuid/id أو person/voice uuid/id."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM voice_samples
                WHERE uuid=? OR id=? OR person_uuid=? OR person_id=?
                ORDER BY (uuid=?) DESC, (id=?) DESC, is_default DESC, id ASC
                LIMIT 1
            """
            s = str(uuid_or_id)
            row = conn.execute(query, (s, s, s, s, s, s)).fetchone()
            return dict(row) if row else None

    def get_noise_sample(self, uuid_or_id: str | int) -> Optional[Dict[str, Any]]:
        """البحث عن عينة ضوضاء بواسطة sample uuid/id أو noise profile uuid/id."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM noise_samples
                WHERE uuid=? OR id=? OR noise_profile_uuid=? OR noise_profile_id=?
                ORDER BY (uuid=?) DESC, (id=?) DESC, is_default DESC, id ASC
                LIMIT 1
            """
            s = str(uuid_or_id)
            row = conn.execute(query, (s, s, s, s, s, s)).fetchone()
            return dict(row) if row else None

    def list_voices(self) -> List[Dict[str, Any]]:
        """استرجاع قائمة الأصوات مع كافة عيناتها."""
        with self._get_connection() as conn:
            voices_rows = conn.execute("SELECT * FROM voices ORDER BY id ASC").fetchall()
            result = []
            for v in voices_rows:
                v_dict = dict(v)
                samples = conn.execute(
                    "SELECT * FROM voice_samples WHERE person_uuid=? ORDER BY is_default DESC, id ASC",
                    (v_dict["uuid"],),
                ).fetchall()
                v_dict["samples"] = [dict(s) for s in samples]
                result.append(v_dict)
            return result

    def list_noise_profiles(self) -> List[Dict[str, Any]]:
        """استرجاع قائمة ملفات الضوضاء مع كافة عيناتها."""
        with self._get_connection() as conn:
            profiles_rows = conn.execute("SELECT * FROM noise_profiles ORDER BY id ASC").fetchall()
            result = []
            for p in profiles_rows:
                p_dict = dict(p)
                samples = conn.execute(
                    "SELECT * FROM noise_samples WHERE noise_profile_uuid=? ORDER BY is_default DESC, id ASC",
                    (p_dict["uuid"],),
                ).fetchall()
                p_dict["samples"] = [dict(s) for s in samples]
                result.append(p_dict)
            return result

    def update_voice_sample_path(self, uuid_or_id: str | int, local_path: str, is_cleaned: bool = False) -> None:
        """تحديث المسار المحلي لعينة الصوت فور تنزيلها."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            col = "local_cleaned_audio_path" if is_cleaned else "local_audio_path"
            s = str(uuid_or_id)
            conn.execute(
                f"UPDATE voice_samples SET {col}=?, updated_at=? WHERE uuid=? OR id=?",
                (local_path, now, s, s),
            )

    def update_noise_sample_path(self, uuid_or_id: str | int, local_path: str) -> None:
        """تحديث المسار المحلي لعينة الضوضاء فور تنزيلها."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            s = str(uuid_or_id)
            conn.execute(
                "UPDATE noise_samples SET local_audio_path=?, updated_at=? WHERE uuid=? OR id=?",
                (local_path, now, s, s),
            )

    def get_summary(self) -> Dict[str, Any]:
        """إرجاع ملخص إحصائي للموارد المخزنة في SQLite وحالة آخر مزامنة."""
        with self._get_connection() as conn:
            v_cnt = conn.execute("SELECT COUNT(*) AS c FROM voices").fetchone()["c"]
            vs_cnt = conn.execute("SELECT COUNT(*) AS c FROM voice_samples").fetchone()["c"]
            np_cnt = conn.execute("SELECT COUNT(*) AS c FROM noise_profiles").fetchone()["c"]
            ns_cnt = conn.execute("SELECT COUNT(*) AS c FROM noise_samples").fetchone()["c"]
            last_sync = self.get_meta("last_sync_at")
            server_registered_at = self.get_meta("server_registered_at")
            last_registered_url = self.get_meta("last_registered_server_url")
            return {
                "total_voices": v_cnt,
                "total_voice_samples": vs_cnt,
                "total_noise_profiles": np_cnt,
                "total_noise_samples": ns_cnt,
                "last_sync_at": last_sync,
                "server_registered_at": server_registered_at,
                "last_registered_server_url": last_registered_url,
                "database_path": str(self.db_path),
            }
