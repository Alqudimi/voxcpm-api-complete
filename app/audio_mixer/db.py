import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None


def now():
    return datetime.now(UTC).isoformat()


def new_id():
    return secrets.token_hex(16)


logger = logging.getLogger(__name__)


SQLITE_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS categories(id TEXT PRIMARY KEY, slug TEXT UNIQUE NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, category_id TEXT, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', original_filename TEXT NOT NULL, storage_key TEXT NOT NULL UNIQUE, content_type TEXT NOT NULL, format TEXT, codec TEXT, duration_ms INTEGER, sample_rate INTEGER, channels INTEGER, bit_depth INTEGER, bit_rate INTEGER, loudness_lufs REAL, true_peak_dbtp REAL, rms_db REAL, sha256 TEXT NOT NULL UNIQUE, source TEXT, license TEXT, attribution TEXT, quality_score REAL, status TEXT NOT NULL, is_recommended INTEGER NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}', audio_bytes BLOB NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT, FOREIGN KEY(category_id) REFERENCES categories(id));
CREATE TABLE IF NOT EXISTS api_keys(id TEXT PRIMARY KEY, key_prefix TEXT NOT NULL, key_hash TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', scopes_json TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT, revoked_at TEXT, last_used_at TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, status TEXT NOT NULL, source_reference TEXT, noise_reference TEXT, configuration_json TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, failed_at TEXT, error_code TEXT, error_message TEXT, output_reference TEXT, metrics_json TEXT NOT NULL DEFAULT '{}', processing_time_ms INTEGER, source_audio BLOB, noise_audio BLOB, output_audio BLOB);
CREATE TABLE IF NOT EXISTS job_assets(job_id TEXT NOT NULL, asset_id TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(job_id, asset_id, role), FOREIGN KEY(job_id) REFERENCES jobs(id), FOREIGN KEY(asset_id) REFERENCES assets(id));
CREATE TABLE IF NOT EXISTS service_settings(service_name TEXT PRIMARY KEY, settings_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS service_admin_events(id TEXT PRIMARY KEY, service_name TEXT NOT NULL, event_type TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_assets_category_status ON assets(category_id,status);
CREATE INDEX IF NOT EXISTS ix_assets_created ON assets(created_at);
CREATE INDEX IF NOT EXISTS ix_jobs_status_created ON jobs(status,created_at);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS categories(id TEXT PRIMARY KEY, slug TEXT UNIQUE NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, category_id TEXT REFERENCES categories(id), title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', original_filename TEXT NOT NULL, storage_key TEXT UNIQUE NOT NULL, content_type TEXT NOT NULL, format TEXT, codec TEXT, duration_ms INTEGER, sample_rate INTEGER, channels INTEGER, bit_depth INTEGER, bit_rate INTEGER, loudness_lufs DOUBLE PRECISION, true_peak_dbtp DOUBLE PRECISION, rms_db DOUBLE PRECISION, sha256 TEXT UNIQUE NOT NULL, source TEXT, license TEXT, attribution TEXT, quality_score DOUBLE PRECISION, status TEXT NOT NULL, is_recommended BOOLEAN NOT NULL DEFAULT FALSE, metadata_json TEXT NOT NULL DEFAULT '{}', audio_bytes BYTEA NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE IF NOT EXISTS api_keys(id TEXT PRIMARY KEY, key_prefix TEXT NOT NULL, key_hash TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', scopes_json TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT, revoked_at TEXT, last_used_at TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, status TEXT NOT NULL, source_reference TEXT, noise_reference TEXT, configuration_json TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, failed_at TEXT, error_code TEXT, error_message TEXT, output_reference TEXT, metrics_json TEXT NOT NULL DEFAULT '{}', processing_time_ms INTEGER, source_audio BYTEA, noise_audio BYTEA, output_audio BYTEA);
CREATE TABLE IF NOT EXISTS job_assets(job_id TEXT NOT NULL, asset_id TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(job_id, asset_id, role), FOREIGN KEY(job_id) REFERENCES assets(id));
CREATE TABLE IF NOT EXISTS service_settings(service_name TEXT PRIMARY KEY, settings_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS service_admin_events(id TEXT PRIMARY KEY, service_name TEXT NOT NULL, event_type TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_assets_category_status ON assets(category_id,status);
CREATE INDEX IF NOT EXISTS ix_assets_created ON assets(created_at);
CREATE INDEX IF NOT EXISTS ix_jobs_status_created ON jobs(status,created_at);
"""


class ConnectionProxy:
    def __init__(self, connection, postgres=False):
        self.connection = connection
        self.postgres = postgres

    def execute(self, sql, params=()):
        if self.postgres:
            sql = sql.replace("?", "%s")
            return self.connection.execute(sql, params)
        return self.connection.execute(sql, params)

    def cursor(self):
        return self.connection.cursor()

    def __getattr__(self, name):
        return getattr(self.connection, name)


class Database:
    RECOVERY_INTERVAL_SECONDS = 2 * 60 * 60

    def __init__(
        self,
        url: str,
        fallback_path=None,
        connection_retries: int = 3,
        retry_delay_seconds: float = 0.5,
        recovery_interval_seconds: int = RECOVERY_INTERVAL_SECONDS,
    ):
        self.url = url
        self.primary_url = url
        self.primary_postgres = url.startswith(("postgres://", "postgresql://"))
        self.postgres = self.primary_postgres
        self.connection_retries = max(0, int(connection_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.recovery_interval_seconds = max(60, int(recovery_interval_seconds))
        self._state_lock = threading.RLock()
        self.fallback_active = False
        self.fallback_since = None
        self.last_fallback_at = None
        self._fallback_since_monotonic = None
        self._next_primary_probe = None
        self.last_primary_error = None
        self.primary_connection_attempts = 0
        self._recovery_stop = threading.Event()
        self._recovery_thread = None

        if self.primary_postgres:
            default_fallback = os.getenv(
                "AUDIO_MIXER_FALLBACK_SQLITE_PATH",
                "./storage/temp/database-fallback/audio_mixer_fallback.sqlite3",
            )
            self.fallback_path = Path(fallback_path or default_fallback).expanduser().resolve()
            self.path = None
        else:
            self.fallback_path = None
            self.path = Path(url.removeprefix("sqlite:///")).resolve()
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @property
    def mode(self):
        if self.fallback_active:
            return "sqlite_fallback"
        return "postgresql" if self.primary_postgres else "sqlite"

    @property
    def active_url(self):
        if self.fallback_active:
            return f"sqlite:///{self.path.as_posix()}"
        return self.primary_url

    def status(self):
        with self._state_lock:
            return {
                "mode": self.mode,
                "backend": "postgresql" if self.postgres else "sqlite",
                "primary_backend": "postgresql" if self.primary_postgres else "sqlite",
                "fallback_active": self.fallback_active,
                "fallback_since": self.fallback_since,
                "last_fallback_at": self.last_fallback_at,
                "recovery_interval_seconds": self.recovery_interval_seconds,
                "primary_connection_attempts": self.primary_connection_attempts,
                "primary_retries_used": max(0, self.primary_connection_attempts - 1),
                "last_primary_error": self.last_primary_error,
            }

    def start_recovery_monitor(self):
        if not self.primary_postgres:
            return
        if self._recovery_thread and self._recovery_thread.is_alive():
            return
        self._recovery_stop.clear()

        def monitor():
            while not self._recovery_stop.wait(30):
                if not self.fallback_active:
                    continue
                try:
                    connection = self._try_recover_primary()
                    if connection is not None:
                        connection.close()
                except Exception as exc:
                    logger.error("Database recovery monitor failed (%s)", type(exc).__name__)

        self._recovery_thread = threading.Thread(
            target=monitor, name="audio-db-recovery", daemon=True
        )
        self._recovery_thread.start()

    def stop_recovery_monitor(self):
        self._recovery_stop.set()
        if self._recovery_thread and self._recovery_thread.is_alive():
            self._recovery_thread.join(timeout=2)

    def _open_primary_with_retries(self):
        last_error = None
        total_attempts = self.connection_retries + 1
        for attempt in range(1, total_attempts + 1):
            self.primary_connection_attempts = attempt
            try:
                if psycopg is None:
                    raise RuntimeError("psycopg is required for PostgreSQL")
                connection = psycopg.connect(self.primary_url, row_factory=dict_row)
                self.last_primary_error = None
                return connection
            except Exception as exc:
                last_error = exc
                self.last_primary_error = type(exc).__name__
                if attempt < total_attempts and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        raise last_error

    def _open_sqlite(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return connection

    @staticmethod
    def _initialize_sqlite_schema(connection):
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SQLITE_SCHEMA)
        if not Database._sqlite_column_exists(connection, "assets", "is_recommended"):
            connection.execute(
                "ALTER TABLE assets ADD COLUMN is_recommended INTEGER NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _initialize_postgres_schema(connection):
        connection.execute(POSTGRES_SCHEMA)
        for statement in [
            "ALTER TABLE assets ADD COLUMN IF NOT EXISTS audio_bytes BYTEA",
            "ALTER TABLE assets ADD COLUMN IF NOT EXISTS is_recommended BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS source_audio BYTEA",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS noise_audio BYTEA",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS output_audio BYTEA",
        ]:
            connection.execute(statement)

    def _activate_fallback(self, reason):
        with self._state_lock:
            if self.fallback_active:
                return
            raw = self._open_sqlite(self.fallback_path)
            try:
                self._initialize_sqlite_schema(raw)
            finally:
                raw.close()
            stamp = datetime.now(UTC).isoformat()
            self.path = self.fallback_path
            self.postgres = False
            self.fallback_active = True
            self.fallback_since = stamp
            self.last_fallback_at = stamp
            self._fallback_since_monotonic = time.monotonic()
            self._next_primary_probe = self._fallback_since_monotonic + self.recovery_interval_seconds
            self.last_primary_error = reason
            logger.error(
                "Primary PostgreSQL unavailable after %s attempts; using isolated SQLite fallback (%s)",
                self.primary_connection_attempts,
                reason,
            )

    def _try_recover_primary(self):
        with self._state_lock:
            if not self.fallback_active:
                return None
            now_monotonic = time.monotonic()
            if self._next_primary_probe and now_monotonic < self._next_primary_probe:
                return None
            self._next_primary_probe = now_monotonic + self.recovery_interval_seconds

        try:
            connection = self._open_primary_with_retries()
            try:
                proxy = ConnectionProxy(connection, True)
                self._initialize_postgres_schema(proxy)
                connection.commit()
            except Exception as exc:
                try:
                    connection.rollback()
                except Exception as rollback_error:
                    logger.debug("Rollback failed during recovery probe (%s)", type(rollback_error).__name__)
                connection.close()
                self.last_primary_error = type(exc).__name__
                logger.warning("Primary PostgreSQL recovery probe failed during schema check (%s)", self.last_primary_error)
                return None
        except Exception as exc:
            self.last_primary_error = type(exc).__name__
            logger.warning("Primary PostgreSQL is still unavailable (%s)", self.last_primary_error)
            return None

        with self._state_lock:
            self.postgres = True
            self.path = None
            self.fallback_active = False
            self._fallback_since_monotonic = None
            self._next_primary_probe = None
            self.fallback_since = None
            self.last_primary_error = None
            logger.warning(
                "Primary PostgreSQL restored; switching back. SQLite fallback file is retained and is not merged automatically."
            )
        return connection

    def probe_primary(self):
        """Check PostgreSQL without changing the currently active backend."""
        if not self.primary_postgres:
            return {
                "available": False,
                "primary_backend": "sqlite",
                "error_code": "primary_is_sqlite",
            }
        try:
            connection = self._open_primary_with_retries()
            try:
                connection.execute("SELECT 1").fetchone()
            finally:
                connection.close()
            return {
                "available": True,
                "primary_backend": "postgresql",
                "connection_attempts": self.primary_connection_attempts,
            }
        except Exception as exc:
            self.last_primary_error = type(exc).__name__
            return {
                "available": False,
                "primary_backend": "postgresql",
                "connection_attempts": self.primary_connection_attempts,
                "error_code": type(exc).__name__,
            }

    def force_primary_recovery(self):
        """Try the primary immediately; preserve, but never merge, SQLite data."""
        if not self.primary_postgres:
            return {
                "status": "not_applicable",
                "mode": self.mode,
                "primary_backend": "sqlite",
            }
        if not self.fallback_active:
            probe = self.probe_primary()
            return {
                "status": "already_primary" if probe["available"] else "primary_unavailable",
                "mode": self.mode,
                "primary_available": probe["available"],
                "primary_connection_attempts": probe.get("connection_attempts", 0),
                "last_primary_error": probe.get("error_code"),
            }
        with self._state_lock:
            self._next_primary_probe = 0
        connection = self._try_recover_primary()
        if connection is not None:
            connection.close()
            return {
                "status": "recovered",
                "mode": "postgresql",
                "fallback_file_retained": True,
                "primary_connection_attempts": self.primary_connection_attempts,
            }
        if not self.fallback_active:
            return {
                "status": "recovered",
                "mode": "postgresql",
                "fallback_file_retained": True,
                "primary_connection_attempts": self.primary_connection_attempts,
            }
        return {
            "status": "fallback_active",
            "mode": "sqlite_fallback",
            "primary_available": False,
            "primary_connection_attempts": self.primary_connection_attempts,
            "last_primary_error": self.last_primary_error,
        }

    @staticmethod
    def _is_connection_error(exc):
        if psycopg is not None:
            error_types = tuple(
                error_type
                for error_type in (
                    getattr(psycopg, "OperationalError", None),
                    getattr(psycopg, "InterfaceError", None),
                )
                if isinstance(error_type, type)
            )
            if error_types and isinstance(exc, error_types):
                return True
        sqlstate = getattr(exc, "sqlstate", None)
        return isinstance(sqlstate, str) and sqlstate.startswith("08")

    def _handle_primary_connection_error(self, exc):
        if not self.primary_postgres or not self._is_connection_error(exc):
            return
        try:
            probe = self._open_primary_with_retries()
        except Exception as probe_error:
            self._activate_fallback(type(probe_error).__name__)
        else:
            probe.close()

    @contextmanager
    def _yield_connection(self, connection, postgres):
        try:
            yield ConnectionProxy(connection, postgres)
            if postgres:
                connection.commit()
        except Exception as exc:
            if postgres:
                try:
                    connection.rollback()
                except Exception as rollback_error:
                    logger.debug("Rollback failed after database error (%s)", type(rollback_error).__name__)
                self._handle_primary_connection_error(exc)
            raise
        finally:
            connection.close()

    @contextmanager
    def connect(self):
        if self.fallback_active:
            primary_connection = self._try_recover_primary()
            if primary_connection is not None:
                with self._yield_connection(primary_connection, True) as connection:
                    yield connection
                return
            connection = self._open_sqlite(self.fallback_path)
            with self._yield_connection(connection, False) as proxy:
                yield proxy
            return

        if self.primary_postgres:
            try:
                connection = self._open_primary_with_retries()
            except Exception as exc:
                self._activate_fallback(type(exc).__name__)
                connection = self._open_sqlite(self.fallback_path)
                with self._yield_connection(connection, False) as proxy:
                    yield proxy
                return

            if self.fallback_active:
                connection.close()
                primary_connection = self._try_recover_primary()
                if primary_connection is not None:
                    with self._yield_connection(primary_connection, True) as proxy:
                        yield proxy
                    return
                connection = self._open_sqlite(self.fallback_path)
                with self._yield_connection(connection, False) as proxy:
                    yield proxy
                return
            with self._yield_connection(connection, True) as proxy:
                yield proxy
            return

        connection = self._open_sqlite(self.path)
        with self._yield_connection(connection, False) as proxy:
            yield proxy

    def init(self):
        with self.connect() as connection:
            if connection.postgres:
                self._initialize_postgres_schema(connection)
            else:
                self._initialize_sqlite_schema(connection)

    @staticmethod
    def _sqlite_column_exists(connection, table, column):
        return any(
            row["name"] == column
            for row in connection.execute(f"PRAGMA table_info({table})")
        )

    def get_service_settings(self, service_name):
        with self.connect() as c:
            row = c.execute(
                "SELECT settings_json,updated_at FROM service_settings WHERE service_name=?",
                (service_name,),
            ).fetchone()
        if not row:
            return {"settings": {}, "updated_at": None}
        return {"settings": json.loads(row["settings_json"]), "updated_at": row["updated_at"]}

    def set_service_settings(self, service_name, values):
        stamp = now()
        with self.connect() as c:
            c.execute(
                "INSERT INTO service_settings(service_name,settings_json,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(service_name) DO UPDATE SET settings_json=excluded.settings_json,updated_at=excluded.updated_at",
                (service_name, json.dumps(values, separators=(",", ":")), stamp),
            )
        return {"service_name": service_name, "settings": values, "updated_at": stamp}

    def create_key(self, name, description, scopes, expires_at=None):
        raw = "amx_" + secrets.token_urlsafe(32)
        kid = new_id()
        prefix = raw[:12]
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.connect() as c:
            c.execute(
                "INSERT INTO api_keys (id,key_prefix,key_hash,name,description,scopes_json,created_at,expires_at,revoked_at,last_used_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    kid,
                    prefix,
                    digest,
                    name,
                    description,
                    json.dumps(scopes),
                    now(),
                    expires_at,
                    None,
                    None,
                ),
            )
        return kid, raw

    def verify_key(self, raw):
        if not raw:
            return None
        prefix = raw[:12]
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.connect() as c:
            row = c.execute(
                "SELECT * FROM api_keys WHERE key_prefix=? AND revoked_at IS NULL",
                (prefix,),
            ).fetchone()
            if not row or not hmac.compare_digest(row["key_hash"], digest):
                return None
            if row["expires_at"] and row["expires_at"] <= now():
                return None
            c.execute(
                "UPDATE api_keys SET last_used_at=? WHERE id=?", (now(), row["id"])
            )
        return row

    def list_keys(self):
        with self.connect() as c:
            rows = c.execute(
                "SELECT id,key_prefix,name,description,scopes_json,created_at,expires_at,revoked_at,last_used_at FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [{**dict(row), "scopes": json.loads(row["scopes_json"])} for row in rows]

    def revoke_key(self, key_id):
        with self.connect() as c:
            cursor = c.execute(
                "UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                (now(), key_id),
            )
            return cursor.rowcount > 0

    def cancel_job(self, job_id):
        with self.connect() as c:
            cursor = c.execute(
                "UPDATE jobs SET status='cancelled',completed_at=? WHERE id=? AND status='queued'",
                (now(), job_id),
            )
            return cursor.rowcount > 0

    def cleanup_expired_payloads(self, cutoff, dry_run=False):
        completed_filter = "status='completed' AND completed_at<? AND (source_audio IS NOT NULL OR noise_audio IS NOT NULL OR output_audio IS NOT NULL)"
        retry_filter = "status IN ('failed','cancelled') AND COALESCE(failed_at,completed_at,created_at)<? AND (source_audio IS NOT NULL OR noise_audio IS NOT NULL)"
        with self.connect() as c:
            completed = c.execute(
                f"SELECT COUNT(*) AS count FROM jobs WHERE {completed_filter}", (cutoff,)
            ).fetchone()["count"]
            retryable = c.execute(
                f"SELECT COUNT(*) AS count FROM jobs WHERE {retry_filter}", (cutoff,)
            ).fetchone()["count"]
            if not dry_run:
                c.execute(
                    f"UPDATE jobs SET source_audio=NULL,noise_audio=NULL,output_audio=NULL WHERE {completed_filter}", (cutoff,)
                )
                c.execute(
                    f"UPDATE jobs SET source_audio=NULL,noise_audio=NULL WHERE {retry_filter}",
                    (cutoff,),
                )
        return {
            "completed_payloads_cleared": int(completed),
            "retry_inputs_cleared": int(retryable),
            "dry_run": bool(dry_run),
        }

    def record_admin_event(self, service_name, event_type, details):
        event_id, stamp = new_id(), now()
        with self.connect() as c:
            c.execute(
                "INSERT INTO service_admin_events(id,service_name,event_type,details_json,created_at) VALUES (?,?,?,?,?)",
                (event_id, service_name, event_type, json.dumps(details, separators=(",", ":")), stamp),
            )
            c.execute(
                "DELETE FROM service_admin_events WHERE service_name=? AND id NOT IN "
                "(SELECT id FROM service_admin_events WHERE service_name=? ORDER BY created_at DESC,id DESC LIMIT 500)",
                (service_name, service_name),
            )
        return event_id

    def list_admin_events(self, service_name, limit=50):
        with self.connect() as c:
            rows = c.execute(
                "SELECT id,service_name,event_type,details_json,created_at FROM service_admin_events WHERE service_name=? ORDER BY created_at DESC LIMIT ?",
                (service_name, min(max(int(limit), 1), 200)),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            events.append(event)
        return events

    def set_recommended(self, asset_id, recommended):
        with self.connect() as c:
            cursor = c.execute(
                "UPDATE assets SET is_recommended=?,updated_at=? WHERE id=? AND deleted_at IS NULL",
                (bool(recommended), now(), asset_id),
            )
            return cursor.rowcount > 0

    def seed_category(self, slug, title, description=""):
        with self.connect() as c:
            row = c.execute(
                "SELECT id FROM categories WHERE slug=?", (slug,)
            ).fetchone()
            if row:
                return row["id"]
            i = new_id()
            c.execute(
                "INSERT INTO categories VALUES (?,?,?,?,?)",
                (i, slug, title, description, now()),
            )
            return i
