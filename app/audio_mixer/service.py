import base64
import json
import logging
import secrets
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from .db import new_id, now
from .errors import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


class AudioService:
    def __init__(self, db, ffmpeg, settings):
        self.db = db
        self.ffmpeg = ffmpeg
        self.settings = settings
        self.settings.temp_path.mkdir(parents=True, exist_ok=True)
        self.executor = threading.BoundedSemaphore(
            settings.max_queue_size + settings.max_concurrent_jobs
        )
        self._cleanup_stop = threading.Event()
        self._cleanup_thread = None
        self._cleanup_last_run_at = None
        self._cleanup_last_result = None
        self._cleanup_last_error = None
        self.admin_config = None

    def _setting(self, service_name, key, default):
        if self.admin_config is None:
            return default
        return self.admin_config.get(service_name, key)

    def resolve_output_format(self, output_format=None):
        return self.ffmpeg.normalize_output_format(
            output_format or self._setting("ffmpeg", "default_output_format", "wav")
        )

    def cleanup_worker_status(self):
        return {
            "worker_alive": bool(self._cleanup_thread and self._cleanup_thread.is_alive()),
            "last_run_at": self._cleanup_last_run_at,
            "last_result": self._cleanup_last_result,
            "last_error": self._cleanup_last_error,
            "interval_seconds": self._setting("cleanup", "interval_seconds", self.settings.cleanup_interval_seconds),
            "retention_hours": self._setting("cleanup", "retention_hours", self.settings.retention_hours),
        }

    def cleanup_expired_data(self, retention_hours=None, dry_run=False):
        retention_hours = retention_hours or self._setting("cleanup", "retention_hours", self.settings.retention_hours)
        cutoff_dt = datetime.now(UTC) - timedelta(hours=retention_hours)
        result = self.db.cleanup_expired_payloads(cutoff_dt.isoformat(), dry_run=dry_run)
        stale_files = 0
        fallback_path = getattr(self.db, "fallback_path", None)
        fallback_path = fallback_path.resolve() if fallback_path else None
        for path in self.settings.temp_path.iterdir():
            try:
                if (
                    path.is_file()
                    and (fallback_path is None or path.resolve() != fallback_path)
                    and path.stat().st_mtime < cutoff_dt.timestamp()
                ):
                    stale_files += 1
                    if not dry_run:
                        path.unlink(missing_ok=True)
            except FileNotFoundError:
                continue
        result["temporary_files_removed"] = 0 if dry_run else stale_files
        result["temporary_files_expired"] = stale_files
        result["retention_hours"] = retention_hours
        self._cleanup_last_run_at = datetime.now(UTC).isoformat()
        self._cleanup_last_result = result
        self._cleanup_last_error = None
        return result

    def start_cleanup_worker(self):
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        self._cleanup_stop.clear()

        def run():
            while not self._cleanup_stop.is_set():
                try:
                    self.cleanup_expired_data()
                except Exception:
                    self._cleanup_last_run_at = datetime.now(UTC).isoformat()
                    self._cleanup_last_error = "cleanup_failed"
                    logger.exception("Automatic expired-data cleanup failed")
                interval = self._setting("cleanup", "interval_seconds", self.settings.cleanup_interval_seconds)
                if self._cleanup_stop.wait(interval):
                    break

        self._cleanup_thread = threading.Thread(
            target=run, name="audio-data-cleanup", daemon=True
        )
        self._cleanup_thread.start()

    def stop_cleanup_worker(self):
        self._cleanup_stop.set()

    def fail_orphaned_memory_jobs(self):
        """Mark volatile-input jobs from a previous process as unrecoverable."""
        if not self.settings.memory_job_inputs:
            return 0
        with self.db.connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status=?,failed_at=?,error_code=?,error_message=? "
                "WHERE status IN (?,?) AND configuration_json LIKE ?",
                (
                    "failed", now(), "worker_restarted",
                    "Volatile job inputs were lost when the service process restarted",
                    "queued", "processing", '%"storage": "worker_memory"%',
                ),
            )
        count = max(0, int(getattr(cursor, "rowcount", 0)))
        if count:
            logger.warning("Marked %d orphaned worker_memory jobs as failed", count)
        return count

    def _temp_file(self, data, suffix=".audio"):
        with NamedTemporaryFile(
            dir=self.settings.temp_path, suffix=suffix, delete=False
        ) as handle:
            handle.write(data)
            handle.flush()
        return Path(handle.name)

    def _read_upload(self, upload):
        size = 0
        chunks = []
        max_upload_size = self._setting("jobs", "max_upload_size_bytes", self.settings.max_upload_size)
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_upload_size:
                raise ValidationError("Upload is too large")
            chunks.append(chunk)
        if not chunks:
            raise ValidationError("Audio upload is empty")
        return b"".join(chunks)

    def _upload_to_temp(self, upload):
        """Copy an upload to a temp file in bounded chunks, avoiding a full RAM copy."""
        size = 0
        max_upload_size = self._setting("jobs", "max_upload_size_bytes", self.settings.max_upload_size)
        path = None
        try:
            with NamedTemporaryFile(
                dir=self.settings.temp_path,
                suffix=Path(upload.filename or "").suffix or ".audio",
                delete=False,
            ) as handle:
                path = Path(handle.name)
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_size:
                        raise ValidationError("Upload is too large")
                    handle.write(chunk)
                if size == 0:
                    raise ValidationError("Audio upload is empty")
            return path
        except Exception:
            if path:
                path.unlink(missing_ok=True)
            raise

    def _validate_input_duration(self, path):
        metadata = self.ffmpeg.probe(path)
        max_duration = self._setting(
            "jobs", "max_audio_duration_seconds", self.settings.max_audio_duration
        )
        if metadata["duration"] > max_duration:
            raise ValidationError("Audio duration exceeds the configured limit")
        return metadata

    def categories(self):
        with self.db.connect() as c:
            return [
                dict(x)
                for x in c.execute("SELECT * FROM categories ORDER BY title").fetchall()
            ]

    def assets(self, category=None, search=None, limit=None, offset=0, recommended=None):
        limit = limit or self._setting("assets", "default_page_size", 50)
        query = "SELECT id,title,description,category_id,duration_ms,format,codec,status,is_recommended,created_at FROM assets WHERE deleted_at IS NULL"
        params = []
        if recommended is not None:
            query += " AND is_recommended=?"
            params.append(bool(recommended))
        if category:
            query += " AND category_id=?"
            params.append(category)
        if search:
            query += " AND (title LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([min(limit, 200), max(offset, 0)])
        with self.db.connect() as c:
            return [dict(x) for x in c.execute(query, params).fetchall()]

    def recommended_assets(self, limit=None, offset=0):
        return self.assets(limit=limit, offset=offset, recommended=True)

    def asset(self, aid):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT * FROM assets WHERE id=? AND deleted_at IS NULL", (aid,)
            ).fetchone()
        if not row:
            raise NotFoundError("asset_not_found", "Audio asset not found")
        result = dict(row)
        result.pop("audio_bytes", None)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def asset_bytes(self, aid):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT audio_bytes,original_filename,content_type FROM assets WHERE id=? AND deleted_at IS NULL",
                (aid,),
            ).fetchone()
        if not row or not row["audio_bytes"]:
            raise NotFoundError(
                "asset_file_not_found", "Audio asset bytes are unavailable"
            )
        return bytes(row["audio_bytes"]), row["original_filename"], row["content_type"]

    def asset_audio_bundle(self, aid):
        """Fetch a library asset and its ingest-validated metadata in one query."""
        with self.db.connect() as c:
            row = c.execute(
                "SELECT audio_bytes,original_filename,content_type,metadata_json FROM assets WHERE id=? AND deleted_at IS NULL",
                (aid,),
            ).fetchone()
        if not row or not row["audio_bytes"]:
            raise NotFoundError("asset_file_not_found", "Audio asset bytes are unavailable")
        return (
            bytes(row["audio_bytes"]),
            row["original_filename"],
            row["content_type"],
            json.loads(row["metadata_json"] or "{}"),
        )

    def ingest(
        self,
        upload,
        title,
        description,
        category_id=None,
        source="upload",
        license_name="",
        license_url="",
        is_recommended=False,
    ):
        data = self._read_upload(upload)
        suffix = Path(upload.filename or "").suffix or ".audio"
        temp = self._temp_file(data, suffix)
        try:
            meta = self.ffmpeg.probe(temp)
        finally:
            temp.unlink(missing_ok=True)
        if (
            meta["duration"] > self._setting("jobs", "max_audio_duration_seconds", self.settings.max_audio_duration)
            or not 1 <= meta["channels"] <= 2
        ):
            raise ValidationError("Audio duration or channels exceed limits")
        aid = new_id()
        t = now()
        storage_key = f"db://assets/{aid}"
        sha = __import__("hashlib").sha256(data).hexdigest()
        category_id = category_id or None
        values = (
            aid,
            category_id,
            title or upload.filename or aid,
            description or "",
            upload.filename or "upload",
            storage_key,
            getattr(upload, "content_type", None) or "audio/*",
            meta["format"],
            meta["codec"],
            int(meta["duration"] * 1000),
            meta["sample_rate"],
            meta["channels"],
            meta["bit_depth"],
            meta["bit_rate"],
            None,
            None,
            None,
            sha,
            source,
            license_name,
            license_url,
            None,
            "ready",
            bool(is_recommended),
            json.dumps(meta),
            data,
            t,
            t,
            None,
        )
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO assets (id,category_id,title,description,original_filename,storage_key,content_type,format,codec,duration_ms,sample_rate,channels,bit_depth,bit_rate,loudness_lufs,true_peak_dbtp,rms_db,sha256,source,license,attribution,quality_score,status,is_recommended,metadata_json,audio_bytes,created_at,updated_at,deleted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
        return self.asset(aid)

    def delete_asset(self, aid):
        self.asset(aid)
        with self.db.connect() as c:
            c.execute(
                "UPDATE assets SET deleted_at=?,status='deleted',updated_at=?,audio_bytes=? WHERE id=?",
                (now(), now(), b"", aid),
            )
        return {"id": aid, "status": "deleted"}

    def submit_mix_bytes(
        self,
        source_bytes,
        source_reference="source.audio",
        noise_bytes=None,
        noise_reference=None,
        source_gain=0,
        noise_gain=-18,
        variation_seed=None,
        output_format="wav",
        noise_metadata=None,
        persist_inputs=True,
    ):
        output_format = self.resolve_output_format(output_format)
        if not -60 <= source_gain <= 6 or not -60 <= noise_gain <= 6:
            raise ValidationError("Gain must be between -60 and +6 dB")
        jid = new_id()
        cfg = {
            "source_gain_db": source_gain,
            "noise_gain_db": noise_gain,
            "noise_extension_seconds": self._setting("jobs", "noise_extension_seconds", self.settings.min_noise_margin_seconds),
            "pipeline_version": "2.0.0",
            "storage": "postgresql_bytea" if persist_inputs else "worker_memory",
            "variation_seed": variation_seed
            if variation_seed is not None
            else secrets.randbits(64),
            "variation_method": "seeded_noise_window_offset",
            "output_format": output_format,
            "noise_metadata": noise_metadata,
        }
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO jobs (id,status,source_reference,noise_reference,configuration_json,created_at,started_at,completed_at,failed_at,error_code,error_message,output_reference,metrics_json,processing_time_ms,source_audio,noise_audio,output_audio) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    "queued",
                    Path(source_reference).name,
                    noise_reference,
                    json.dumps(cfg),
                    now(),
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"db://jobs/{jid}/output",
                    "{}",
                    None,
                    source_bytes if persist_inputs else None,
                    noise_bytes if persist_inputs else None,
                    None,
                ),
            )
        threading.Thread(
            target=self._run,
            args=(jid, None if persist_inputs else source_bytes, None if persist_inputs else noise_bytes),
            daemon=True,
        ).start()
        return jid

    def submit_mix(
        self, source_path, noise_asset_id=None, source_gain=0, noise_gain=-18
    ):
        source_bytes = Path(source_path).read_bytes()
        noise_bytes = self.asset_bytes(noise_asset_id)[0] if noise_asset_id else None
        return self.submit_mix_bytes(
            source_bytes,
            Path(source_path).name,
            noise_bytes,
            noise_asset_id,
            source_gain,
            noise_gain,
        )

    def process_sync_bytes(
        self,
        source_bytes,
        noise_bytes,
        source_reference="source.audio",
        noise_reference="noise.audio",
        source_gain=0,
        noise_gain=-18,
        variation_seed=None,
        output_format="wav",
        noise_metadata=None,
    ):
        output_format = self.resolve_output_format(output_format)
        if not -60 <= source_gain <= 6 or not -60 <= noise_gain <= 6:
            raise ValidationError("Gain must be between -60 and +6 dB")
        seed = variation_seed if variation_seed is not None else secrets.randbits(64)
        source = noise = output = None
        try:
            source = self._temp_file(
                source_bytes, Path(source_reference).suffix or ".audio"
            )
            noise = self._temp_file(
                noise_bytes, Path(noise_reference).suffix or ".audio"
            )
            return self.process_sync_files(
                source, noise, source_gain, noise_gain, seed, output_format, noise_metadata
            )
        finally:
            for path in (source, noise, output):
                if path:
                    path.unlink(missing_ok=True)

    def process_sync_uploads(
        self,
        source_upload,
        noise_upload=None,
        noise_bytes=None,
        noise_reference="noise.audio",
        source_gain=0,
        noise_gain=-18,
        variation_seed=None,
        output_format="wav",
        noise_metadata=None,
    ):
        """Process request uploads from bounded temp-file copies instead of RAM blobs."""
        if (noise_upload is None) == (noise_bytes is None):
            raise ValidationError("Provide exactly one noise upload or noise byte payload")
        source = noise = None
        try:
            source = self._upload_to_temp(source_upload)
            if noise_upload is not None:
                noise = self._upload_to_temp(noise_upload)
            else:
                noise = self._temp_file(noise_bytes, Path(noise_reference).suffix or ".audio")
            return self.process_sync_files(
                source,
                noise,
                source_gain,
                noise_gain,
                variation_seed,
                output_format,
                noise_metadata,
            )
        finally:
            for path in (source, noise):
                if path:
                    path.unlink(missing_ok=True)

    def process_sync_files(
        self, source, noise, source_gain=0, noise_gain=-18,
        variation_seed=None, output_format="wav", noise_metadata=None,
    ):
        output_format = self.resolve_output_format(output_format)
        if not -60 <= source_gain <= 6 or not -60 <= noise_gain <= 6:
            raise ValidationError("Gain must be between -60 and +6 dB")
        seed = variation_seed if variation_seed is not None else secrets.randbits(64)
        output = None
        try:
            source_meta = self._validate_input_duration(source)
            noise_meta = noise_metadata or self._validate_input_duration(noise)
            if noise_metadata:
                duration = float(noise_meta.get("duration") or 0)
                channels = int(noise_meta.get("channels") or 0)
                max_duration = self._setting(
                    "jobs", "max_audio_duration_seconds", self.settings.max_audio_duration
                )
                if duration <= 0 or duration > max_duration or not 1 <= channels <= 2:
                    raise ValidationError("Stored noise metadata is invalid")
            output = self._temp_file(b"", self.ffmpeg.output_spec(output_format)["suffix"])
            self.ffmpeg.mix(
                source, noise, output,
                source_gain=source_gain,
                noise_gain=noise_gain,
                variation_seed=seed,
                output_format=output_format,
                noise_extension_seconds=self._setting(
                    "jobs", "noise_extension_seconds", self.settings.min_noise_margin_seconds
                ),
                source_metadata=source_meta,
                noise_metadata=noise_meta,
            )
            return output.read_bytes()
        finally:
            if output:
                output.unlink(missing_ok=True)

    def submit_mix_files(
        self,
        source_path,
        noise_path=None,
        noise_reference=None,
        source_gain=0,
        noise_gain=-18,
    ):
        return self.submit_mix_bytes(
            Path(source_path).read_bytes(),
            Path(source_path).name,
            Path(noise_path).read_bytes() if noise_path else None,
            noise_reference,
            source_gain,
            noise_gain,
        )

    def _run(self, jid, source_bytes=None, noise_bytes=None):
        if not self.executor.acquire(timeout=1):
            with self.db.connect() as c:
                c.execute(
                    "UPDATE jobs SET status=?,error_code=?,error_message=? WHERE id=?",
                    ("failed", "resource_limit", "Job queue is full", jid),
                )
            return
        started = time.time()
        logger.info("audio_job_stage job_id=%s stage=mix_started", jid)
        source = noise = output = None
        try:
            with self.db.connect() as c:
                row = c.execute(
                    "SELECT source_audio,noise_audio,source_reference,noise_reference,configuration_json FROM jobs WHERE id=?",
                    (jid,),
                ).fetchone()
                c.execute(
                    "UPDATE jobs SET status=?,started_at=? WHERE id=?",
                    ("processing", now(), jid),
                )
            configuration = json.loads(row["configuration_json"] or "{}")
            source_payload = source_bytes if source_bytes is not None else row["source_audio"]
            noise_payload = noise_bytes if noise_bytes is not None else row["noise_audio"]
            if source_payload is None:
                raise ValidationError("Job input payload is unavailable after process restart")
            source = self._temp_file(bytes(source_payload), Path(row["source_reference"] or "source.audio").suffix or ".source")
            noise = (
                self._temp_file(bytes(noise_payload), Path(row["noise_reference"] or "noise.audio").suffix or ".noise")
                if noise_payload is not None
                else None
            )
            source_meta = self._validate_input_duration(source)
            if noise:
                noise_meta = configuration.get("noise_metadata")
                if noise_meta:
                    duration = float(noise_meta.get("duration") or 0)
                    channels = int(noise_meta.get("channels") or 0)
                    max_duration = self._setting(
                        "jobs", "max_audio_duration_seconds", self.settings.max_audio_duration
                    )
                    if duration <= 0 or duration > max_duration or not 1 <= channels <= 2:
                        raise ValidationError("Stored noise metadata is invalid")
                else:
                    noise_meta = self._validate_input_duration(noise)
            else:
                noise_meta = None
            output_format = self.ffmpeg.normalize_output_format(configuration.get("output_format", "wav"))
            output = self._temp_file(b"", self.ffmpeg.output_spec(output_format)["suffix"])
            mix_metrics = self.ffmpeg.mix(
                source,
                noise,
                output,
                source_gain=configuration["source_gain_db"],
                noise_gain=configuration["noise_gain_db"],
                variation_seed=configuration["variation_seed"],
                output_format=output_format,
                noise_extension_seconds=configuration.get(
                    "noise_extension_seconds",
                    configuration.get(
                        "min_noise_margin_seconds",
                        self._setting("jobs", "noise_extension_seconds", self.settings.min_noise_margin_seconds),
                    ),
                ),
                source_metadata=source_meta,
                noise_metadata=noise_meta,
            )
            meta = self.ffmpeg.probe(output)
            meta.update(mix_metrics or {})
            output_bytes = output.read_bytes()
            with self.db.connect() as c:
                c.execute(
                    "UPDATE jobs SET status=?,completed_at=?,metrics_json=?,processing_time_ms=?,output_audio=?,source_audio=NULL,noise_audio=NULL WHERE id=?",
                    (
                        "completed",
                        now(),
                        json.dumps(meta),
                        int((time.time() - started) * 1000),
                        output_bytes,
                        jid,
                    ),
                )
            logger.info("audio_job_stage job_id=%s stage=mix_completed seconds=%.4f", jid, time.time() - started)
        except Exception as exc:
            try:
                with self.db.connect() as c:
                    c.execute(
                        "UPDATE jobs SET status=?,failed_at=?,error_code=?,error_message=?,processing_time_ms=? WHERE id=?",
                        (
                            "failed",
                            now(),
                            getattr(exc, "code", "processing_failed"),
                            str(exc),
                            int((time.time() - started) * 1000),
                            jid,
                        ),
                    )
            except Exception:
                logger.exception(
                    "Could not persist failed job state", extra={"job_id": jid}
                )
        finally:
            for path in (source, noise, output):
                if path:
                    path.unlink(missing_ok=True)
            self.executor.release()

    def job(self, jid):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT id,status,source_reference,noise_reference,configuration_json,created_at,started_at,completed_at,failed_at,error_code,error_message,output_reference,metrics_json,processing_time_ms,output_audio FROM jobs WHERE id=?",
                (jid,),
            ).fetchone()
        if not row:
            raise NotFoundError("job_not_found", "Job not found")
        result = dict(row)
        output_audio = result.pop("output_audio", None)
        result["configuration"] = json.loads(result.pop("configuration_json"))
        result["metrics"] = json.loads(result.pop("metrics_json"))
        if result["status"] == "completed" and output_audio is not None:
            output_format = self.ffmpeg.normalize_output_format(
                result["configuration"].get("output_format", "wav")
            )
            spec = self.ffmpeg.output_spec(output_format)
            result["output_audio_base64"] = base64.b64encode(
                bytes(output_audio)
            ).decode("ascii")
            result["output_audio_format"] = output_format
            result["output_audio_media_type"] = spec["media_type"]
            result["output_audio_filename"] = f"mix.{output_format}"
        return result

    def job_status(self, jid):
        """Minimal polling payload: never fetch the potentially large output blob."""
        with self.db.connect() as c:
            row = c.execute(
                "SELECT id,status,created_at,started_at,completed_at,failed_at,error_code,processing_time_ms FROM jobs WHERE id=?",
                (jid,),
            ).fetchone()
        if not row:
            raise NotFoundError("job_not_found", "Job not found")
        return dict(row)

    def job_result_info(self, jid):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT status,output_audio,source_reference,configuration_json FROM jobs WHERE id=?", (jid,)
            ).fetchone()
        if not row:
            raise NotFoundError("job_not_found", "Job not found")
        if row["status"] != "completed" or row["output_audio"] is None:
            raise ValidationError("Result is not ready")
        configuration = json.loads(row["configuration_json"] or "{}")
        output_format = self.ffmpeg.normalize_output_format(configuration.get("output_format", "wav"))
        spec = self.ffmpeg.output_spec(output_format)
        return {
            "data": bytes(row["output_audio"]),
            "format": output_format,
            "media_type": spec["media_type"],
            "filename": f"mix.{output_format}",
        }

    def job_result_bytes(self, jid):
        return self.job_result_info(jid)["data"]

    def jobs(self, status=None, limit=50, offset=0):
        query = "SELECT id,status,source_reference,noise_reference,created_at,started_at,completed_at,failed_at,error_code,error_message,output_reference,processing_time_ms FROM jobs"
        params = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([min(limit, 200), max(offset, 0)])
        with self.db.connect() as c:
            return [dict(row) for row in c.execute(query, params).fetchall()]

    def stats(self):
        with self.db.connect() as c:
            assets = c.execute(
                "SELECT COUNT(*) AS count, COALESCE(SUM(LENGTH(audio_bytes)),0) AS bytes FROM assets WHERE deleted_at IS NULL"
            ).fetchone()
            jobs = c.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        return {
            "assets": {"count": int(assets["count"]), "bytes": int(assets["bytes"])},
            "jobs": {row["status"]: int(row["count"]) for row in jobs},
            "storage": "postgresql_bytea",
        }

    def retry_job(self, jid):
        with self.db.connect() as c:
            row = c.execute(
                "SELECT status,source_reference,noise_reference,configuration_json,source_audio,noise_audio FROM jobs WHERE id=?",
                (jid,),
            ).fetchone()
        if not row:
            raise NotFoundError("job_not_found", "Job not found")
        if row["status"] not in {"failed", "cancelled"}:
            raise ValidationError("Only failed or cancelled jobs can be retried")
        cfg = json.loads(row["configuration_json"])
        return self.submit_mix_bytes(
            bytes(row["source_audio"]),
            row["source_reference"] or "source.audio",
            bytes(row["noise_audio"]) if row["noise_audio"] is not None else None,
            row["noise_reference"],
            cfg.get("source_gain_db", 0),
            cfg.get("noise_gain_db", -18),
            output_format=cfg.get("output_format", "wav"),
            noise_metadata=cfg.get("noise_metadata"),
        )

    def cancel_job(self, jid):
        job = self.job(jid)
        if job["status"] == "processing":
            raise ValidationError("Running jobs require cooperative cancellation")
        if job["status"] != "queued" or not self.db.cancel_job(jid):
            raise ValidationError("Job is not cancellable")
        return self.job(jid)
