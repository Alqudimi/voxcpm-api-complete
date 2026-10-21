from datetime import UTC, datetime

from .errors import ValidationError


class ServiceAdmin:
    """Allowlisted service diagnostics, persistent runtime settings, and safe maintenance."""

    SERVICE_NAMES = ("database", "ffmpeg", "jobs", "assets", "catalog", "cleanup")
    FORMATS = ("wav", "ogg", "opus", "mp3", "flac", "m4a", "aac", "webm")

    def __init__(self, db, audio_service, ffmpeg, catalog_import):
        self.db = db
        self.audio_service = audio_service
        self.ffmpeg = ffmpeg
        self.catalog_import = catalog_import
        self.audio_service.admin_config = self
        self._specs = {
            "database": {"error_limit": (int, 1, 100)},
            "ffmpeg": {"default_output_format": (str, self.FORMATS)},
            "jobs": {
                "max_upload_size_bytes": (int, 1_048_576, 268_435_456),
                "max_audio_duration_seconds": (int, 1, 1800),
                "noise_extension_seconds": (float, 0, 60),
            },
            "assets": {"default_page_size": (int, 1, 200)},
            "catalog": {
                "import_workers": (int, 1, 32),
                "import_timeout_seconds": (int, 5, 300),
            },
            "cleanup": {
                "retention_hours": (int, 1, 8760),
                "interval_seconds": (int, 60, 86400),
            },
        }
        self._defaults = {
            "database": {"error_limit": 20},
            "ffmpeg": {"default_output_format": audio_service.settings.default_output_format},
            "jobs": {
                "max_upload_size_bytes": audio_service.settings.max_upload_size,
                "max_audio_duration_seconds": int(audio_service.settings.max_audio_duration),
                "noise_extension_seconds": audio_service.settings.min_noise_margin_seconds,
            },
            "assets": {"default_page_size": 50},
            "catalog": {
                "import_workers": audio_service.settings.import_workers,
                "import_timeout_seconds": audio_service.settings.import_timeout,
            },
            "cleanup": {
                "retention_hours": audio_service.settings.retention_hours,
                "interval_seconds": audio_service.settings.cleanup_interval_seconds,
            },
        }
        saved_catalog = db.get_service_settings("catalog")["settings"]
        self.catalog_import.import_workers = saved_catalog.get(
            "import_workers", self._defaults["catalog"]["import_workers"]
        )
        self.catalog_import.import_timeout = saved_catalog.get(
            "import_timeout_seconds", self._defaults["catalog"]["import_timeout_seconds"]
        )

    def has_service(self, service_name):
        return service_name in self.SERVICE_NAMES

    def get(self, service_name, key):
        if service_name not in self._defaults or key not in self._defaults[service_name]:
            raise KeyError(service_name)
        stored = self.db.get_service_settings(service_name)["settings"]
        return stored.get(key, self._defaults[service_name][key])

    def settings(self, service_name):
        if not self.has_service(service_name):
            raise KeyError(service_name)
        stored = self.db.get_service_settings(service_name)
        effective = {**self._defaults[service_name], **stored["settings"]}
        return {
            "service": service_name,
            "settings": effective,
            "mutable": sorted(self._specs[service_name]),
            "updated_at": stored["updated_at"],
        }

    def _validate(self, service_name, proposed):
        if not self.has_service(service_name):
            raise KeyError(service_name)
        specs = self._specs[service_name]
        if not proposed:
            raise ValidationError("At least one setting is required")
        unknown = sorted(set(proposed) - set(specs))
        if unknown:
            raise ValidationError(f"Settings are not remotely configurable: {', '.join(unknown)}")
        cleaned = {}
        for key, value in proposed.items():
            spec = specs[key]
            if spec[0] is str:
                if not isinstance(value, str) or value not in spec[1]:
                    raise ValidationError(f"Invalid value for {key}")
                cleaned[key] = value
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"Invalid numeric value for {key}")
            if spec[0] is int and not isinstance(value, int):
                raise ValidationError(f"{key} must be an integer")
            number = int(value) if spec[0] is int else float(value)
            if not spec[1] <= number <= spec[2]:
                raise ValidationError(f"{key} is outside its permitted range")
            cleaned[key] = number
        return cleaned

    def update_settings(self, service_name, proposed, dry_run=True):
        validated = self._validate(service_name, proposed)
        before = self.settings(service_name)["settings"]
        after = {**before, **validated}
        result = {
            "service": service_name,
            "dry_run": dry_run,
            "changes": {
                key: {"before": before[key], "after": after[key]}
                for key in validated
                if before[key] != after[key]
            },
            "settings": after,
        }
        if dry_run:
            return result
        persisted = self.db.set_service_settings(service_name, after)
        if service_name == "catalog":
            self.catalog_import.import_workers = after["import_workers"]
            self.catalog_import.import_timeout = after["import_timeout_seconds"]
        self.db.record_admin_event(service_name, "settings_updated", {"changes": result["changes"]})
        result["updated_at"] = persisted["updated_at"]
        return result

    def _database_status(self):
        with self.db.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        state = self.db.status()
        return {"status": "degraded" if state["fallback_active"] else "ok", **state}

    def health(self, service_name):
        if not self.has_service(service_name):
            raise KeyError(service_name)
        checked_at = datetime.now(UTC).isoformat()
        try:
            if service_name == "database":
                details = self._database_status()
            elif service_name == "ffmpeg":
                cap = self.ffmpeg.capabilities()
                details = {
                    "available": cap["available"],
                    "version": cap["ffmpeg_version"],
                    "supported_output_formats": cap["supported_output_formats"],
                }
                details["status"] = "ok" if cap["available"] else "error"
            elif service_name == "jobs":
                with self.db.connect() as connection:
                    rows = connection.execute("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
                counts = {row["status"]: int(row["count"]) for row in rows}
                details = {"status": "ok", "counts": counts, "active": counts.get("queued", 0) + counts.get("processing", 0)}
            elif service_name == "assets":
                with self.db.connect() as connection:
                    active = connection.execute("SELECT COUNT(*) AS count FROM assets WHERE deleted_at IS NULL").fetchone()["count"]
                    deleted = connection.execute("SELECT COUNT(*) AS count FROM assets WHERE deleted_at IS NOT NULL").fetchone()["count"]
                    size = connection.execute("SELECT COALESCE(SUM(LENGTH(audio_bytes)),0) AS bytes FROM assets WHERE deleted_at IS NULL").fetchone()["bytes"]
                details = {"status": "ok", "active_count": int(active), "deleted_count": int(deleted), "active_audio_bytes": int(size)}
            elif service_name == "catalog":
                raw = self.catalog_import.status()
                state = raw.get("status", "not_started")
                details = {
                    "status": "error" if state == "failed" else ("ok" if state in {"completed", "running", "disabled", "not_started"} else "unknown"),
                    "import_status": state,
                    "auto_import_enabled": raw.get("auto_import_enabled", self.audio_service.settings.auto_import_mixkit),
                    "updated_at": raw.get("updated_at"),
                    "error_code": "catalog_import_failed" if state == "failed" else None,
                }
            else:
                details = self.audio_service.cleanup_worker_status()
                details["status"] = "error" if details.get("last_error") else ("ok" if details.get("worker_alive") else "degraded")
            return {"service": service_name, "status": details.pop("status", "ok"), "checked_at": checked_at, "details": details, "errors": self.errors(service_name)}
        except Exception as exc:
            return {
                "service": service_name,
                "status": "error",
                "checked_at": checked_at,
                "details": {},
                "errors": [{"code": "health_check_failed", "message": type(exc).__name__}],
            }

    def list_services(self):
        return [self.health(name) for name in self.SERVICE_NAMES]

    def errors(self, service_name):
        if not self.has_service(service_name):
            raise KeyError(service_name)
        if service_name == "jobs":
            limit = self.get("database", "error_limit")
            with self.db.connect() as connection:
                rows = connection.execute(
                    "SELECT id,status,error_code,failed_at,created_at FROM jobs WHERE status='failed' ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [{"code": row["error_code"] or "job_failed", "job_id": row["id"], "at": row["failed_at"] or row["created_at"]} for row in rows]
        if service_name == "catalog":
            state = self.catalog_import.status()
            if state.get("status") == "failed":
                return [{"code": "catalog_import_failed", "at": state.get("updated_at")}]
            return []
        if service_name == "cleanup":
            error = self.audio_service.cleanup_worker_status().get("last_error")
            return [{"code": "cleanup_failed", "message": error}] if error else []
        if service_name == "database":
            try:
                self._database_status()
                return []
            except Exception:
                return [{"code": "database_unavailable"}]
        if service_name == "ffmpeg":
            try:
                return [] if self.ffmpeg.capabilities()["available"] else [{"code": "ffmpeg_unavailable"}]
            except Exception as exc:
                return [{"code": "ffmpeg_check_failed", "message": type(exc).__name__}]
        return []

    def maintenance(self, service_name, action, dry_run=True):
        if not self.has_service(service_name):
            raise KeyError(service_name)
        result = {"service": service_name, "action": action, "dry_run": dry_run}
        if service_name == "database" and action == "connectivity_check":
            result["result"] = self._database_status()
        elif service_name == "database" and action == "reconnect_primary":
            result["result"] = (
                {"active_mode": self.db.mode, "primary_probe": self.db.probe_primary()}
                if dry_run
                else self.db.force_primary_recovery()
            )
        elif service_name == "ffmpeg" and action == "capabilities_check":
            cap = self.ffmpeg.capabilities()
            result["result"] = {"available": cap["available"], "version": cap["ffmpeg_version"], "supported_output_formats": cap["supported_output_formats"]}
        elif service_name == "jobs" and action == "cleanup_expired_payloads":
            result["result"] = self.audio_service.cleanup_expired_data(dry_run=dry_run)
        elif service_name == "assets" and action == "audit":
            result["result"] = self.health("assets")["details"]
        elif service_name == "catalog" and action == "import_catalog":
            status = self.catalog_import.status()
            if dry_run:
                result["result"] = {"would_start": status.get("status") != "running", "import_status": status.get("status"), "dataset_available": self.catalog_import.dataset_path.is_file()}
            else:
                result["result"] = self.catalog_import.start_manual()
        elif service_name == "cleanup" and action == "cleanup_expired_payloads":
            result["result"] = self.audio_service.cleanup_expired_data(dry_run=dry_run)
        else:
            raise ValidationError("Unsupported maintenance action for this service")
        self.db.record_admin_event(service_name, "maintenance_preview" if dry_run else "maintenance_executed", {"action": action, "result": result["result"]})
        return result

    def audit(self, service_name, limit=50):
        if not self.has_service(service_name):
            raise KeyError(service_name)
        return self.db.list_admin_events(service_name, limit)
