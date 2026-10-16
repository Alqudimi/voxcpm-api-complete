import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path


def utc_now():
    return datetime.now(UTC).isoformat()


class CatalogImportManager:
    def __init__(self, settings):
        self.settings = settings
        self.dataset_path = Path(settings.dataset_path).resolve()
        self.root = settings.temp_path / "catalog"
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "import_status.json"
        self.lock_path = self.root / "import.lock"
        self._thread = None
        self.import_workers = settings.import_workers
        self.import_timeout = settings.import_timeout
        self.database_url_provider = None

    def _write_status(self, **changes):
        current = {}
        if self.status_path.exists():
            try:
                current = json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current.update(changes)
        current["updated_at"] = utc_now()
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.status_path)

    def status(self):
        if not self.status_path.exists():
            return {
                "status": "not_started",
                "auto_import_enabled": self.settings.auto_import_mixkit,
            }
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def start_if_enabled(self):
        if not self.settings.auto_import_mixkit:
            self._write_status(
                status="disabled",
                reason="AUDIO_MIXER_AUTO_IMPORT_MIXKIT is false",
                auto_import_enabled=False,
            )
            return
        self._start()

    def start_manual(self):
        return self._start()

    def _start(self):
        if not self.dataset_path.is_file():
            self._write_status(
                status="failed",
                error="dataset_not_found",
                dataset=str(self.dataset_path),
            )
            return {"status": "failed", "error": "dataset_not_found"}
        if self._thread and self._thread.is_alive():
            return {"status": "running", "already_running": True}
        self._thread = threading.Thread(
            target=self._run, name="mixkit-auto-import", daemon=True
        )
        self._thread.start()
        return {"status": "started"}

    def _run(self):
        lock_file = self.lock_path.open("a+")
        try:
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._write_status(status="running", owner="another-process")
                return
            self._write_status(
                status="running", dataset=str(self.dataset_path), started_at=utc_now()
            )
            project_root = Path(__file__).resolve().parents[2]
            script = project_root / "scripts" / "import_mixkit.py"
            command = [
                sys.executable,
                str(script),
                "--dataset",
                str(self.dataset_path),
                "--database",
                self.database_url_provider() if self.database_url_provider else self.settings.database_url,
                "--storage",
                str(self.settings.temp_path / "catalog-import"),
                "--workers",
                str(self.import_workers),
                "--timeout",
                str(self.import_timeout),
                "--retries",
                "2",
            ]
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=None,
                check=False,
            )
            if result.returncode == 0:
                self._write_status(
                    status="completed",
                    completed_at=utc_now(),
                    output=result.stdout[-2000:],
                )
            else:
                self._write_status(
                    status="failed",
                    completed_at=utc_now(),
                    returncode=result.returncode,
                    output=(result.stdout + "\n" + result.stderr)[-4000:],
                )
        except Exception as exc:
            self._write_status(status="failed", completed_at=utc_now(), error=repr(exc))
        finally:
            lock_file.close()
