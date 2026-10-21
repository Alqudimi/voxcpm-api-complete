import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from app.audio_mixer import service as service_module
from app.audio_mixer.service import AudioService


class _DB:
    def __init__(self):
        self.params = None

    @contextmanager
    def connect(self):
        class Connection:
            def __init__(inner):
                inner.db = self

            def execute(inner, _sql, params=()):
                self.params = params
        yield Connection()


class _FFmpeg:
    def normalize_output_format(self, fmt):
        return fmt or "wav"


class _Thread:
    created = []

    def __init__(self, target, args, **_kwargs):
        self.created.append((target, args))

    def start(self):
        pass


def _build_service(tmp_path):
    db = _DB()
    settings = SimpleNamespace(
        temp_path=Path(tmp_path),
        max_queue_size=8,
        max_concurrent_jobs=2,
        min_noise_margin_seconds=2,
    )
    return AudioService(db, _FFmpeg(), settings), db


def test_memory_inputs_skip_blob_persistence_and_are_passed_to_worker(tmp_path, monkeypatch):
    _Thread.created.clear()
    monkeypatch.setattr(service_module.threading, "Thread", _Thread)
    service, db = _build_service(tmp_path)
    source, noise = b"source pcm", b"noise pcm"

    job_id = service.submit_mix_bytes(
        source,
        source_reference="generated.wav",
        noise_bytes=noise,
        noise_reference="noise.wav",
        persist_inputs=False,
    )

    params = db.params
    assert params[0] == job_id
    assert params[14] is None and params[15] is None
    cfg = json.loads(params[4])
    assert cfg["storage"] == "worker_memory"
    target, args = _Thread.created[-1]
    assert target == service._run
    assert args == (job_id, source, noise)


def test_persistent_inputs_remain_the_default(tmp_path, monkeypatch):
    _Thread.created.clear()
    monkeypatch.setattr(service_module.threading, "Thread", _Thread)
    service, db = _build_service(tmp_path)
    source, noise = b"source", b"noise"

    job_id = service.submit_mix_bytes(source, noise_bytes=noise)

    params = db.params
    assert params[0] == job_id
    assert params[14] == source and params[15] == noise
    assert json.loads(params[4])["storage"] == "postgresql_bytea"
    assert _Thread.created[-1][1] == (job_id, None, None)
