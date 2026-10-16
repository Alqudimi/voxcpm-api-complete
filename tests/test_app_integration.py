"""CPU-only smoke tests for the combined VoxCPM and Audio Mixer ASGI app.

Model packages are replaced with import-time stubs so these tests exercise the
real FastAPI routing, lifecycle, authentication, SQLite and FFmpeg paths without
loading or downloading multi-gigabyte model weights.
"""

from __future__ import annotations

import atexit
import io
import os
import shutil
import sys
import tempfile
import types
import wave
from pathlib import Path
from urllib.parse import urlparse

import pytest

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="audio-mixer-app-smoke-"))
atexit.register(shutil.rmtree, _TEST_ROOT, ignore_errors=True)
_TEST_API_KEY = "cpu-smoke-test-key"
os.environ["API_KEY"] = _TEST_API_KEY
os.environ.pop("AUDIO_MIXER_API_KEY", None)
os.environ["AUDIO_MIXER_TEMP_PATH"] = str(_TEST_ROOT / "audio-temp")
os.environ["AUDIO_MIXER_STORAGE_PATH"] = str(_TEST_ROOT / "audio-storage")
os.environ["AUDIO_MIXER_DATABASE_URL"] = f"sqlite:///{_TEST_ROOT / 'audio.sqlite3'}"
os.environ["AUDIO_MIXER_DATASET_PATH"] = str(_TEST_ROOT / "missing-dataset.json")
os.environ["AUDIO_MIXER_AUTO_IMPORT_MIXKIT"] = "false"
os.environ["STORAGE_ROOT"] = str(_TEST_ROOT / "host-storage")
os.environ["UPLOAD_DIR"] = str(_TEST_ROOT / "host-storage" / "uploads")
os.environ["OUTPUT_DIR"] = str(_TEST_ROOT / "host-storage" / "outputs")
os.environ["TEMP_DIR"] = str(_TEST_ROOT / "host-storage" / "temp")
os.environ["ECAPA_SAVEDIR"] = str(_TEST_ROOT / "host-storage" / "ecapa")


def _install_model_import_stubs() -> None:
    """Avoid loading ML packages; this suite tests the application shell only."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = torch

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda *args, **kwargs: None
    sys.modules["whisper"] = whisper

    voxcpm = types.ModuleType("voxcpm")
    voxcpm.VoxCPM = type("VoxCPM", (), {})
    sys.modules["voxcpm"] = voxcpm

    speechbrain = types.ModuleType("speechbrain")
    speechbrain.__path__ = []
    inference = types.ModuleType("speechbrain.inference")
    inference.__path__ = []
    speaker = types.ModuleType("speechbrain.inference.speaker")
    speaker.SpeakerRecognition = type("SpeakerRecognition", (), {})
    sys.modules.update(
        {
            "speechbrain": speechbrain,
            "speechbrain.inference": inference,
            "speechbrain.inference.speaker": speaker,
        }
    )


_install_model_import_stubs()

from fastapi.testclient import TestClient

from app.core.model_manager import model_manager
from app.main import app


def _wav_bytes(frequency: int, seconds: float = 1.0, sample_rate: int = 24_000) -> bytes:
    import math
    import struct

    frames = bytearray()
    for index in range(int(seconds * sample_rate)):
        value = int(8_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return output.getvalue()


def test_combined_app_auth_lifecycle_and_cpu_audio_endpoints(monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.fail("FFmpeg and ffprobe are required for the CPU integration test")

    calls = {"load": 0, "unload": 0}

    async def fake_load_all():
        calls["load"] += 1

    async def fake_unload_all():
        calls["unload"] += 1

    monkeypatch.setattr(model_manager, "load_all", fake_load_all)
    monkeypatch.setattr(model_manager, "unload_all", fake_unload_all)

    assert os.environ["AUDIO_MIXER_API_KEY"] == _TEST_API_KEY
    paths = set(app.openapi()["paths"])
    assert "/api/v1/audio/merge" in paths
    assert "/api/v1/audio/assets" in paths
    assert "/api/v1/audio/process/from-upload" in paths

    headers = {"X-API-Key": _TEST_API_KEY}
    source = _wav_bytes(440)
    noise = _wav_bytes(880)

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200

        assert client.get("/api/v1/audio/assets").status_code == 401
        assets = client.get("/api/v1/audio/assets", headers=headers)
        assert assets.status_code == 200
        assert isinstance(assets.json(), list)

        mixed = client.post(
            "/api/v1/audio/process/from-upload",
            headers=headers,
            files={
                "source_audio": ("source.wav", source, "audio/wav"),
                "noise_audio": ("noise.wav", noise, "audio/wav"),
            },
            data={"output_format": "wav", "variation_seed": "17"},
        )
        assert mixed.status_code == 200, mixed.text
        assert mixed.content[:4] == b"RIFF"
        assert int(mixed.headers["content-length"]) > 44

        merged = client.post(
            "/api/v1/audio/merge",
            headers=headers,
            files=[
                ("files", ("first.wav", source, "audio/wav")),
                ("files", ("second.wav", noise, "audio/wav")),
            ],
        )
        assert merged.status_code == 200, merged.text
        result = merged.json()
        assert result["success"] is True
        output_path = urlparse(result["audio_url"]).path
        fetched = client.get(output_path, headers=headers)
        assert fetched.status_code == 200
        assert fetched.content.startswith(b"OggS")

        capabilities = client.get("/api/v1/audio/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert capabilities.json()["available"] is True

    assert calls == {"load": 1, "unload": 1}
