"""اختبارات وحدوية لطبقة v3-max: AutoTuner + GPUScheduler + الخصائص المضبوطة.

تعمل على CPU خالص (بلا torch حقيقي) — كل القرارات تُختبر كدوال نقية
على بروفايلات عتاد مصطنعة (T4 / RTX 4090 / 2×3090 / CPU فقط).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.core.autotune import AutoTuner, probe_hardware, solve
from app.core.gpu_scheduler import GPUScheduler


def _hw_cpu_only() -> dict:
    return {
        "platform": "test", "machine": "x86_64", "python": "3.11.0",
        "cpu_logical": 8, "ram_mb": 16384, "shm_free_mb": 2048,
        "cuda": False, "gpu_count": 0, "gpus": [],
        "torch": None, "cuda_version": None,
        "packages": {"torchao": False, "faster_whisper": False, "nanovllm_voxcpm": False, "uvloop": False, "soxr": True},
    }


def _hw_single(cc_major: float, vram_mb: int, name: str = "GPU", nanovllm: bool = True) -> dict:
    hw = _hw_cpu_only()
    hw.update({"cuda": True, "gpu_count": 1, "cuda_version": "12.4", "torch": "2.6.0"})
    hw["gpus"] = [{
        "index": 0, "name": name, "vram_mb": vram_mb,
        "cc_major": int(cc_major), "cc_minor": int(round((cc_major % 1) * 10)),
        "sm_count": 40,
    }]
    hw["packages"] = {"torchao": True, "faster_whisper": True, "nanovllm_voxcpm": nanovllm, "uvloop": True, "soxr": True}
    return hw


def _hw_multi() -> dict:
    hw = _hw_single(8.6, 24576, "RTX 3090")
    hw["gpu_count"] = 2
    hw["gpus"].append(dict(hw["gpus"][0], index=1))
    return hw


# ─────────────────────────── solve(): قواعد العتاد ───────────────────────────

def test_solve_cpu_only_is_conservative():
    t = solve(_hw_cpu_only())
    assert t["dtype"] == "fp32"
    assert t["compile_mode"] == "off"
    assert t["quant"] == "off"
    assert t["engine"] == "official"
    assert t["whisper_model"] == "small"
    assert t["ecapa_device"] == "cpu"
    assert t["gpu_ids"] == []
    assert t["gpu_permits"] == 1


def test_solve_turing_t4():
    t = solve(_hw_single(7.5, 16110, "T4", nanovllm=False))
    assert t["dtype"] == "fp16"
    assert t["tf32"] is True
    assert t["quant"] == "int8"          # cc 7.5 + torchao متوفر
    assert t["compile_mode"] == "reduce-overhead"  # vram ≥ 16000
    assert t["engine"] == "official"     # nanovllm يتطلب cc≥8


def test_solve_ada_4090_takes_fp8_and_bf16():
    t = solve(_hw_single(8.9, 24576, "RTX 4090", nanovllm=False))
    assert t["dtype"] == "bf16"
    assert t["quant"] == "fp8"           # cc ≥ 8.9
    assert t["compile_mode"] == "reduce-overhead"
    assert t["gpu_permits"] == 2         # vram ≥ 16000
    assert t["whisper_model"] == "keep"  # VRAM كافية → يحافظ على large-v3
    assert t["whisper_batch_size"] == 8
    assert t["ecapa_device"] == "cuda"


def test_solve_dual_gpu_pool_and_scatter():
    t = solve(_hw_multi())
    assert t["gpu_ids"] == [0, 1]
    assert t["chunk_parallel_max"] == 4  # permits(2) × gpus(2)
    assert t["executor_workers"] >= 4
    assert t["quant"] == "int8"          # cc 8.6 < 8.9 → int8


def test_solve_ampere_below_fp8_gets_int8():
    t = solve(_hw_single(8.0, 10240, "RTX 3080", nanovllm=False))
    assert t["dtype"] == "bf16"
    assert t["quant"] == "int8"
    assert t["compile_mode"] == "default"  # 10GB ≥ 10000 و < 16000
    assert t["gpu_permits"] == 1


def test_solve_nanovllm_disables_compile_and_quant():
    t = solve(_hw_single(8.9, 24576))
    t["engine"] = "nanovllm"  # المحرك البديل يدير دقته بنفسه
    s = solve(_hw_single(8.9, 24576))
    # نتحقق من القاعدة عبر شرط المحرك في الحل مباشرة:
    hw = _hw_single(8.9, 24576)
    hw["packages"]["nanovllm_voxcpm"] = True
    s2 = solve(hw)
    assert s2["engine"] == "nanovllm"
    assert s2["compile_mode"] == "off"
    assert s2["quant"] == "off"


def test_solve_budgets_scale_with_ram():
    small = solve(dict(_hw_cpu_only(), ram_mb=4096))
    big = solve(dict(_hw_cpu_only(), ram_mb=131072))
    assert small["result_cache_max_mb"] < big["result_cache_max_mb"]
    assert small["speaker_cache_max"] < big["speaker_cache_max"]
    assert small["ram_staging_max_mb"] < big["ram_staging_max_mb"]
    assert big["result_cache_max_mb"] <= 512 and big["ram_staging_max_mb"] <= 1024


# ─────────────────── أولوية القيمة الصريحة فوق المضبوط ───────────────────

def _tuner_with(tuned: dict) -> AutoTuner:
    t = AutoTuner()
    t._tuned = tuned
    t._hw = _hw_cpu_only()
    return t


def test_explicit_string_beats_tuned():
    t = _tuner_with({"dtype": "bf16", "compile_mode": "reduce-overhead"})
    assert t.resolve_str("fp16", "dtype", "fp32") == "fp16"
    assert t.resolve_str("off", "compile_mode", "reduce-overhead") == "off"
    assert t.resolve_str("auto", "dtype", "fp32") == "bf16"
    assert t.resolve_str(None, "dtype", "fp32") == "bf16"


def test_explicit_int_beats_tuned_and_zero_means_auto():
    t = _tuner_with({"executor_workers": 7})
    assert t.resolve_int(0, "executor_workers", 2) == 7
    assert t.resolve_int(4, "executor_workers", 2) == 4
    assert t.resolve_int("3", "executor_workers", 2) == 3


def test_gpu_ids_explicit_vs_auto():
    t = _tuner_with({"gpu_ids": [0, 1]})
    assert t.resolve_gpu_ids("auto") == [0, 1]
    # القيمة الصريحة تمر عبر parse_gpu_ids (تُفلتر على الأجهزة الموجودة — لا كروت في بيئة الاختبار → [])
    assert t.resolve_gpu_ids("0,1") == []


def test_whisper_model_resolution():
    t = _tuner_with({"whisper_model": "keep"})
    assert t.resolve_whisper_model("auto", "large-v3") == "large-v3"
    t._tuned = {"whisper_model": "small"}
    assert t.resolve_whisper_model("auto", "large-v3") == "small"
    assert t.resolve_whisper_model("medium", "large-v3") == "medium"


# ─────────────────────────── استمرارية البروفايل ───────────────────────────

def test_profile_persist_and_reload(tmp_path: Path):
    t = AutoTuner()
    t._profile_path = tmp_path / "profile.json"
    t._hw = _hw_single(8.9, 24576)
    t._tuned = solve(t._hw)
    t._calibration = {"rtf": 0.21}
    t._persist()
    assert t._profile_path.exists()

    t2 = AutoTuner()
    t2._profile_path = t._profile_path
    assert t2._load_cached_profile() is True
    assert t2._tuned == t._tuned
    assert t2._calibration["rtf"] == 0.21


def test_profile_rejects_foreign_fingerprint(tmp_path: Path):
    t = AutoTuner()
    t._profile_path = tmp_path / "profile.json"
    t._hw = _hw_single(8.9, 24576)
    t._tuned = solve(t._hw)
    t._persist()

    t2 = AutoTuner()
    t2._profile_path = t._profile_path
    assert t2._load_cached_profile() is True
    # عتاد مختلف (بصمة لا تطابق) → يُرفض الكاش
    from app.core.autotune import json as _json
    data = _json.loads(t._profile_path.read_text("utf-8"))
    data["hardware"]["gpus"][0]["vram_mb"] = 8000
    t._profile_path.write_text(_json.dumps(data), "utf-8")
    t3 = AutoTuner()
    t3._profile_path = t._profile_path
    assert t3._load_cached_profile() is False


def test_probe_never_raises():
    hw = probe_hardware()
    assert isinstance(hw, dict)
    assert hw["cpu_logical"] >= 1
    assert hw["ram_mb"] > 0
    assert isinstance(hw["gpus"], list)


# ─────────────────────────── GPUScheduler ───────────────────────────

def test_scheduler_limits_concurrency_per_device():
    s = GPUScheduler()
    s.configure([0], permits_per_gpu=2)
    running = []
    entered = threading.Event()
    release = threading.Event()

    def work():
        with s.slot(0):
            running.append(1)
            entered.set()
            release.wait(5)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
        time.sleep(0.02)
    entered.wait(5)
    time.sleep(0.1)
    # تصريحان فقط يدخلان؛ الباقي ينتظر
    assert len(running) == 2
    release.set()
    for th in threads:
        th.join(5)
    assert len(running) == 4
    assert s.stats()["active_total"] == 0


def test_scheduler_least_loaded_spreads():
    s = GPUScheduler()
    s.configure([0, 1], permits_per_gpu=2)
    # اشغل الكرت 0 بالكامل
    with s.slot(0):
        with s.slot(0):
            assert s.least_loaded() == 1
    assert s.least_loaded() in (0, 1)


def test_scheduler_unconfigured_slot_passthrough():
    s = GPUScheduler()
    with s.slot(5):
        pass
    assert s.configured() is False


# ─────────────────── خصائص الإعدادات المضبوطة (تكامل) ───────────────────

def test_settings_properties_resolve(monkeypatch):
    from app.core.config import settings

    # الصريح يتقدم
    monkeypatch.setattr(settings, "VM_EXECUTOR_WORKERS", 5)
    assert settings.executor_workers == 5
    monkeypatch.setattr(settings, "VM_EXECUTOR_WORKERS", 0)
    assert settings.executor_workers >= 2  # auto: من المضبوط/الافتراضي

    monkeypatch.setattr(settings, "VM_AUTO_CHUNK", "off")
    assert settings.auto_chunk_chars == 0
    monkeypatch.setattr(settings, "VM_AUTO_CHUNK", "on")
    assert settings.auto_chunk_chars >= 100 or settings.auto_chunk_chars == 600

    monkeypatch.setattr(settings, "VM_COMPILE", "off")
    assert settings.compile_mode == "off"
    assert settings.compile_enabled is False
    monkeypatch.setattr(settings, "VM_COMPILE", "auto")
    assert settings.compile_mode in {"off", "default", "reduce-overhead", "max-autotune-no-cudagraphs"}

    monkeypatch.setattr(settings, "VM_WHISPER_GREEDY", "auto")
    assert settings.whisper_greedy in (True, False)


def test_calibrate_keeps_best_rtf(tmp_path: Path):
    t = AutoTuner()
    t._profile_path = tmp_path / "p.json"
    t._hw = _hw_cpu_only()
    t._tuned = solve(t._hw)
    t.calibrate(rtf=0.30, gen_seconds=3.0, audio_seconds=10.0, engine="official")
    t.calibrate(rtf=0.20, gen_seconds=2.0, audio_seconds=10.0, engine="official")
    t.calibrate(rtf=0.25, gen_seconds=2.5, audio_seconds=10.0, engine="official")
    assert t._calibration["rtf"] == 0.20  # الأفضل يبقى
    assert t._calibration["last_rtf"] == 0.25
