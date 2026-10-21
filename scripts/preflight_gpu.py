#!/usr/bin/env python3
"""Validate the local Python/PyTorch CUDA runtime before starting VoxCPM2."""
from __future__ import annotations
import os
import shutil
import subprocess
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        print("ERROR: VoxCPM2 documents Python 3.10–3.12.")
        return 2
    try:
        import torch
    except Exception as exc:
        print(f"ERROR: Cannot import torch: {type(exc).__name__}: {exc}")
        return 2
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is false. Install a CUDA-enabled PyTorch build and verify the NVIDIA driver/container GPU passthrough.")
        return 2
    count = torch.cuda.device_count()
    print(f"CUDA devices: {count}")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        bf16 = bool(torch.cuda.is_bf16_supported())
        print(f"GPU {index}: {props.name}")
        print(f"  compute capability: {props.major}.{props.minor}")
        print(f"  VRAM total/free: {total_bytes / 1024**3:.2f}/{free_bytes / 1024**3:.2f} GiB")
        print(f"  BF16 supported: {bf16}")
    engine = os.environ.get("TTS_ENGINE")
    if engine is None:
        try:
            for line in open(".env", encoding="utf-8"):
                if line.strip().startswith("TTS_ENGINE="):
                    engine = line.split("=", 1)[1].strip().strip("'\"")
                    break
        except OSError:
            pass
    engine = (engine or "nanovllm").lower()
    print(f"TTS engine: {engine}")
    if engine == "nanovllm":
        for module in ("nanovllm_voxcpm", "flash_attn"):
            try:
                __import__(module)
            except Exception as exc:
                print(f"ERROR: {module} is missing or cannot load: {type(exc).__name__}: {exc}")
                return 2
        print("NanoVLLM and FlashAttention imports passed")
    elif engine != "pytorch":
        print("ERROR: TTS_ENGINE must be nanovllm or pytorch")
        return 2
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        print("ERROR: ffmpeg and ffprobe must be installed and available on PATH")
        return 2
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    if encoders.returncode != 0 or "libopus" not in encoders.stdout:
        print("ERROR: the installed FFmpeg build must include the libopus encoder")
        return 2
    print("CUDA preflight passed. Next verify the VoxCPM2 package, model weights, FFmpeg and API key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
