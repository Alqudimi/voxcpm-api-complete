#!/usr/bin/env python3
"""Validate the local Python/PyTorch CUDA runtime before starting VoxCPM2."""
from __future__ import annotations
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
    print("CUDA preflight passed. Next verify the VoxCPM2 package, model weights, FFmpeg and API key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
