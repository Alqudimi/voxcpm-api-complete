#!/usr/bin/env python3
"""Measure end-to-end clone-and-mix latency against a running API.

Example:
  API_BASE=http://127.0.0.1:8000 API_KEY=... python scripts/benchmark_clone_and_mix.py \
    --reference reference.wav --noise noise.wav --prompt-text '...' \
    --target-text '...' --runs 30 --warmups 2
"""
from __future__ import annotations
import argparse
import os
import statistics
import time
from pathlib import Path
import requests


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def one_run(base: str, key: str, args: argparse.Namespace) -> tuple[float, float]:
    headers = {"X-API-Key": key}
    url = base.rstrip("/") + "/api/v1/voice/clone-and-mix"
    data = {
        "target_text": args.target_text,
        "prompt_text": args.prompt_text,
        "cfg_value": "2.0",
        "inference_timesteps": "30",
        "candidates": "1",
        "retry_badcase": "true",
        "output_format": args.output_format,
    }
    started = time.perf_counter()
    with open(args.reference, "rb") as ref, open(args.noise, "rb") as noise:
        response = requests.post(
            url,
            headers=headers,
            files={
                "reference_audio": (Path(args.reference).name, ref),
                "noise_audio": (Path(args.noise).name, noise),
            },
            data=data,
            timeout=args.timeout,
        )
    accepted_at = time.perf_counter()
    if response.status_code != 202:
        raise RuntimeError(f"POST failed: HTTP {response.status_code}: {response.text[:1000]}")
    payload = response.json()
    status_url, result_url = payload["status_url"], payload["result_url"]
    while True:
        status_response = requests.get(status_url, headers=headers, timeout=30)
        status_response.raise_for_status()
        status = status_response.json()
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            raise RuntimeError(f"Job failed: {status}")
        if time.perf_counter() - started > args.timeout:
            raise TimeoutError(f"Job did not finish within {args.timeout}s: {status}")
        time.sleep(args.poll_ms / 1000)
    result = requests.get(result_url, headers=headers, timeout=args.timeout)
    result.raise_for_status()
    if not result.content:
        raise RuntimeError("Result download was empty")
    finished = time.perf_counter()
    return accepted_at - started, finished - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-format", choices=["wav", "mp3", "ogg", "flac"], default="wav")
    args = parser.parse_args()
    base = os.environ.get("API_BASE", "http://127.0.0.1:8000")
    key = os.environ.get("API_KEY")
    if not key:
        raise SystemExit("Set API_KEY in the environment; never put secrets in command-line arguments.")
    for path in (args.reference, args.noise):
        if not Path(path).is_file():
            raise SystemExit(f"Input file not found: {path}")
    for i in range(args.warmups):
        one_run(base, key, args)
        print(f"warmup {i + 1}/{args.warmups}: done", flush=True)
    accepted, total = [], []
    for i in range(args.runs):
        post_time, total_time = one_run(base, key, args)
        accepted.append(post_time)
        total.append(total_time)
        print(f"run {i + 1}/{args.runs}: accept={post_time:.3f}s e2e={total_time:.3f}s", flush=True)
    for label, values in (("202 response", accepted), ("end-to-end incl. result download", total)):
        print(f"\n{label}: n={len(values)} mean={statistics.mean(values):.3f}s")
        for q in (0.50, 0.95, 0.99):
            print(f"  p{int(q * 100)}={percentile(values, q):.3f}s")


if __name__ == "__main__":
    main()
