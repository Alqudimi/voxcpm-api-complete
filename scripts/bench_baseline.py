#!/usr/bin/env python3
"""قياس خط الأساس (المرحلة 0 من خطة التسريع).

يرسل N طلباً إلى نقطة /clone أو /clone-and-mix ويجمع P50/P95/P99 زمن
الاستجابة الكامل من جهة العميل. توقيتات المراحل الداخلية تُقرأ من سجل
الخادم (سطور STAGES JSON المطبوعة عند VM_LOG_STAGES=on).

الاستخدام:
  python scripts/bench_baseline.py --api-key KEY --runs 30 \
      --reference tests/fixtures/voices/voice_a.wav \
      --text "مرحبا هذا اختبار" [--noise noise.wav --endpoint clone-and-mix]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


def percentiles(values, ps=(50, 95, 99)):
    out = {}
    for p in ps:
        if not values:
            out[f"p{p}"] = None
            continue
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, round((p / 100) * len(ordered) + 0.5) - 1))
        out[f"p{p}"] = round(ordered[k], 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--noise", default=None, help="مطلوب مع endpoint=clone-and-mix")
    ap.add_argument("--endpoint", default="clone", choices=["clone", "clone-and-mix"])
    ap.add_argument("--text", default="مرحبا، هذا اختبار قياس للأداء.")
    ap.add_argument("--prompt-text", default=None)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.is_file():
        raise SystemExit(f"reference not found: {ref_path}")
    noise_path = Path(args.noise) if args.noise else None
    if args.endpoint == "clone-and-mix" and (noise_path is None or not noise_path.is_file()):
        raise SystemExit("--noise مطلوب مع endpoint=clone-and-mix")

    url = f"{args.base_url.rstrip('/')}/api/v1/voice/{args.endpoint}"
    headers = {"X-API-Key": args.api_key}

    latencies, errors = [], []
    for i in range(args.runs):
        data = {
            "target_text": args.text,
            "cfg_value": "2.0",
            "inference_timesteps": "30",
        }
        if args.prompt_text:
            data["prompt_text"] = args.prompt_text
        files = {"reference_audio": (ref_path.name, ref_path.read_bytes(), "audio/wav")}
        if noise_path is not None:
            files["noise_audio"] = (noise_path.name, noise_path.read_bytes(), "audio/wav")
        t0 = time.perf_counter()
        try:
            r = requests.post(url, headers=headers, data=data, files=files, timeout=600)
            elapsed = time.perf_counter() - t0
            if r.status_code in (200, 202):
                latencies.append(elapsed)
                print(f"[{i+1}/{args.runs}] {elapsed:.2f}s OK")
            else:
                errors.append(f"{r.status_code}: {r.text[:200]}")
                print(f"[{i+1}/{args.runs}] FAIL {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            print(f"[{i+1}/{args.runs}] ERROR {exc}")

    report = {
        "date": datetime.now().isoformat(),
        "endpoint": args.endpoint,
        "runs": args.runs,
        "successful": len(latencies),
        "failed": len(errors),
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "min": round(min(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
            **percentiles(latencies),
        },
        "errors_sample": errors[:5],
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baseline_{args.endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport -> {out_file}")


if __name__ == "__main__":
    main()
