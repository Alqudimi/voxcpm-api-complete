#!/usr/bin/env python3
"""bench_v3.py — قياس أداء النسخة v3-max.

وضعان:
  1) --profile : تقرير المضبوط التلقائي محلياً (بلا خادم) — يعرض كل القيم
     المحسوبة لعتادك + توقعات زمنية من المعايرة إن وُجدت.
  2) --url ... : قياس حي ضد خادم يعمل — يرسل طلبات clone حقيقية ويعرض
     p50/p90/min/max + زمن كاش التكرار (HIT).

أمثلة:
  python scripts/bench_v3.py --profile
  python scripts/bench_v3.py --url http://localhost:8000 --api-key KEY \
      --reference voice.wav --text "مرحبا بكم في الخدمة" --runs 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _percentile(values, p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def mode_profile() -> int:
    from app.core.autotune import probe_hardware, solve, tuner

    hw = probe_hardware()
    tuned = solve(hw)
    print("=" * 70)
    print("AutoTuner v3 — تقرير العتاد والقيم المحسوبة")
    print("=" * 70)
    gpus = hw.get("gpus", [])
    if gpus:
        for g in gpus:
            print(f"  GPU cuda:{g['index']}: {g['name']} {g['vram_mb']}MB (cc {g['cc_major']}.{g['cc_minor']}, {g['sm_count']} SM)")
    else:
        print("  GPU: لا يوجد — وضع CPU")
    print(f"  CPU logical cores: {hw.get('cpu_logical')} | RAM: {hw.get('ram_mb')}MB | /dev/shm free: {hw.get('shm_free_mb')}MB")
    print(f"  الحزم: {json.dumps(hw.get('packages', {}), ensure_ascii=False)}")
    print("-" * 70)
    print("  القيم المحسوبة (كل متغير = auto ما لم تثبّته صراحةً):")
    for k in sorted(tuned):
        print(f"    {k:26s} = {tuned[k]}")
    cal = tuner._calibration if isinstance(getattr(tuner, "_calibration", None), dict) else {}
    if cal.get("rtf"):
        print("-" * 70)
        rtf = cal["rtf"]
        print(f"  معايرة محفوظة: RTF={rtf:.3f} ({cal.get('engine')}, {cal.get('measured_at')})")
        for label, secs in (("نص قصير ~3s", 3), ("نص متوسط ~10s", 10), ("نص طويل ~30s", 30)):
            print(f"    توقع التوليد لـ{label}: ~{rtf * secs:.2f}s (+1-2s تحضير/ترميز)")
    else:
        print("-" * 70)
        print("  لا معايرة بعد — ستُقاس تلقائياً أثناء أول إحماء بإقلاع الخادم.")
    print("=" * 70)
    return 0


def mode_live(args) -> int:
    ref = Path(args.reference)
    if not ref.exists():
        print(f"ERROR: reference file not found: {ref}", file=sys.stderr)
        return 2
    base = args.url.rstrip("/")
    endpoint = f"{base}/api/v1/voice/clone"
    latencies = []
    hit_latency = None
    payload_fields = {
        "prompt_text": args.prompt,
        "target_text": args.text,
    }
    for i in range(args.runs):
        # بناء multipart يدوي بسيط (بلا اعتماديات خارجية)
        boundary = f"----benchv3{i}{int(time.time()*1000)}"
        parts = []
        for k, v in payload_fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            )
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="reference_audio"; '
                f'filename="ref.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            ).encode()
            + ref.read_bytes()
            + b"\r\n"
        )
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-API-Key": args.api_key,
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                status = resp.status
                resp.read()
        except Exception as exc:
            print(f"run {i+1}: FAILED {exc}", file=sys.stderr)
            continue
        dt = time.perf_counter() - t0
        latencies.append(dt)
        print(f"run {i+1}/{args.runs}: {dt:.2f}s (HTTP {status})")
    if not latencies:
        print("No successful runs.", file=sys.stderr)
        return 1
    # الطلب الأخير = تكرار مطابق → زمن كاش النتائج (إن فُعّل)
    hit_latency = latencies[-1]
    misses = latencies[:-1] if len(latencies) > 1 else latencies
    print("=" * 70)
    print(f"نتائج v3 ({len(misses)} توليد فعلي + 1 تكرار)")
    print(f"  min={min(misses):.2f}s  p50={_percentile(misses, 50):.2f}s  "
          f"p90={_percentile(misses, 90):.2f}s  max={max(misses):.2f}s  "
          f"mean={statistics.mean(misses):.2f}s")
    if len(latencies) > 1:
        speedup = misses[0] / hit_latency if hit_latency else 0
        print(f"  كاش التكرار (نفس الطلب): {hit_latency:.2f}s (تسريع ~{speedup:.0f}×)")
    try:
        with urllib.request.urlopen(f"{base}/api/v1/autotune", timeout=10) as resp:
            report = json.loads(resp.read().decode())
        rt = report.get("runtime", {})
        cal = report.get("calibration", {})
        print(f"  محرك/دقة/تكميم: {report.get('tuned', {}).get('engine')} / "
              f"{rt.get('precision')} / {rt.get('quantization')} | compile={rt.get('compile_mode')}")
        if cal.get("rtf"):
            print(f"  RTF معاير: {cal['rtf']:.3f}")
    except Exception:
        pass
    print("=" * 70)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="bench_v3 — قياس أداء النسخة v3-max")
    ap.add_argument("--profile", action="store_true", help="عرض تقرير المضبوط محلياً (بلا خادم)")
    ap.add_argument("--url", help="عنوان الخادم للقياس الحي")
    ap.add_argument("--api-key", default="change-me")
    ap.add_argument("--reference", default="voice.wav", help="ملف مرجع WAV")
    ap.add_argument("--prompt", default="", help="نص المرجع (اختياري مع auto_transcribe)")
    ap.add_argument("--text", default="مرحبا بكم في خدمة الاستنساخ الصوتي السريع.")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if args.url:
        return mode_live(args)
    return mode_profile()


if __name__ == "__main__":
    raise SystemExit(main())
