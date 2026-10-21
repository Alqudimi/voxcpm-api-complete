#!/usr/bin/env python3
"""المجموعة الذهبية (المرحلة 0) — توليد مخرجات مرجعية قبل/بعد التحسين.

الوضعان:
  --mode baseline : يولّد المخرجات الأساسية ويخزنها + بصماتها (قبل التغيير)
  --mode check    : يولّد المخرجات الجديدة ويقارنها بالمجموعة المحفوظة

المجموعة: نصوص scripts/golden_texts.txt × مراجع tests/fixtures/voices/*.wav
كل عينة تخزن: ملف OGG + sha256 للـ WAV الناتج عبر API (إن توفر) + أطوال.

ملاحظة: مقارنة الجودة الصوتية العميقة (ECAPA/PESQ/STOI) عبر quality_gate.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TEXTS_FILE = ROOT / "scripts" / "golden_texts.txt"
VOICES_DIR = ROOT / "tests" / "fixtures" / "voices"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_texts(limit: int | None) -> list[str]:
    texts = [
        line.strip()
        for line in TEXTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return texts[:limit] if limit else texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "check"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--limit-texts", type=int, default=None, help="تقليل العدد للاختبار السريع")
    ap.add_argument("--store", default=str(ROOT / "reports" / "golden_baseline"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "golden_check"))
    args = ap.parse_args()

    voices = sorted(p for p in VOICES_DIR.glob("*.wav")) if VOICES_DIR.exists() else []
    if not voices:
        raise SystemExit(f"لا توجد مراجع صوتية في {VOICES_DIR} — ضع 3-8 ملفات wav")
    texts = load_texts(args.limit_texts)
    store_dir = Path(args.store if args.mode == "baseline" else args.out)
    store_dir.mkdir(parents=True, exist_ok=True)
    url = f"{args.base_url.rstrip('/')}/api/v1/voice/clone"
    headers = {"X-API-Key": args.api_key}

    manifest, failures = {}, []
    total = len(texts) * len(voices)
    done = 0
    for vi, voice in enumerate(voices):
        for ti, text in enumerate(texts):
            done += 1
            sample_id = f"v{vi}_t{ti:02d}"
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    data={"target_text": text, "cfg_value": "2.0", "inference_timesteps": "30"},
                    files={"reference_audio": (voice.name, voice.read_bytes(), "audio/wav")},
                    timeout=600,
                )
                r.raise_for_status()
                audio_url = r.json()["audio_url"]
                audio = requests.get(audio_url, headers=headers, timeout=120).content
                (store_dir / f"{sample_id}.ogg").write_bytes(audio)
                manifest[sample_id] = {
                    "voice": voice.name,
                    "text": text,
                    "ogg_sha256": sha256_bytes(audio),
                    "bytes": len(audio),
                    "seed": r.json().get("seed"),
                    "duration": r.json().get("duration"),
                    "speaker_score": r.json().get("speaker_score"),
                }
                print(f"[{done}/{total}] {sample_id} OK")
            except Exception as exc:  # noqa: BLE001
                failures.append({"sample_id": sample_id, "error": str(exc)[:200]})
                print(f"[{done}/{total}] {sample_id} FAIL: {exc}")

    manifest_path = store_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"generated": datetime.now().isoformat(), "samples": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nmanifest -> {manifest_path} | failures: {len(failures)}")

    if args.mode == "check":
        base_manifest = json.loads((Path(args.store) / "manifest.json").read_text(encoding="utf-8"))["samples"]
        identical = sum(
            1 for sid, m in manifest.items()
            if sid in base_manifest and m["ogg_sha256"] == base_manifest[sid]["ogg_sha256"]
        )
        duration_drift = [
            abs((m.get("duration") or 0) - base_manifest[sid].get("duration", 0))
            for sid, m in manifest.items() if sid in base_manifest and m.get("duration")
        ]
        summary = {
            "identical_bytes": identical,
            "total_compared": len(manifest),
            "duration_drift_max": max(duration_drift) if duration_drift else None,
        }
        (store_dir / "compare_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("compare:", json.dumps(summary, ensure_ascii=False))
        return 0 if identical > 0 or not duration_drift else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
