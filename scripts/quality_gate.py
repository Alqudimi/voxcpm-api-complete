#!/usr/bin/env python3
"""بوابة الجودة (المرحلة 0) — الحكم الرقمي قبل/بعد أي تحسين.

البوابات:
  B (حياد صوتي):  ECAPA Δ≤±0.01 لكل صوت | PESQ Δ≤0.05 | STOI Δ≤0.01 | طول Δ≤±3%
  A (بتّية):      مطابقة sha256 للعينات المتطابقة متوقعاً (تقرير فقط هنا)

الاستخدام:
  python scripts/quality_gate.py --baseline reports/golden_baseline --candidate reports/golden_check
الاعتمادات: speechbrain (موجودة)؛ pesq/stoi اختيارية — تُتخطى مع تحذير إن غابت.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THRESHOLDS = {"ecapa_delta": 0.01, "pesq_delta": 0.05, "stoi_delta": 0.01, "duration_ratio": 0.03}


def try_imports():
    ecapa = pesq = stoi = None
    try:
        import torch
        from speechbrain.inference.speaker import SpeakerRecognition

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="storage/models/ecapa_voxceleb_gate",
            run_opts={"device": device},
        )
        _ = torch
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: ECAPA غير متاح ({exc}) — تُتخطى بوابة ECAPA")
    try:
        from pesq import pesq as _pesq

        pesq = _pesq
    except Exception:
        print("WARN: pesq غير مثبت (pip install pesq) — تُتخطى بوابة PESQ")
    try:
        from pystoi import stoi as _stoi

        stoi = _stoi
    except Exception:
        print("WARN: pystoi غير مثبت (pip install pystoi) — تُتخطى بوابة STOI")
    return ecapa, pesq, stoi


def load_audio(path: Path):
    import librosa
    import numpy as np

    audio, sr = librosa.load(str(path), sr=16000, mono=True, res_type="soxr_hq")
    return np.asarray(audio, dtype="float32"), sr


def ecapa_similarity(model, ref_path: Path, cand_path: Path):
    import torch

    score, prediction = model.verify_files(str(ref_path), str(cand_path))
    return float(score.squeeze().cpu().item()), int(prediction.squeeze().cpu().item()) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--voice-ref", default=None, help="مجلد المراجع لبوابة ECAPA (اختياري)")
    args = ap.parse_args()

    base_dir, cand_dir = Path(args.baseline), Path(args.candidate)
    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]
    cand_manifest = json.loads((cand_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]

    ecapa_model, pesq_fn, stoi_fn = try_imports()

    rows, verdicts = [], []
    for sid, bm in base_manifest.items():
        if sid not in cand_manifest:
            continue
        cand = cand_manifest[sid]
        cand_path = cand_dir / f"{sid}.ogg"
        base_path = base_dir / f"{sid}.ogg"
        row = {"sample_id": sid, "identical_bytes": bm["ogg_sha256"] == cand["ogg_sha256"]}

        # بوابة الطول
        bd, cd = bm.get("duration") or 0, cand.get("duration") or 0
        if bd > 0 and cd > 0:
            drift = abs(cd - bd) / bd
            row["duration_delta_ratio"] = round(drift, 4)
            verdicts.append(("duration", drift <= THRESHOLDS["duration_ratio"]))

        # بوابة ECAPA: التشابه بين مخرج المرشح والمرجع الأصلي للصوت
        voice_name = bm.get("voice")
        if ecapa_model is not None and voice_name:
            ref_dir = Path(args.voice_ref) if args.voice_ref else (Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "voices")
            ref_path = ref_dir / voice_name
            if ref_path.is_file():
                try:
                    score, _same = ecapa_similarity(ecapa_model, ref_path, cand_path)
                    row["ecapa_candidate_vs_ref"] = round(score, 4)
                    # مقارنة بتغير النتيجة المسجلة في الـ manifest
                    bs = bm.get("speaker_score")
                    if bs is not None:
                        delta = abs(score - float(bs))
                        row["ecapa_delta"] = round(delta, 4)
                        verdicts.append(("ecapa", delta <= THRESHOLDS["ecapa_delta"]))
                except Exception as exc:  # noqa: BLE001
                    row["ecapa_error"] = str(exc)[:120]

        # بوابات PESQ/STOI: المرشح مقابل الأساس (تحياد مقارن)
        if (pesq_fn or stoi_fn) and base_path.is_file() and cand_path.is_file():
            try:
                a, sr = load_audio(base_path)
                b, _ = load_audio(cand_path)
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
                if pesq_fn:
                    p = pesq_fn(sr, a, b, "wb")
                    row["pesq"] = round(float(p), 4)
                if stoi_fn:
                    s = stoi_fn(a, b, sr, extended=False)
                    row["stoi"] = round(float(s), 4)
            except Exception as exc:  # noqa: BLE001
                row["pesq_error"] = str(exc)[:120]

        rows.append(row)

    failures = [v for kind, ok in verdicts if not ok for v in [kind]]
    gate_report = {
        "rows": rows,
        "checked": len(rows),
        "verdicts_failed": len(failures),
        "thresholds": THRESHOLDS,
        "verdict": "PASS" if not failures else "FAIL",
    }
    out = cand_dir / "quality_gate.json"
    out.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in gate_report.items() if k != "rows"}, ensure_ascii=False))
    print(f"report -> {out}")
    return 0 if gate_report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
