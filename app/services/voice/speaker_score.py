"""تقييم تشابه المتحدث من الذاكرة (v2-ultra) — ECAPA يخرج من المسار الحرج نهائياً.

قبل v1: verify_files (كتابة WAV → تحميل ملفين → إعادة عينة → encode ×2 → cosine)
بعد v1: نفس العملية متوازية مع OGG.
بعد v2:
  1. embedding المرجع يُحسب **مرة واحدة لكل متحدث** ويُخزَّن (كاش bit-exact) —
     الطلبات اللاحقة تحسب embedding المولَّد فقط (نصف العمل).
  2. كلا الطرفين يُحمَّلان من RAM (BytesIO) عبر torchaudio/soundfile —
     صفر ملفات وسيطة، صفر قرص.
  3. عند أي عائق (torchaudio غائب/بيئة اختبار/خطأ غير متوقع) سقوط آمن
     إلى verify_files الأصلي حرفياً — عقد الجودة (ECAPA ±0.01) محفوظ.

رياضياً نفس حساب verify_files: torchaudio.load → resample إلى معدل ECAPA →
encode_batch → CosineSimilarity(eps=1e-8) → العتبة من hparams إن وجدت.
"""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.model_manager import model_manager
from app.services.voice.speaker_cache import speaker_state_cache

logger = logging.getLogger(__name__)


def _resample_torchaudio(wav: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
    """إعادة عينة بنفس مسار speechbrain (torchaudio.functional.resample الافتراضي)."""
    import torch
    import torchaudio.functional as ta_F

    t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    if orig_sr != target_sr:
        t = ta_F.resample(t, orig_sr, target_sr)
    return t


def _load_wav_bytes(data: bytes) -> tuple["np.ndarray", int]:
    """تحميل WAV من الذاكرة: torchaudio أولاً (مطابقة verify_files) ثم soundfile."""
    try:
        import torch  # noqa: F401
        import torchaudio

        wav, sr = torchaudio.load(io.BytesIO(data))
        return wav.numpy().squeeze(0).astype(np.float32), int(sr)
    except ImportError:
        pass
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


class SpeakerScorer:
    """نسّاب متحدث RAM-first مع كاش embedding مرجعي وسقوط آمن."""

    def __init__(self):
        self._ref_emb_fail = False

    # ─── embedding المرجع (مرة واحدة لكل متحدث) ───
    def reference_embedding(self, clean_wav_bytes: bytes):
        """يعيد tensor embedding المرجع (CPU) أو None عند الفشل."""
        if self._ref_emb_fail or model_manager.speaker_verification is None:
            return None
        try:
            if not getattr(settings, "embed_cache_enabled", True):
                emb_np = self._compute_embedding(clean_wav_bytes)
                if emb_np is None:
                    return None
                import torch

                return torch.from_numpy(emb_np)
            cached = speaker_state_cache.get_ecapa_embedding(clean_wav_bytes)
            if cached is not None:
                import torch

                return torch.from_numpy(np.asarray(cached["emb"], dtype=np.float32))
            emb_np = self._compute_embedding(clean_wav_bytes)
            if emb_np is None:
                return None
            speaker_state_cache.put_ecapa_embedding(clean_wav_bytes, emb_np)
            import torch

            return torch.from_numpy(emb_np)
        except Exception as exc:
            logger.debug("reference_embedding cache path failed: %s", exc)
            self._ref_emb_fail = True
            return None

    def _compute_embedding(self, wav_bytes: bytes) -> Optional["np.ndarray"]:
        """encode_batch على بايتات WAV — نفس خطوات verify_files الداخلية."""
        import torch

        try:
            wav, sr = _load_wav_bytes(wav_bytes)
            verifier = model_manager.speaker_verification
            target_sr = int(getattr(getattr(verifier, "hparams", None), "sample_rate", 16000))
            t = _resample_torchaudio(wav, sr, target_sr)
            with torch.inference_mode():
                emb = verifier.encode_batch(t, wav_lens=torch.ones(1))
            emb = emb.squeeze(0).squeeze(0).detach().cpu().float().numpy()
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            logger.debug("embedding compute failed (RAM path): %s", exc)
            return None

    # ─── التقييم الكامل من الذاكرة ───
    def score_ram(self, clean_wav_bytes: bytes, generated_wav_bytes: bytes) -> Optional[dict]:
        """score/same_speaker بالكامل من الذاكرة، أو None ليُسقط المتصل للمسار الأصلي."""
        ref_emb = self.reference_embedding(clean_wav_bytes)
        if ref_emb is None:
            return None
        gen_emb_np = self._compute_embedding(generated_wav_bytes)
        if gen_emb_np is None:
            return None
        try:
            import torch

            gen_emb = torch.from_numpy(gen_emb_np)
            score_t = torch.nn.functional.cosine_similarity(
                ref_emb.unsqueeze(0), gen_emb.unsqueeze(0), dim=-1, eps=1e-8
            )
            score = float(score_t.item())
            return {"score": score, "same_speaker": self._prediction(score)}
        except Exception as exc:
            logger.debug("cosine scoring failed: %s", exc)
            return None

    def _prediction(self, score: float) -> int:
        """نفس منطق speechbrain: عتبة hparams إن وجدت وإلا sigmoid>0.5."""
        try:
            thr = getattr(getattr(model_manager.speaker_verification, "hparams", None),
                          "similarity_threshold", None)
        except Exception:
            thr = None
        if thr is None:
            import math

            score = 1.0 / (1.0 + math.exp(-score))
            thr = 0.5
        return 1 if score > float(thr) else 0

    # ─── المسار الأصلي (السقوط الآمن — سلوك v1 حرفياً) ───
    def score_files(self, reference_path, generated_path) -> dict:
        score, prediction = model_manager.speaker_verification.verify_files(
            str(reference_path), str(generated_path)
        )
        return {
            "score": float(score.squeeze().cpu().item()),
            "same_speaker": int(prediction.squeeze().cpu().item()) == 1,
        }


def write_wav_atomic(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    """كتابة WAV بذرة ذرّية (نفس المكتبة، للسقوط الآمن فقط)."""
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.part")
    import soundfile as sf

    sf.write(str(tmp), audio, sr, subtype=subtype)
    tmp.replace(path)
