import logging
import os
import uuid
from pathlib import Path

import numpy as np

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ModelManager:
    def __init__(self):
        self.voxcpm = None
        self.whisper = None
        self.whisper_backend = "openai-whisper"
        self.whisper_batched = None
        self.speaker_verification = None
        self.ready = False
        self.precision_applied = "fp32"
        self.compiled = False
        self.compile_mode = "off"
        self.quantized = "none"
        self.whisper_model_used = ""
        self._device_instances: list = []  # [(device_idx, model)] — بركة multi-GPU (v2)

    # ─────────────────────────────────────────────────────────────
    # تحميل النماذج — مع طبقة الأداء: dtype/TF32/cudnn/compile/إحماء
    # ─────────────────────────────────────────────────────────────
    async def load_all(self):
        if self.ready:
            return

        settings.ensure_directories()

        # v3-max: المضبوط التلقائي أولاً — يفحص العتاد ويحل/يعيّن كل المتغيرات
        # (أو يحمّل البروفايل المحفوظ) قبل أي قرار تحميل.
        try:
            from app.core.autotune import tuner

            tuner.ensure()
        except Exception as exc:
            logger.warning("AutoTuner unavailable (legacy defaults used): %s", exc)

        cuda_available = torch.cuda.is_available()

        self._apply_backend_flags(cuda_available)

        logger.info("Loading VoxCPM2: %s", settings.VOXCPM_MODEL_ID)
        from voxcpm import VoxCPM

        self.voxcpm = VoxCPM.from_pretrained(
            settings.VOXCPM_MODEL_ID,
            load_denoiser=False,
        )
        logger.info("VoxCPM2 loaded")

        self._apply_precision()
        self._try_quantize()
        self._load_device_pool()

        logger.info("Loading Whisper (backend=%s): %s", self._resolve_whisper_backend(), settings.WHISPER_MODEL_NAME)
        self._load_whisper(cuda_available)
        logger.info("Whisper loaded via %s", self.whisper_backend)

        device = settings.ecapa_device if (cuda_available and settings.ecapa_device == "cuda") else "cpu"
        logger.info("Loading ECAPA speaker verification on %s", device)
        from speechbrain.inference.speaker import SpeakerRecognition

        self.speaker_verification = SpeakerRecognition.from_hparams(
            source=settings.ECAPA_SOURCE,
            savedir=settings.ECAPA_SAVEDIR,
            run_opts={"device": device},
        )
        logger.info("ECAPA loaded")

        self.ready = True
        logger.info(
            "All AI models are ready (precision=%s, compile=%s, quant=%s, whisper=%s/%s)",
            self.precision_applied, self.compile_mode, self.quantized,
            self.whisper_backend, self.whisper_model_used,
        )

    # ── أعلام الخلفية (TF32 / cudnn.benchmark) — آمنة على أي بيئة ──
    def _apply_backend_flags(self, cuda_available: bool):
        if not cuda_available:
            return
        try:
            backends = getattr(torch, "backends", None)
            if backends is None:
                return
            if settings.tf32_enabled:
                matmul = getattr(backends.cuda, "matmul", None) if hasattr(backends, "cuda") else None
                cudnn = getattr(backends, "cudnn", None)
                if matmul is not None and hasattr(matmul, "allow_tf32"):
                    matmul.allow_tf32 = True
                if cudnn is not None:
                    if hasattr(cudnn, "allow_tf32"):
                        cudnn.allow_tf32 = True
                    if settings.cudnn_benchmark and hasattr(cudnn, "benchmark"):
                        cudnn.benchmark = True
            # v2: اختيار خوارزمية matmul عالية الأداء (مكسب إضافي على Ampere+)
            try:
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            logger.info("TF32/cudnn.benchmark/matmul-precision enabled (Ampere+ benefits most)")
        except Exception as exc:  # أي فشل هنا ليس قاتلاً — نكمل بالإعدادات الافتراضية
            logger.warning("Backend flags not applied: %s", exc)

    # ── دقة الحساب: المضبوط يحل (bf16 على Ampere+ / fp16 على Turing) — VM_PRECISION صريح يتفوق ──
    def _apply_precision(self):
        try:
            from app.core.autotune import tuner

            mode = tuner.resolve_str(settings.VM_PRECISION, "dtype", "fp32").lower()
        except Exception:
            mode = (settings.VM_PRECISION or "auto").strip().lower()
            if mode in {"", "auto"}:
                mode = "bf16" if torch.cuda.is_available() else "fp32"
        if not torch.cuda.is_available() or mode in {"", "off", "none", "fp32"}:
            self.precision_applied = "fp32"
            return
        try:
            dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            tts_model = getattr(self.voxcpm, "tts_model", None)
            if tts_model is not None:
                tts_model.to(dtype)
                self.precision_applied = mode
                logger.info("Model precision applied: %s (auto-tuned unless pinned)", mode)
            else:
                logger.info("tts_model attribute not found; keeping fp32 weights")
        except Exception as exc:
            logger.warning("Precision %s not applied (keeping fp32): %s", mode, exc)
            self.precision_applied = "fp32"

    # ── torch.compile اختياري — v3: الوضع يحلّه المضبوط (reduce-overhead = CUDA Graphs) ──
    def _try_compile(self):
        mode = (settings.compile_mode or "off").strip().lower()
        if mode in {"", "off", "none", "no", "false", "0"} or not torch.cuda.is_available():
            self.compile_mode = "off"
            return
        if mode == "on":
            mode = "reduce-overhead"  # القيمة التاريخية "on" = وضع CUDA Graphs
        if mode == "max-autotune":
            mode = "max-autotune-no-cudagraphs"  # الاسم الصحيح في torch ≥ 2.x
        try:
            tts_model = getattr(self.voxcpm, "tts_model", None)
            if tts_model is None:
                logger.info("torch.compile skipped: tts_model not exposed")
                return
            self.voxcpm.tts_model = torch.compile(tts_model, mode=mode, dynamic=False)
            self.compiled = True
            self.compile_mode = mode
            logger.info("torch.compile(mode=%s) applied to tts_model", mode)
        except Exception as exc:
            self.compiled = False
            self.compile_mode = "off"
            logger.warning("torch.compile failed (continuing uncompiled): %s", exc)

    # ── تكميم أوزان weight-only — v3: الوضع يحلّه المضبوط تلقائياً ──
    # fp8 (Ada+): ~1.3-1.6× | int8 (≥7.5): ~1.2-1.4×. AudioVAE مستثنى دائماً — العقد الصلب.
    def _try_quantize(self):
        try:
            from app.core.autotune import tuner

            mode = tuner.resolve_str(settings.VM_QUANT, "quant", "off").strip().lower()
        except Exception:
            mode = (settings.VM_QUANT or "off").strip().lower()
        if mode in {"", "on", "true", "yes", "1"}:
            mode = "int8"  # القيمة التاريخية "on" = الأكثر توافراً
        if mode in {"", "off", "none", "no", "0", "false"} or not torch.cuda.is_available():
            return
        try:
            from torchao.quantization import (  # توفر اختياري — سقوط آمن
                float8_weight_only,
                int8_weight_only,
                quantize_,
            )

            tts_model = getattr(self.voxcpm, "tts_model", None)
            if tts_model is None:
                logger.info("VM_QUANT skipped: tts_model not exposed")
                return

            # استثناء كل وحدات VAE/فك التشفير الصوتي من التكميم (العقد الصلب: AudioVAE bf16/fp16 فقط)
            def _skip_vae(mod, name):
                n = name.lower()
                return any(tag in n for tag in ("vae", "audiovae", "decoder", "vocoder"))

            spec = float8_weight_only() if ("fp8" in mode or "float8" in mode) else int8_weight_only()
            quantize_(tts_model, spec, filter_fn=_skip_vae)
            self.quantized = mode
            logger.info("Quantization applied (%s, weight-only, VAE excluded) — قيد بوابة C", mode)
        except Exception as exc:
            self.quantized = "none"
            logger.warning("Quantization failed (continuing unquantized): %s", exc)

    # ── بركة multi-GPU (v2/v3): auto = كل الكروت عبر المضبوط ──
    def _load_device_pool(self):
        from app.core.autotune import tuner

        gpu_ids = tuner.resolve_gpu_ids(getattr(settings, "VM_GPU_IDS", "auto"))
        if len(gpu_ids) <= 1:
            return
        self.load_device_instances(gpu_ids[1:])

    def load_device_instances(self, device_ids) -> list:
        """يحمّل نسخة VoxCPM إضافية على كل جهاز مطلوب (تُستخدم من EnginePool).
        يعيد قائمة [(device_idx, model)] للنسخ الجديدة فقط. فشل أي جهاز غير قاتل."""
        loaded = []
        for dev in device_ids:
            try:
                if any(idx == dev for idx, _ in self._device_instances):
                    continue
                with torch.cuda.device(dev):
                    from voxcpm import VoxCPM

                    instance = VoxCPM.from_pretrained(
                        settings.VOXCPM_MODEL_ID,
                        load_denoiser=False,
                    )
                    self._device_instances.append((int(dev), instance))
                    loaded.append((int(dev), instance))
                    logger.info("VoxCPM2 instance loaded on cuda:%d (pool)", dev)
            except Exception as exc:
                logger.warning("Device cuda:%s pool load failed (skipped): %s", dev, exc)
        return loaded

    # ── اختيار محرك Whisper ──
    def _resolve_whisper_backend(self) -> str:
        requested = (settings.VM_WHISPER_BACKEND or "auto").strip().lower()
        if requested in {"auto", "fw-int8", "fw-fp16", "faster-whisper"}:
            try:
                import faster_whisper  # noqa: F401
                return "faster-whisper"
            except Exception:
                if requested == "auto":
                    return "openai-whisper"
                logger.warning("faster-whisper requested but not installed; falling back to openai-whisper")
        return "openai-whisper"

    def _load_whisper(self, cuda_available: bool):
        backend = self._resolve_whisper_backend()
        # v3: النموذج/الدقة يحلّهما المضبوط حسب VRAM — القيم الصريحة تتفوق.
        try:
            from app.core.autotune import tuner

            model_name = tuner.resolve_whisper_model(settings.VM_WHISPER_MODEL, settings.WHISPER_MODEL_NAME)
            compute = tuner.resolve_str("auto", "whisper_compute", "int8")
        except Exception:
            model_name = settings.WHISPER_MODEL_NAME
            compute = "int8"
        if (settings.VM_WHISPER_BACKEND or "").strip().lower() == "fw-fp16" and cuda_available:
            compute = "float16"  # تعريف صريح يتقدم
        if not cuda_available:
            compute = "int8"  # CPU: int8 هو الأسرع دائماً
        self.whisper_model_used = model_name
        if backend == "faster-whisper":
            from faster_whisper import WhisperModel

            self.whisper = WhisperModel(
                model_name,
                device="cuda" if cuda_available else "cpu",
                compute_type=compute,
            )
            self.whisper_backend = "faster-whisper"
            if settings.whisper_batched:
                try:
                    from faster_whisper import BatchedInferencePipeline

                    self.whisper_batched = BatchedInferencePipeline(model=self.whisper)
                    logger.info(
                        "faster-whisper BatchedInferencePipeline ready (batch=%s, ~3.7x على دفعات)",
                        settings.whisper_batch_size,
                    )
                except Exception as exc:
                    logger.warning("Batched whisper pipeline unavailable: %s", exc)
        else:
            import whisper

            self.whisper = whisper.load_model(model_name)
            self.whisper_backend = "openai-whisper"

    # ─────────────────────────────────────────────────────────────
    # إحماء عند الإقلاع: تمريرة توليد حقيقية قصيرة تسخّن النوى
    # (cuBLAS/Triton/compile) قبل أن تفتح الخدمة مقبس الاستقبال.
    # تحترم عقد الجودة: نفس timesteps=30 وcfg=2.0.
    # ─────────────────────────────────────────────────────────────
    async def warmup(self):
        """الإحماء + المعايرة (v3): كل تمريرة تُقاس (زمن/ثواني صوت = RTF) وتُحفظ
        في بروفايل المضبوط — كلفة الإقلاع تشتري رقماً حقيقياً بدل التخمين."""
        if not self.ready or self.voxcpm is None:
            return
        if not torch.cuda.is_available():
            return
        import time as _time

        compile_passes = max(1, int(settings.compile_warmup_passes or 1))
        tmp_dir = settings.resolve_temp_dir()
        wav_path = Path(tmp_dir) / f"warmup_{uuid.uuid4().hex}.wav"
        rtfs = []
        try:
            import soundfile as sf

            sr = int(getattr(self.voxcpm.tts_model, "sample_rate", settings.OUTPUT_SAMPLE_RATE))
            rng = np.random.default_rng(7)
            tone = (0.15 * np.sin(2 * np.pi * 220.0 * np.arange(sr) / sr)).astype(np.float32)
            sf.write(str(wav_path), tone, sr, subtype="PCM_16")

            def _measured_pass(instance, label: str):
                nonlocal _tot_gen, _tot_audio
                t0 = _time.perf_counter()
                wav = instance.generate(
                    text="مرحبا.",
                    prompt_wav_path=str(wav_path),
                    prompt_text="مرحبا.",
                    cfg_value=settings.VOXCPM_CFG,
                    inference_timesteps=settings.VOXCPM_TIMESTEPS,
                    retry_badcase=False,
                )
                dt = _time.perf_counter() - t0
                audio_s = 0.0
                try:
                    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
                    audio_s = float(arr.size) / float(sr) if arr.size else 0.0
                except Exception:
                    pass
                rtf = (dt / audio_s) if audio_s > 0 else None
                rtfs.append(rtf)
                _tot_gen += dt
                _tot_audio += audio_s
                extra = f", RTF={rtf:.3f}" if rtf else ""
                logger.info("Warmup [%s] %.2fs (audio=%.2fs%s)", label, dt, audio_s, extra)
                return rtf

            _tot_gen = 0.0
            _tot_audio = 0.0

            # تمريرة أساس باردة (تسخّن النواة وتقيس الخط الأساس)
            _measured_pass(self.voxcpm, "base")
            # التجميع يُفعَّل بعد الإحماء الأولي (بلا مطالب من أول خطوة)
            if not self.compiled:
                self._try_compile()
                if self.compiled:
                    for i in range(compile_passes):
                        _measured_pass(self.voxcpm, f"compile {i + 1}/{compile_passes}")
                    logger.info("Compiled warmup complete (mode=%s)", self.compile_mode)
            # v2/v3: إحماء نسخ البركة على الأجهزة الإضافية (تمريرة مقاسة لكل نسخة)
            for dev_idx, instance in getattr(self, "_device_instances", []):
                try:
                    with torch.cuda.device(dev_idx):
                        _measured_pass(instance, f"pool cuda:{dev_idx}")
                    logger.info("Pool engine warm on cuda:%d", dev_idx)
                except Exception as exc:
                    logger.warning("Pool warmup skipped for cuda:%s: %s", dev_idx, exc)

            # المعايرة: أفضل RTF مقاس + المجاميع الفعلية تُحفظ في بروفايل المضبوط
            if settings.autotune_calibrate:
                valid = [r for r in rtfs if r]
                if valid and _tot_audio > 0:
                    try:
                        from app.core.autotune import tuner

                        tuner.calibrate(
                            rtf=min(valid),
                            gen_seconds=_tot_gen,
                            audio_seconds=_tot_audio,
                            engine="official-compiled" if self.compiled else "official",
                        )
                    except Exception as exc:
                        logger.debug("calibration persist failed: %s", exc)
        except Exception as exc:
            logger.warning("Warmup skipped/failed (non-fatal): %s", exc)
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def unload_all(self):
        self.voxcpm = None
        self.whisper = None
        self.whisper_batched = None
        self.speaker_verification = None
        self._device_instances = []
        self.ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()
