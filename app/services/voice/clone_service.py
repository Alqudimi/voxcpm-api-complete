import asyncio
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings
from app.core.model_manager import model_manager
from app.core.tts_engine import EnginePool, get_engine
from app.services.voice.speaker_cache import speaker_state_cache
from app.services.voice.speaker_score import SpeakerScorer
from app.services.voice.result_cache import result_cache
from app.utils.audio import (
    normalize_peak, prepare_reference, reference_quality_report,
    normalize_iraqi_target, convert_to_ogg,
    split_text_at_question_marks, concatenate_with_silence,
)
from app.utils.ram_pipeline import wav_bytes_from_array, encode_ogg_from_wav_bytes
from app.utils.stopwatch import start_timer, clear_timer

logger = logging.getLogger(__name__)

# بركة تنفيذ واحدة بحجم مضبوط تلقائياً (v3): تُبنى كسولاً بعد أن يكون
# AutoTuner قد عرف العتاد — العمال يغطون تصاريح كل الكروت + عمال CPU.
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=max(2, int(settings.executor_workers)),
                    thread_name_prefix="voxgpu",
                )
    return _executor


_scorer = SpeakerScorer()


class VoiceCloneService:
    """خدمة الاستنساخ — النسخة v3-max (أضعاف مضاعفة فوق v2-ultra):

    طبقات v1+v2 (محفوظة كاملة): كاش المتحدث bit-exact، توازي QC∥Whisper،
    tmpfs، ECAPA∥OGG، كاش النتائج، خط أنابيب صفراوي، كاش embedding،
    مقاطع متوازية ببذور ثابتة، سقوط آمن ملفي دائم.

    طبقات v3 الجديدة:
      S) مجدول GPU: كل توليد على الكرت **الأقل تحميلاً** مع تصريح تزامن
         لكل كرت — لا تدافع VRAM، وتناثر تلقائي على كل الكروت.
      C) مرشحو count>1 يتولّدون **بالتوازي** (بذور ثابتة — نفس نتيجة
         التسلسل حرفياً، ترتيب أعلى-نتيجة محفوظ).
      A) تقسيم تلقائي للنص الطويل فوق 600 حرفاً على فواصل طبيعية
         (VM_AUTO_CHUNK=off للتعطيل) — يُفعّل توازي المقاطع وحده.
      M) تخزين كاش النتائج من الذاكرة مباشرة (بلا قراءة قرص عكسية).
      T) كل الأحجام (عمال/كاشات/سقوف) محسوبة من AutoTuner.

    العقد الصامت المحفوظ: timesteps=30، cfg=2.0، نفس حقول الاستجابة حرفياً
    (+ حقل "performance" الإضافي غير المخرِب).
    """

    def _speaker_score(self, reference_path, generated_path):
        """المسار الأصلي الملفي — يُستخدم لسقوط الأمان وللمسار count>1."""
        return _scorer.score_files(reference_path, generated_path)

    def _engine_fingerprint(self, engine) -> str:
        """بصمة المحرك داخل مفتاح كاش النتائج (محرك/دقة/تكميم مختلف = مفتاح مختلف)."""
        return "|".join((
            getattr(engine, "name", "official"),
            getattr(model_manager, "precision_applied", "fp32"),
            getattr(model_manager, "quantized", "none"),
            getattr(model_manager, "compile_mode", "off") if getattr(model_manager, "compiled", False) else "-",
        ))

    def _generate_one(self, target_text, prompt_text, reference_path, cfg, timesteps,
                      seed, retry_badcase, retry_max_times, retry_ratio_threshold):
        """متوافق رجعياً: يفوّض إلى المحرك (نفس سلوك النسخة السابقة)."""
        engine = get_engine()
        wav = engine.generate_sync(
            target_text, prompt_text.strip(), reference_path,
            cfg, timesteps, seed, retry_badcase, retry_max_times, retry_ratio_threshold,
        )
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_peak(wav)

    def _launch_generate(self, engine, loop, chunk: str, seg_seed: int, kwargs_builder):
        """يطلق توليد مقطع واحد على المحرك الأقل تحميلاً داخل تصريح المجدول."""
        target = engine.pick() if isinstance(engine, EnginePool) else engine
        kwargs = kwargs_builder(chunk, seg_seed)

        def _work():
            try:
                from app.core.gpu_scheduler import gpu_scheduler

                if gpu_scheduler.configured():
                    with gpu_scheduler.slot(getattr(target, "device_index", 0)):
                        return target.generate_sync(**kwargs)
            except Exception:
                pass
            return target.generate_sync(**kwargs)

        return loop.run_in_executor(_get_executor(), _work)

    # ─── توليد المقاطع: توازي آمن بذور ثابتة + تناثر على الكروت (v3) ───
    async def _generate_segments(self, engine, loop, chunks, prompt_text, clean,
                                 options, base_seed):
        """يعيد قائمة المقاطع بترتيبها الأصلي دائماً (التوازي لا يغيّر النتيجة).

        شرط التوازي: seed أصلي في generate() (كل خيط يضمن بذرته بنفسه) —
        وإلا تسلسل حرفياً كما في v1 لضمان الحتمية.
        v3: كل مهمة تُرسل للكرت الأقل تحميلاً عبر المجدول (لا دوران أعمى)،
        مع سقف توازي مضبوط تلقائياً (chunk_parallel_max).
        """
        cfg = options.get("cfg_value", settings.DEFAULT_CFG)
        timesteps = options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS)
        retry_badcase = options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE)
        retry_max_times = options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES)
        retry_ratio_threshold = options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD)

        def _kwargs(chunk: str, seg_seed: int) -> dict:
            return dict(
                target_text=chunk,
                prompt_text=prompt_text.strip(),
                reference_path=clean,
                cfg_value=cfg,
                inference_timesteps=timesteps,
                seed=seg_seed,
                retry_badcase=retry_badcase,
                retry_max_times=retry_max_times,
                retry_ratio_threshold=retry_ratio_threshold,
            )

        def _post(wav) -> np.ndarray:
            wav = np.asarray(wav, dtype=np.float32)
            wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            return normalize_peak(wav)

        can_parallel = (
            len(chunks) > 1
            and settings.parallel_chunks_enabled
            and bool(getattr(engine, "supports_native_seed", False))
        )

        if getattr(engine, "is_async", False):
            # محرك غير متزامن (NanoVLLM): يدير تزامنه داخلياً
            if can_parallel:
                tasks = [
                    engine.generate_async(**_kwargs(chunk, base_seed + seg_index))
                    for seg_index, chunk in enumerate(chunks)
                ]
                wavs = await asyncio.gather(*tasks)
            else:
                wavs = []
                for seg_index, chunk in enumerate(chunks):
                    wavs.append(await engine.generate_async(**_kwargs(chunk, base_seed + seg_index)))
            return [_post(w) for w in wavs]

        if can_parallel:
            limit = max(1, int(settings.chunk_parallel_max or 4))
            sem = asyncio.Semaphore(limit)

            async def _bounded(c, s):
                async with sem:
                    return await self._launch_generate(engine, loop, c, s, _kwargs)

            wavs = await asyncio.gather(*[_bounded(c, base_seed + i) for i, c in enumerate(chunks)])
        else:
            wavs = []
            for seg_index, chunk in enumerate(chunks):
                wavs.append(await self._launch_generate(engine, loop, chunk, base_seed + seg_index, _kwargs))
        return [_post(w) for w in wavs]

    # ─── الإنهاء الصفراوي: تقييم ∥ ترميز في RAM ───
    async def _finalize_ram(self, loop, wav, model_sr, clean_bytes, bitrate):
        """يعيد (ogg_bytes, similarity) من الذاكرة بالكامل، أو None ليُسقط للسقوط الآمن.

        التقييم (embedding المولَّد فقط — المرجع مخزّن) يتوازى مع ترميز OGG،
        نفس تداخل v1 لكن بلا أي ملف وسيط.
        """
        try:
            wav_bytes = wav_bytes_from_array(wav, model_sr, "PCM_24")
            encode_task = loop.run_in_executor(
                _get_executor(),
                lambda: encode_ogg_from_wav_bytes(
                    wav_bytes, bitrate, settings.OUTPUT_SAMPLE_RATE, settings.opus_fastenc
                ),
            )
            score_task = loop.run_in_executor(
                _get_executor(), lambda: _scorer.score_ram(clean_bytes, wav_bytes)
            )
            ogg_bytes, similarity = await asyncio.gather(encode_task, score_task)
            if not ogg_bytes:
                return None
            if similarity is None:
                return None  # سقوط آمن: نريد نتيجة تقييم حقيقية متطابقة مع العقد
            return ogg_bytes, similarity
        except Exception as exc:
            logger.warning("RAM pipeline finalize failed (safe fallback to files): %s", exc)
            return None

    async def clone(self, reference_audio, prompt_text, target_text, **options):
        if not model_manager.ready:
            raise RuntimeError("AI models are not ready.")

        loop = asyncio.get_running_loop()
        timer = start_timer(uuid.uuid4().hex[:8])
        raw = clean = None
        candidate_paths = []
        cache_key = None
        try:
            tmp_dir = settings.resolve_temp_dir()

            # ── المرحلة أ: استلام الطلب (قراءة واحدة لبايتات المرجع) ──
            with timer.stage("receive"):
                ref_bytes = await reference_audio.read()
                raw = tmp_dir / f"{uuid.uuid4().hex}_raw"
                clean = tmp_dir / f"{uuid.uuid4().hex}_reference.wav"
                raw.write_bytes(ref_bytes)

            # ── المرحلة ب: تحضير المرجع — مع كاش bit-exact ──
            with timer.stage("prepare_reference"):
                prep = speaker_state_cache.get_prepare(
                    ref_bytes, settings.REFERENCE_SAMPLE_RATE, 35
                )
                if prep is not None:
                    clean.write_bytes(prep["clean_wav_bytes"])
                    prep_info = {
                        "path": str(clean),
                        "sample_rate": settings.REFERENCE_SAMPLE_RATE,
                        "duration": float(prep["duration"]),
                        "samples": int(prep["duration"] * settings.REFERENCE_SAMPLE_RATE),
                    }
                    timer.note("speaker_cache", "HIT")
                else:
                    prep_info = await loop.run_in_executor(
                        _get_executor(),
                        lambda: prepare_reference(raw, clean, settings.REFERENCE_SAMPLE_RATE),
                    )
                    speaker_state_cache.put_prepare(
                        ref_bytes, settings.REFERENCE_SAMPLE_RATE, 35,
                        clean.read_bytes(), prep_info["duration"],
                    )
                    timer.note("speaker_cache", "MISS")

            # ── المرحلة ج: تقرير الجودة + نسخ المرجع — بالتوازي ──
            clean_bytes = clean.read_bytes()

            async def _qc_stage():
                with timer.stage("reference_qc"):
                    cached = speaker_state_cache.get_qc(clean_bytes)
                    if cached is not None:
                        return cached
                    qc_text = await loop.run_in_executor(
                        _get_executor(), lambda: reference_quality_report(clean)
                    )
                    speaker_state_cache.put_qc(clean_bytes, qc_text)
                    return qc_text

            async def _prompt_stage():
                if not options.get("auto_transcribe"):
                    return prompt_text
                with timer.stage("whisper"):
                    cached = speaker_state_cache.get_transcript(
                        clean_bytes,
                        getattr(model_manager, "whisper_backend", "openai-whisper"),
                        settings.whisper_vad,
                    )
                    if cached is not None:
                        timer.note("whisper_cache", "HIT")
                        return cached
                    from app.services.transcription.transcription_service import TranscriptionService
                    text = await TranscriptionService().transcribe_file(clean)
                    speaker_state_cache.put_transcript(
                        clean_bytes,
                        getattr(model_manager, "whisper_backend", "openai-whisper"),
                        settings.whisper_vad,
                        text,
                    )
                    timer.note("whisper_cache", "MISS")
                    return text

            if settings.parallel_prep:
                qc, prompt_text = await asyncio.gather(_qc_stage(), _prompt_stage())
            else:
                qc = await _qc_stage()
                prompt_text = await _prompt_stage()

            if not prompt_text or not prompt_text.strip():
                raise ValueError("prompt_text is required unless auto_transcribe=true")
            if not target_text or not target_text.strip():
                raise ValueError("target_text is required")

            target_text = normalize_iraqi_target(target_text)
            base_seed = int(options.get("seed") if options.get("seed") is not None else 12345)
            count = 1

            engine = get_engine()
            model_sr = int(engine.sample_rate)
            chunk_long_text = bool(options.get("chunk_long_text", False))
            silence_ms = int(options.get("silence_ms", 350))
            bitrate = settings.OUTPUT_BITRATE

            # ── التقسيم: يدوي كما كان، أو تلقائي فوق عتبة مضبوطة (v3) ──
            if chunk_long_text:
                chunks = split_text_at_question_marks(target_text)
                if not chunks:
                    chunks = [target_text]
            else:
                auto_chars = int(settings.auto_chunk_chars or 0)
                if auto_chars and len(target_text) > auto_chars:
                    chunks = split_text_at_question_marks(target_text)
                    if len(chunks) <= 1:
                        chunks = [target_text]
                    else:
                        logger.info(
                            "auto-chunk (v3): %d chars > %d threshold → %d segments",
                            len(target_text), auto_chars, len(chunks),
                        )
                else:
                    chunks = [target_text]

            logger.info("Text chunks (%d): %s", len(chunks), chunks)

            # ── المرحلة R: كاش النتائج الكاملة — رد فوري على التكرار ──
            cache_key = result_cache.build_key(
                target_text, prompt_text.strip(), clean_bytes,
                options.get("cfg_value", settings.DEFAULT_CFG),
                options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS),
                base_seed,
                options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE),
                options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES),
                options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD),
                silence_ms, bitrate, settings.OUTPUT_SAMPLE_RATE,
                chunks_count=len(chunks),
                engine_fingerprint=self._engine_fingerprint(engine),
            )
            cached = result_cache.get(cache_key)
            if cached is not None and count == 1:
                with timer.stage("cache_writeback"):
                    output = Path(settings.OUTPUT_DIR) / f"{uuid.uuid4().hex}.ogg"
                    await loop.run_in_executor(_get_executor(), lambda: output.write_bytes(cached["ogg"]))
                meta = cached["meta"]
                timer.note("result_cache", "HIT")
                timer.finish_and_log()
                return {
                    "request_id": output.stem,
                    "output": output,
                    "prompt_text": prompt_text.strip(),
                    "target_text": target_text,
                    "seed": int(meta.get("seed", base_seed)),
                    "duration": float(meta.get("duration", 0.0)),
                    "speaker_score": float(meta.get("score", 0.0)),
                    "same_speaker": int(meta.get("same_speaker", 0)),
                    "reference_qc": qc,
                    "chunks_count": int(meta.get("chunks_count", len(chunks))),
                    "ranking": list(meta.get("ranking", [])),
                    "performance": {"result_cache": "hit", "engine": getattr(engine, "name", "official")},
                }

            # ── المرحلة د: التوليد (عقد الجودة: نفس المعاملات حرفياً) ──
            ranked = []
            final_ogg_bytes = None
            with timer.stage("generate"):
                can_parallel_candidates = (
                    count > 1
                    and settings.parallel_chunks_enabled
                    and bool(getattr(engine, "supports_native_seed", False))
                )
                if can_parallel_candidates:
                    # v3: كل المرشحين يتولّدون بالتوازي — بذور ثابتة لكل مرشح
                    # (نفس نتيجة التسلسل حرفياً؛ الترتيب لا يؤثر على الحتمية)
                    cand_seeds = [base_seed + index * 1009 for index in range(count)]
                    cand_segments = await asyncio.gather(*[
                        self._generate_segments(engine, loop, chunks, prompt_text, clean, options, s)
                        for s in cand_seeds
                    ])
                    for candidate_seed, segment_wavs in zip(cand_seeds, cand_segments):
                        wav = concatenate_with_silence(segment_wavs, model_sr, silence_ms=silence_ms)
                        wav = normalize_peak(wav)
                        candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
                        candidate_paths.append(candidate_wav)
                        await loop.run_in_executor(
                            _get_executor(),
                            lambda w=wav, p=candidate_wav: sf.write(str(p), w, model_sr, subtype="PCM_24"),
                        )
                        similarity = self._speaker_score(clean, candidate_wav)
                        ranked.append({
                            "path": candidate_wav,
                            "wav": wav,
                            "score": similarity["score"],
                            "same_speaker": similarity["same_speaker"],
                            "seed": candidate_seed,
                            "duration": len(wav) / model_sr,
                        })
                else:
                    for index in range(count):
                        candidate_seed = base_seed + index * 1009

                        segment_wavs = await self._generate_segments(
                            engine, loop, chunks, prompt_text, clean, options, candidate_seed
                        )

                        wav = concatenate_with_silence(segment_wavs, model_sr, silence_ms=silence_ms)
                        wav = normalize_peak(wav)
                        if count == 1:
                            # المسار الشائع: التقييم يُؤجَّل ويعمل متوازياً مع الترميز (في RAM)
                            ranked = [{
                                "wav": wav,
                                "score": 0.0,
                                "same_speaker": 0,
                                "seed": candidate_seed,
                                "duration": len(wav) / model_sr,
                            }]
                        else:
                            # المسار النادر (count>1): سلوك v1 الملفي حرفياً
                            candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
                            candidate_paths.append(candidate_wav)
                            await loop.run_in_executor(
                                _get_executor(),
                                lambda w=wav, p=candidate_wav: sf.write(str(p), w, model_sr, subtype="PCM_24"),
                            )
                            similarity = self._speaker_score(clean, candidate_wav)
                            ranked.append({
                                "path": candidate_wav,
                                "wav": wav,
                                "score": similarity["score"],
                                "same_speaker": similarity["same_speaker"],
                                "seed": candidate_seed,
                                "duration": len(wav) / model_sr,
                            })

            best = ranked[0] if count == 1 else max(ranked, key=lambda x: x["score"])
            output = Path(settings.OUTPUT_DIR) / f"{uuid.uuid4().hex}.ogg"

            # ── المرحلة هـ: التقييم ∥ الترميز ──
            #    v2 (count=1): كلاهما في RAM عبر pipes، التقييم بنصف التكلفة
            #    (embedding المرجع مخزّن). أي فشل → السقوط الملفي v1 حرفياً.
            similarity = None
            with timer.stage("finalize"):
                if count == 1 and settings.ram_pipeline_enabled:
                    finalized = await self._finalize_ram(loop, best["wav"], model_sr, clean_bytes, bitrate)
                    if finalized is not None:
                        ogg_bytes, similarity = finalized
                        final_ogg_bytes = ogg_bytes  # v3: يدخل الكاش من RAM بلا قراءة عكسية
                        await loop.run_in_executor(_get_executor(), lambda: output.write_bytes(ogg_bytes))
                        timer.note("pipeline", "ram")
                    else:
                        # السقوط الآمن: كتابة WAV وسيط ومسار الملفات الكامل
                        candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
                        candidate_paths.append(candidate_wav)
                        await loop.run_in_executor(
                            _get_executor(),
                            lambda w=best["wav"], p=candidate_wav: sf.write(str(p), w, model_sr, subtype="PCM_24"),
                        )
                        if settings.ecapa_async:
                            score_task = loop.run_in_executor(
                                _get_executor(), lambda: self._speaker_score(clean, candidate_wav)
                            )
                            encode_task = loop.run_in_executor(
                                _get_executor(),
                                lambda: convert_to_ogg(
                                    candidate_wav, output, bitrate, settings.OUTPUT_SAMPLE_RATE
                                ),
                            )
                            similarity, _ = await asyncio.gather(score_task, encode_task)
                        else:
                            similarity = self._speaker_score(clean, candidate_wav)
                            await loop.run_in_executor(
                                _get_executor(),
                                lambda: convert_to_ogg(
                                    candidate_wav, output, bitrate, settings.OUTPUT_SAMPLE_RATE
                                ),
                            )
                        timer.note("pipeline", "file_fallback")
                else:
                    # count>1 أو pipeline معطّل: مسار v1 الملفي حرفياً
                    if count == 1:
                        candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
                        candidate_paths.append(candidate_wav)
                        await loop.run_in_executor(
                            _get_executor(),
                            lambda w=best.get("wav"), p=candidate_wav, sr=model_sr: sf.write(str(p), w, sr, subtype="PCM_24"),
                        )
                    else:
                        candidate_wav = best["path"]  # كُتبت أثناء الترتيب (سلوك v1)
                    if settings.ecapa_async and count == 1:
                        score_task = loop.run_in_executor(
                            _get_executor(), lambda: self._speaker_score(clean, candidate_wav)
                        )
                        encode_task = loop.run_in_executor(
                            _get_executor(),
                            lambda: convert_to_ogg(
                                candidate_wav, output, bitrate, settings.OUTPUT_SAMPLE_RATE
                            ),
                        )
                        similarity, _ = await asyncio.gather(score_task, encode_task)
                    else:
                        if count == 1:
                            similarity = self._speaker_score(clean, candidate_wav)
                        else:
                            similarity = {"score": best["score"], "same_speaker": best["same_speaker"]}
                        await loop.run_in_executor(
                            _get_executor(),
                            lambda: convert_to_ogg(
                                candidate_wav, output, bitrate, settings.OUTPUT_SAMPLE_RATE
                            ),
                        )

            best["score"] = similarity["score"]
            best["same_speaker"] = int(similarity["same_speaker"])

            timer.finish_and_log()

            # ── المرحلة R+: تخزين النتيجة الكاملة في الكاش ──
            ranking = [
                {
                    "rank": i + 1,
                    "seed": x["seed"],
                    "score": x["score"],
                    "same_speaker": x["same_speaker"],
                    "duration": x["duration"],
                }
                for i, x in enumerate(sorted(ranked, key=lambda x: x["score"], reverse=True))
            ]
            if count == 1:
                # v3: من الذاكرة مباشرة عند توفرها (توفير قراءة قرص كاملة)
                if final_ogg_bytes:
                    result_cache.put(cache_key, final_ogg_bytes, {
                        "score": best["score"],
                        "same_speaker": best["same_speaker"],
                        "seed": best["seed"],
                        "duration": best["duration"],
                        "chunks_count": len(chunks),
                        "ranking": ranking,
                    })
                else:
                    await loop.run_in_executor(
                        _get_executor(),
                        lambda: result_cache.put(cache_key, output.read_bytes(), {
                            "score": best["score"],
                            "same_speaker": best["same_speaker"],
                            "seed": best["seed"],
                            "duration": best["duration"],
                            "chunks_count": len(chunks),
                            "ranking": ranking,
                        }),
                    )
                timer.note("result_cache", "STORED")

            return {
                "request_id": output.stem,
                "output": output,
                "prompt_text": prompt_text.strip(),
                "target_text": target_text,
                "seed": best["seed"],
                "duration": best["duration"],
                "speaker_score": best["score"],
                "same_speaker": best["same_speaker"],
                "reference_qc": qc,
                "chunks_count": len(chunks),
                "ranking": ranking,
                "performance": {
                    "result_cache": "stored",
                    "engine": getattr(engine, "name", "official"),
                    "quantization": getattr(model_manager, "quantized", "none"),
                },
            }
        finally:
            clear_timer()
            # تنظيف الملفات الوسيطة (tmpfs = ذاكرة — التنظيف أهم هنا)
            for path in [raw, clean, *candidate_paths]:
                try:
                    if path is not None:
                        Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
