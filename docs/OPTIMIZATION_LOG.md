# سجل التنفيذ — خطة تسريع `clone_voice_and_mix` (v1.0)

> هذا السجل يوثق ما نُفِّذ فعلياً في الكود، وأثره المتوقع، وكيفية التحقق، وكيفية التراجع.
> العقد الصامت المحفوظ في كل بند: `inference_timesteps=30` و`cfg_value=2.0` وتوافق API 100%.

## خريطة البنود المنفذة

| # | البند | الملفات | المفتاح | الأثر المتوقع | التحقق | التراجع |
|---|---|---|---|---|---|---|
| 1 | قراءة المرجع مرة واحدة + tmpfs تلقائي | `clone_service.py`, `config.py` | `VM_TMPFS_DIR` | 0.1–0.4s | سطر STAGES: مرحلة واحدة `prepare_reference` | الافتراضي `auto` يتراجع ذاتياً بلا /dev/shm |
| 2 | soxr_hq بدل kaiser_best | `utils/audio.py`, `requirements.txt` | `VM_RESAMPLER` | 0.05–0.3s | زمن المرحلة في STAGES | `VM_RESAMPLER=kaiser_best` |
| 3 | تقرير الجودة بلا إعادة تحميل + كاش QC | `utils/audio.py`, `speaker_cache.py` | `VM_SPEAKER_CACHE` | 0.1–0.3s (تكرار: 0) | `speaker_cache=HIT` في السجل | `VM_SPEAKER_CACHE=off` |
| 4 | فك حجب event loop كاملاً | `clone_service.py`, `voice.py`, `transcription_service.py` | — | استجابة الخدمة تحت الحمل | طلبان متزامنان لا يحجبان بعضهما | داخلي — لا مفتاح |
| 5 | توازي QC ∥ Whisper | `clone_service.py` | `VM_PARALLEL_PREP` | 0.2–0.8s في فرع auto_transcribe | STAGES: `whisper` ≤ `reference_qc` | `VM_PARALLEL_PREP=off` |
| 6 | ECAPA ∥ ترميز OGG (الحالة الشائعة) | `clone_service.py` | `VM_ECAPA_ASYNC` | 0.15–0.5s | STAGES: `finalize` ≈ max(الاثنان) | `VM_ECAPA_ASYNC=off` |
| 7 | كاش التحضير + كاش نص Whisper (bit-exact) | `speaker_cache.py` | `VM_SPEAKER_CACHE` | 0.3–1.2s للطلبات المتكررة | طلب ثانٍ بنفس المرجع = لا Whisper ولا تحضير | `VM_SPEAKER_CACHE=off` |
| 8 | دقة bf16/fp16 تلقائية + TF32 + cudnn.benchmark | `model_manager.py` | `VM_PRECISION`, `VM_TF32` | 1.3–2.2× على التوليد (Ampere+) | `precision=` في سجل الإقلاع + بوابة الجودة | `VM_PRECISION=off`, `VM_TF32=off` |
| 9 | torch.compile reduce-overhead (افتراضي off) | `model_manager.py` | `VM_COMPILE` | 1.2–1.65× إضافية عند التفعيل | `compiled=True` في سجل الإقلاع + بوابة | `VM_COMPILE=off` |
| 10 | إحماء عند الإقلاع | `model_manager.py`, `lifecycle.py` | `VM_WARMUP_ON_BOOT` | يلغي 0.5–2s من أول طلب | أول طلب بعد التشغيل بزمن طبيعي | `VM_WARMUP_ON_BOOT=off` |
| 11 | faster-whisper int8 + VAD (سقوط آمن) | `model_manager.py`, `transcription_service.py` | `VM_WHISPER_BACKEND/VAD` | 3–5× على النسخ | `whisper=faster-whisper` في سجل الإقلاع | `VM_WHISPER_BACKEND=openai-whisper` |
| 12 | SQLite WAL | `audio_mixer/db.py` | `VM_DB_WAL` | استطلاع بلا حجب | ملفات -wal إلى جوار القاعدة | `VM_DB_WAL=off` |
| 13 | طبقة محرك TTS (رسمي/نانوفي إل إل إم اختياري) | `core/tts_engine.py` | `VM_ENGINE` | حتى ~2× عند nanovllm | `VM_ENGINE=nanovllm` بعد تثبيت الحزمة والبوابات | `VM_ENGINE=official` |
| 14 | منصة القياس والجودة | `scripts/*`, `stopwatch.py` | `VM_LOG_STAGES` | (أداة) | `bench_baseline.py` + `quality_gate.py` | — |

## بروتوكول التحقق قبل الإعلان عن الأرقام

1. شغّل الخدمة على عتادك الحقيقي ورسّخ خط الأساس: `python scripts/bench_baseline.py --api-key ... --reference <ref.wav> --runs 30`
2. التقط المجموعة الذهبية قبل أي تعديل لاحق: `python scripts/golden_set.py --mode baseline --api-key ...` (ضع 3–8 مراجع في `tests/fixtures/voices/`)
3. بعد أي تغيير: `python scripts/golden_set.py --mode check` ثم `python scripts/quality_gate.py --baseline reports/golden_baseline --candidate reports/golden_check`
4. اقرأ توزيع المراحل من سجل الخادم (سطور `STAGES {...}`) وقارن مرحلة-بمرحلة.

## البنود الموثقة كخارطة طريق (لم تُنفذ عمداً — تتطلب عتاد اختبار أو عقد توصيل جديد)

- وضع البث التدريجي للعميل (`http_chunked`) وخلط المقاطع المتداخل (`file_overlap`): يتطلبان توليداً بمقاطع من كود النموذج واختباراً سمعياً على عتاد GPU حقيقي — موثقة في الدليل الرئيسي (القسم 29) كبند تالٍ.
- تكميم العمود اللغوي (INT8/FP8 عبر torchao): بند بوابة C — يتطلب مجموعة ذهبية كاملة على GPU الهدف قبل تفعيله.
- CUDA MPS وتثبيت الساعات: أوامر تشغيل بشري موثقة في `docs/HARDWARE_PLAYBOOK.md` لا كود.

---

# سجل التنفيذ — طبقة v2-ultra (أضعاف مضاعفة فوق v1)

> كل بند خلف مفتاح مستقل مع سقوط آمن. العقد الصامت محفوظ حرفياً:
> `inference_timesteps=30` و`cfg_value=2.0` وAudioVAE بلا تكميم وتوافق API 100%
> (الإضافات على الاستجابة: حقل `performance` الإضافي غير المخرِب).

## خريطة بنود v2

| # | البند | الملفات | المفتاح | الأثر المتوقع | التحقق | التراجع |
|---|---|---|---|---|---|---|
| 15 | كاش النتائج الكاملة (L1 RAM LRU بميزانية بايتات + L2 tmpfs) | `services/voice/result_cache.py`, `clone_service.py` | `VM_RESULT_CACHE*` | تكرار الطلب < 100ms (حتى ~100×) | `result_cache=HIT` في STAGES + إحصاءات `result_cache.stats()` | `VM_RESULT_CACHE=off` |
| 16 | خط أنابيب صفراوي: WAV/OGG وسيط في RAM عبر ffmpeg pipes | `utils/ram_pipeline.py`, `clone_service.py` | `VM_RAM_PIPELINE` | −0.5 إلى −1.5s I/O | `pipeline=ram` في STAGES | `VM_RAM_PIPELINE=off` |
| 17 | ترميز Opus سريع (compression_level=4 + application=audio) | `ram_pipeline.py` | `VM_OPUS_FASTENC` | ~2× ترميز أسرع | زمن مرحلة finalize + بوابة PESQ/STOI | `VM_OPUS_FASTENC=off` |
| 18 | كاش embedding ECAPA المرجعي + تقييم من الذاكرة (رياضيات verify_files نفسها) | `services/voice/speaker_score.py`, `speaker_cache.py` | `VM_EMBED_CACHE` | ~50% تكلفة التقييم | عدم إعادة حساب embedding لمرجع متكرر | `VM_EMBED_CACHE=off` |
| 19 | توازي مقاطع النص الطويل (بذور أصلية ثابتة — يسقط للتسلسل بلا seed أصلي) | `clone_service.py` | `VM_PARALLEL_CHUNKS` | حتى ~1.5–2× للنص المقطع | 3 مقاطع = 3 استدعاءات متوازية بنفس البذور | `VM_PARALLEL_CHUNKS=off` |
| 20 | بركة محركات متعددة GPUs (round-robin thread-safe + إحماء لكل نسخة) | `core/tts_engine.py` (EnginePool), `model_manager.py` | `VM_GPU_IDS` | ~N× إنتاجية تحت الحمل | سجل `EnginePool active` عند الإقلاع | `VM_GPU_IDS=` (فارغ) |
| 21 | تكميم أوزان weight-only (fp8/int8) مع استثناء VAE/decoder دائماً | `model_manager.py` | `VM_QUANT` (off افتراضياً) | 1.2–1.6× عند التفعيل | `quantized=` في سجل الإقلاع + بوابة C كاملة | `VM_QUANT=off` |
| 22 | Whisper المجمع batch=8 | `model_manager.py`, `transcription_service.py` | `VM_WHISPER_BATCHED` | ~3.7× على النسخ الطويلة | سجل `BatchedInferencePipeline ready` | `VM_WHISPER_BATCHED=off` |
| 23 | RAM staging لمدخلات الوظائف (SQLite فقط — Postgres يبقى BLOB) + سقف ميزانية | `audio_mixer/service.py` | `VM_JOB_RAM_STAGING`, `VM_RAM_STAGING_MAX_MB` | −2 رحلات BLOB كاملتين | `storage=ram_staging` في configuration_json | `VM_JOB_RAM_STAGING=off` |
| 24 | تجاوز ffprobe على خرج الخلط (خصائص مشتقة حسابياً) | `audio_mixer/service.py` | `VM_SKIP_OUTPUT_PROBE` | −30 إلى −80ms لكل وظيفة | `probe=derived` في metrics | `VM_SKIP_OUTPUT_PROBE=off` |
| 25 | -nostdin في كل استدعاءات ffmpeg/ffprobe المهمة | `audio_mixer/ffmpeg.py`, `ram_pipeline.py` | — | منع تعليقات stdin محتملة | داخلي | داخلي |
| 26 | بث SSE لحالة الوظائف | `audio_mixer/api.py` | — (نقطة إضافية) | إلغاء تأخير الاستطلاع | `GET /api/v1/audio/jobs/{id}/events` | عدم استدعاء النقطة |
| 27 | matmul precision=high + PYTORCH_CUDA_ALLOC_CONF قبل تحميل torch | `model_manager.py`, `main.py` | — | 5–15% على أحجام متغيرة | سجل الإقلاع | حذف السطر من main.py |
| 28 | حقول `performance` الإضافية في ردود clone (غير مخرِبة) | `clone_service.py`, `api/routes/voice.py` | — | قابلية رصد أفضل | حقل جديد في JSON | تجاهل من العميل |

## ملاحظات هندسية مهمة (v2)

1. **مفتاح كاش النتائج يتضمن بصمة المحرك** (name|precision|quant|compile) — أي تغيير في طبقة الاستدلال يبني مفاتيح جديدة تلقائياً، فلا تتلوث النتائج عبر تغييرات الإعداد.
2. **توازي المقاطع مشروط بـseed أصلي في `model.generate()`** — إن غاب، النظام يسقط للتسلسل تلقائياً لأن البذر اليدوي العام (`torch.manual_seed`) غير آمن خيطياً والحتمية عقد.
3. **تقييم ECAPA من الذاكرة يكرر رياضيات `verify_files` حرفياً** (torchaudio.load → resample إلى 16k → encode_batch → cosine eps=1e-8 → عتبة hparams) — وأي عائق يسقط إلى `verify_files` الأصلي، فبوابة ±0.01 محفوظة بالتصميم.
4. **RAM staging للوظائع محصور بـSQLite** (الثبات عبر العمليات مطلوب على PostgreSQL)، وبسقف ميزانية افتراضي 512MB فوقه يعود مسار BLOB تلقائياً. إعادة تشغيل العملية مع وظائف معلقة في RAM تعيد فشلاً صريحاً قابلاً لإعادة الإرسال بدل بيانات مسربة.
5. **الإحماء يغطي كل نسخ البركة** على كل الأجهزة — لا طلب يصطدم بنسخة باردة.
6. بنود قيد التنفيذ المتقدم (بث تدريجي داخل مقطع واحد، continuous batching حقيقي على مستوى DiT) تبقى موثقة كخارطة طريق لأنها تعيد تشكيل عقد التوصيل مع كود النموذج — انظر الدليل الرئيسي القسم 29.

## اختبارات v2 (أُجريت في بيئة CPU بلا GPU)

- `pytest tests/` — 2/2 (الغلاف الكامل للتطبيق عبر ASGI + مصادقة + ffmpeg حقيقي).
- اختبار وظيفي 19/19: كاش النتائج (L1/L2/LRU/TTL/حساسية المفتاح)، ram_pipeline (WAV→OGG عبر pipe حقيقي + فك عكسي)، كاش embedding (bit-exact)، EnginePool (دوران/معدل عينة/فلترة أجهزة)، RAM staging (تفعيل/تعطيل/فشل/تنظيف).
- اختبار طرف-لطرف 19/19: مسار clone كامل بمحرك مزيف — MISS كامل → HIT بصفر GPU، مقاطع متوازية ببذور ثابتة، سقوط آمن من RAM إلى الملفات، حساسية المفتاح للنص/المرجع/البنية، حقول العقد كاملة في كل الردود.

---

# سجل الجولة v3-max (بنود 29–38): المضبوط التلقائي + أضعاف مضاعفة

| # | التحسين | الملفات | المفتاح (auto افتراضياً) | الأثر المتوقع | كيف تتحقق | التراجع |
|---|---|---|---|---|---|---|
| 29 | **AutoTuner**: فحص عتاد كامل + حلّال نقي يحسب/يعيّن كل متغيرات الأداء + بروفايل JSON محفوظ ببصمة عتاد | `core/autotune.py` (جديد), `config.py` | `VM_AUTOTUNE` | إلغاء خطأ الضبط اليدوي كلياً + إقلاع سريع بالكاش | `GET /api/v1/autotune` + سجل الإقلاع | `VM_AUTOTUNE=off` |
| 30 | اختيار محرك NanoVLLM **تلقائياً** (RTF≈0.13 ≈ 2.3×) مع عقد timesteps/cfg وسقوط إجباري للرسمي | `core/tts_engine.py` | `VM_ENGINE=auto` | ~2.3× على مرحلة التوليد | سجل `Alternative engine loaded` / قياس RTF | `VM_ENGINE=official` |
| 31 | إصلاح فجوة v2: `pending_load` لم يكن مستهلكاً — `ensure_engine_loaded()` يحمل البديل فعلياً مع إعادة بناء فورية عند الفشل | `core/tts_engine.py`, `core/lifecycle.py` | — | يجعل البند 30 حقيقياً بدلاً نظرياً | سجل الإقلاع | داخلي |
| 32 | **CUDA Graphs تلقائياً** عبر `torch.compile(mode=reduce-overhead)` على ≥16GB (`default` على ≥10GB) بعد إحماء كامل | `core/model_manager.py` | `VM_COMPILE=auto` | ~1.1–1.65× (تقليل عبء الإطلاق) | `compile_mode=` في سجل/تقرير | `VM_COMPILE=off` |
| 33 | **تكميم تلقائي** weight-only: fp8 على Ada+ / int8 على cc≥7.5 بشرط torchao — AudioVAE مستثنى دائماً | `core/model_manager.py` | `VM_QUANT=auto` | 1.2–1.6× | `quantization=` في التقرير + بوابة C موصى بها | `VM_QUANT=off` |
| 34 | **مجدول GPU**: تصاريح تزامن لكل كرت + اختيار الأقل تحميلاً بدل round-robin الأعمى | `core/gpu_scheduler.py` (جديد), `tts_engine.py`, `clone_service.py` | `VM_GPU_PERMITS` | منع تدافع VRAM + توزيع أمثل تحت حمل متزامن | `scheduler` في `GET /api/v1/autotune` | `VM_GPU_PERMITS=1` |
| 35 | بركة **كل الكروت تلقائياً** + مقاطع النص تتناثر على الكروت عبر المجدول | `tts_engine.py`, `model_manager.py`, `clone_service.py` | `VM_GPU_IDS=auto` | ~N× إنتاجية/تقسيم بدون أي إعداد | سجل `EnginePool active` | `VM_GPU_IDS=0` |
| 36 | **Whisper مضبوط تلقائياً**: نموذج حسب VRAM (keep/distil/small) + دفعة مضبوطة + فك حتمي greedy (~1.5–2× قصير) | `model_manager.py`, `transcription_service.py`, `config.py` | `VM_WHISPER_MODEL/BATCH_SIZE/GREEDY` | 2–4× على مرحلة النسخ الساخنة | `whisper_model=` في التقرير | ثبّت القيم صراحة |
| 37 | **تقسيم تلقائي للنص الطويل** (>600 حرفاً على فواصل طبيعية) يفعّل توازي المقاطع وحده + مرشحو count>1 بالتوازي (بذور ثابتة) | `clone_service.py` | `VM_AUTO_CHUNK` | 1.5–2× للنص الطويل / ~count× للمرشحين | سجل `auto-chunk` | `VM_AUTO_CHUNK=off` |
| 38 | تخزين كاش النتائج **من الذاكرة مباشرة** (إلغاء القراءة العكسية من القرص) + تنفيذ كسول للبركة بحجم مضبوط | `clone_service.py` | — | −1 رحلة قرص لكل طلب مخزَّن | داخلي | داخلي |

## ملاحظات هندسية (v3)

1. **قانون الأولوية مطبق في كل نقطة استهلاك**: صريح ≠ auto يفوز دائماً؛
   `auto`/0 يقرأ من بروفايل المضبوط؛ الافتراضي القديم شبكة أمان أخيرة.
2. الحلّال `solve()` دالة نقية على بروفايل العتاد — 19 اختباراً وحدوياً
   يغطي T4/3080/4090/2×3090/CPU فقط + أولوية القيم الصريحة + استمرارية
   البروفايل وبصمته + المجدول (حد التزامن + الأقل تحميلاً) + المعايرة.
3. المعايرة تُقاس من تمريرات الإحماء الموجودة أصلاً — صفر كلفة إضافية على
   الإقلاع، وتُحفظ بأفضل قياس.
4. محرك NanoVLLM يُختار فقط على كرت واحد (المحرك يدير أجهزته بنفسه)،
   وبمنعه يصير compile/quant مساراً رسمياً خالصاً.
5. `compile on` التاريخية تُفسَّر `reduce-overhead`، و`quant on` تُفسَّر
   `int8` — توافق رجعي مع قيم v1/v2.

## اختبارات v3 (بيئة CPU بلا GPU)

- `pytest tests/` — **21/21** (2 غلاف تطبيق + 19 وحدوية جديدة للمضبوط/المجدول/الخصائص).
- `py_compile` ناجح لكل الملفات المعدلة والجديدة.
- سكربت القياس الجديد: `scripts/bench_v3.py` (وضع `--profile` محلي + وضع `--url` حي مع p50/p90 وكاش التكرار).
