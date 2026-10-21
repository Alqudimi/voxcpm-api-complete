# تنفيذ دليل تسريع Clone & Mix — سجل التنفيذ والتحقق

## الحالة

جهزت المشروع لتشغيل GPU محلياً، مع مسارات اختيارية للأداء وعودة آمنة حيث أمكن. لا يمكنني الادعاء بأن الأرقام أو الجودة تحققت على جهازك حتى تقيسها عليه؛ لكن التنفيذ والإعدادات والسكربتات أصبحت داخل الحزمة. ظلت قيم التوليد الأساسية `inference_timesteps=30` و`cfg_value=2.0`، ولم يُعطل `retry_badcase`.

## التغييرات المنفذة

- **GPU وتهيئة VoxCPM2:** يمرر التطبيق `DEVICE=cuda` و`VOXCPM_OPTIMIZE=true` صراحة إلى واجهة `from_pretrained` الرسمية، ويجري warm-up عند الإقلاع باستخدام CFG والخطوات نفسيهما. يمكن إيقافهما عبر `VOXCPM_OPTIMIZE=false` أو `VOXCPM_WARMUP=false`. دعم `optimize` و`device` موثق في API الرسمية [1] [2].
- **الوضع التلقائي للمرجع:** `REFERENCE_MODE=auto` يستخدم prompt-conditioned hi-fi عند توفر النص المرجعي؛ وإن لم يتوفر، يستخدم `reference_wav_path` وحده بدلاً من النسخ التلقائي. الوضعان مدعومان رسمياً، لكنهما ليسا مضمونين للتكافؤ السمعي؛ يمكن فرض `hifi` أو `reference` وفحص النتائج على مجموعة الصوت المستهدفة [1] [3].
- **Whisper كسول:** لا يُحمّل نموذج النسخ عند الإقلاع افتراضياً، بل عند أول طلب يختار `auto_transcribe=true`. هذا يوفر زمن إقلاع وذاكرة GPU عندما لا يلزم ASR. يمكن تحميله عند الإقلاع عبر `WHISPER_PRELOAD=true`.
- **مسار Whisper اختياري أسرع:** يمكن اختيار `WHISPER_BACKEND=faster-whisper`، مع `int8_float16` على CUDA و`int8` على CPU، وVAD اختياري (`WHISPER_VAD=false` افتراضياً). الاعتمادات الاختيارية في `requirements-performance.txt`. يتطلب GPU توافق CTranslate2 مع CUDA/cuDNN؛ راجع تعليمات المشروع الرسمية [4].
- **مسار PCM للخليط:** نقطة `/voice/clone-and-mix` تمرر WAV PCM_24 دون ترميز OGG وسيط، وتبدأ مهمة الخلط عند جاهزية PCM، بالتوازي مع تقييم ECAPA للمرشح الوحيد. المرشحون المتعددون ما زالوا يُقيّمون قبل إرسال الأفضل للخليط.
- **استجابة التطبيق:** نقل تجهيز المرجع والكتابة والتوليد وتقييم ECAPA والقراءة وإدراج المهمة الثقيلة إلى workers يمنع حجب حلقة asyncio. قفل ينظم استدعاءات التوليد على GPU واحد، ويحيط الاستدلال بـ`torch.inference_mode()`.
- **تخفيف عبء المرجع والقرص:** يقرأ `soundfile` مباشرة؛ ويُستخدم `soxr_hq` عند توفره مع fallback لـlibrosa، ويمكن العودة إلى librosa عبر `AUDIO_RESAMPLER=librosa`. يدعم مسار الخلط مجلد tmpfs عبر `AUDIO_MIXER_TEMP_PATH` ويدعم تطبيق VoxCPM مسار `TEMP_DIR`.
- **إدخالات الطابور:** خيار `AUDIO_MIXER_MEMORY_JOB_INPUTS=true` يتجنب تخزين إدخال المهمة في BLOB ويمرره مباشرة إلى worker. الافتراضي `false` كي تبقى الإدخالات قابلة للاستعادة بعد إعادة التشغيل؛ الخيار الذاكري حصري لخدمة أحادية العملية ويخسر مدخلات المهام المعلقة عند تعطلها.
- **SQLite:** WAL عند تهيئة المخطط، و`synchronous=NORMAL` لكل اتصال. لا يؤثر هذا الإعداد في PostgreSQL.
- **المراقبة والقياس:** أضيفت قياسات مراحل السجل وبيانات GPU/VRAM والدقة وbackend Whisper إلى health endpoint، وأداة قياس p50/p95/p99 للطرف إلى الطرف.

## تجهيز GPU وتشغيل الخدمة

على Linux مع Python 3.10–3.12 وNVIDIA، استخدم `bash scripts/install_gpu.sh`. يثبت السكربت PyTorch 2.8.0 بعجلات CUDA 12.8، ثم اعتمادات المشروع ويشغّل `scripts/preflight_gpu.py`. بعد ضبط `.env` بمفتاح قوي، ابدأ عبر `scripts/start_gpu.sh`. يتأكد المشغّل من CUDA ويستخدم عامل Uvicorn واحداً كي لا يحمل كل عامل نسخة مستقلة من النموذج. أوزان النموذج تُنزّل عند أول تحميل ما لم تحدد مساراً محلياً. واجهة VoxCPM الرسمية تذكر PyTorch ≥2.5 وCUDA ≥12 وتوثق `optimize=True` كخيار تسريع CUDA [1] [2].

إذا كان الجهاز لا يدعم cu128 أو كنت على Windows، أنشئ البيئة وحمّل عجلات PyTorch الملائمة للجهاز يدوياً، ثم استخدم `scripts/preflight_gpu.py` قبل بدء الخدمة. لا تغيّر `inference_timesteps` أو `cfg_value` بغرض السرعة.

## ما بقي غير مدمج أو غير متحقق منه

لم أدمج NanoVLLM أو vLLM-Omni كـbackend داخل API الحالية، ولم أضف بث HTTP مقطعياً/Opus أو كاش حالة المتحدث أو TensorRT-Whisper أو تكميم BF16/FP8 مخصصاً. الـAPI الحالي للنموذج المرجعي لا يقدّم بث HTTP جاهزاً؛ إعادة هيكلة الخدمة لهذه المحركات تحتاج محولات واعتمادات/خدمة منفصلة واختبارات توافق. لم أغيّر dtypes أو أحمل نموذجاً مكمماً غير مدقق.

ملف المجموعة الذهبية وGPU غير متاحين في بيئة البناء؛ لذا لم تُختبر جودة reference-only أو VAD/faster-whisper أو `torch.compile` على جهاز العميل. لا توجد نتيجة فعلية لزمن الاستجابة هنا. اختبر 30 طلباً دافئاً مع الأصوات والنصوص التي تهمك، وسجّل p50/p95/p99 وECAPA والمدة والاستماع المقارن قبل اعتبار الأرقام وعداً إنتاجياً.

## القياس على جهازك

```bash
API_BASE=http://127.0.0.1:8000 API_KEY='…' \
python scripts/benchmark_clone_and_mix.py \
  --reference reference.wav --noise ambience.wav \
  --prompt-text 'نص مرجعي مطابق للتسجيل' \
  --target-text 'النص المطلوب توليده' --warmups 2 --runs 30 --poll-ms 200
```

لا تضع مفاتيح API أو ملفات الصوت الحساسة في مستودع عام.

## المصادر الرسمية

[1] [VoxCPM API Reference](https://voxcpm.readthedocs.io/en/latest/reference/api.html) — `from_pretrained`, `optimize`, `device`, `generate`، و`reference_wav_path`.

[2] [VoxCPM Installation](https://voxcpm.readthedocs.io/en/latest/installation.html) — متطلبات Python/PyTorch/CUDA وسلوك `optimize`.

[3] [VoxCPM Usage Guide](https://voxcpm.readthedocs.io/en/latest/usage_guide.html) — أوضاع hi-fi وreference-only والنسخ والبث.

[4] [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper) — واجهة الاستخدام ومتطلبات cuBLAS/CUDA/cuDNN.
