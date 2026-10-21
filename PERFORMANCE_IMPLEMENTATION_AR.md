# تنفيذ دليل تسريع Clone & Mix — سجل التنفيذ والتحقق

## الحالة

جهزت المشروع بتكامل فعلي لمحرك NanoVLLM كخيار افتراضي، مع PyTorch كخطة رجوع، وملفات تشغيل GPU وDocker. لا يمكنني الادعاء بأن الأداء أو الجودة قيسا على جهازك دون تشغيله على GPU فعلي. قيم التوليد الأساسية الافتراضية بقيت `inference_timesteps=30` و`cfg_value=2.0`، وحارس `retry_badcase` لا يمكن تعطيله من واجهات API.

## التغييرات المنفذة

- **محرك NanoVLLM الافتراضي:** يحمّل `nano-vllm-voxcpm` كـasync server pool، يمرر مرجع الصوت في صورة latents، ويستقبل waveform chunks، مع batching/stream scheduling داخلي؛ يحتفظ PyTorch بخطة رجوع قابلة للاختيار عبر `TTS_ENGINE=pytorch`. تُختار `max_num_seqs` وVRAM utilization تلقائياً من الذاكرة المتاحة، ويمكن تثبيتهما من `.env`.
- **GPU وتهيئة VoxCPM2:** مسار PyTorch الاحتياطي يمرر `optimize=True` إلى VoxCPM الرسمي، الذي يستخدم إعداد dtype الافتراضي للنموذج، ويجري warm-up. NanoVLLM يدار بواسطة engine مستقل ولا نفرض عليه dtype/compile غير مكشوفين في API؛ تُضبط خطواته عند الإقلاع عبر `DEFAULT_TIMESTEPS`.
- **الوضع التلقائي للمرجع:** `REFERENCE_MODE=auto` يستخدم prompt-conditioned hi-fi عند توفر النص المرجعي؛ وإن لم يتوفر، يستخدم `reference_wav_path` وحده بدلاً من النسخ التلقائي. الوضعان مدعومان رسمياً، لكنهما ليسا مضمونين للتكافؤ السمعي؛ يمكن فرض `hifi` أو `reference` وفحص النتائج على مجموعة الصوت المستهدفة [1] [3].
- **Whisper كسول:** لا يُحمّل نموذج النسخ عند الإقلاع افتراضياً، بل عند أول طلب يختار `auto_transcribe=true`. هذا يوفر زمن إقلاع وذاكرة GPU عندما لا يلزم ASR. يمكن تحميله عند الإقلاع عبر `WHISPER_PRELOAD=true`.
- **مسار Whisper اختياري أسرع:** `WHISPER_BACKEND=faster-whisper` افتراضياً في ملف البيئة، مع `int8_float16` على CUDA و`int8` على CPU؛ يظل Whisper كسولاً ولا يدخل المسار افتراضياً عند وضع reference-only. الاعتمادات في `requirements-performance.txt` [4].
- **مسار PCM للخليط:** نقطة `/voice/clone-and-mix` تمرر WAV PCM_24 دون ترميز OGG وسيط، وتبدأ مهمة الخلط عند جاهزية PCM، بالتوازي مع تقييم ECAPA للمرشح الوحيد. المرشحون المتعددون ما زالوا يُقيّمون قبل إرسال الأفضل للخليط.
- **التزامن والاستجابة:** NanoVLLM يتولى جدولة الطلبات المتزامنة؛ مسار PyTorch يبقى مقفلاً على GPU واحد. تجهيز المرجع والكتابة وECAPA وقراءة المدخلات تعمل في workers. أضيف SSE لمتابعة مرحلة الخلط بعد الاستجابة؛ لا يمثل ذلك بث waveform من التوليد إلى العميل.
- **تخفيف عبء المرجع والقرص:** يقرأ `soundfile` مباشرة؛ ويستخدم `soxr_hq` عند توفره مع fallback محافظ. يستخدم tmpfs فقط عندما تتوفر مساحة كافية، وإلا يعود إلى القرص. ملف recovery الخاص بـSQLite يبقى على التخزين الدائم.
- **إدخالات الطابور:** خيار `AUDIO_MIXER_MEMORY_JOB_INPUTS=true` يتجنب تخزين إدخال المهمة في BLOB ويمرره مباشرة إلى worker. الافتراضي `false` كي تبقى الإدخالات قابلة للاستعادة بعد إعادة التشغيل؛ الخيار الذاكري حصري لخدمة أحادية العملية ويخسر مدخلات المهام المعلقة عند تعطلها.
- **SQLite:** WAL عند تهيئة المخطط، و`synchronous=NORMAL` لكل اتصال. لا يؤثر هذا الإعداد في PostgreSQL.
- **الترميز والطابور:** Opus عند 128 kbps و`application=audio` و`compression_level=4` هو الافتراضي. يرفض Clone-and-Mix الطلب مبكراً بـ429 و`Retry-After` عندما لا تتوفر سعة لخدمة الخلط.
- **المراقبة والقياس:** أضيفت قياسات مراحل السجل وبيانات GPU/VRAM وbackend الفعلي ونمط Nano إلى health endpoint، وأداة قياس p50/p95/p99 للطرف إلى الطرف.

## تجهيز GPU وتشغيل الخدمة

على Linux مع Python 3.10–3.12 وNVIDIA، انسخ `.env.example` إلى `.env` ثم استخدم `bash scripts/install_gpu.sh`؛ سيولد المفتاح العشوائي تلقائياً إذا بقي placeholder، ويضبط الملف بصلاحيات خاصة. يثبت PyTorch 2.8.0 CUDA 12.8، الاعتمادات الاختيارية، NanoVLLM وFlashAttention، ثم يشغّل preflight يشمل CUDA والمحرك وlibopus. بعده ابدأ بـ`scripts/start_gpu.sh`. لتشغيل Docker استخدم `python3 scripts/prepare_env.py` ثم `docker compose up --build`؛ Compose يخصص shared memory أكبر ويحفظ الأوزان والكاش والبيانات في volumes.

على Windows أو عند تعذر تثبيت FlashAttention، استخدم `TTS_ENGINE=pytorch` وخطة التثبيت المناسبة لنظامك؛ NanoVLLM يتطلب CUDA وFlashAttention. لا تغيّر `inference_timesteps` أو `cfg_value` بغرض السرعة.

## ما بقي غير مدمج أو غير متحقق منه

لم أدمج vLLM-Omni، FP8، MIG/قفل ترددات GPU، TensorRT-Whisper، أو بث waveform/Opus إلى العميل أثناء التوليد، كما أن تداخل ffmpeg mix مع أول مقطع يحتاج خط أنابيب صوتياً متدفقاً منفصلاً. أبقيت هذه البنود خارج المسار الافتراضي لأنها تحتاج بوابات جودة صوتية واختبارات توافق/عتاد؛ SSE الموجود يتابع مرحلة المهمة وليس مقاطع الصوت. حُفظ BF16/compile في المسار المرجعي عندما يدعمه النموذج، دون فرض تكميم FP8.

نجح `compileall` و`bash -n`، كما نجح اختبار مولد المفتاح الخاص، واختبارا تخزين مدخلات مهام الخلط، واختباران صغيران لـsplitter ومحول NanoVLLM باستخدام محرك وهمي. لم تتوفر `pytest` ولا GPU أو NVIDIA في sandbox، ولم تُبنَ صورة Docker هنا؛ لذلك لم تُختبر جودة الصوت/مرجع المستخدم، ولا سرعة المحرك أو FFmpeg على جهاز فعلي. اختبر 30 طلباً دافئاً على جهازك مع الأصوات والنصوص المطلوبة، وسجّل p50/p95/p99 وECAPA والمدة والاستماع المقارن قبل اعتماد أرقام إنتاجية.

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

[5] [Nano-vLLM-VoxCPM README](https://github.com/a710128/nanovllm-voxcpm) — متطلبات GPU وFlashAttention، API async، التسلسل، ومرجع الصوت.

[6] [Nano-vLLM VoxCPM2 server source](https://github.com/a710128/nanovllm-voxcpm/blob/main/nanovllm_voxcpm/models/voxcpm2/server.py) — توقيع `generate`, `encode_latents`, وتدفق المقاطع.

[7] [VoxCPM2 source](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/model/voxcpm2.py) — نوع BF16 الافتراضي، تحسين `torch.compile` بوضع `reduce-overhead`، ومعيار retry الأصلي.
