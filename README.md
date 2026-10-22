# VoxCPM2 Professional API

خدمة FastAPI احترافية لاستنساخ الصوت باستخدام VoxCPM2، التفريغ العربي باستخدام Whisper، التحقق من تشابه المتحدث باستخدام ECAPA-TDNN، ودمج الصوتيات وإخراجها بصيغة OGG/Opus.

## المبدأ

يتم تحميل النماذج الثلاثة عند Startup:

1. VoxCPM2
2. Whisper large-v3
3. ECAPA speaker verification

وبالتالي لا يحدث تحميل للنماذج عند أول طلب.

## الهيكل

```text
app/
├── api/routes/              # HTTP فقط
├── core/                    # Config / Security / Lifecycle / Models
├── schemas/                 # API contracts
├── services/
│   ├── voice/               # Voice cloning + Best-of-N
│   ├── transcription/       # Whisper
│   ├── audio/               # Merge + FFmpeg
│   └── storage/             # Files
└── utils/                   # Audio utilities
storage/
├── uploads/
├── outputs/
└── temp/
```

## Endpoints

### Health
`GET /api/v1/health`

### Voice cloning
`POST /api/v1/voice/clone`

Multipart:
- `reference_audio`
- `prompt_text`
- `target_text`
- `cfg_value`
- `inference_timesteps`
- `candidates`
- `seed`
- `retry_badcase`
- `retry_max_times`
- `retry_ratio_threshold`
- `auto_transcribe`

### Voice Cloning & Audio Mixing (Clone and Mix)
`POST /api/v1/voice/clone-and-mix` (انظر التوثيق الكامل في [CLONE_AND_MIX_API.md](CLONE_AND_MIX_API.md))

Multipart:
- نفس معاملات `clone` بالإضافة إلى:
  - `noise_audio` (ملف صوت الضوضاء)
  - `source_gain_db`
  - `noise_gain_db`
  - `variation_seed`
  - `output_format`

### Transcription
`POST /api/v1/audio/transcribe`

### Merge
`POST /api/v1/audio/merge`

### Audio
`GET /api/v1/voice/audio/{filename}`
`GET /api/v1/audio/output/{filename}`

### Platform Resources & Sync
(انظر التوثيق الكامل في [PLATFORM_SYNC_INTEGRATION.md](PLATFORM_SYNC_INTEGRATION.md))
- `GET /api/v1/resources/summary`: إحصائيات الموارد المخزنة محلياً في SQLite وحالة المنصة.
- `GET /api/v1/resources/voices`: استعراض كافة الأصوات وعيناتها المخزنة محلياً.
- `GET /api/v1/resources/noise`: استعراض كافة ملفات وعينات الضوضاء المخزنة محلياً.
- `POST /api/v1/resources/sync`: تشغيل المزامنة وإعادة فحص الملفات يدوياً.
- `POST /api/v1/resources/register-server`: إعادة إرسال دومين الخادم ومفتاح API للمنصة يدوياً.

## Authentication

أرسل:

`X-API-Key: YOUR_SECRET`

## تشغيل

```bash
cp .env.example .env
# عدّل API_KEY
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger:
`http://SERVER:8000/docs`

## ملاحظة

تم فصل طبقة VoxCPM2 عن FastAPI. منطق `model.generate()` الموجود في Notebook المستخدم تم الحفاظ عليه، بما في ذلك:
- prompt_wav_path
- prompt_text
- reference_wav_path
- cfg_value
- inference_timesteps
- retry_badcase
- Best-of-N
- seed compatibility
- ECAPA speaker scoring
- إرسال النص كاملاً بدون Smart Chunking

كما تم دمج Whisper وFFmpeg في الخدمات المنفصلة.

## طبقة الأداء (v1.0) — خطة تسريع `clone_voice_and_mix`

النسخة الحالية تتضمن طبقة تسريع كاملة مبنية على دليل التدقيق (`docs/OPTIMIZATION_LOG.md`). أهم ما تغير:

- **كاش حالة المتحدث (bit-exact):** تحضير المرجع + تقرير جودته + نص Whisper تُحسب مرة لكل صوت مرجعي وتُعاد مطابقة بتّياً في الطلبات المتكررة (`VM_SPEAKER_CACHE`).
- **عزل الـ event loop:** كل عمل حاجب (تحضير/توليد/تقييم/ترميز) يعمل في `ThreadPoolExecutor` — الخدمة تستجيب تحت الحمل دائماً.
- **توازي:** تقرير الجودة ∥ Whisper، وتقييم ECAPA ∥ ترميز OGG.
- **دقة الحساب:** bf16 تلقائي على Ampere+ مع TF32 وcudnn.benchmark (`VM_PRECISION=auto`)، و`torch.compile` اختياري (`VM_COMPILE=on`).
- **faster-whisper int8 + VAD** مع سقوط آمن إلى openai-whisper إن لم تُثبت.
- **SQLite WAL + tmpfs + soxr_hq** لإزالة هدر الإدخال/الإخراج.
- **إحماء عند الإقلاع** فيلغي البداية الباردة من أول طلب.

العقد الثابت: `inference_timesteps=30` و`cfg_value=2.0` ومخطط API لم يتغير حرفياً.

## طبقة الأداء (v2-ultra) — أضعاف مضاعفة فوق v1

طبقة ثانية كاملة بنيت فوق طبقة v1 (كل مفاتيح v1 باقية وتعمل). الجديد:

- **كاش النتائج الكاملة (`VM_RESULT_CACHE`):** الطلب المتطابق تماماً (نص + متحدث + معاملات) يعيد OGG النهائي والبيانات المرافقة من الذاكرة في **أقل من 100ms** — التوليد والتقييم والترميز كلها تُتخطى. أكبر معامل مضاعف في النظام (حتى ~100× للطلبات المتكررة: عبارات IVR/إشعارات/إعادة محاولة). طبقتان: RAM LRU بميزانية بايتات + ملفات على tmpfs تنجو من إعادة التشغيل. المفتاح حساس لبصمة المحرك والدقة والتكميم — تغيّر أيٍّ منها يبني مفاتيح جديدة تلقائياً.
- **خط أنابيب صفراوي (`VM_RAM_PIPELINE`):** WAV الوسيط وترميز OGG يتمان كلياً في الذاكرة عبر ffmpeg stdin/stdout pipes — صفر ملفات وسيطة على القرص. الملف الوحيد المكتوب هو OGG النهائي في storage/outputs. أي فشل في الـpipe يسقط تلقائياً إلى المسار الملفي v1 حرفياً.
- **كاش embedding ECAPA (`VM_EMBED_CACHE`):** متجه المتحدث المرجعي يُحسب مرة واحدة لكل صوت ويُخزَّن — الطلبات اللاحقة تحسب متجه المولَّد فقط (نصف تكلفة التقييم)، والتقييم نفسه من الذاكرة بنفس رياضيات `verify_files`.
- **توازي مقاطع النص الطويل (`VM_PARALLEL_CHUNKS`):** نص «س1؟ س2؟ س3؟» يُولَّد بمقاطعه متوازية ببذور ثابتة لكل مقطع — نفس النتيجة تماماً كالتسلسل. يسقط تلقائياً إلى التسلسل إذا المحرك بلا `seed` أصلي (حماية الحتمية).
- **بركة محركات متعددة GPUs (`VM_GPU_IDS=0,1,...`):** نسخة نموذج لكل كرت ودوران round-robin thread-safe — إنتاجية N× حقيقية تحت الحمل وانتظار طابور ÷N. الإحماء يغطي كل نسخ البركة.
- **تكميم أوزان اختياري (`VM_QUANT=fp8|int8`، الافتراضي off):** weight-only عبر torchao مع استثناء دائم لوحدات VAE/decoder (العقد الصلب). فعّله فقط بعد اجتياز بوابة الجودة C على عتادك.
- **Whisper مجمع (`VM_WHISPER_BATCHED`):** BatchedInferencePipeline(batch=8) — ~3.7× موثق، مع سقوط آمن للوضع الأحادي.
- **مسار خلط أسرع:** مدخلات الوظيفة في RAM بدل BLOB مزدوج في SQLite (`VM_JOB_RAM_STAGING`)، وخصائص الخرج تُشتق حسابياً بدل ffprobe إضافي (`VM_SKIP_OUTPUT_PROBE`)، و`-nostdin` في كل استدعاءات ffmpeg.
- **ترميز Opus أسرع (`VM_OPUS_FASTENC`):** compression_level=4 بدل 10 — ~2× ترميز أسرع بشفافية عملية عند 96k (قابل للتحقق ببوابات PESQ/STOI).
- **بث حالة الوظائف (SSE):** `GET /api/v1/audio/jobs/{id}/events` يدفع تغيّرات الحالة لحظياً — يلغي تأخير الاستطلاع الدوري (نقطة إضافية، الاستطلاع يبقى صالحاً).
- **إعداد CUDA allocator:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` يُضبط تلقائياً قبل تحميل torch.

### مصفوفة الأثر المتوقع (v2 فوق v1)

| البند | الحالة | المعامل |
|---|---|---|
| كاش النتائج (تكرار الطلب) | افتراضي on | < 100ms ثابت (حتى ~100×) |
| خط أنابيب صفراوي | افتراضي on | −0.5 إلى −1.5s I/O لكل طلب |
| كاش embedding المرجعي | افتراضي on | ~50% من تكلفة ECAPA |
| توازي المقاطع (3 مقاطع) | افتراضي on | حتى ~1.5–2× على النص الطويل |
| بركة multi-GPU (N كرت) | `VM_GPU_IDS` | ~N× إنتاجية تحت الحمل |
| Whisper المجمع | افتراضي on | ~3.7× على النسخ الطويلة |
| fastenc Opus | افتراضي on | ~2× ترميز أسرع |
| RAM staging للوظائف | افتراضي on | −2 رحلات BLOB كاملتين |
| تكميم fp8/int8 | off (بوابة C) | 1.2–1.6× إضافية عند التفعيل |

### القياس

```bash
# خط الأساس (P50/P95/P99 من جهة العميل)
python scripts/bench_baseline.py --api-key $API_KEY --reference voice.wav --runs 30

# توقيتات المراحل لكل طلب: تظهر في سجل الخادم كسطور STAGES {...} JSON

# مجموعة ذهبية + بوابة الجودة قبل/بعد أي تعديل
python scripts/golden_set.py --mode baseline --api-key $API_KEY
python scripts/golden_set.py --mode check   --api-key $API_KEY
python scripts/quality_gate.py --baseline reports/golden_baseline --candidate reports/golden_check
```

### متغيرات البيئة

السجل الكامل في `.env.example` مع التعليقات. القاعدة: كل تحسين خلف مفتاح مستقل، والتراجع بأمر واحد.

### عتاد الهدف

الأهداف الزمنية لكل فئة كرت وأوامر تشغيل النظام في `docs/HARDWARE_PLAYBOOK.md`.

## طبقة الأداء (v3-max) — المضبوط التلقائي + أضعاف مضاعفة فوق v2

الجديد الجوهري: **لم تعد بحاجة لضبط أي متغير بيئة**. عند الإقلاع، `AutoTuner`
يفحص عتادك (كروت/VRAM/CC/CPU/RAM/الحزم المثبتة) ويحسب ويُعيّن كل متغيرات
الأداء بالقيمة المثلى لهذا العتاد بالذات، ويقيس RTF فعلياً أثناء الإحماء،
ويحفظ بروفايل JSON يُعاد استخدامه في الإقلاعات التالية.

- **قانون الأولوية الصارم:** قيمة صريحة في البيئة تتفوق دائماً على المضبوط.
  `auto` (أو `0`) = اتركه للمضبوط. التراجع الكامل: `VM_AUTOTUNE=off`.
- **اختيار المحرك تلقائياً (`VM_ENGINE=auto`):** إن كانت حزمة `nano-vllm-voxcpm`
  مثبتة والكرت كابي (cc≥8، VRAM≥8GB) → يُستخدم المحرك السريع (RTF ≈ 0.13 الموثق
  ≈ **2.3×** مقابل الرسمي) — والعقد محفوظ: إن رفض قبول timesteps=30/cfg=2.0
  سقط النظام تلقائياً للرسمي عند الإقلاع.
- **CUDA Graphs تلقائياً (`VM_COMPILE=auto`):** `torch.compile(mode=reduce-overhead)`
  على كروت ≥16GB (وضع `default` على ≥10GB) بعد إحماء كامل — مع سقوط آمن عند أي فشل.
- **تكميم تلقائي (`VM_QUANT=auto`):** fp8 على Ada+ (cc≥8.9)، int8 على cc≥7.5،
  بشرط وجود torchao — **AudioVAE مستثنى دائماً** (العقد الصلب). موصى بتشغيل
  بوابة الجودة C مرة واحدة بعد أول إقلاع للتوثيق.
- **مجدول GPU (`gpu_scheduler.py`):** تصاريح تزامن لكل كرت (2 على ≥16GB) +
  اختيار **الأقل تحميلاً** بدل الدوران الأعمى — المقاطع والطلبات تتناثر
  تلقائياً على كل الكروت دون تدافع VRAM.
- **بركة كل الكروت تلقائياً (`VM_GPU_IDS=auto`):** كرتان = بركة محركين بدون أي إعداد.
- **Whisper مضبوط تلقائياً (`VM_WHISPER_MODEL=auto`):** يحافظ على large-v3
  عند VRAM≥11GB، `distil-large-v3` على 6–11GB، `small` تحت ذلك أو على CPU —
  مع دفعة مضبوطة حسب VRAM وفك حتمي greedy (~1.5–2× على المقاطع القصيرة).
- **تقسيم تلقائي للنص الطويل (`VM_AUTO_CHUNK`):** نص > 600 حرفاً يُقسم على
  فواصل طبيعية فيفعال توازي المقاطع وحده (`VM_AUTO_CHUNK=off` للتعطيل).
- **مرشحو count>1 بالتوازي:** كل المرشحين يتولّدون متوازيين ببذور ثابتة —
  نفس نتيجة التسلسل حرفياً.
- **معايرة RTF حقيقية:** كل تمريرة إحماء تُقاس (زمن ÷ ثواني صوت) وتُحفظ في
  البروفايل — راجع `GET /api/v1/autotune`.
- **إصلاح فجوة v2:** محرك NanoVLLM يُحمّل فعلياً عند الإقلاع (`ensure_engine_loaded`)
  مع إعادة بناء فورية على الرسمي عند أي فشل.

### نقاط النهاية الجديدة

| نقطة | الوصف |
|---|---|
| `GET /api/v1/autotune` | التقرير الكامل: العتاد + كل القيم المحسوبة + المعايرة + الكاشات + المجدول |
| `POST /api/v1/autotune/retune` | إعادة فحص العتاد وحل جديد (قرارات التحميل الثقيلة عند الإقلاع التالي) |

### القياس (v3)

```bash
# تقرير المضبوط محلياً (بلا خادم) + توقعات زمنية
python scripts/bench_v3.py --profile

# قياس حي ضد خادم يعمل: p50/p90/min/max + زمن كاش التكرار
python scripts/bench_v3.py --url http://localhost:8000 --api-key $API_KEY \
  --reference voice.wav --text "النص المطلوب" --runs 10
```

### التفاصيل الكاملة

كل ما يخص المضبوط التلقائي (جدول كل متغير وكيف يُحسب وكيف تثبّته يدوياً)
في `docs/AUTOTUNE.md`.
