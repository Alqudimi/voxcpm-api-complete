# VoxCPM2 Professional API

خدمة FastAPI احترافية لاستنساخ الصوت باستخدام VoxCPM2، التفريغ العربي باستخدام Whisper، التحقق من تشابه المتحدث باستخدام ECAPA-TDNN، ودمج الصوتيات وإخراجها بصيغة OGG/Opus.

## المبدأ

يتم تحميل محرك VoxCPM2 وECAPA عند Startup؛ Whisper يُحمّل كسولاً فقط عند طلب النسخ:

1. VoxCPM2
2. ECAPA speaker verification
3. Whisper large-v3 عند الحاجة فقط

لا يحدث تحميل لـVoxCPM2 أو ECAPA عند أول طلب. قد يحمّل أول طلب نسخ Whisper إذا لم يكن محمّلاً مسبقاً.

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
- `reference_mode` (`auto`, `hifi`, أو `reference`)
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

## Authentication

أرسل:

`X-API-Key: YOUR_SECRET`

## تشغيل

```bash
cp .env.example .env
# المثبت ينشئ API_KEY عشوائياً إذا بقي placeholder؛ يتطلب Linux وNVIDIA وPython 3.10–3.12:
bash scripts/install_gpu.sh
source .venv/bin/activate
scripts/start_gpu.sh
```

أو عبر Docker Compose (يتطلب NVIDIA Container Toolkit):

```bash
cp .env.example .env
python3 scripts/prepare_env.py
# سيُنشأ مفتاح API عشوائي داخل .env دون طباعته:
docker compose up --build
```

يثبت السكربت PyTorch 2.8.0 بعجلات CUDA 12.8 وNanoVLLM/FlashAttention وملحقات الأداء، ثم يتحقق من GPU. أول تشغيل ينزّل أوزان VoxCPM2 ويجري الإحماء. يستخدم NanoVLLM افتراضياً؛ للعودة إلى المحرك القياسي ضع `TTS_ENGINE=pytorch` في `.env`. يبدأ `start_gpu.sh` بعامل Uvicorn واحد، ويتولى محرك NanoVLLM التزامن.

Swagger:
`http://SERVER:8000/docs`

## ملاحظة

يدعم التطبيق محرك NanoVLLM المتدفق ومحرك VoxCPM PyTorch كخطة رجوع. في مسار PyTorch حُفظت معاملات `model.generate()`، وفي NanoVLLM تُمرر إعدادات المرجع والنص وCFG والبذرة مع سقف طول وإعادة محاولة للحالات الشاذة، بما في ذلك:
- prompt_wav_path
- prompt_text
- reference_wav_path
- cfg_value
- inference_timesteps
- retry_badcase
- Best-of-N
- seed compatibility
- ECAPA speaker scoring
- تقطيع النص الطويل عند علامات الاستفهام العربية فقط عند تفعيل الخيار

كما تم دمج Whisper وFFmpeg في الخدمات المنفصلة.


## تفاصيل تشغيل Clone and Mix

يقبل `POST /api/v1/voice/clone-and-mix` طلب `multipart/form-data` ومصادقة `X-API-Key`، ويعيد `202 Accepted` مع `job_id` وروابط الحالة والأحداث والنتيجة بعد اكتمال التوليد. يتابع SSE حالة مرحلة الخلط؛ لا يرسل مقاطع الصوت إلى العميل. استخدم `events_url` مع `fetch` وترويسة `X-API-Key`، أو `status_url` للتوافق، ثم نزّل النتيجة عبر `result_url`.

تم ضبط `candidates` فعلياً (من 1 إلى 8) لاختيار المرشح الأعلى في تشابه المتحدث. يجري المسار فحص حدود النص وصلاحية صيغة الإخراج ومجال الكسب وسعة طابور الخلط قبل تشغيل التوليد. الصيغة الافتراضية Opus عند 128 kbps؛ وتبقى الصيغ الأخرى قابلة للاختيار. تستخدم الملفات المؤقتة tmpfs عند توفر مساحة كافية، وإلا يعود التطبيق تلقائياً إلى القرص.

استخدم `.env.example` كقالب، واستبدل مفتاح API قبل التشغيل. وضع `REFERENCE_MODE=auto` يستخدم hi-fi cloning عند إرسال النص المرجعي، وإلا يستخدم المرجع الصوتي مباشرة لتجاوز Whisper. اختبر المخرجات على أصواتك قبل اعتماد هذا الوضع إنتاجياً.


لتفاصيل التحسينات وخيارات التشغيل والقياس راجع [PERFORMANCE_IMPLEMENTATION_AR.md](PERFORMANCE_IMPLEMENTATION_AR.md). فحص الجهاز عبر `scripts/preflight_gpu.py` والقياس الطرفي عبر `scripts/benchmark_clone_and_mix.py`.
