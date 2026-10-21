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
# عدّل API_KEY، ثم نفّذ في Linux مع NVIDIA GPU وPython 3.10–3.12:
bash scripts/install_gpu.sh
source .venv/bin/activate
scripts/start_gpu.sh
```

يثبت السكربت PyTorch 2.8.0 بعجلات CUDA 12.8 ثم اعتمادات المشروع ويتحقق من GPU. أول تشغيل ينزّل أوزان VoxCPM2 والنماذج المساعدة ويجري إحماءً للنموذج. يبدأ `start_gpu.sh` بعامل Uvicorn واحد، كي لا تُحمّل نسخة GPU كاملة لكل عامل.

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


## تفاصيل تشغيل Clone and Mix

يقبل `POST /api/v1/voice/clone-and-mix` طلب `multipart/form-data` ومصادقة `X-API-Key`، ويعيد `202 Accepted` مع `job_id` وروابط الحالة والنتيجة. تتوفر متابعة المهمة عبر `GET /api/v1/audio/jobs/{job_id}/status` وتنزيلها عبر `GET /api/v1/audio/jobs/{job_id}/result`.

تم ضبط `candidates` فعلياً (من 1 إلى 8) لاختيار المرشح الأعلى في تشابه المتحدث. يجري المسار فحص حجم/وجود الضوضاء، صلاحية صيغة الإخراج، ومجال الكسب من -60 إلى +6 dB قبل تشغيل التوليد. الصيغة الافتراضية WAV ما لم يُحدد خلاف ذلك. ملفات المرشحين المؤقتة وملف الكلام الوسيط تزال بعد الاستخدام.

استخدم `.env.example` كقالب، واستبدل مفتاح API قبل التشغيل. وضع `REFERENCE_MODE=auto` يستخدم hi-fi cloning عند إرسال النص المرجعي، وإلا يستخدم المرجع الصوتي مباشرة لتجاوز Whisper. اختبر المخرجات على أصواتك قبل اعتماد هذا الوضع إنتاجياً.


لتفاصيل التحسينات وخيارات التشغيل والقياس راجع [PERFORMANCE_IMPLEMENTATION_AR.md](PERFORMANCE_IMPLEMENTATION_AR.md). فحص الجهاز عبر `scripts/preflight_gpu.py` والقياس الطرفي عبر `scripts/benchmark_clone_and_mix.py`.
