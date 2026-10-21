# المضبوط التلقائي (AutoTuner) — v3-max

> «اجعله يحسب ويعيّن جميع المتغيرات الذي يحتاجها بشكل آلي تلقائي» — هذا
> الملف يوثّق كيف يتحقق ذلك حرفياً: عند الإقلاع يُفحص العتاد، تُحل كل
> متغيرات الأداء بالقيمة المثلى لهذا العتاد، تُطبَّق، تُقاس فعلياً، وتُحفظ.

## 1. الفكرة في ثلاث جمل

1. **فحص** — `probe_hardware()` يجمع: عدد الكروت وأسماؤها وVRAM وCompute
   Capability وعدد SM، نوى CPU، حجم RAM، مساحة `/dev/shm` الحرة، إصدارات
   torch/CUDA، والحزم المثبتة (torchao / faster-whisper / nano-vllm-voxcpm
   / uvloop / soxr). يعمل بلا GPU وبلا torch (بيئات CPU/الاختبار).
2. **حلّ** — `solve()` دالة نقية: (عتاد) → (كل القيم). قابلة للاختبار
   الوحدوي على بروفايلات مصطنعة (انظر `tests/test_autotune.py`).
3. **تطبيق + معايرة + حفظ** — القرارات تُطبق على التحميل، كل تمريرة إحماء
   تُقاس RTF وتُحفظ، والبروفايل كله يُخزن في
   `storage/autotune_profile.json` — إقلاع تالٍ بنفس بصمة العتاد يقفز فوق
   الفحص والمعايرة فوراً.

## 2. قانون الأولوية (الأهم)

```
قيمة صريحة في البيئة/.env  >  قيمة المضبوط  >  الافتراضي القديم
```

- «صريح» = أي قيمة غير `auto` (أو غير `0` للحقول العددية).
- `VM_AUTOTUNE=off` يطفئ المضبوط كلياً ويعيد سلوك v1/v2 بالضبط.
- لا يوجد أي متغير لا يمكن تثبيته يدوياً — المضبوط يملأ **الفراغ فقط**.

## 3. جدول كل متغير وكيف يُحسب

| المتغير | كيف يحسبه المضبوط | كيف تثبّته يدوياً |
|---|---|---|
| `dtype` | bf16 إذا cc≥8.0، fp16 إذا cc≥7.0، وإلا fp32 | `VM_PRECISION=bf16\|fp16\|fp32` |
| `tf32` | true إذا cc≥8.0 | `VM_TF32=on/off` |
| `cudnn_benchmark` | true مع أي GPU | `VM_CUDNN_BENCHMARK` |
| `engine` | `nanovllm` إذا الحزمة مثبتة + cc≥8 + VRAM≥8GB + كرت واحد؛ وإلا `official` | `VM_ENGINE=official\|nanovllm` |
| `compile_mode` | `reduce-overhead` (CUDA Graphs) إذا VRAM≥16GB وcc≥7.5؛ `default` إذا ≥10GB؛ `off` دون ذلك؛ `off` مع nanovllm (يدير نفسه) | `VM_COMPILE=reduce-overhead\|default\|off` |
| `compile_warmup` | 3 مع compile، 1 بدونه | `VM_COMPILE_WARMUP=n` |
| `quant` | fp8 إذا cc≥8.9؛ int8 إذا cc≥7.5؛ يتطلب torchao؛ `off` مع nanovllm أو بلا GPU. **AudioVAE مستثنى دائماً** | `VM_QUANT=fp8\|int8\|off` |
| `gpu_ids` | كل الكروت المكتشفة | `VM_GPU_IDS=0,1` |
| `gpu_permits` | 2 إذا VRAM≥16GB لكل كرت، وإلا 1 | `VM_GPU_PERMITS=n` |
| `executor_workers` | (كروت × تصاريح) + عمال CPU `max(2, min(8, cores//4))` — سقف 24 | `VM_EXECUTOR_WORKERS=n` |
| `chunk_parallel_max` | تصاريح × كروت — سقف 8 | `VM_CHUNK_PARALLEL_MAX=n` |
| `whisper_model` | `keep` (large-v3) إذا VRAM≥11GB؛ `distil-large-v3` إذا 6–11GB؛ `small` دون ذلك أو CPU | `VM_WHISPER_MODEL=<اسم>` |
| `whisper_compute` | int8 دائماً (الأسرع)؛ float16 فقط عند `VM_WHISPER_BACKEND=fw-fp16` صراحة | عبر `VM_WHISPER_BACKEND` |
| `whisper_batch_size` | 8 (≥11GB) / 4 (6–11GB) / 2 (<6GB) / 4 (CPU) | `VM_WHISPER_BATCH_SIZE=n` |
| `whisper_greedy` | true (beam=1، بلا حمل سياق — ~1.5–2× على المقاطع القصيرة) | `VM_WHISPER_GREEDY=off` |
| `ecapa_device` | cuda مع أي GPU وإلا cpu | `VM_ECAPA_DEVICE=cuda\|cpu` |
| `result_cache_max` / `result_cache_max_mb` | من RAM: `clamp(ram//64, 16, 128)` / `clamp(2%ram, 64, 512)MB` | `VM_RESULT_CACHE_MAX(_MB)=n` |
| `speaker_cache_max` | `clamp(ram//512, 8, 48)` | `VM_SPEAKER_CACHE_MAX=n` |
| `ram_staging_max_mb` | `clamp(4%ram, 128, 1024)` | `VM_RAM_STAGING_MAX_MB=n` |
| `auto_chunk_chars` | 600 حرفاً | `VM_AUTO_CHUNK=off` أو `VM_AUTO_CHUNK_CHARS=n` |
| `tmpfs_usable` | `/dev/shm` حرة ≥1GB | `VM_TMPFS_DIR=<مسار>` |
| `rtf` (معايرة) | قياس فعلي: أسرع تمريرة إحماء (زمن ÷ ثواني صوت) | — (يُقاس، لا يُضبط) |

## 4. العقد الصلبة التي لا يلمسها المضبوط

1. `inference_timesteps=30` و`cfg_value=2.0` — مثبتان في الكود والتوثيق،
   لا يوجد مسار في المضبوط يغيّرهما.
2. **AudioVAE/decoder/vocoder خارج كل تكميم** — فلتر الاستثناء في
   `_try_quantize` يعمل في كل الأوضاع.
3. **سقوط آمن في كل قرار** — فشل compile/quant/nanovllm/المجدول = استمرار
   بسلوك v1/v2 حرفياً، والإقلاع لا يفشل أبداً.
4. توافق API رجعي كامل — نفس حقول الاستجابة (+ حقل `performance` الإضافي).

## 5. المعايرة (Calibration)

- تُقاس أثناء الإحماء الافتراضي (بلا أي خطوة إضافية): كل تمريرة توليد
  تُقاس `RTF = زمن التوليد ÷ ثواني الصوت المولَّد`.
- يُحفظ **أفضل** قياس (يمثل أداء الخدمة بعد تسخين النوى) + آخر قياس +
  مجاميع الفحص.
- تشاهد النتيجة في `GET /api/v1/autotune` → `calibration`، وفي سجل
  الإقلاع كسطور `Warmup [compile 1/3] 0.42s (audio=1.00s, RTF=0.421)`.
- إعادة الفحص والحل بدون إقلاع: `POST /api/v1/autotune/retune`
  (قرارات التحميل الثقيلة — dtype/compile/quant — تُطبق في الإقلاع التالي).

## 6. مسار الإقلاع الكامل (v3)

```
uvicorn app.main:app
  └─ lifespan
      ├─ model_manager.load_all()
      │    ├─ tuner.ensure()          ← فحص العتاد / تحميل بروفايل مخبّأ
      │    ├─ VoxCPM2.from_pretrained
      │    ├─ dtype (من المضبوط) + quant (من المضبوط، VAE مستثنى)
      │    ├─ بركة كل الكروت (auto) + نسخة لكل كرت
      │    ├─ Whisper (نموذج/دقة/دفعة من المضبوط)
      │    └─ ECAPA (جهاز من المضبوط)
      ├─ model_manager.warmup()       ← تمريرات مقاسة → معايرة RTF → حفظ
      │    └─ torch.compile(mode=المضبوط) بعد أول تمريرة + إحماء مجمع
      └─ ensure_engine_loaded()       ← NanoVLLM تلقائي؛ أي فشل → رسمي فوراً
```

## 7. أسئلة متكررة

**س: كيف أرجع كل شيء كما كان في دقيقة واحدة؟**
ج: `VM_AUTOTUNE=off` (أو حذف `storage/autotune_profile.json`) — كل سلوك
v1/v2 يعود، لأن كل بند محسوب يمر عبر قانون الأولوية.

**س: كيف أثبّت قيمة واحدة فقط وأترك الباقي تلقائياً؟**
ج: ثبّت ذلك المتغير وحده في `.env` — مثال: `VM_QUANT=off` والباقي auto.

**س: هل التكميم التلقائي يغير الجودة؟**
ج: weight-only على وحدات النص/الانتشار فقط مع استثناء VAE كلياً. موصى
بتشغيل `scripts/quality_gate.py` مرة واحدة بعد أول إقلاع للتوثيق على
عتادك، وإن رغبت بالإلغاء: `VM_QUANT=off`.

**س: أين يُحفظ البروفايل؟ وكيف أفرض إعادة حساب؟**
ج: `storage/autotune_profile.json` (متغير `VM_AUTOTUNE_FILE`). إعادة
حساب: `POST /api/v1/autotune/retune` أو حذف الملف.

**س: تغيّر الكرت — هل يتعرف تلقائياً؟**
ج: نعم. البروفايل يحمل بصمة العتاد (أنواع الكروت + VRAM + CPU + RAM)؛
بصمة مختلفة = إعادة فحص وحل كامل تلقائياً.

**س: لماذا compile=off عندما يُختار محرك NanoVLLM؟**
ج: المحرك البديل يدير تنفيذه بنفسه (تنفيذ مخصص أسرع) — تجميع فوقه مضاعفة
جهد بلا مكسب. إن أردت compile مع الرسمي: `VM_ENGINE=official`.
