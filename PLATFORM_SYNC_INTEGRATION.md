# دليل تكامل المنصة الخارجية: المزامنة، تخزين العينات في SQLite، وتسجيل الخادم

يوثق هذا الدليل التحسينات المضافة لنظام **VoxCPM API** لتمكينه من:
1. **التحميل التلقائي** لكافة عينات الأصوات والضوضاء وبياناتها وتخزينها محلياً في قاعدة بيانات **SQLite** سريعة (`storage/platform_resources.db`) مع حفظ الملفات الصوتية على القرص لمرة واحدة عند الإقلاع.
2. **التسجيل الذكي للخادم**: إرسال دومين/رابط هذا الخادم الحالي ومفتاح الـ API تلقائياً إلى المنصة الخارجية عبر نقطة النهاية `POST /api/voice-cloning/server-settings`.
3. **الاستنساخ الفوري عبر المعرّف (UUID)**: إمكانية استنساخ الصوت ودمج الضوضاء مباشرة عبر معرّف العينة `reference_sample_uuid` و `noise_sample_uuid` دون الحاجة لإعادة رفع الملفات الصوتية في كل طلب.

---

## 1. متغيرات البيئة في `.env` (Configuration)

أضف أو عدّل المتغيرات التالية في ملف `.env`:

```bash
# رابط المنصة الخارجية الأساسي
PLATFORM_BASE_URL=https://your-main-platform.com

# رابط خادم الذكاء الاصطناعي الحالي (اختياري، يكتشف تلقائياً في RunPod / Cloudflare / Localhost إذا تُرِك فارغاً)
SERVER_PUBLIC_URL=https://your-ai-worker.runpod.net

# تفعيل المزامنة وتحميل العينات عند الإقلاع لأول مرة (افتراضي: True)
PLATFORM_SYNC_ON_STARTUP=True

# تفعيل إرسال إعدادات الخادم ومفتاح الـ API للمنصة عند الإقلاع (افتراضي: True)
PLATFORM_REGISTER_ON_STARTUP=True

# مهلة الاتصال بالمنصة بالثواني
PLATFORM_TIMEOUT=600

# مسارات التخزين المحلي
PLATFORM_DB_PATH=storage/platform_resources.db
VOICE_SAMPLES_DIR=storage/voice_samples
NOISE_SAMPLES_DIR=storage/noise_samples
```

---

## 2. آلية العمل عند الإقلاع (Startup & Lifecycle Flow)

عند تشغيل الخادم (`uvicorn app.main:app`):
1. **تحميل النماذج والإحماء:** يتم تشغيل نماذج الذكاء الاصطناعي (VoxCPM2, Whisper, ECAPA).
2. **المزامنة في الخلفية (`run_startup_workflow`):**
   * **الخطوة الأولى:** يكتشف النظام رابط الخادم الحالي ويرسل طلب `POST` إلى `{PLATFORM_BASE_URL}/api/voice-cloning/server-settings` بالحمولة:
     ```json
     {
       "server_url": "https://your-ai-worker.runpod.net",
       "api_key": "YOUR_API_KEY",
       "timeout": 600
     }
     ```
     فتقوم المنصة بتحديث إعداداتها وتطهير الكاش فوراً.
   * **الخطوة الثانية:** يتصل الخادم بـ `{PLATFORM_BASE_URL}/api/voice-cloning/voices-and-noise` ويجلب كافة الأشخاص والعينات وملفات الضوضاء.
   * **الخطوة الثالثة:** يقوم بتنزيل الملفات الصوتية إلى:
     - `storage/voice_samples/{sample_uuid}.wav`
     - `storage/noise_samples/{noise_uuid}.mp3`  
     *(ملاحظة: إذا كان الملف محملاً مسبقاً وبحجم سليم، يتم تخطي تنزيله لتوفير الباندويث وضمان إقلاع فوري)*.
   * **الخطوة الرابعة:** يحفظ كافة البيانات والنصوص المرجعية والمسارات المحلية في جداول SQLite داخل `storage/platform_resources.db`.

---

## 3. نقاط نهاية إدارة الموارد (Resources Endpoints)

تمت إضافة مجموعة نقاط نهاية لفحص الموارد المخزنة أو إعادة المزامنة يدوياً:

### أ. فحص إحصائيات الموارد المخزنة وحالة المنصة:
```http
GET /api/v1/resources/summary
X-API-Key: YOUR_API_KEY
```
**الرد:**
```json
{
  "success": true,
  "platform_base_url": "https://your-main-platform.com",
  "detected_server_url": "https://my-pod-8000.proxy.runpod.net",
  "summary": {
    "total_voices": 8,
    "total_voice_samples": 24,
    "total_noise_profiles": 5,
    "total_noise_samples": 12,
    "last_sync_at": "2026-10-22T20:30:00Z",
    "server_registered_at": "2026-10-22T20:30:01Z",
    "last_registered_server_url": "https://my-pod-8000.proxy.runpod.net",
    "database_path": "storage/platform_resources.db"
  }
}
```

### ب. استعراض كافة الأصوات والعينات المتاحة محلياً:
```http
GET /api/v1/resources/voices
X-API-Key: YOUR_API_KEY
```

### ج. استعراض كافة ملفات وعينات الضوضاء المتاحة محلياً:
```http
GET /api/v1/resources/noise
X-API-Key: YOUR_API_KEY
```

### د. تشغيل إعادة المزامنة يدوياً:
```http
POST /api/v1/resources/sync?force=false
X-API-Key: YOUR_API_KEY
```

### هـ. إعادة إرسال إعدادات الخادم والمفتاح يدوياً للمنصة:
```http
POST /api/v1/resources/register-server
X-API-Key: YOUR_API_KEY
```

---

## 4. الاستنساخ والدمج المباشر عبر الـ UUID

أصبح بإمكان العميل استدعاء نقاط الاستنساخ والدمج عبر تمرير معرّف العينة `reference_sample_uuid` أو `noise_sample_uuid` بدون رفع أي ملفات:

### مثال: استنساخ صوت باستخدام معرّف العينة المخزنة
## 4. الاستنساخ والدمج الفوري عبر الـ ID والـ RAM Cache (Zero-Disk I/O)

بفضل كاش الذاكرة العشوائية السريعة `FastResourceCache`، يتم تحميل كافة بايتات عينات الأصوات والضوضاء ونصوصها في RAM عند الإقلاع، ويمكن للعميل إرسال طلبات **JSON خفيفة جداً (<200 بايت)** تحتوي فقط على المعرفات:

### أ. استنساخ الصوت عبر JSON (فائق السرعة - Sub-microsecond RAM Retrieval):
```http
POST /api/v1/voice/clone
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "voice_sample_id": 12,
  "target_text": "مرحباً بكم، هذا توليد صوتي فوري عبر الآيدي المسجل في الذاكرة الحية."
}
```
*أو عبر الـ UUID النصي:*
```json
{
  "voice_sample_id": "485d25f7-d890-4c96-a1c1-0e925eb7b4bb",
  "target_text": "مرحباً بكم، هذا توليد صوتي فوري."
}
```
*يتم استرجاع الصوت المرجعي والنص المرجعي من الذاكرة RAM في زمن أقل من 0.001ms بدون قراءة قرص وبدون استعلام قاعدة بيانات.*

### ب. استنساخ الصوت ودمج الضوضاء بالمعرفات عبر JSON:
```http
POST /api/v1/voice/clone-and-mix
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "voice_sample_id": 12,
  "noise_sample_id": 5,
  "target_text": "نص التوليد المطلوب دمجه مع صوت المطر الهادئ",
  "source_gain_db": 0.0,
  "noise_gain_db": -18.0,
  "output_format": "wav"
}
```

### ج. دمج ملف مرفوع مع ضوضاء مسجلة بالـ ID في الـ Audio Mixer:
```http
POST /api/v1/audio/mixes/from-upload
X-API-Key: YOUR_API_KEY
Content-Type: multipart/form-data

Form fields:
- source_audio: (الملف الصوتي المرفوع)
- noise_sample_id: 5
- source_gain_db: 0
- noise_gain_db: -18
```
*(ملاحظة: لا زال بالإمكان إرسال `reference_audio` أو `noise_audio` كملفات مرفوعة يدوياً كـ Fallback متوافق رجعياً بنسبة 100%)*
