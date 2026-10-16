# توثيق نقطة النهاية: توليد الصوت ودمج الضوضاء (Clone & Mix Audio)

نقطة نهاية موحدة تقوم باستنساخ نبرة الصوت من ملف صوتي مرجعي ونطقه بالنص المطلوب، ثم تمرير الصوت الناتج تلقائياً لدمجه مع ملف ضوضاء/خلفية صوتية محددة بنسب تحكم دقيقة، وجدولة عملية المعالجة.

---

## 1. معلومات نقطة النهاية (Endpoint Information)

* **العنوان (Endpoint URL):**  
  `POST /api/v1/voice/clone-and-mix`
* **كود الحالة المتوقع:**  
  `202 Accepted`
* **نوع المحتوى (Content-Type):**  
  `multipart/form-data` (لوجود ملفات صوتية مرفوعة)
* **المصادقة (Headers):**
  ```http
  X-API-Key: YOUR_API_KEY
  ```

---

## 2. جسم الطلب (Request Body)

بما أن الطلب يحتوي على ملفات صوتية مرفوعة، فإنه يُرسل بصيغة `multipart/form-data`. 

### تمثيل الحقول وقيمها بصيغة JSON (للتوضيح والتوثيق):

```json
{
  "reference_audio": "@path/to/reference_voice.wav (ملف صوتي - إجباري)",
  "target_text": "مرحباً بكم، هذا تسجيل تجريبي للصوت المستنسخ المدموج مع الضوضاء.",
  "noise_audio": "@path/to/background_noise.wav (ملف صوتي - إجباري)",
  "prompt_text": "النص المنطوق داخل الصوت المرجعي إن وجد",
  "auto_transcribe": false,
  "cfg_value": 2.0,
  "inference_timesteps": 30,
  "candidates": 1,
  "seed": 12345,
  "retry_badcase": true,
  "retry_max_times": 3,
  "retry_ratio_threshold": 6.0,
  "silence_ms": 350,
  "source_gain_db": 0.0,
  "noise_gain_db": -18.0,
  "variation_seed": 42,
  "output_format": "wav"
}
```

### تفاصيل الحقول (Parameters Specification):

| الحقل | النوع | إجباري؟ | القيمة الافتراضية | الوصف |
| :--- | :--- | :--- | :--- | :--- |
| **`reference_audio`** | `File (Binary)` | **نعم** | - | ملف الصوت المرجعي المراد استنساخ نبرته (WAV / MP3 / OGG). |
| **`target_text`** | `string` | **نعم** | - | النص المطلوب توليده بصوت المتحدث المستنسخ. |
| **`noise_audio`** | `File (Binary)` | **نعم** | - | ملف صوت الضوضاء أو الخلفية الموسيقية المراد دمجها مع الصوت المستنسخ. |
| **`prompt_text`** | `string` | اختياري | `null` | التفريغ النصي للصوت المرجعي (مطلوب إذا لم يتم تفعيل `auto_transcribe`). |
| **`auto_transcribe`** | `boolean` | اختياري | `false` | استخدام Whisper لتفريغ الصوت المرجعي تلقائياً في حال عدم إرسال `prompt_text`. |
| **`cfg_value`** | `float` | اختياري | `2.0` | قيمة الـ Classifier-Free Guidance للتوجيه في التوليد. |
| **`inference_timesteps`** | `integer` | اختياري | `30` | عدد خطوات الاستدلال والتوليد. |
| **`candidates`** | `integer` | اختياري | `1` | عدد العينات المرشحة للمفاضلة بينها واختيار الأفضل. |
| **`seed`** | `integer` | اختياري | `null` | البذرة العشوائية لتوليد الصوت المستنسخ لضمان تكرارية النتيجة. |
| **`retry_badcase`** | `boolean` | اختياري | `true` | إعادة المحاولة تلقائياً إذا كانت العينة رديئة الجودة. |
| **`retry_max_times`** | `integer` | اختياري | `3` | أقصى عدد لإعادة المحاولة في حال فشل الجودة. |
| **`retry_ratio_threshold`** | `float` | اختياري | `6.0` | عتبة نسبة الطول المقبولة بين الإدخال والإخراج. |
| **`silence_ms`** | `integer` | اختياري | `350` | مدة الصمت المضافة بين المقاطع (عند تقسيم النص بـ "؟"). |
| **`source_gain_db`** | `float` | اختياري | `0.0` | تعديل مستوى صوت الكلام المستنسخ بالديسيبل (dB) من `-60` إلى `+6`. |
| **`noise_gain_db`** | `float` | اختياري | `-18.0` | تعديل مستوى صوت الضوضاء بالديسيبل (dB) من `-60` إلى `+6`. |
| **`variation_seed`** | `integer` | اختياري | `null` | بذرة عشوائية لتحديد موضع اقتطاع الضوضاء بشكل مختلف في كل مرة. |
| **`output_format`** | `string` | اختياري | `null` (افتراضي `wav`) | صيغة الإخراج النهائية للملف المدموج (`wav`, `mp3`, `ogg`, `flac`). |

---

## 3. شكل الاستجابة (Response Body)

تُرجع النقطة استجابة بصيغة JSON تحتوي على بيانات مهمة الدمج وروابط المتابعة، بالإضافة إلى تقرير فحص ومقاييس جودة الصوت المستنسخ.

### مثال على الاستجابة الناجحة (`202 Accepted`):

```json
{
  "success": true,
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "queued",
  "status_url": "http://example.com/api/v1/audio/jobs/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d/status",
  "result_url": "http://example.com/api/v1/audio/jobs/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d/result",
  "clone_info": {
    "request_id": "a823f6c128914b1f9b329402ab84d632",
    "prompt_text": "هذا تسجيل صوتي مرجعي بصوتي",
    "seed": 12345,
    "duration": 5.42,
    "speaker_score": 0.884,
    "same_speaker": true,
    "reference_qc": {
      "duration": 4.1,
      "sample_rate": 24000,
      "channels": 1,
      "peak": -1.2,
      "rms": -22.5
    },
    "chunks_count": 1,
    "ranking": [
      {
        "rank": 1,
        "seed": 12345,
        "score": 0.884,
        "same_speaker": true,
        "duration": 5.42
      }
    ]
  }
}
```

### شرح حقول الاستجابة:
* **`success`**: تأكيد نجاح معالجة الطلب واستنساخ الصوت وبدء مهمة الدمج.
* **`job_id`**: المعرف الفريد لمهمة الدمج الصوتية في الخلفية.
* **`status`**: حالة المهمة الحالية (تكون `queued` أو `processing`).
* **`status_url`**: رابط للاستعلام عن حالة المعالجة الحالية.
* **`result_url`**: رابط تحميل الملف الصوتي المدمج النهائي بمجرد اكتماله.
* **`clone_info`**:
  * **`request_id`**: المعرف الداخلي لملف الاستنساخ الناتج.
  * **`prompt_text`**: النص المعتمد كمرجع.
  * **`duration`**: مدة الصوت المستنسخ بالثواني.
  * **`speaker_score`**: درجة تطابق بصمة الصوت مع الصوت المرجعي من خلال نموذج ECAPA-TDNN.
  * **`same_speaker`**: هل النبرة تعتبر لنفس المتحدث (`true`/`false`).
  * **`reference_qc`**: تقرير الجودة والفحص للصوت المرجعي الأصلي.
  * **`chunks_count`**: عدد الأجزاء النصية المنطوقة.

---

## 4. كيفية متابعة النتيجة وتحميل الملف المدموج

بعد استلام الاستجابة السابقة:

### أ. التحقق من حالة المعالجة (Status Check):
```http
GET /api/v1/audio/jobs/{job_id}/status
X-API-Key: YOUR_API_KEY
```
**الرد:**
```json
{
  "id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "completed",
  "created_at": "2026-10-16T20:10:00Z",
  "completed_at": "2026-10-16T20:10:02Z",
  "processing_time_ms": 1820
}
```

### ب. تنزيل الصوت المدموج النهائي (Download Final Result):
```http
GET /api/v1/audio/jobs/{job_id}/result
X-API-Key: YOUR_API_KEY
```
يُرجع الملف الصوتي المدمج مباشرة بالصيغة المحددة (`output_format`).

---

## 5. أمثلة الاستدعاء البرمجية (Code Examples)

### مثال باستخدام `cURL`:
```bash
curl -X POST "http://localhost:8000/api/v1/voice/clone-and-mix" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "reference_audio=@sample_voice.wav" \
  -F "noise_audio=@city_traffic_noise.wav" \
  -F "target_text=أهلاً بكم في النظام الصوتي المتقدم" \
  -F "prompt_text=هذا صوتي الأصلي للتدريب" \
  -F "source_gain_db=0" \
  -F "noise_gain_db=-20" \
  -F "output_format=wav"
```

### مثال باستخدام `Python (requests)`:
```python
import requests

url = "http://localhost:8000/api/v1/voice/clone-and-mix"
headers = {"X-API-Key": "YOUR_API_KEY"}

files = {
    "reference_audio": open("voice.wav", "rb"),
    "noise_audio": open("noise.wav", "rb"),
}

data = {
    "target_text": "السلام عليكم ورحمة الله، مرحباً بكم.",
    "prompt_text": "نص الصوت المرجعي هنا",
    "source_gain_db": 0.0,
    "noise_gain_db": -18.0,
    "output_format": "wav",
}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.status_code)
print(response.json())
```
