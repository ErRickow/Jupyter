# 🎙️ Deploy F5-TTS (PapaRazi/Ijazah_Palsu_V2) di Modal.com

**Guide lengkap untuk deploy Text-to-Speech model dengan GPU T4 inference**

---

## 📋 Table of Contents

- [Tentang F5-TTS](#tentang-f5-tts)
- [Tentang Modal.com](#tentang-modalcom)
- [Arsitektur Deployment](#arsitektur-deployment)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Deployment](#deployment)
- [Cara Menggunakan API](#cara-menggunakan-api)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)

---

## 🎯 Tentang F5-TTS

**F5-TTS** (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching) adalah state-of-the-art Text-to-Speech model yang dikembangkan oleh SWivid.

### ✨ Fitur Utama:

- **Zero-Shot Voice Cloning** - Clone suara dari audio reference 3-12 detik
- **Multilingual Support** - Trained dengan 100K hours multilingual dataset
- **Non-Autoregressive** - Lebih cepat dari model autoregressive tradisional
- **High Quality** - Natural dan expressive speech synthesis
- **Flow Matching** - Menggunakan Diffusion Transformer (DiT) architecture
- **Fast Inference** - RTF (Real-Time Factor) 0.15

### 🔧 Model: PapaRazi/Ijazah_Palsu_V2

Model ini adalah **fine-tuned version** dari F5-TTS base model, kemungkinan:
- Optimized untuk bahasa Indonesia
- Custom voice atau speaking style
- Specific domain adaptation

---

## 🚀 Tentang Modal.com

**Modal** adalah platform serverless compute yang dioptimalkan untuk AI/ML workloads.

### 💪 Keunggulan Modal untuk TTS Inference:

| Feature | Benefit |
|---------|---------|
| **Pay-per-use** | Hanya bayar saat inference berjalan, tidak ada idle cost |
| **Auto-scaling** | Scale from 0 to N containers otomatis |
| **Fast Cold Start** | Sub-second container startup |
| **GPU Pool** | Instant access ke T4, A10G, A100, H100 |
| **Simple Code** | Deploy dengan Python decorators saja |
| **Volume Caching** | Model weights di-cache untuk fast loading |

### 💰 Pricing Comparison:

| Platform | T4 GPU Cost/hour | Idle Cost |
|----------|------------------|-----------|
| Modal | ~$0.60 | $0 (pay-per-second) |
| AWS SageMaker | ~$0.70 | Yes (always running) |
| GCP Vertex AI | ~$0.65 | Yes (min 1 hour) |
| Azure ML | ~$0.90 | Yes (always running) |

---

## 🏗️ Arsitektur Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                      MODAL.COM CLOUD                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Container (GPU T4)                                 │    │
│  │                                                      │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  F5-TTS Model (PapaRazi/Ijazah_Palsu_V2)    │  │    │
│  │  │  - Loaded di GPU memory                       │  │    │
│  │  │  - Model weights cached di Volume            │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                                                      │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  REST API Endpoints                          │  │    │
│  │  │  - POST /tts_api (TTS inference)             │  │    │
│  │  │  - GET  /health_check                        │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                                                      │    │
│  │  Auto-scaling: 0 → N containers based on load      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Modal Volume (Persistent Storage)                  │    │
│  │  /root/.cache/huggingface/                          │    │
│  │  - Model weights                                     │    │
│  │  - Tokenizer files                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ HTTPS
                            │
                   ┌────────┴─────────┐
                   │   Your Client    │
                   │   Application    │
                   └──────────────────┘
```

---

## ✅ Prerequisites

### 1. **Modal Account**

```bash
# Install Modal CLI
pip install modal

# Setup Modal account (akan buka browser untuk login)
modal setup
```

Anda akan mendapat **$30 free credits** untuk mulai!

### 2. **Hugging Face Account** (Optional)

Jika model `PapaRazi/Ijazah_Palsu_V2` adalah private:

```bash
# Get token dari: https://huggingface.co/settings/tokens
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxx
```

### 3. **Python Environment**

```bash
# Python 3.11+ recommended
python --version

# Install dependencies
pip install modal requests
```

---

## 🛠️ Setup & Installation

### 1. Clone atau Download Files

Anda memerlukan 3 files:

```
your-project/
├── modal_f5_tts_inference.py    # Modal deployment code
├── modal_f5_tts_client.py       # Client code untuk test
└── MODAL_F5_TTS_GUIDE.md        # Dokumentasi ini
```

### 2. Edit Configuration

Edit `modal_f5_tts_inference.py`, line 21-23:

```python
# Model configuration
MODEL_NAME = "PapaRazi/Ijazah_Palsu_V2"  # Ganti dengan model ID Anda
BASE_MODEL = "SWivid/F5-TTS"  # Fallback ke base model
GPU_TYPE = "T4"  # T4, A10G, A100, dll
```

**Pilihan GPU:**

| GPU Type | VRAM | Speed | Cost/hour | Recommended For |
|----------|------|-------|-----------|-----------------|
| T4 | 16GB | 1x | $0.60 | Development, light inference |
| A10G | 24GB | 3x | $1.10 | Production, medium load |
| A100 | 40GB | 8x | $4.00 | High-volume production |

### 3. Model Compatibility

Jika `PapaRazi/Ijazah_Palsu_V2` tidak tersedia atau error, code akan otomatis fallback ke base model `SWivid/F5-TTS`.

**Untuk custom model:**

```python
# Option 1: Hugging Face model
MODEL_NAME = "your-username/your-model"

# Option 2: Local checkpoint
# Upload checkpoint ke Modal Volume, lalu:
MODEL_NAME = "/cache/your-checkpoint.pth"
```

---

## 🚀 Deployment

### Step 1: Test Locally

```bash
# Test function untuk verify code works
modal run modal_f5_tts_inference.py
```

Output expected:
```
🧪 Testing F5-TTS inference...
🚀 Loading F5-TTS model: PapaRazi/Ijazah_Palsu_V2
✅ Successfully loaded custom model
🎮 Using device: cuda
   GPU: Tesla T4
   Memory: 14.76 GB
✅ Model loaded and ready for inference!
🎤 Generating speech for text: Halo, ini adalah test...
✅ Speech generated! Sample rate: 24000 Hz
📊 Audio duration: 5.23 seconds
✅ Test successful!
💾 Audio saved to: test_output.wav
```

### Step 2: Deploy ke Modal

```bash
# Deploy as serverless app
modal deploy modal_f5_tts_inference.py
```

Output:
```
✓ Created objects.
├── 🔨 Created mount /path/to/modal_f5_tts_inference.py
├── 🔨 Created f5_tts_image => Image
├── 🔨 Created model_cache_vol => Volume
├── 🔨 Created F5TTSModel => Class
├── 🔨 Created tts_api => Function
└── 🔨 Created health_check => Function

✓ App deployed! 🎉

View Deployment: https://modal.com/apps/your-username/f5-tts-inference

Endpoints:
  tts_api         => https://your-username--f5-tts-inference-tts-api.modal.run
  health_check    => https://your-username--f5-tts-inference-health-check.modal.run
```

**SAVE URL ENDPOINTS INI!** Anda akan menggunakannya untuk API calls.

### Step 3: Verify Deployment

```bash
# Check health
curl https://your-username--f5-tts-inference-health-check.modal.run

# Expected response:
{
  "status": "healthy",
  "gpu_available": true,
  "model": "PapaRazi/Ijazah_Palsu_V2",
  "gpu_type": "T4"
}
```

---

## 📡 Cara Menggunakan API

### Method 1: Python Client (Recommended)

Edit `modal_f5_tts_client.py`, ganti API URL:

```python
API_URL = "https://your-username--f5-tts-inference-tts-api.modal.run"
```

Run:

```bash
python modal_f5_tts_client.py
```

### Method 2: Python Requests

```python
import requests
import base64

# Simple TTS (tanpa voice cloning)
response = requests.post(
    "https://your-url.modal.run",
    json={
        "text": "Halo, ini adalah test TTS dalam bahasa Indonesia.",
        "remove_silence": True
    },
    timeout=120
)

result = response.json()

if result["success"]:
    # Decode dan save audio
    audio_bytes = base64.b64decode(result["audio_base64"])

    with open("output.wav", "wb") as f:
        f.write(audio_bytes)

    print(f"Duration: {result['duration']:.2f}s")
    print(f"Sample rate: {result['sample_rate']} Hz")
```

### Method 3: Voice Cloning

```python
# Read reference audio
with open("reference_voice.wav", "rb") as f:
    ref_audio = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://your-url.modal.run",
    json={
        "text": "Text yang ingin di-clone dengan suara reference.",
        "ref_audio_base64": ref_audio,
        "ref_text": "Transcription of reference audio.",  # Optional tapi recommended
        "remove_silence": True
    },
    timeout=120
)
```

### Method 4: cURL

```bash
curl -X POST https://your-url.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from cURL!",
    "remove_silence": true
  }' \
  | jq -r '.audio_base64' \
  | base64 -d > output.wav
```

### Method 5: JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

async function generateSpeech(text) {
    const response = await axios.post(
        'https://your-url.modal.run',
        {
            text: text,
            remove_silence: true
        },
        {
            timeout: 120000
        }
    );

    if (response.data.success) {
        // Decode base64 audio
        const audioBuffer = Buffer.from(
            response.data.audio_base64,
            'base64'
        );

        // Save to file
        fs.writeFileSync('output.wav', audioBuffer);

        console.log(`Duration: ${response.data.duration}s`);
    }
}

generateSpeech("Hello from JavaScript!");
```

---

## 📊 API Reference

### POST /tts_api

**Request:**

```json
{
    "text": "string (required)",
    "ref_audio_base64": "string (optional) - base64 encoded WAV",
    "ref_text": "string (optional) - transcription of reference audio",
    "remove_silence": "boolean (optional, default: true)"
}
```

**Response (Success):**

```json
{
    "success": true,
    "audio_base64": "base64 encoded WAV audio",
    "sample_rate": 24000,
    "duration": 5.23
}
```

**Response (Error):**

```json
{
    "success": false,
    "error": "Error message"
}
```

### GET /health_check

**Response:**

```json
{
    "status": "healthy",
    "gpu_available": true,
    "model": "PapaRazi/Ijazah_Palsu_V2",
    "gpu_type": "T4"
}
```

---

## 🎯 Best Practices

### 1. **Reference Audio untuk Voice Cloning**

✅ **DO:**
- Gunakan audio 3-12 detik
- Clear speech, minimal background noise
- Single speaker
- Ada sedikit silence di akhir (~0.5s)

❌ **DON'T:**
- Audio lebih dari 12 detik (akan di-trim)
- Multiple speakers
- Heavy background music
- Audio dengan echo/reverb yang kuat

### 2. **Text Input**

✅ **DO:**
```python
# Add punctuation untuk pauses natural
text = "Halo, nama saya John. Apa kabar hari ini?"

# Uppercase untuk spell letter-by-letter
text = "Model AI ini bernama GPT."
```

❌ **DON'T:**
```python
# No punctuation
text = "halo nama saya john apa kabar hari ini"

# Terlalu panjang (>500 chars, akan di-chunk otomatis tapi kurang optimal)
text = "very long text..." * 100
```

### 3. **Performance Optimization**

**Cold Start:**
- First request: ~10-15 detik (load model)
- Subsequent requests: ~1-3 detik (warm container)
- Container idle timeout: 5 menit (configurable)

**Warm Container:**
```python
# Set idle timeout lebih lama untuk production
container_idle_timeout=600,  # 10 menit
```

**Batching:**
```python
# Process multiple texts dalam satu container warm
client = F5TTSClient(API_URL)

for text in texts:
    result = client.synthesize(text)  # Reuse warm container
```

### 4. **Error Handling**

```python
import time

def generate_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = client.synthesize(text, timeout=120)

            if result["success"]:
                return result

            # API error, retry
            print(f"Attempt {attempt + 1} failed: {result['error']}")

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")

        # Exponential backoff
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return {"success": False, "error": "Max retries exceeded"}
```

### 5. **Cost Optimization**

```python
# Batch processing untuk efficiency
texts = ["text1", "text2", "text3", ...]

# Process dalam chunks untuk reuse warm container
CHUNK_SIZE = 10

for i in range(0, len(texts), CHUNK_SIZE):
    chunk = texts[i:i+CHUNK_SIZE]

    for text in chunk:
        result = client.synthesize(text)
        # Container stays warm between requests

    # Optional: sleep antara chunks
    time.sleep(1)
```

---

## 🔧 Troubleshooting

### ❌ Error: "Model not found"

**Problem:** Model `PapaRazi/Ijazah_Palsu_V2` tidak ditemukan.

**Solution:**
```python
# Option 1: Gunakan base model
MODEL_NAME = "SWivid/F5-TTS"

# Option 2: Check model name di Hugging Face
# https://huggingface.co/PapaRazi/Ijazah_Palsu_V2

# Option 3: Jika private, tambahkan HF token
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

### ❌ Error: "CUDA out of memory"

**Problem:** GPU memory insufficient.

**Solution:**
```python
# Option 1: Upgrade GPU
GPU_TYPE = "A10G"  # 24GB instead of T4's 16GB

# Option 2: Reduce batch size or sequence length
# (jika custom implementation)
```

### ❌ Error: "Blank audio output"

**Problem:** FFmpeg tidak terinstall atau audio processing error.

**Solution:**
```python
# Verify FFmpeg di image
.apt_install("git", "ffmpeg", "libsndfile1")

# Check remove_silence setting
remove_silence=False  # Disable jika masalah persist
```

### ❌ Slow cold starts

**Problem:** Container startup terlalu lama.

**Solution:**
```python
# Option 1: Pre-download model weights saat build
.run_commands(
    "python -c 'from f5_tts.api import F5TTS; F5TTS()'"
)

# Option 2: Increase idle timeout
container_idle_timeout=900,  # 15 menit

# Option 3: Use keep_warm untuk production
@app.function(
    keep_warm=1,  # Always keep 1 container warm
)
```

### ❌ Reference audio not working

**Problem:** Voice cloning tidak menggunakan reference voice.

**Solution:**
```python
# Ensure:
# 1. Audio format WAV
# 2. Duration 3-12 seconds
# 3. Base64 encoding correct
# 4. ref_text provided (recommended)

# Convert audio ke WAV jika perlu:
from pydub import AudioSegment
audio = AudioSegment.from_file("input.mp3")
audio.export("output.wav", format="wav")
```

---

## 💰 Cost Estimation

### Pricing Breakdown (Modal.com)

| Resource | Unit Cost |
|----------|-----------|
| T4 GPU | $0.000167/second = $0.60/hour |
| CPU (8 vCPU) | $0.0001/CPU-second |
| Memory (16 GB) | Included with GPU |
| Network egress | First 10GB free, then $0.10/GB |
| Storage (Volume) | $0.10/GB/month |

### Example Cost Calculation

**Scenario:** 1000 requests/day, avg 5 seconds inference each

```
Daily GPU time: 1000 × 5s = 5000s = 1.39 hours
Daily cost: 1.39 × $0.60 = $0.83

Monthly cost: $0.83 × 30 = $24.90

Plus:
- Storage (10GB model): $1.00/month
- Network (50GB/month): $4.00/month

Total: ~$30/month for 1000 requests/day
```

**Cost per request:** $0.03 cents/request

### Cost Comparison

| Provider | 1000 req/day | Notes |
|----------|--------------|-------|
| Modal (serverless) | $30/month | Pay-per-second, auto-scale |
| AWS SageMaker (dedicated) | $500/month | ml.g4dn.xlarge 24/7 |
| GCP Vertex AI | $470/month | Always-on instance |
| ElevenLabs API | $200/month | 500K chars ≈ 1000 requests |

**Modal saves 94% vs always-on GPU instances!**

---

## 📈 Monitoring & Logging

### View Logs

```bash
# Real-time logs
modal app logs f5-tts-inference --follow

# Filter by function
modal app logs f5-tts-inference --function F5TTSModel
```

### Metrics Dashboard

Visit: https://modal.com/apps/your-username/f5-tts-inference

Metrics available:
- Request count
- Success rate
- Average latency
- GPU utilization
- Cost per request

### Add Custom Logging

```python
import logging

@modal.enter()
def load_model(self):
    logging.info("Model loading started")
    # ... load model
    logging.info(f"Model loaded: {MODEL_NAME}")
```

---

## 🛑 Stop/Delete Deployment

### Temporary Stop

```bash
# Stop app (masih bisa restart)
modal app stop f5-tts-inference
```

### Delete Deployment

```bash
# Delete app completely
modal app delete f5-tts-inference
```

### Delete Volume (freed storage)

```bash
# Delete cached model weights
modal volume delete f5-tts-model-cache
```

---

## 🎓 Kesimpulan

Anda sekarang memiliki:

✅ F5-TTS model deployed di Modal.com dengan GPU T4
✅ RESTful API untuk TTS inference
✅ Zero-shot voice cloning capability
✅ Auto-scaling serverless infrastructure
✅ Pay-per-use pricing (no idle cost)
✅ Production-ready dengan monitoring

### Next Steps:

1. **Integrate ke aplikasi** - Gunakan API dari web/mobile app
2. **Fine-tune model** - Custom training untuk bahasa/voice spesifik
3. **Optimize performance** - Tune batch size, caching, dll
4. **Add features** - Speech speed control, emotion, dll
5. **Monitor usage** - Track costs dan optimize

---

## 📚 Resources

- **F5-TTS Paper:** https://huggingface.co/papers/2410.06885
- **F5-TTS GitHub:** https://github.com/SWivid/F5-TTS
- **Modal Docs:** https://modal.com/docs
- **Modal Examples:** https://github.com/modal-labs/modal-examples
- **Modal Discord:** https://discord.gg/modal

---

## 🤝 Support

Jika ada pertanyaan atau issue:

1. Check **Troubleshooting** section di atas
2. Review Modal logs: `modal app logs f5-tts-inference`
3. Check Modal Discord untuk community support
4. Open issue di GitHub repository

---

**Happy Inferencing!** 🎉

Created with ❤️ by Claude | 2025
