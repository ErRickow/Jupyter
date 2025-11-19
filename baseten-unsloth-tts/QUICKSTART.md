# 🚀 Quick Start Guide - 5 Menit dari Zero ke Production!

## Prerequisites

1. **Baseten Account** - Daftar gratis di [baseten.co](https://baseten.co)
2. **Python 3.11+**
3. **Minimal T4 GPU** untuk deployment

## Step 1: Install Truss (30 detik)

```bash
pip install truss
```

## Step 2: Login ke Baseten (30 detik)

```bash
truss login
```

Atau gunakan API key:
```bash
export BASETEN_API_KEY="your_api_key_here"
```

Dapatkan API key di: https://app.baseten.co/settings/api_keys

## Step 3: Deploy! (5-10 menit)

### Opsi A: Otomatis dengan Script

```bash
cd baseten-unsloth-tts
./deploy.sh
```

### Opsi B: Manual

```bash
cd baseten-unsloth-tts
truss push
```

**Output yang akan Anda dapatkan:**
```
✅ Building model...
✅ Deploying to Baseten...
✅ Model deployed successfully!

📋 Model Details:
   Model ID: model-abc123xyz
   URL: https://model-abc123xyz.api.baseten.co/production/predict
   Status: ACTIVE
```

## Step 4: Test Inference (1 menit)

### Setup Environment

```bash
export BASETEN_MODEL_URL="https://model-<your-id>.api.baseten.co/production/predict"
export BASETEN_API_KEY="your_api_key"
```

### Run Test

```bash
python test_inference.py
```

**Output:**
```
============================================================
🎤 Unsloth TTS Inference Test
============================================================

📝 Input text: Halo, ini adalah test text to speech...
🎯 Model URL: https://model-xxx.api.baseten.co/production/predict
💾 Output file: output_20250119_143022.wav

🚀 Sending request to Baseten...

✅ Response received!
   Sample rate: 24000 Hz
   Duration: 2.45 seconds
   Format: wav

💾 Audio saved to: output_20250119_143022.wav
   File size: 117.32 KB

============================================================
✨ Test completed successfully!
============================================================
```

## Step 5: Integrate ke Aplikasi Anda

### Python

```python
import requests
import base64

MODEL_URL = "https://model-<id>.api.baseten.co/production/predict"
API_KEY = "your_api_key"

def text_to_speech(text):
    response = requests.post(
        MODEL_URL,
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json={"text": text, "temperature": 0.7}
    )

    result = response.json()
    audio_bytes = base64.b64decode(result["audio"])

    with open("speech.wav", "wb") as f:
        f.write(audio_bytes)

    return result["duration"]

# Usage
duration = text_to_speech("Halo dunia!")
print(f"Generated {duration:.2f}s of audio")
```

### Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

const MODEL_URL = 'https://model-<id>.api.baseten.co/production/predict';
const API_KEY = 'your_api_key';

async function textToSpeech(text) {
  const response = await axios.post(
    MODEL_URL,
    { text, temperature: 0.7 },
    { headers: { 'Authorization': `Api-Key ${API_KEY}` } }
  );

  const audioBuffer = Buffer.from(response.data.audio, 'base64');
  fs.writeFileSync('speech.wav', audioBuffer);

  return response.data.duration;
}

// Usage
textToSpeech('Hello world!')
  .then(duration => console.log(`Generated ${duration}s of audio`));
```

### cURL

```bash
curl -X POST https://model-<id>.api.baseten.co/production/predict \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Halo, ini test TTS!",
    "temperature": 0.7,
    "return_format": "base64"
  }' | jq -r '.audio' | base64 -d > output.wav
```

## 🎛️ Advanced: Customize Model

### 1. Gunakan Model Fine-tuned Anda

Edit `model/model.py`:

```python
# Line 41
model_name = "your-username/your-finetuned-tts"  # HuggingFace path
```

### 2. Adjust Resources

Edit `config.yaml`:

```yaml
resources:
  accelerator: A10G  # Upgrade ke A10G atau A100
  memory: 32Gi       # Increase memory
```

### 3. Add Emotion Support

Request dengan emotion tags:

```python
response = requests.post(
    MODEL_URL,
    headers={"Authorization": f"Api-Key {API_KEY}"},
    json={
        "text": "This is amazing!",
        "emotion": "<happy>",  # atau <laugh>, <sigh>, <whisper>
        "temperature": 0.8
    }
)
```

### 4. Streaming Response (untuk real-time)

Untuk latency ultra-rendah, gunakan WebSocket streaming (advanced):

```python
# Modify model.py untuk streaming chunks
# See: https://docs.baseten.co/examples/text-to-speech
```

## 📊 Monitoring & Debugging

### Check Model Status

```bash
truss watch
```

### View Logs

Di Baseten dashboard: https://app.baseten.co/models

Filter by:
- Request logs
- Error logs
- Performance metrics

### Common Issues

**1. Out of Memory**
```yaml
# Reduce model size atau increase GPU
resources:
  accelerator: A10G  # Upgrade dari T4
```

**2. Slow Inference**
```python
# Pastikan fast inference enabled
FastLanguageModel.for_inference(self._model)
```

**3. Model Not Loading**
```python
# Check HuggingFace token untuk private models
secrets:
  hf_token: "your_token"
```

## 🎉 Done!

Anda sekarang punya TTS API production-ready yang:
- ✅ 2x faster dengan Unsloth
- ✅ 50% less memory
- ✅ Perfect untuk T4 GPU
- ✅ Scalable di Baseten infrastructure

## Next Steps

1. **Benchmark performance**: Test dengan berbagai input lengths
2. **Fine-tune model**: Train dengan voice data custom Anda
3. **Optimize latency**: Experiment dengan model size trade-offs
4. **Add features**: Voice cloning, multi-language, dll

## 📚 Resources

- [Full README](README.md)
- [Baseten Docs](https://docs.baseten.co/)
- [Unsloth Docs](https://docs.unsloth.ai/)
- [Model Monitoring](https://app.baseten.co/)

---

**Happy shipping! 🚀**
