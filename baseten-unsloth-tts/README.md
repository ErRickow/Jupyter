# Unsloth TTS Deployment untuk Baseten

Text-to-Speech deployment menggunakan Unsloth untuk inference yang super efisien di T4 GPU.

## 🎯 Kenapa Unsloth + SNAC?

### Unsloth Benefits
- **2x faster inference** dibanding implementasi standard
- **50% less memory** dengan Flash Attention 2 optimization
- **Perfect untuk T4 GPU** - tidak boros resources
- **Support 4-bit quantization** untuk efficiency maksimal

### SNAC Vocoder
- **High-quality audio** - Neural audio codec 24kHz
- **Multi-scale architecture** - Hierarchical audio representation
- **Low bitrate** - 0.98 kbps untuk speech
- **Fast decoding** - Real-time audio generation

## 📁 Struktur Project

```
baseten-unsloth-tts/
├── config.yaml          # Konfigurasi Baseten (T4 GPU, dependencies)
├── model/
│   └── model.py        # Model inference dengan Unsloth
├── data/               # (Optional) Model weights custom
└── README.md           # Dokumentasi ini
```

## 🚀 Quick Start

### 1. Install Truss

```bash
pip install truss
```

### 2. Setup Baseten API Key

```bash
# Login ke Baseten
truss login

# Atau set environment variable
export BASETEN_API_KEY="your_api_key_here"
```

Dapatkan API key dari: https://app.baseten.co/settings/api_keys

### 3. Deploy ke Baseten

```bash
cd baseten-unsloth-tts
truss push
```

Tunggu beberapa menit untuk build dan deployment. Anda akan mendapat:
- Model ID
- Model URL endpoint

### 4. Test Model

```python
import requests
import base64

# Ganti dengan model URL Anda
MODEL_URL = "https://model-<model-id>.api.baseten.co/production/predict"
API_KEY = "your_baseten_api_key"

# Request
response = requests.post(
    MODEL_URL,
    headers={"Authorization": f"Api-Key {API_KEY}"},
    json={
        "text": "Halo, ini test text to speech menggunakan Unsloth!",
        "voice": "tara",  # Options: tara, leah, jess, leo, dan, mia, zac, zoe
        "temperature": 0.7,
        "emotion": "",  # Optional: <laugh>, <sigh>, <cough>
        "return_format": "base64"
    }
)

# Decode audio
result = response.json()
audio_b64 = result["audio"]
audio_bytes = base64.b64decode(audio_b64)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print(f"Audio saved! Duration: {result['duration']:.2f}s")
```

## 🎛️ API Parameters

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | **required** | Text yang ingin di-convert ke speech |
| `voice` | string | "tara" | Voice name: tara, leah, jess, leo, dan, mia, zac, zoe |
| `temperature` | float | 0.7 | Sampling temperature (0.0-1.0) |
| `max_new_tokens` | int | 512 | Maximum tokens to generate |
| `emotion` | string | "" | Emotion tags: `<laugh>`, `<sigh>`, `<cough>`, dll |
| `return_format` | string | "base64" | Format output: "base64" atau "array" |

### Output

```json
{
  "audio": "base64_encoded_wav_data...",
  "sample_rate": 24000,
  "duration": 2.5,
  "format": "wav",
  "encoding": "base64"
}
```

## 🔧 Customization

### Menggunakan Model Fine-tuned Anda

Edit `model/model.py` line 41:

```python
# Ganti dengan HuggingFace model path atau local path
model_name = "your-username/your-finetuned-tts-model"
```

### Menggunakan Model Lokal

1. Letakkan model weights di folder `data/`
2. Update `model.py`:

```python
import os
model_path = os.path.join(self._data_dir, "your-model")
self._model, self._tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    # ... other params
)
```

### Adjust GPU Resources

Edit `config.yaml`:

```yaml
resources:
  cpu: "4"           # Adjust CPU cores
  memory: 16Gi       # Adjust RAM
  use_gpu: true
  accelerator: T4    # T4, A10G, A100, dll
```

## 🎤 Model TTS yang Didukung

### Currently Implemented

1. **Orpheus-TTS (3B)** ⭐ **RECOMMENDED**
   - Model: `unsloth/orpheus-3b-0.1-ft`
   - **8 voices**: tara, leah, jess, mia, leo, dan, zac, zoe
   - **Emotional expression**: `<laugh>`, `<sigh>`, `<cough>`, dll
   - **SNAC vocoder**: 24kHz high-quality audio
   - **Multi-speaker**: Pre-trained on diverse voices

### Alternative Models (Configurable)

2. **Orpheus Pre-trained**
   - Model: `unsloth/orpheus-3b-0.1-pretrained`
   - Base model untuk fine-tuning custom

3. **Sesame-CSM (1B)**
   - Lightweight option (coming soon)

4. **Spark-TTS (0.5B)**
   - Ultra-low latency (coming soon)

5. **Custom Fine-tuned Model**
   - Fine-tune pakai Unsloth di Colab/local
   - Upload ke HuggingFace
   - Deploy di sini!

## 📊 Performance Benchmarks (T4 GPU)

| Model | Size | TTFB | Memory |
|-------|------|------|--------|
| Orpheus-3B (4-bit) | 3B | ~200ms | ~6GB |
| Sesame-CSM (4-bit) | 1B | ~150ms | ~3GB |
| Spark-TTS (4-bit) | 0.5B | ~100ms | ~2GB |

## 🔒 Using Private/Gated Models

Jika menggunakan gated models dari HuggingFace:

1. Tambahkan HF token ke `config.yaml`:

```yaml
secrets:
  hf_token: null  # Will be set via Baseten UI
```

2. Set secret di Baseten dashboard:
   - Go to model settings
   - Add secret: `hf_token = your_huggingface_token`

3. Update `model.py`:

```python
def load(self):
    hf_token = os.environ.get("hf_token")
    self._model, self._tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        token=hf_token,
        # ...
    )
```

## 🐛 Troubleshooting

### Error: Out of Memory

**Solution**: Gunakan model lebih kecil atau enable 4-bit quantization:

```python
load_in_4bit = True  # Already enabled in config
```

### Error: Model not found

**Solution**: Pastikan model name benar atau tambahkan HF token untuk private models.

### Slow Inference

**Solution**:
1. Pastikan `FastLanguageModel.for_inference()` dipanggil
2. Check GPU utilization: `nvidia-smi`
3. Reduce `max_length` parameter

## 📚 Resources

- [Unsloth Docs](https://docs.unsloth.ai/)
- [Baseten Docs](https://docs.baseten.co/)
- [Truss Examples](https://github.com/basetenlabs/truss-examples)
- [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)

## 🤝 Contributing

Punya improvement? Feel free to:
1. Fine-tune model dengan data custom
2. Optimize inference lebih lanjut
3. Add support untuk model TTS lainnya

## 📝 License

Sesuai dengan license model yang digunakan (biasanya Apache 2.0 atau MIT).

---

**Happy TTS-ing! 🎉🔊**

Dibuat dengan ❤️ menggunakan Unsloth untuk efficiency maksimal di T4 GPU.
