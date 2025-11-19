# F5-TTS Indonesian Model Deployment (PapaRazi/Ijazah_Palsu_V2)

Deployment guide untuk Indonesian TTS model di Modal.com

## ⚠️ CRITICAL REQUIREMENTS

**Model PapaRazi/Ijazah_Palsu_V2 fine-tuned untuk Indonesian (95%)**

- ✅ Reference audio dalam **BAHASA INDONESIA** adalah **REQUIRED**
- ❌ **TIDAK BISA** pakai English default voice
- ❌ **TIDAK BISA** tanpa reference audio (akan error)

### Why Reference Audio is REQUIRED?

Model ini mengalami **"catastrophic forgetting"** setelah fine-tuning:
- Model **"lupa"** bahasa aslinya (English/Chinese)
- English reference audio → **Output NOISE saja**, bukan vocal
- F5-TTS adalah **voice cloning model**, bukan simple TTS
- Quality terbaik dengan **clear Indonesian reference audio**

## 🚀 Quick Start

### 1. Prepare Indonesian Reference Audio

Siapkan audio file dalam bahasa Indonesia (WAV format recommended):

```bash
# Example: Record your own voice saying something in Indonesian
# File: my_indonesian_voice.wav
# Content: "Halo, nama saya adalah contoh suara untuk referensi."
```

**Tips untuk reference audio yang bagus:**
- Duration: 3-10 detik
- Clear pronunciation (no background noise)
- Natural speaking pace
- WAV format (24kHz recommended)

### 2. Deploy to Modal

```bash
# Set Modal credentials
export MODAL_TOKEN_ID='your-token-id'
export MODAL_TOKEN_SECRET='your-token-secret'

# Deploy
modal deploy modal_f5_tts_inference.py
```

### 3. Use the API

**Python Client:**

```python
from modal_f5_tts_client import F5TTSClient

# Initialize client with your Modal URL
API_URL = "https://your-username--f5-tts-inference-tts-api.modal.run"
client = F5TTSClient(API_URL)

# Generate speech (HARUS dengan Indonesian reference audio!)
result = client.synthesize(
    text="Suatu hari nanti, suara ini mungkin tidak bisa dibedakan lagi dari suara manusia asli.",
    ref_audio_path="my_indonesian_voice.wav",  # REQUIRED: Indonesian audio
    ref_text="Halo, nama saya adalah contoh suara untuk referensi.",  # REQUIRED: Transcription
    output_path="output.wav"
)

if result["success"]:
    print(f"✅ Success! Duration: {result['duration']:.2f}s")
    print(f"   Saved to: {result['output_path']}")
else:
    print(f"❌ Error: {result['error']}")
```

**cURL Request:**

```bash
# First, encode your Indonesian reference audio to base64
REF_AUDIO_BASE64=$(base64 -w 0 my_indonesian_voice.wav)

# Send request
curl -X POST "https://your-username--f5-tts-inference-tts-api.modal.run" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Suatu hari nanti, suara ini mungkin tidak bisa dibedakan lagi dari suara manusia asli.\",
    \"ref_audio_base64\": \"$REF_AUDIO_BASE64\",
    \"ref_text\": \"Halo, nama saya adalah contoh suara untuk referensi.\",
    \"remove_silence\": true
  }"
```

## 📖 Model Information

**Model:** PapaRazi/Ijazah_Palsu_V2
**Base Model:** SWivid/F5-TTS
**Training:**
- Dataset: PapaRazi/id-tts-v2
- Size: ~70,000 samples (70 hours)
- Language: Indonesian (95%), English (5%)
- Training time: ~3 days
- Epochs: 34

**Model README:** https://huggingface.co/PapaRazi/Ijazah_Palsu_V2

## 🎧 Audio Quality Examples

From model README:

1. **Natural Sentence** (Excellent):
   - Text: "Suatu hari nanti, suara ini mungkin tidak bisa dibedakan lagi dari suara manusia asli."
   - [Listen](https://voca.ro/18y7FTzxcbta)

2. **Number Pronunciation** (Good):
   - Text: "Serius?! Tiket konsernya habis dalam waktu 3 menit?!"
   - [Listen](https://voca.ro/19daRAoMs0oD)

3. **Large Numbers** (Imperfect):
   - Text: "Masa cuma buat beli kursi kantor aja harus bayar Rp 2.500.000,-?! Gila sih itu!"
   - [Listen](https://voca.ro/1nbtoyUOGWJP)
   - ⚠️ Large number pronunciation masih tidak akurat

## 🐛 Troubleshooting

### Problem: Output hanya noise, bukan vocal

**Root Cause:** Reference audio bukan bahasa Indonesia

**Solution:**
```python
# ❌ WRONG: Using English reference audio
result = client.synthesize(
    text="Halo, apa kabar?",
    ref_audio_path="english_voice.wav",  # English audio
    ref_text="Hello, how are you?",
    ...
)
# Output: NOISE only (catastrophic forgetting)

# ✅ CORRECT: Using Indonesian reference audio
result = client.synthesize(
    text="Halo, apa kabar?",
    ref_audio_path="indonesian_voice.wav",  # Indonesian audio
    ref_text="Halo, nama saya Budi.",
    ...
)
# Output: Proper Indonesian speech
```

### Problem: "Missing required field: ref_audio_base64"

**Solution:** Reference audio is REQUIRED. Tidak ada "default voice mode".

```python
# You MUST provide Indonesian reference audio
result = client.synthesize(
    text="...",
    ref_audio_path="indonesian_voice.wav",  # REQUIRED
    ref_text="...",  # REQUIRED
)
```

### Problem: Model generates English pronunciation for Indonesian text

**Possible causes:**
1. Reference audio contains English speech
2. Reference text is in English
3. Mixed language in reference

**Solution:** Ensure BOTH ref_audio and ref_text are purely Indonesian.

## 📚 Technical Background

### Why Fine-tuned Models "Forget" Original Languages?

This is called **"catastrophic forgetting"** - a known issue in fine-tuning:

1. **Original F5-TTS** trained on English/Chinese data
2. **PapaRazi fine-tuned** on Indonesian data (70 hours)
3. **Model weights updated** to favor Indonesian pronunciation
4. **English capability degraded** (neural network "forgets" old patterns)

From F5-TTS GitHub issues:
> "After fine-tuning the original model with other languages, I can no longer speak Chinese and English properly... However, my newly tuned language is becoming more and more similar."

### Why Reference Audio Language Must Match?

F5-TTS uses reference audio for:
- Voice timbre/tone
- **Pronunciation patterns** (phonemes)
- Prosody (rhythm/intonation)

When you use English reference audio with Indonesian model:
- ✅ Voice timbre copied
- ❌ Pronunciation **FAILS** (model trained on Indonesian phonemes)
- ❌ Output = noise/garbled audio

## 🔗 Resources

- **Model Card:** https://huggingface.co/PapaRazi/Ijazah_Palsu_V2
- **Dataset:** PapaRazi/id-tts-v2
- **Base F5-TTS:** https://github.com/SWivid/F5-TTS
- **Modal Docs:** https://modal.com/docs

## 📝 License

Non-commercial use only (cc-by-nc-4.0)

## 👤 Credits

- **Model Author:** [PapaRazi](https://huggingface.co/PapaRazi)
- **Base Model:** [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **Deployment:** Claude (this implementation)
