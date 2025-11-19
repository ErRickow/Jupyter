# 🎤 Unsloth TTS Inference Guide

Panduan lengkap untuk inference Text-to-Speech menggunakan Unsloth + SNAC vocoder.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [API Parameters](#api-parameters)
- [Voice Options](#voice-options)
- [Emotion Tags](#emotion-tags)
- [Code Examples](#code-examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Basic Request

```python
import requests
import base64

MODEL_URL = "https://model-<id>.api.baseten.co/production/predict"
API_KEY = "your_api_key"

response = requests.post(
    MODEL_URL,
    headers={"Authorization": f"Api-Key {API_KEY}"},
    json={
        "text": "Hello, this is a test!",
        "voice": "tara",
        "temperature": 0.7
    }
)

result = response.json()
audio_bytes = base64.b64decode(result["audio"])

with open("speech.wav", "wb") as f:
    f.write(audio_bytes)

print(f"✓ Generated {result['duration']:.2f}s of audio")
```

## 🎛️ API Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | Text to convert to speech |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | string | `"tara"` | Voice to use (see options below) |
| `temperature` | float | `0.7` | Sampling temperature (0.0-1.0) |
| `max_new_tokens` | int | `512` | Maximum tokens to generate |
| `emotion` | string | `""` | Emotion tag (see options below) |
| `return_format` | string | `"base64"` | Return format: "base64" or "array" |

### Response Format

```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAA...",
  "sample_rate": 24000,
  "duration": 2.45,
  "format": "wav",
  "encoding": "base64",
  "text": "tara: Hello, this is a test!"
}
```

## 🎤 Voice Options

Orpheus-TTS mendukung 8 voices berbeda:

| Voice | Gender | Characteristics |
|-------|--------|-----------------|
| **tara** | Female | Warm, professional (default) |
| **leah** | Female | Clear, energetic |
| **jess** | Female | Friendly, casual |
| **mia** | Female | Soft, gentle |
| **leo** | Male | Deep, authoritative |
| **dan** | Male | Clear, neutral |
| **zac** | Male | Young, dynamic |
| **zoe** | Female | Bright, expressive |

### Example: Different Voices

```python
voices = ["tara", "leo", "jess", "dan"]
text = "Welcome to our text to speech demo!"

for voice in voices:
    response = requests.post(
        MODEL_URL,
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json={"text": text, "voice": voice}
    )

    result = response.json()
    audio_bytes = base64.b64decode(result["audio"])

    with open(f"speech_{voice}.wav", "wb") as f:
        f.write(audio_bytes)

    print(f"✓ Generated {voice}: {result['duration']:.2f}s")
```

## 😊 Emotion Tags

Orpheus-TTS mendukung emotional expression dengan special tags:

| Tag | Effect | Example |
|-----|--------|---------|
| `<laugh>` | Laughing | "This is hilarious!" |
| `<sigh>` | Sighing | "I'm so tired..." |
| `<cough>` | Coughing | "Excuse me..." |
| `<gasp>` | Gasping | "Oh my!" |
| `<whisper>` | Whispering | "Keep it secret" |
| `<happy>` | Happy tone | "I'm so excited!" |
| `<sad>` | Sad tone | "That's unfortunate" |
| `<angry>` | Angry tone | "This is unacceptable!" |

### Usage

**Method 1: In text parameter**
```python
{
    "text": "<laugh> This is amazing!",
    "voice": "tara"
}
```

**Method 2: Separate emotion parameter**
```python
{
    "text": "This is amazing!",
    "voice": "tara",
    "emotion": "<laugh>"
}
```

### Example: Emotional Speech

```python
emotions = {
    "neutral": "",
    "happy": "<laugh> I'm so excited about this!",
    "contemplative": "<sigh> Let me think about that...",
    "surprised": "<gasp> I can't believe it!",
}

for emotion_name, text in emotions.items():
    response = requests.post(
        MODEL_URL,
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json={"text": text, "voice": "tara"}
    )

    result = response.json()
    # Save audio...
    print(f"✓ {emotion_name}: {result['duration']:.2f}s")
```

## 💻 Code Examples

### Python - Advanced Usage

```python
import requests
import base64
from pathlib import Path

class UnslothTTSClient:
    def __init__(self, model_url: str, api_key: str):
        self.model_url = model_url
        self.headers = {"Authorization": f"Api-Key {api_key}"}

    def synthesize(
        self,
        text: str,
        voice: str = "tara",
        temperature: float = 0.7,
        emotion: str = "",
        output_file: str = None
    ) -> dict:
        """Generate speech from text"""

        payload = {
            "text": text,
            "voice": voice,
            "temperature": temperature,
            "emotion": emotion,
            "return_format": "base64"
        }

        response = requests.post(
            self.model_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        # Save audio if output file specified
        if output_file:
            audio_bytes = base64.b64decode(result["audio"])
            Path(output_file).write_bytes(audio_bytes)

        return result

# Usage
client = UnslothTTSClient(
    model_url="https://model-xxx.api.baseten.co/production/predict",
    api_key="your_api_key"
)

result = client.synthesize(
    text="Hello world!",
    voice="tara",
    temperature=0.8,
    output_file="output.wav"
)

print(f"Duration: {result['duration']:.2f}s")
print(f"Sample rate: {result['sample_rate']} Hz")
```

### Node.js / TypeScript

```typescript
import axios from 'axios';
import fs from 'fs';

interface TTSRequest {
  text: string;
  voice?: string;
  temperature?: number;
  emotion?: string;
  return_format?: string;
}

interface TTSResponse {
  audio: string;
  sample_rate: number;
  duration: number;
  format: string;
  encoding: string;
  text: string;
}

class UnslothTTSClient {
  constructor(
    private modelUrl: string,
    private apiKey: string
  ) {}

  async synthesize(request: TTSRequest, outputFile?: string): Promise<TTSResponse> {
    const response = await axios.post<TTSResponse>(
      this.modelUrl,
      {
        voice: 'tara',
        temperature: 0.7,
        return_format: 'base64',
        ...request,
      },
      {
        headers: { 'Authorization': `Api-Key ${this.apiKey}` },
        timeout: 60000,
      }
    );

    if (outputFile) {
      const audioBuffer = Buffer.from(response.data.audio, 'base64');
      fs.writeFileSync(outputFile, audioBuffer);
    }

    return response.data;
  }
}

// Usage
const client = new UnslothTTSClient(
  'https://model-xxx.api.baseten.co/production/predict',
  'your_api_key'
);

const result = await client.synthesize(
  { text: 'Hello world!', voice: 'tara' },
  'output.wav'
);

console.log(`Duration: ${result.duration.toFixed(2)}s`);
```

### cURL

```bash
# Basic request
curl -X POST https://model-<id>.api.baseten.co/production/predict \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world!",
    "voice": "tara",
    "temperature": 0.7
  }' | jq -r '.audio' | base64 -d > output.wav

# With emotion
curl -X POST https://model-<id>.api.baseten.co/production/predict \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is amazing!",
    "voice": "leah",
    "emotion": "<laugh>",
    "temperature": 0.8
  }' | jq -r '.audio' | base64 -d > output_happy.wav
```

### Go

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
)

type TTSRequest struct {
    Text          string  `json:"text"`
    Voice         string  `json:"voice,omitempty"`
    Temperature   float64 `json:"temperature,omitempty"`
    Emotion       string  `json:"emotion,omitempty"`
    ReturnFormat  string  `json:"return_format,omitempty"`
}

type TTSResponse struct {
    Audio      string  `json:"audio"`
    SampleRate int     `json:"sample_rate"`
    Duration   float64 `json:"duration"`
    Format     string  `json:"format"`
    Encoding   string  `json:"encoding"`
    Text       string  `json:"text"`
}

func synthesizeSpeech(modelURL, apiKey, text, voice string) (*TTSResponse, error) {
    req := TTSRequest{
        Text:         text,
        Voice:        voice,
        Temperature:  0.7,
        ReturnFormat: "base64",
    }

    jsonData, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequest("POST", modelURL, bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("Authorization", "Api-Key "+apiKey)
    httpReq.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    resp, err := client.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result TTSResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return &result, nil
}

func saveAudio(audioB64, filename string) error {
    audioData, err := base64.StdEncoding.DecodeString(audioB64)
    if err != nil {
        return err
    }

    return os.WriteFile(filename, audioData, 0644)
}

func main() {
    result, err := synthesizeSpeech(
        "https://model-xxx.api.baseten.co/production/predict",
        "your_api_key",
        "Hello world!",
        "tara",
    )
    if err != nil {
        panic(err)
    }

    if err := saveAudio(result.Audio, "output.wav"); err != nil {
        panic(err)
    }

    fmt.Printf("Duration: %.2fs\n", result.Duration)
}
```

## 🐛 Troubleshooting

### Error: "No text provided"

**Cause:** Missing `text` parameter in request

**Solution:**
```python
# ❌ Wrong
{"voice": "tara"}

# ✅ Correct
{"text": "Hello!", "voice": "tara"}
```

### Error: Request timeout

**Cause:** Text too long atau model masih cold start

**Solution:**
- Reduce `max_new_tokens` (default: 512)
- Split long text into chunks
- Wait for warm-up (~30s first request)

### Audio quality issues

**Cause:** Temperature too high/low

**Solution:**
```python
# For consistent quality
{"text": "...", "temperature": 0.5}

# For more variation
{"text": "...", "temperature": 0.9}
```

### Voice doesn't match

**Cause:** Invalid voice name

**Solution:**
```python
# ❌ Wrong
{"voice": "sarah"}  # Not a valid voice

# ✅ Correct
{"voice": "tara"}  # Valid voice from list
```

### Emotion tags not working

**Cause:** Improper tag format

**Solution:**
```python
# ❌ Wrong
{"text": "laugh This is funny"}

# ✅ Correct
{"text": "<laugh> This is funny"}
```

## 📊 Performance Tips

### 1. Batch Processing

```python
texts = ["Text 1", "Text 2", "Text 3", ...]

# Sequential (slower)
for text in texts:
    result = client.synthesize(text)

# Parallel (faster)
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(client.synthesize, text) for text in texts]
    results = [f.result() for f in futures]
```

### 2. Caching

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_synthesize(text: str, voice: str):
    return client.synthesize(text, voice)
```

### 3. Optimize Token Count

```python
# Shorter texts = faster inference
long_text = "Very long paragraph..."

# Split into sentences
sentences = long_text.split(". ")
audios = [client.synthesize(s) for s in sentences]

# Concatenate audio files later
```

## 🔗 Resources

- [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) - Technical details
- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick deployment guide

---

**Happy TTS-ing! 🎉**
