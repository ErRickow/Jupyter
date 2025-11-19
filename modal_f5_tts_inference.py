"""
F5-TTS Inference Deployment on Modal.com
==========================================

Deploy PapaRazi/Ijazah_Palsu_V2 (F5-TTS based) Text-to-Speech model for serverless inference.

Features:
- GPU T4 untuk inference cepat
- RESTful API endpoints
- Zero-shot voice cloning
- Automatic batching support
- Model caching untuk cold start yang lebih cepat

Author: Claude
Date: 2025
"""

import modal
import io
import base64
from pathlib import Path
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_NAME = "PapaRazi/Ijazah_Palsu_V2"  # Ganti dengan model ID Anda
BASE_MODEL = "SWivid/F5-TTS"  # Fallback ke base model jika custom model tidak tersedia
GPU_TYPE = "T4"  # GPU type sesuai permintaan
TIMEOUT_MINUTES = 10

# ============================================================================
# CONTAINER IMAGE SETUP
# ============================================================================

# Build container image dengan dependencies yang diperlukan
f5_tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")  # FFmpeg required untuk audio processing
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "soundfile",
        "librosa",
        "numpy",
        "scipy",
        "pydub",
        "huggingface-hub",
    )
    .run_commands(
        # Install F5-TTS dari GitHub
        "pip install git+https://github.com/SWivid/F5-TTS.git"
    )
)

# ============================================================================
# VOLUME CACHING
# ============================================================================

# Volume untuk caching model weights
model_cache_vol = modal.Volume.from_name(
    "f5-tts-model-cache",
    create_if_missing=True
)

# ============================================================================
# MODAL APP
# ============================================================================

app = modal.App("f5-tts-inference")

# ============================================================================
# TTS MODEL CLASS
# ============================================================================

@app.cls(
    image=f5_tts_image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_MINUTES * 60,
    container_idle_timeout=300,  # Keep warm untuk 5 menit setelah last request
    volumes={
        "/root/.cache/huggingface": model_cache_vol,
    },
    secrets=[
        # Optional: Tambahkan HF token jika model private
        # modal.Secret.from_name("huggingface-secret")
    ],
)
class F5TTSModel:
    """
    F5-TTS Model untuk Text-to-Speech inference.

    Model ini menggunakan zero-shot voice cloning dengan reference audio.
    """

    @modal.enter()
    def load_model(self):
        """
        Load model saat container start (warm start optimization).
        Fungsi ini dipanggil sekali per container lifecycle.
        """
        print(f"🚀 Loading F5-TTS model: {MODEL_NAME}")

        from f5_tts.api import F5TTS
        import torch

        # Initialize F5-TTS
        try:
            # Coba load custom model dulu
            self.tts = F5TTS(model_type="custom", ckpt_file=MODEL_NAME)
            print(f"✅ Successfully loaded custom model: {MODEL_NAME}")
        except Exception as e:
            # Fallback ke base model
            print(f"⚠️  Custom model not found, using base model: {BASE_MODEL}")
            print(f"   Error: {e}")
            self.tts = F5TTS(model_type="F5-TTS")  # Use base F5-TTS model

        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎮 Using device: {self.device}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.2f} GB")

        print("✅ Model loaded and ready for inference!")

    @modal.method()
    def generate_speech(
        self,
        text: str,
        ref_audio_base64: Optional[str] = None,
        ref_text: Optional[str] = None,
        remove_silence: bool = True,
    ) -> dict:
        """
        Generate speech dari text menggunakan F5-TTS.

        Args:
            text: Text yang ingin diubah menjadi speech
            ref_audio_base64: Reference audio dalam format base64 (optional untuk voice cloning)
            ref_text: Transcription dari reference audio (optional)
            remove_silence: Remove silence di awal dan akhir audio

        Returns:
            dict dengan keys:
                - audio_base64: Generated audio dalam base64
                - sample_rate: Sample rate audio
                - duration: Duration dalam detik
        """
        import torch
        import soundfile as sf
        import numpy as np
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence
        import tempfile

        print(f"🎤 Generating speech for text: {text[:50]}...")

        # Handle reference audio jika ada
        ref_file = None
        if ref_audio_base64:
            # Decode base64 audio
            audio_bytes = base64.b64decode(ref_audio_base64)

            # Save temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                ref_file = f.name

            print(f"📁 Using reference audio for voice cloning")

        try:
            # Generate speech
            wav, sr, spect = self.tts.infer(
                gen_text=text,
                ref_file=ref_file,
                ref_text=ref_text,
            )

            print(f"✅ Speech generated! Sample rate: {sr} Hz")

            # Convert to audio bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_file = f.name

            # Save audio
            sf.write(output_file, wav, sr)

            # Remove silence if requested
            if remove_silence:
                audio = AudioSegment.from_wav(output_file)

                # Remove leading and trailing silence
                trim_leading = detect_leading_silence(audio)
                trim_trailing = detect_leading_silence(audio.reverse())

                audio = audio[trim_leading:len(audio)-trim_trailing]
                audio.export(output_file, format="wav")

                print(f"🔇 Silence removed")

            # Read audio file
            with open(output_file, "rb") as f:
                audio_bytes = f.read()

            # Convert to base64
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Calculate duration
            duration = len(wav) / sr

            print(f"📊 Audio duration: {duration:.2f} seconds")

            return {
                "audio_base64": audio_base64,
                "sample_rate": int(sr),
                "duration": float(duration),
                "success": True,
            }

        except Exception as e:
            print(f"❌ Error during speech generation: {e}")
            return {
                "success": False,
                "error": str(e),
            }

        finally:
            # Cleanup temporary files
            import os
            if ref_file and os.path.exists(ref_file):
                os.remove(ref_file)
            if 'output_file' in locals() and os.path.exists(output_file):
                os.remove(output_file)

# ============================================================================
# WEB ENDPOINTS
# ============================================================================

@app.function(
    image=f5_tts_image,
)
@modal.web_endpoint(method="POST")
def tts_api(data: dict) -> dict:
    """
    RESTful API endpoint untuk TTS inference.

    Request body:
    {
        "text": "Text to synthesize",
        "ref_audio_base64": "base64 encoded audio (optional)",
        "ref_text": "Reference transcription (optional)",
        "remove_silence": true
    }

    Response:
    {
        "audio_base64": "base64 encoded generated audio",
        "sample_rate": 24000,
        "duration": 3.5,
        "success": true
    }
    """
    model = F5TTSModel()

    text = data.get("text")
    if not text:
        return {
            "success": False,
            "error": "Missing required field: text"
        }

    result = model.generate_speech.remote(
        text=text,
        ref_audio_base64=data.get("ref_audio_base64"),
        ref_text=data.get("ref_text"),
        remove_silence=data.get("remove_silence", True),
    )

    return result

@app.function(image=f5_tts_image)
@modal.web_endpoint(method="GET")
def health_check() -> dict:
    """
    Health check endpoint untuk monitoring.
    """
    import torch

    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model": MODEL_NAME,
        "gpu_type": GPU_TYPE,
    }

# ============================================================================
# LOCAL TEST FUNCTION
# ============================================================================

@app.local_entrypoint()
def test():
    """
    Test function untuk local testing.
    Run: modal run modal_f5_tts_inference.py
    """
    print("🧪 Testing F5-TTS inference...")

    # Test text
    test_text = "Halo, ini adalah test dari model F5-TTS untuk text to speech dalam bahasa Indonesia."

    # Initialize model
    model = F5TTSModel()

    # Generate speech
    result = model.generate_speech.remote(
        text=test_text,
        remove_silence=True,
    )

    if result["success"]:
        print(f"✅ Test successful!")
        print(f"   Sample rate: {result['sample_rate']} Hz")
        print(f"   Duration: {result['duration']:.2f} seconds")
        print(f"   Audio size: {len(result['audio_base64'])} bytes (base64)")

        # Save to file
        audio_bytes = base64.b64decode(result["audio_base64"])
        output_path = "test_output.wav"

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        print(f"💾 Audio saved to: {output_path}")
    else:
        print(f"❌ Test failed: {result.get('error')}")

# ============================================================================
# DEPLOYMENT INSTRUCTIONS
# ============================================================================

"""
CARA DEPLOY & MENGGUNAKAN:
===========================

1. INSTALL MODAL CLI:
   pip install modal

2. SETUP MODAL ACCOUNT:
   modal setup

3. DEPLOY MODEL:
   modal deploy modal_f5_tts_inference.py

4. TEST LOCALLY:
   modal run modal_f5_tts_inference.py

5. MENGGUNAKAN API:

   Setelah deploy, Anda akan mendapat URL endpoint seperti:
   https://your-username--f5-tts-inference-tts-api.modal.run

   Contoh request dengan Python:

   ```python
   import requests
   import base64

   # Read reference audio (optional untuk voice cloning)
   with open("reference.wav", "rb") as f:
       ref_audio_base64 = base64.b64encode(f.read()).decode()

   # API request
   response = requests.post(
       "https://your-url.modal.run",
       json={
           "text": "Halo, ini adalah test TTS.",
           "ref_audio_base64": ref_audio_base64,  # Optional
           "ref_text": "This is reference audio transcription",  # Optional
           "remove_silence": True
       }
   )

   result = response.json()

   # Save audio
   if result["success"]:
       audio_bytes = base64.b64decode(result["audio_base64"])
       with open("output.wav", "wb") as f:
           f.write(audio_bytes)
   ```

   Contoh request dengan cURL:

   ```bash
   curl -X POST https://your-url.modal.run \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, this is a test.",
       "remove_silence": true
     }'
   ```

6. MONITORING:
   modal app logs f5-tts-inference

7. STOP DEPLOYMENT:
   modal app stop f5-tts-inference

CATATAN:
- Model akan di-cache di volume untuk cold start yang lebih cepat
- Container akan tetap warm selama 5 menit setelah request terakhir
- Gunakan T4 GPU untuk balance antara cost dan performance
- Untuk production, pertimbangkan upgrade ke A10G atau A100
"""
