"""
F5-TTS Inference Deployment on Modal.com (FIXED VERSION)
=========================================================

Deploy PapaRazi/Ijazah_Palsu_V2 (F5-TTS based) Text-to-Speech model for serverless inference.

FIXED: Proper model download dari HuggingFace sebelum loading

Features:
- Download model dari HuggingFace dulu
- GPU T4 untuk inference cepat
- RESTful API endpoints
- Zero-shot voice cloning
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
MODEL_NAME = "PapaRazi/Ijazah_Palsu_V2"  # HuggingFace model ID
BASE_MODEL = "SWivid/F5-TTS"  # Fallback ke base model jika custom model tidak tersedia
GPU_TYPE = "T4"  # GPU type sesuai permintaan
TIMEOUT_MINUTES = 10

# Paths
MODEL_CACHE_DIR = "/models"
HF_CACHE_DIR = "/root/.cache/huggingface"

# ============================================================================
# CONTAINER IMAGE SETUP
# ============================================================================

# Build container image dengan dependencies yang diperlukan
f5_tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")  # FFmpeg required untuk audio processing
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
        "huggingface-hub>=0.20.0",
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
# HELPER FUNCTION: DOWNLOAD MODEL
# ============================================================================

def download_model_from_hf(model_id: str, cache_dir: str) -> str:
    """
    Download model dari HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "PapaRazi/Ijazah_Palsu_V2")
        cache_dir: Directory untuk cache model

    Returns:
        str: Path ke downloaded model
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    import os

    print(f"📥 Downloading model from HuggingFace: {model_id}")

    try:
        # Try snapshot_download untuk download semua files
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            # token=os.getenv("HF_TOKEN"),  # Uncomment jika model private
        )
        print(f"✅ Model downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise

# ============================================================================
# TTS MODEL CLASS
# ============================================================================

@app.cls(
    image=f5_tts_image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_MINUTES * 60,
    container_idle_timeout=300,  # Keep warm untuk 5 menit setelah last request
    volumes={
        MODEL_CACHE_DIR: model_cache_vol,
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
        Download dan load model saat container start (warm start optimization).
        Fungsi ini dipanggil sekali per container lifecycle.
        """
        import torch
        import os
        from pathlib import Path

        print(f"🚀 Initializing F5-TTS model: {MODEL_NAME}")

        # Check GPU availability first
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎮 Using device: {self.device}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.2f} GB")

        # Download model dari HuggingFace
        model_loaded = False
        model_path = None

        try:
            # Try download custom model
            print(f"\n📥 Step 1: Downloading model from HuggingFace...")
            model_path = download_model_from_hf(MODEL_NAME, MODEL_CACHE_DIR)

            # Check if model files exist
            model_path_obj = Path(model_path)
            print(f"\n📁 Model files in {model_path}:")

            if model_path_obj.exists():
                files = list(model_path_obj.rglob("*"))[:20]  # Show first 20 files
                for f in files:
                    if f.is_file():
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"   - {f.name} ({size_mb:.2f} MB)")

            # Load model dengan F5-TTS
            print(f"\n🔄 Step 2: Loading model into memory...")

            from f5_tts.api import F5TTS

            # Check for model checkpoint files
            ckpt_files = list(model_path_obj.rglob("*.pt")) + list(model_path_obj.rglob("*.pth")) + list(model_path_obj.rglob("*.safetensors"))

            if ckpt_files:
                # Load dari checkpoint yang di-download
                ckpt_file = str(ckpt_files[0])
                print(f"   Using checkpoint: {ckpt_file}")
                self.tts = F5TTS(model_type="custom", ckpt_file=ckpt_file)
                model_loaded = True
                print(f"✅ Successfully loaded custom model: {MODEL_NAME}")
            else:
                print(f"⚠️  No checkpoint files found in downloaded model")
                raise FileNotFoundError("No model checkpoint found")

        except Exception as e:
            # Fallback ke base model
            print(f"\n⚠️  Could not load custom model: {e}")
            print(f"🔄 Falling back to base model: {BASE_MODEL}")

            try:
                from f5_tts.api import F5TTS
                self.tts = F5TTS(model_type="F5-TTS")  # Use base F5-TTS model
                model_loaded = True
                print(f"✅ Successfully loaded base F5-TTS model")
            except Exception as base_error:
                print(f"❌ Failed to load base model: {base_error}")
                raise

        if not model_loaded:
            raise RuntimeError("Failed to load any TTS model")

        print("\n✅ Model loaded and ready for inference!")

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
        import os

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
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
            }

        finally:
            # Cleanup temporary files
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

3. (Optional) SETUP HF TOKEN jika model private:
   modal secret create huggingface-secret HF_TOKEN=hf_xxxxx

4. DEPLOY MODEL:
   modal deploy modal_f5_tts_inference.py

5. TEST LOCALLY:
   modal run modal_f5_tts_inference.py

6. MENGGUNAKAN API:
   Gunakan modal_f5_tts_client.py atau curl/requests

CHANGELOG:
==========

v2 (FIXED):
- ✅ Download model dari HuggingFace dengan snapshot_download
- ✅ Proper model caching di Modal Volume
- ✅ Better error handling dan fallback
- ✅ Show downloaded model files
- ✅ Support untuk checkpoint files (.pt, .pth, .safetensors)
- ✅ More verbose logging untuk debugging

v1:
- Initial version (buggy - tidak download model dulu)
"""
