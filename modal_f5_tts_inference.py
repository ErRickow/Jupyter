"""
F5-TTS Inference Deployment on Modal.com
=========================================

Deploy PapaRazi/Ijazah_Palsu_V2 - Indonesian fine-tuned F5-TTS model.

IMPORTANT:
- Model PapaRazi/Ijazah_Palsu_V2 adalah REQUIRED (fine-tuned untuk bahasa Indonesia)
- TIDAK ada fallback ke base model - model Indo ini HARUS digunakan
- Jika model gagal download/load, deployment akan error (by design)

Features:
- Download model PapaRazi dari HuggingFace
- GPU T4 untuk inference cepat
- RESTful API endpoints
- Zero-shot voice cloning
- Model caching untuk cold start yang lebih cepat
- Optimized untuk bahasa Indonesia

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
MODEL_NAME = "PapaRazi/Ijazah_Palsu_V2"  # Indonesian fine-tuned F5-TTS model (REQUIRED)
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

        # Download model dari HuggingFace - HARUS pakai PapaRazi (fine-tuned Indo)
        print(f"\n📥 Step 1: Downloading model from HuggingFace...")
        print(f"   REQUIRED: {MODEL_NAME} (Indonesian fine-tuned version)")

        # Download model - akan error jika gagal
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
        ckpt_files = list(model_path_obj.rglob("*.pt")) + \
                     list(model_path_obj.rglob("*.pth")) + \
                     list(model_path_obj.rglob("*.safetensors"))

        if not ckpt_files:
            error_msg = f"No checkpoint files (.pt/.pth/.safetensors) found in {model_path}"
            print(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        # Load dari checkpoint yang di-download
        ckpt_file = str(ckpt_files[0])
        print(f"   Using checkpoint: {ckpt_file}")

        # F5TTS API: model parameter (not model_type), ckpt_file for custom checkpoint
        self.tts = F5TTS(
            model="F5TTS_Base",  # Model config name
            ckpt_file=ckpt_file,  # Custom checkpoint path (PapaRazi Indonesian model)
            device=self.device,
        )

        print(f"✅ Successfully loaded Indonesian TTS model: {MODEL_NAME}")
        print("✅ Model loaded and ready for inference!")

    @modal.method()
    def generate_speech(
        self,
        text: str,
        ref_audio_base64: str,  # REQUIRED! F5-TTS needs reference audio
        ref_text: str,  # REQUIRED! Transcription of reference audio
        remove_silence: bool = True,
    ) -> dict:
        """
        Generate speech dari text menggunakan F5-TTS.

        IMPORTANT: F5-TTS adalah zero-shot TTS yang MEMERLUKAN reference audio!
        Tidak ada default voice - user HARUS provide reference audio.

        Args:
            text: Text yang ingin diubah menjadi speech
            ref_audio_base64: Reference audio dalam format base64 (REQUIRED!)
            ref_text: Transcription dari reference audio (REQUIRED!)
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

        # Validate required parameters
        if not ref_audio_base64:
            error_msg = "ref_audio_base64 is REQUIRED! F5-TTS needs reference audio for voice cloning."
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
            }

        if not ref_text:
            error_msg = "ref_text is REQUIRED! Provide transcription of reference audio."
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
            }

        # Handle reference audio
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

    IMPORTANT: F5-TTS requires reference audio for voice cloning!

    Request body:
    {
        "text": "Text to synthesize (REQUIRED)",
        "ref_audio_base64": "base64 encoded WAV audio (REQUIRED!)",
        "ref_text": "Transcription of reference audio (REQUIRED!)",
        "remove_silence": true
    }

    Response (Success):
    {
        "audio_base64": "base64 encoded generated audio",
        "sample_rate": 24000,
        "duration": 3.5,
        "success": true
    }

    Response (Error):
    {
        "success": false,
        "error": "Error message"
    }
    """
    model = F5TTSModel()

    # Validate required fields
    text = data.get("text")
    if not text:
        return {
            "success": False,
            "error": "Missing required field: text"
        }

    ref_audio_base64 = data.get("ref_audio_base64")
    if not ref_audio_base64:
        return {
            "success": False,
            "error": "Missing required field: ref_audio_base64. F5-TTS needs reference audio for voice cloning!"
        }

    ref_text = data.get("ref_text")
    if not ref_text:
        return {
            "success": False,
            "error": "Missing required field: ref_text. Provide transcription of reference audio!"
        }

    result = model.generate_speech.remote(
        text=text,
        ref_audio_base64=ref_audio_base64,
        ref_text=ref_text,
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

    NOTE: F5-TTS requires reference audio! You need to provide:
    1. ref_audio_path: Path to reference audio WAV file (3-12 seconds)
    2. ref_text: Transcription of the reference audio
    """
    import sys

    print("🧪 Testing F5-TTS inference...")
    print("\n⚠️  IMPORTANT: F5-TTS requires reference audio for voice cloning!")
    print("Please provide reference audio file to test.")
    print("\nUsage:")
    print("  1. Prepare reference audio (WAV, 3-12 seconds)")
    print("  2. Prepare transcription of reference audio")
    print("  3. Update this test function with your reference audio path")
    print("\nExample:")
    print('  ref_audio_path = "reference.wav"')
    print('  ref_text = "This is the transcription of reference audio"')

    # Test configuration - UPDATE THESE!
    ref_audio_path = None  # TODO: Set to your reference audio path
    ref_text = None  # TODO: Set to transcription
    test_text = "Halo, ini adalah test dari model F5-TTS untuk text to speech dalam bahasa Indonesia."

    if not ref_audio_path or not ref_text:
        print("\n❌ Test skipped: No reference audio provided")
        print("   Update ref_audio_path and ref_text in the test() function")
        print("\nFor production usage, see modal_f5_tts_client.py")
        return

    # Read reference audio
    print(f"\n📁 Reading reference audio: {ref_audio_path}")
    try:
        with open(ref_audio_path, "rb") as f:
            ref_audio_bytes = f.read()
        ref_audio_base64 = base64.b64encode(ref_audio_bytes).decode()
    except FileNotFoundError:
        print(f"❌ Error: Reference audio file not found: {ref_audio_path}")
        return

    # Initialize model
    model = F5TTSModel()

    # Generate speech
    print(f"\n🔊 Generating speech with voice cloning...")
    result = model.generate_speech.remote(
        text=test_text,
        ref_audio_base64=ref_audio_base64,
        ref_text=ref_text,
        remove_silence=True,
    )

    if result["success"]:
        print(f"\n✅ Test successful!")
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
        print(f"\n❌ Test failed: {result.get('error')}")

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

v5 (REFERENCE AUDIO REQUIRED):
- ✅ Fix TypeError: ref_file cannot be None
- ✅ Make ref_audio_base64 and ref_text REQUIRED parameters
- ✅ Add validation untuk required fields di API endpoint
- ✅ Update test function dengan reference audio example
- ✅ Better error messages untuk missing reference audio
- ✅ Clarify bahwa F5-TTS adalah zero-shot TTS (needs reference)

v4 (INDONESIAN REQUIRED):
- ✅ REMOVE fallback ke base model - PapaRazi model adalah REQUIRED
- ✅ Model PapaRazi/Ijazah_Palsu_V2 fine-tuned untuk bahasa Indonesia
- ✅ Fail fast jika model gagal download/load (no silent fallback)
- ✅ Update documentation untuk clarify Indonesian-only model
- ✅ Better error messages untuk checkpoint tidak ditemukan

v3 (FIXED API):
- ✅ Fix F5TTS API call - use 'model' parameter instead of 'model_type'
- ✅ Correct initialization: F5TTS(model="F5TTS_Base", ckpt_file=path)
- ✅ Add device parameter to F5TTS initialization

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
