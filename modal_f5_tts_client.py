"""
F5-TTS Modal Client
===================

Client code untuk menggunakan F5-TTS API yang di-deploy di Modal.com

Usage:
    python modal_f5_tts_client.py
"""

import requests
import base64
import json
from pathlib import Path
from typing import Optional

class F5TTSClient:
    """Client untuk F5-TTS API di Modal.com"""

    def __init__(self, api_url: str):
        """
        Initialize client dengan API URL.

        Args:
            api_url: URL endpoint dari Modal deployment
                    Format: https://your-username--f5-tts-inference-tts-api.modal.run
        """
        self.api_url = api_url.rstrip('/')
        self.health_url = api_url.replace('tts-api', 'health-check')

    def health_check(self) -> dict:
        """
        Check health status dari API.

        Returns:
            dict: Health status response
        """
        try:
            response = requests.get(self.health_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        remove_silence: bool = True,
        output_path: str = "output.wav",
        timeout: int = 120,
    ) -> dict:
        """
        Synthesize speech dari text.

        Args:
            text: Text yang ingin diubah menjadi speech
            ref_audio_path: Path ke reference audio untuk voice cloning (optional)
            ref_text: Transcription dari reference audio (optional)
            remove_silence: Remove silence di awal dan akhir
            output_path: Path untuk save generated audio
            timeout: Request timeout dalam detik

        Returns:
            dict: Response dari API dengan info audio yang di-generate
        """
        # Prepare request data
        data = {
            "text": text,
            "remove_silence": remove_silence,
        }

        # Add reference audio jika ada
        if ref_audio_path:
            ref_path = Path(ref_audio_path)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

            # Read and encode audio
            with open(ref_path, "rb") as f:
                audio_bytes = f.read()
                data["ref_audio_base64"] = base64.b64encode(audio_bytes).decode()

            print(f"📁 Using reference audio: {ref_audio_path}")

        # Add reference text jika ada
        if ref_text:
            data["ref_text"] = ref_text

        # Send request
        print(f"📤 Sending request to API...")
        print(f"   Text length: {len(text)} chars")

        try:
            response = requests.post(
                self.api_url,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            # Check if successful
            if not result.get("success"):
                print(f"❌ API Error: {result.get('error')}")
                return result

            # Save audio
            audio_bytes = base64.b64decode(result["audio_base64"])

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            print(f"✅ Speech generated successfully!")
            print(f"   Duration: {result['duration']:.2f} seconds")
            print(f"   Sample rate: {result['sample_rate']} Hz")
            print(f"   Saved to: {output_path}")

            result["output_path"] = str(output_file)
            return result

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Request timeout after {timeout} seconds"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def batch_synthesize(
        self,
        texts: list[str],
        output_dir: str = "outputs",
        **kwargs
    ) -> list[dict]:
        """
        Batch synthesis untuk multiple texts.

        Args:
            texts: List of texts untuk di-synthesize
            output_dir: Directory untuk save generated audio files
            **kwargs: Additional arguments untuk synthesize()

        Returns:
            list[dict]: List of results untuk setiap text
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for i, text in enumerate(texts, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(texts)}")
            print(f"{'='*60}")

            output_file = output_path / f"output_{i:03d}.wav"

            result = self.synthesize(
                text=text,
                output_path=str(output_file),
                **kwargs
            )

            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.get("success"))
        print(f"\n{'='*60}")
        print(f"📊 BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(texts)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(texts) - successful}")

        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Main function dengan contoh penggunaan"""

    # GANTI dengan URL Modal deployment Anda
    API_URL = "https://your-username--f5-tts-inference-tts-api.modal.run"

    # Initialize client
    client = F5TTSClient(API_URL)

    # 1. Health check
    print("🏥 Checking API health...")
    health = client.health_check()
    print(f"Status: {health.get('status')}")
    print(f"GPU Available: {health.get('gpu_available')}")
    print()

    # 2. Simple TTS (tanpa voice cloning)
    print("="*60)
    print("TEST 1: Simple Text-to-Speech")
    print("="*60)

    result = client.synthesize(
        text="Halo, ini adalah test dari model F5-TTS. Model ini dapat mengubah text menjadi speech dengan kualitas yang sangat baik.",
        output_path="output_simple.wav",
    )

    # 3. TTS dengan voice cloning
    print("\n" + "="*60)
    print("TEST 2: Voice Cloning")
    print("="*60)

    # NOTE: Ganti dengan path ke reference audio Anda
    REF_AUDIO_PATH = "reference_audio.wav"  # Harus ada file ini
    REF_TEXT = "This is the transcription of reference audio."

    # Uncomment jika punya reference audio
    # result = client.synthesize(
    #     text="Ini adalah cloning dari suara reference audio yang diberikan.",
    #     ref_audio_path=REF_AUDIO_PATH,
    #     ref_text=REF_TEXT,
    #     output_path="output_cloned.wav",
    # )

    # 4. Batch processing
    print("\n" + "="*60)
    print("TEST 3: Batch Processing")
    print("="*60)

    texts = [
        "Ini adalah kalimat pertama.",
        "Ini adalah kalimat kedua yang lebih panjang dan mengandung lebih banyak informasi.",
        "Kalimat ketiga untuk testing batch processing.",
    ]

    results = client.batch_synthesize(
        texts=texts,
        output_dir="batch_outputs",
    )


if __name__ == "__main__":
    print("🎙️  F5-TTS Modal Client")
    print("="*60)
    print()

    main()

    print("\n✅ All tests completed!")
