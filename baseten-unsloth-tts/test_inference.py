#!/usr/bin/env python3
"""
Test script untuk Unsloth TTS di Baseten
Usage: python test_inference.py
"""

import os
import sys
import requests
import base64
from datetime import datetime


def test_tts_inference(
    model_url: str,
    api_key: str,
    text: str = "Halo, ini adalah test text to speech menggunakan Unsloth untuk inference yang efisien!",
    output_file: str = None
):
    """
    Test TTS inference endpoint

    Args:
        model_url: Baseten model URL endpoint
        api_key: Baseten API key
        text: Text to convert to speech
        output_file: Output WAV file path (default: output_TIMESTAMP.wav)
    """
    print("=" * 60)
    print("🎤 Unsloth TTS Inference Test")
    print("=" * 60)

    # Generate output filename with timestamp
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.wav"

    print(f"\n📝 Input text: {text}")
    print(f"🎯 Model URL: {model_url}")
    print(f"💾 Output file: {output_file}")

    # Prepare request
    payload = {
        "text": text,
        "voice": "tara",  # Options: tara, leah, jess, leo, dan, mia, zac, zoe
        "temperature": 0.7,
        "max_new_tokens": 512,
        "emotion": "",  # Optional: <laugh>, <sigh>, <cough>, etc.
        "return_format": "base64"
    }

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json"
    }

    print("\n🚀 Sending request to Baseten...")

    try:
        # Make request
        response = requests.post(
            model_url,
            headers=headers,
            json=payload,
            timeout=60  # 60 seconds timeout
        )

        # Check response
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

        # Parse result
        result = response.json()

        if "error" in result:
            print(f"❌ Model error: {result['error']}")
            if "message" in result:
                print(f"Message: {result['message']}")
            return False

        # Decode audio
        print("\n✅ Response received!")
        print(f"   Sample rate: {result.get('sample_rate', 'N/A')} Hz")
        print(f"   Duration: {result.get('duration', 'N/A'):.2f} seconds")
        print(f"   Format: {result.get('format', 'N/A')}")

        # Save audio file
        audio_b64 = result["audio"]
        audio_bytes = base64.b64decode(audio_b64)

        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        file_size_kb = len(audio_bytes) / 1024
        print(f"\n💾 Audio saved to: {output_file}")
        print(f"   File size: {file_size_kb:.2f} KB")

        print("\n" + "=" * 60)
        print("✨ Test completed successfully!")
        print("=" * 60)

        return True

    except requests.exceptions.Timeout:
        print("❌ Error: Request timeout (>60s)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Error: Connection failed. Check your internet or model URL.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    # Get credentials from environment or prompt
    model_url = os.environ.get("BASETEN_MODEL_URL")
    api_key = os.environ.get("BASETEN_API_KEY")

    if not model_url:
        print("⚠️  BASETEN_MODEL_URL not found in environment variables")
        model_url = input("Enter Baseten Model URL: ").strip()

    if not api_key:
        print("⚠️  BASETEN_API_KEY not found in environment variables")
        api_key = input("Enter Baseten API Key: ").strip()

    if not model_url or not api_key:
        print("\n❌ Error: Model URL and API Key are required!")
        print("\nUsage:")
        print("  export BASETEN_MODEL_URL='https://model-<id>.api.baseten.co/production/predict'")
        print("  export BASETEN_API_KEY='your_api_key'")
        print("  python test_inference.py")
        sys.exit(1)

    # Example texts to test
    test_texts = [
        "Halo, ini adalah test text to speech menggunakan Unsloth!",
        "Unsloth membuat inference TTS dua kali lebih cepat dengan memory 50 persen lebih hemat.",
        "Perfect untuk deployment di T4 GPU tanpa boros resources!",
        "<laugh> This is amazing! The voice quality is so natural.",
        "<sigh> Finally, an efficient TTS solution for production."
    ]

    # Run test
    if len(sys.argv) > 1:
        # Use custom text from command line
        custom_text = " ".join(sys.argv[1:])
        success = test_tts_inference(model_url, api_key, text=custom_text)
    else:
        # Use default test text
        success = test_tts_inference(model_url, api_key, text=test_texts[0])

    # Additional tests (optional)
    run_multiple = input("\n🔄 Run additional tests? (y/n): ").strip().lower()
    if run_multiple == 'y':
        for i, text in enumerate(test_texts[1:], start=2):
            print(f"\n\n{'='*60}")
            print(f"Test {i}/{len(test_texts)}")
            print(f"{'='*60}")
            test_tts_inference(
                model_url,
                api_key,
                text=text,
                output_file=f"output_test{i}.wav"
            )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
