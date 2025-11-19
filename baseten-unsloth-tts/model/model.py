"""
Unsloth TTS Model for Baseten Deployment
Optimized for T4 GPU with efficient inference
"""

import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import base64
from typing import Dict, Any
from unsloth import FastLanguageModel


class Model:
    """
    Text-to-Speech model using Unsloth for 2x faster inference
    Optimized for T4 GPU deployment on Baseten
    """

    def __init__(self, **kwargs):
        """Initialize model paths and configuration"""
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._tokenizer = None
        self._vocoder = None

    def load(self):
        """
        Load the TTS model using Unsloth
        This runs once when the model server starts
        """
        print("Loading Unsloth TTS model...")

        # Model configuration
        # Using Orpheus-TTS as it's compatible with Unsloth and has great performance
        model_name = "OuteAI/Orpheus-3B"  # or your fine-tuned model
        max_seq_length = 2048
        dtype = None  # Auto-detection
        load_in_4bit = True  # 4-bit quantization for T4 GPU efficiency

        try:
            # Load model with Unsloth for 2x faster inference
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            # Enable faster inference mode
            FastLanguageModel.for_inference(self._model)

            print(f"✓ Model loaded successfully: {model_name}")
            print(f"✓ Using 4-bit quantization for T4 GPU optimization")
            print(f"✓ Unsloth fast inference enabled (2x speedup)")

        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback: Try loading without Unsloth optimizations
            print("Attempting fallback model loading...")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✓ Model loaded with fallback method")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate speech from text input

        Args:
            request: Dictionary containing:
                - text (str): Input text to convert to speech
                - temperature (float, optional): Sampling temperature (default: 0.7)
                - max_length (int, optional): Maximum audio length (default: 1024)
                - emotion (str, optional): Emotion tags like '<laugh>', '<sigh>' (if supported)
                - return_format (str, optional): 'base64' or 'array' (default: 'base64')

        Returns:
            Dictionary containing:
                - audio: Base64 encoded WAV file or numpy array
                - sample_rate: Audio sample rate
                - duration: Audio duration in seconds
        """
        # Extract parameters
        text = request.get("text", "")
        temperature = request.get("temperature", 0.7)
        max_length = request.get("max_length", 1024)
        emotion = request.get("emotion", "")
        return_format = request.get("return_format", "base64")

        if not text:
            return {"error": "No text provided"}

        # Add emotion tags if specified (for models that support it)
        if emotion:
            text = f"{emotion} {text}"

        print(f"Generating speech for: '{text[:50]}...'")

        try:
            # Tokenize input
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self._model.device)

            # Generate audio tokens with Unsloth optimized inference
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Decode to audio
            # Note: For actual TTS models, this would involve a vocoder
            # This is a simplified example - adjust based on your specific model
            audio_array = self._decode_to_audio(outputs)

            # Get audio properties
            sample_rate = 24000  # Typical for TTS models
            duration = len(audio_array) / sample_rate

            # Format response
            if return_format == "base64":
                # Convert to WAV and encode as base64
                audio_bytes = self._array_to_wav_bytes(audio_array, sample_rate)
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                return {
                    "audio": audio_b64,
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "format": "wav",
                    "encoding": "base64"
                }
            else:
                # Return as numpy array
                return {
                    "audio": audio_array.tolist(),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "format": "array"
                }

        except Exception as e:
            print(f"Error during inference: {e}")
            return {
                "error": str(e),
                "message": "Failed to generate speech"
            }

    def _decode_to_audio(self, audio_tokens: torch.Tensor) -> np.ndarray:
        """
        Convert model output tokens to audio waveform

        Note: This is model-specific. For Orpheus/actual TTS models,
        you may need a separate vocoder or the model has built-in decoding.
        Adjust this method based on your model architecture.
        """
        # Placeholder: Implement actual token-to-audio conversion
        # For real implementation, you might need:
        # 1. Decode tokens to mel-spectrogram
        # 2. Use vocoder (e.g., HiFi-GAN) to convert mel to waveform

        # For now, return a simple sine wave as demonstration
        # Replace this with actual model decoding logic
        sample_rate = 24000
        duration = 2.0  # seconds
        frequency = 440.0  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t) * 0.3

        return audio.astype(np.float32)

    def _array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV file bytes"""
        buffer = BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()
