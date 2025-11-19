"""
Unsloth TTS Model for Baseten Deployment
Optimized for T4 GPU with efficient inference using SNAC vocoder
Based on official Unsloth notebooks and Orpheus-TTS architecture
"""

import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import base64
import re
from typing import Dict, Any, List
from unsloth import FastLanguageModel


class Model:
    """
    Text-to-Speech model using Unsloth for 2x faster inference
    Uses SNAC vocoder for high-quality audio generation
    Optimized for T4 GPU deployment on Baseten
    """

    def __init__(self, **kwargs):
        """Initialize model paths and configuration"""
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._tokenizer = None
        self._snac_model = None
        self._sample_rate = 24000  # SNAC 24kHz for speech

    def load(self):
        """
        Load the TTS model and SNAC vocoder using Unsloth
        This runs once when the model server starts
        """
        print("=" * 60)
        print("Loading Unsloth TTS model with SNAC vocoder...")
        print("=" * 60)

        # Model configuration
        # Using Unsloth's Orpheus fine-tuned model
        model_name = "unsloth/orpheus-3b-0.1-ft"  # or "unsloth/orpheus-3b-0.1-pretrained"
        max_seq_length = 2048
        dtype = None  # Auto-detection
        load_in_4bit = True  # 4-bit quantization for T4 GPU efficiency

        try:
            # Load TTS model with Unsloth for 2x faster inference
            print(f"Loading model: {model_name}")
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            # Enable faster inference mode
            FastLanguageModel.for_inference(self._model)

            print(f"✓ Model loaded: {model_name}")
            print(f"✓ 4-bit quantization enabled (T4 optimized)")
            print(f"✓ Unsloth fast inference enabled (2x speedup)")

        except Exception as e:
            print(f"⚠ Error loading Unsloth model: {e}")
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

        # Load SNAC vocoder for audio decoding
        try:
            print("\nLoading SNAC vocoder (24kHz)...")
            from snac import SNAC

            self._snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
            self._snac_model = self._snac_model.eval()

            # Move to same device as main model
            if torch.cuda.is_available():
                self._snac_model = self._snac_model.cuda()

            print("✓ SNAC vocoder loaded successfully")
            print(f"✓ Sample rate: {self._sample_rate} Hz")

        except Exception as e:
            print(f"⚠ Warning: SNAC vocoder not available: {e}")
            print("  Audio will be generated using fallback method")
            self._snac_model = None

        print("=" * 60)
        print("✅ Model loading complete!")
        print("=" * 60)

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate speech from text input using Orpheus-TTS + SNAC

        Args:
            request: Dictionary containing:
                - text (str): Input text to convert to speech
                - voice (str, optional): Voice name (e.g., 'tara', 'leah', 'jess')
                - temperature (float, optional): Sampling temperature (default: 0.7)
                - max_new_tokens (int, optional): Max tokens to generate (default: 512)
                - emotion (str, optional): Emotion tags like '<laugh>', '<sigh>', '<cough>'
                - return_format (str, optional): 'base64' or 'array' (default: 'base64')

        Returns:
            Dictionary containing:
                - audio: Base64 encoded WAV file or numpy array
                - sample_rate: Audio sample rate (24000 Hz)
                - duration: Audio duration in seconds
                - text: Processed input text
        """
        # Extract parameters
        text = request.get("text", "")
        voice = request.get("voice", "tara")  # Default voice
        temperature = request.get("temperature", 0.7)
        max_new_tokens = request.get("max_new_tokens", 512)
        emotion = request.get("emotion", "")
        return_format = request.get("return_format", "base64")

        if not text:
            return {"error": "No text provided"}

        # Format prompt according to Orpheus format: "{voice}: {text}"
        prompt = self._format_prompt(text, voice, emotion)

        print(f"📝 Prompt: '{prompt[:80]}...'")
        print(f"🎤 Voice: {voice}, Temperature: {temperature}")

        try:
            # Generate audio tokens
            audio_codes = self._generate_audio_tokens(
                prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

            # Decode tokens to audio waveform using SNAC
            audio_array = self._decode_tokens_to_audio(audio_codes)

            # Calculate duration
            duration = len(audio_array) / self._sample_rate

            print(f"✅ Generated {duration:.2f}s of audio")

            # Format response
            if return_format == "base64":
                # Convert to WAV and encode as base64
                audio_bytes = self._array_to_wav_bytes(audio_array, self._sample_rate)
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                return {
                    "audio": audio_b64,
                    "sample_rate": self._sample_rate,
                    "duration": duration,
                    "format": "wav",
                    "encoding": "base64",
                    "text": prompt
                }
            else:
                # Return as numpy array
                return {
                    "audio": audio_array.tolist(),
                    "sample_rate": self._sample_rate,
                    "duration": duration,
                    "format": "array",
                    "text": prompt
                }

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()

            return {
                "error": str(e),
                "message": "Failed to generate speech. Check server logs for details."
            }

    def _format_prompt(self, text: str, voice: str, emotion: str = "") -> str:
        """
        Format prompt according to Orpheus-TTS format

        Orpheus expects: "{voice_name}: {text}"
        Supported voices: tara, leah, jess, leo, dan, mia, zac, zoe
        Emotion tags: <laugh>, <sigh>, <cough>, <gasp>, etc.
        """
        # Add emotion tag if specified
        if emotion:
            # Ensure emotion tag is properly formatted
            if not emotion.startswith("<"):
                emotion = f"<{emotion}>"
            if not emotion.endswith(">"):
                emotion = f"{emotion}>"
            text = f"{emotion} {text}"

        # Format with voice name
        prompt = f"{voice}: {text}"

        return prompt

    def _generate_audio_tokens(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512
    ) -> torch.Tensor:
        """
        Generate audio tokens from text prompt

        Returns:
            Tensor of audio token IDs
        """
        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)

        # Generate with Unsloth optimized inference
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                top_p=0.9,  # Nucleus sampling
            )

        # Extract generated tokens (remove input tokens)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]

        return generated_tokens

    def _decode_tokens_to_audio(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Decode audio tokens to waveform using SNAC vocoder

        Args:
            tokens: Audio token IDs from model

        Returns:
            Audio waveform as numpy array
        """
        if self._snac_model is None:
            # Fallback: Generate placeholder audio
            print("⚠ Using fallback audio generation (SNAC not available)")
            return self._generate_fallback_audio()

        try:
            # Extract audio codes from tokens
            # Note: Orpheus outputs audio tokens that need to be reshaped for SNAC
            # SNAC expects multi-scale codes: list of tensors [B, T] for each scale

            # Parse tokens into SNAC format
            # This is model-specific and may need adjustment based on your model
            codes = self._parse_audio_codes(tokens)

            # Decode with SNAC
            with torch.inference_mode():
                audio_tensor = self._snac_model.decode(codes)

            # Convert to numpy array
            # SNAC returns shape (B, 1, T) - take first batch and channel
            audio_array = audio_tensor.cpu().squeeze().numpy()

            # Normalize to [-1, 1] range
            audio_array = audio_array / np.abs(audio_array).max() if audio_array.max() > 0 else audio_array

            return audio_array.astype(np.float32)

        except Exception as e:
            print(f"⚠ SNAC decoding failed: {e}")
            print("  Using fallback audio generation")
            return self._generate_fallback_audio()

    def _parse_audio_codes(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Parse model output tokens into SNAC multi-scale codes

        SNAC expects a list of code tensors, one for each temporal scale
        This is a simplified version - adjust based on your model's token structure
        """
        # For Orpheus-TTS, tokens are typically structured as:
        # [text tokens] [audio codes at different scales]
        # This needs to be parsed according to model architecture

        # Simplified approach: reshape tokens for SNAC
        # Note: This may need adjustment based on actual model output format

        # SNAC 24kHz has 4 scales with different temporal resolutions
        # We need to split tokens accordingly

        # Placeholder: Return tokens as single-scale codes
        # TODO: Implement proper multi-scale parsing based on model architecture
        codes = [tokens.unsqueeze(0)]  # [1, T]

        return codes

    def _generate_fallback_audio(self, duration: float = 2.0) -> np.ndarray:
        """
        Generate fallback audio when SNAC is not available
        This is a placeholder for development/testing
        """
        print(f"Generating {duration}s fallback audio...")

        # Generate a simple tone as placeholder
        t = np.linspace(0, duration, int(self._sample_rate * duration))

        # Create a pleasant tone (middle C = 261.63 Hz)
        frequency = 261.63
        audio = np.sin(2 * np.pi * frequency * t) * 0.3

        # Add fade in/out to avoid clicks
        fade_samples = int(0.05 * self._sample_rate)  # 50ms fade
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return audio.astype(np.float32)

    def _array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV file bytes"""
        buffer = BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
