"""
Bantz — Local STT via faster-whisper (#36, The Ghost Loop)

Transcribes raw PCM audio to text using faster-whisper, a CTranslate2-
accelerated Whisper implementation.  Runs completely offline on CPU
(or GPU if available).

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                   STTEngine                             │
    │                                                         │
    │  1. Lazy-load model on first transcribe() call          │
    │  2. Accept raw PCM bytes (16-bit, 16 kHz, mono)         │
    │  3. Convert to float32 numpy array                      │
    │  4. Run Whisper inference → segments                    │
    │  5. Concatenate segment texts → return string           │
    └─────────────────────────────────────────────────────────┘

Model sizes (approximate, CPU):
    tiny   — ~39 MB  — fast, good for English
    base   — ~74 MB  — better accuracy
    small  — ~244 MB — best quality for resource-constrained

Dependencies:
    pip install faster-whisper

Graceful degradation: if faster-whisper is not installed,
``transcribe()`` returns None and logs a warning.
"""
from __future__ import annotations

import logging
import tempfile
import struct
import wave
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.stt")

# Audio format matching VoiceCapture output
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # 16-bit
_CHANNELS = 1


class STTEngine:
    """Local speech-to-text via faster-whisper."""

    def __init__(self) -> None:
        self._model = None
        self._model_name: str = ""
        self._device: str = "cpu"
        self._language: str = "en"
        self._available: bool | None = None  # None = not checked yet

    def _ensure_model(self) -> bool:
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return True

        if self._available is False:
            return False  # previously failed

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.warning("STT: faster-whisper not installed — pip install faster-whisper")
            self._available = False
            return False

        try:
            from bantz.config import config
            self._model_name = config.stt_model
            self._device = config.stt_device
            self._language = config.stt_language
        except Exception:
            self._model_name = "tiny"
            self._device = "cpu"
            self._language = "en"

        try:
            log.info("STT: loading model '%s' on '%s'...", self._model_name, self._device)
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type="int8" if self._device == "cpu" else "float16",
            )
            self._available = True
            log.info("STT: model '%s' loaded successfully", self._model_name)
            return True
        except Exception as exc:
            log.error("STT: model load failed — %s", exc)
            self._available = False
            return False

    def transcribe(self, pcm_bytes: bytes) -> Optional[str]:
        """Transcribe raw PCM audio (16-bit, 16 kHz, mono) to text.

        Args:
            pcm_bytes: Raw PCM audio data from VoiceCapture.

        Returns:
            Transcribed text string, or None on failure.
        """
        if not pcm_bytes:
            return None

        if not self._ensure_model():
            return None

        # Convert raw PCM to a WAV file in a temp location
        # (faster-whisper needs a file path or numpy array)
        try:
            import numpy as np
        except ImportError:
            log.warning("STT: numpy not available, falling back to WAV file")
            return self._transcribe_via_wav(pcm_bytes)

        # Convert PCM bytes to float32 numpy array
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            log.error("STT: PCM to numpy conversion failed — %s", exc)
            return None

        return self._transcribe_array(samples)

    def _transcribe_array(self, audio: "np.ndarray") -> Optional[str]:
        """Run inference on a float32 numpy array."""
        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                ),
                # Initial prompt biases the model toward assistant-context vocab,
                # reducing common Whisper errors like "what" → "but".
                initial_prompt="The user is speaking a command to Bantz, an AI assistant.",
            )

            # Collect all segment texts
            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)

            result = " ".join(texts).strip()

            # Whisper sometimes regurgitates the initial_prompt verbatim when
            # audio is unclear or silent. Detect and discard those.
            _PROMPT_ECHO = "The user is speaking a command to Bantz"
            if _PROMPT_ECHO.lower() in result.lower():
                log.warning("STT: discarding prompt-echo transcript: %r", result[:80])
                return None

            if result:
                log.info("STT: transcribed %d chars (lang=%s, prob=%.2f)",
                         len(result), info.language, info.language_probability)
            else:
                log.info("STT: no speech detected in audio")
                return None

            return result

        except Exception as exc:
            log.error("STT: transcription failed — %s", exc)
            return None

    def _transcribe_via_wav(self, pcm_bytes: bytes) -> Optional[str]:
        """Fallback: write PCM to a temp WAV file, then transcribe."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                with wave.open(f, "wb") as wf:
                    wf.setnchannels(_CHANNELS)
                    wf.setsampwidth(_SAMPLE_WIDTH)
                    wf.setframerate(_SAMPLE_RATE)
                    wf.writeframes(pcm_bytes)

            segments, info = self._model.transcribe(
                tmp_path,
                language=self._language,
                beam_size=3,
                vad_filter=True,
            )

            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)

            result = " ".join(texts).strip()
            if not result:
                return None

            log.info("STT: transcribed %d chars via WAV fallback", len(result))
            return result

        except Exception as exc:
            log.error("STT: WAV fallback transcription failed — %s", exc)
            return None
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def available(self) -> bool:
        """Check if STT is available (faster-whisper importable)."""
        if self._available is not None:
            return self._available
        try:
            import faster_whisper  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def diagnose(self) -> dict:
        """Diagnostic info for --doctor."""
        result: dict = {
            "faster_whisper_available": False,
            "numpy_available": False,
            "model": self._model_name or "(not loaded)",
            "device": self._device,
            "language": self._language,
        }
        try:
            import faster_whisper  # noqa: F401
            result["faster_whisper_available"] = True
        except ImportError:
            pass
        try:
            import numpy  # noqa: F401
            result["numpy_available"] = True
        except ImportError:
            pass
        return result


# Module singleton
stt_engine = STTEngine()
