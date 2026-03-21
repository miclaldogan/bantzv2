"""
Bantz — Voice Capture with VAD (#36, The Ghost Loop)

Records microphone audio and uses Voice Activity Detection (WebRTC VAD)
to automatically stop recording when the user finishes speaking.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │               VoiceCapture                              │
    │                                                         │
    │  1. Open mic (PyAudio, 16 kHz mono)                     │
    │  2. Feed frames to WebRTC VAD                           │
    │  3. Accumulate voiced frames into buffer                │
    │  4. When silence exceeds VAD_SILENCE_MS → stop          │
    │  5. Return raw PCM bytes (16-bit 16 kHz mono)           │
    └─────────────────────────────────────────────────────────┘

The capture is synchronous (blocking) and designed to run on a
background thread — never on the asyncio event loop.

Dependencies:
  - pyaudio (already installed for wake_word)
  - webrtcvad (pip install webrtcvad)

Graceful degradation: if either dependency is missing, ``capture()``
returns None and logs a warning.
"""
from __future__ import annotations

import logging
import os
import struct
import sys
import time
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("bantz.voice_capture")

# Audio parameters — must match WebRTC VAD requirements
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
FRAME_DURATION_MS = 30  # WebRTC VAD supports 10, 20, or 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples

# Maximum recording duration (safety limit)
MAX_RECORD_SECONDS = 30.0


@contextmanager
def suppress_alsa_stderr():
    """Temporarily silence C-level ALSA warnings on fd 2.

    PyAudio (via PortAudio) enumerates every ALSA device on init and
    spews dozens of harmless warnings (``ALSA lib pcm…``) directly to
    stderr at the file-descriptor level — Python's ``logging`` or
    ``contextlib.redirect_stderr`` can't catch them.  This redirects
    fd 2 to ``/dev/null`` for the duration of the block.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


class VoiceCapture:
    """Record audio from the mic until user stops speaking (VAD)."""

    def __init__(self) -> None:
        self._pa = None  # PyAudio instance
        self._vad = None  # WebRTC VAD instance

    def _ensure_init(self) -> bool:
        """Lazy-initialize PyAudio and WebRTC VAD. Returns True if ready."""
        if self._vad is not None:
            return True

        try:
            import webrtcvad
        except ImportError:
            log.warning("VoiceCapture: webrtcvad not installed — pip install webrtcvad")
            return False

        try:
            from bantz.config import config
            aggressiveness = config.vad_aggressiveness
        except Exception:
            aggressiveness = 2

        # Clamp aggressiveness to valid range [0, 3]
        aggressiveness = max(0, min(3, aggressiveness))

        try:
            self._vad = webrtcvad.Vad(aggressiveness)
            log.info("VoiceCapture: VAD initialized (aggressiveness=%d)", aggressiveness)
            return True
        except Exception as exc:
            log.error("VoiceCapture: VAD init failed — %s", exc)
            return False

    def capture(self) -> Optional[bytes]:
        """Record audio until the user stops speaking.

        Blocks the calling thread.  Returns raw PCM bytes (16-bit,
        16 kHz, mono) or None on failure.

        The recording starts immediately and ends after ``vad_silence_ms``
        of continuous silence is detected.  A safety cap of 30 seconds
        prevents infinite recording.
        """
        if not self._ensure_init():
            return None

        try:
            import pyaudio
        except ImportError:
            log.warning("VoiceCapture: pyaudio not installed — pip install pyaudio")
            return None

        try:
            from bantz.config import config
            silence_ms = config.vad_silence_ms
        except Exception:
            silence_ms = 800

        # Open microphone (suppress ALSA spam on stderr)
        pa = None
        stream = None
        try:
            with suppress_alsa_stderr():
                pa = pyaudio.PyAudio()
            stream = pa.open(
                rate=SAMPLE_RATE,
                channels=CHANNELS,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=FRAME_SIZE,
            )
        except Exception as exc:
            log.error("VoiceCapture: mic open failed — %s", exc)
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass
            return None

        log.info("VoiceCapture: recording started (silence_threshold=%dms)", silence_ms)

        voiced_frames: list[bytes] = []
        silent_frames = 0
        frames_for_silence = int(silence_ms / FRAME_DURATION_MS)
        started_speaking = False
        start_time = time.monotonic()

        try:
            while True:
                # Safety timeout
                elapsed = time.monotonic() - start_time
                if elapsed > MAX_RECORD_SECONDS:
                    log.info("VoiceCapture: hit max duration (%.0fs)", MAX_RECORD_SECONDS)
                    break

                # Read a frame
                try:
                    pcm_bytes = stream.read(FRAME_SIZE, exception_on_overflow=False)
                except Exception as exc:
                    log.warning("VoiceCapture: read error — %s", exc)
                    break

                # Feed to VAD
                is_speech = self._vad.is_speech(pcm_bytes, SAMPLE_RATE)

                if is_speech:
                    started_speaking = True
                    silent_frames = 0
                    voiced_frames.append(pcm_bytes)
                else:
                    if started_speaking:
                        # Still accumulate silent frames (natural pauses)
                        voiced_frames.append(pcm_bytes)
                        silent_frames += 1

                        if silent_frames >= frames_for_silence:
                            log.info(
                                "VoiceCapture: silence detected after %.1fs",
                                elapsed,
                            )
                            break
                    # else: pre-speech silence, discard

        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass

        if not voiced_frames:
            log.info("VoiceCapture: no speech detected")
            return None

        audio_data = b"".join(voiced_frames)
        duration = len(audio_data) / (SAMPLE_RATE * SAMPLE_WIDTH)
        log.info("VoiceCapture: captured %.1fs of audio (%d bytes)",
                 duration, len(audio_data))
        return audio_data

    def diagnose(self) -> dict:
        """Diagnostic info for --doctor."""
        result: dict = {
            "webrtcvad_available": False,
            "pyaudio_available": False,
        }
        try:
            import webrtcvad  # noqa: F401
            result["webrtcvad_available"] = True
        except ImportError:
            pass
        try:
            import pyaudio  # noqa: F401
            result["pyaudio_available"] = True
        except ImportError:
            pass
        return result


# Module singleton
voice_capture = VoiceCapture()
