"""
Bantz — Wake Word Listener (#165)

Always-on "Hey Bantz" detection via Picovoice Porcupine.
Runs in a dedicated daemon thread with its own audio stream —
completely independent of APScheduler and the Textual event loop.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │              WakeWordListener                        │
    │                                                      │
    │  threading.Thread (daemon=True)                      │
    │  ┌─ while not _stop.is_set():                        │
    │  │    pcm = stream.read(frame_length)  # blocking    │
    │  │    idx = porcupine.process(pcm)                   │
    │  │    if idx >= 0:                                   │
    │  │        tts.stop()              # interrupt TTS    │
    │  │        _play_ack()             # "Yes boss?"      │
    │  │        fire callback           # → TUI message    │
    │  └──────────────────────────────────────────────────  │
    └──────────────────────────────────────────────────────┘

Key decisions:
  - Dedicated thread (NOT APScheduler) — audio stream.read() is blocking
    and continuous; an interval job would overflow the audio buffer.
  - Mic stays open during TTS — Porcupine is excellent at rejecting
    non-wake-word audio.  This lets users interrupt Bantz mid-speech
    with "Hey Bantz" → tts.stop() → immediate silence.
  - No STT in this issue — wake word detection only.  Speech-to-text
    (Whisper/Ollama) is a separate future issue.
  - Graceful degradation: if pvporcupine or pyaudio are missing,
    the listener simply logs a warning and never starts.
"""
from __future__ import annotations

import logging
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from bantz.core.event_bus import bus

log = logging.getLogger("bantz.wake_word")

# Cooldown between detections to prevent rapid re-triggers
_COOLDOWN_SECONDS = 2.0

# Default built-in keyword (Porcupine ships "Hey Google", "Alexa", etc.
# but for a custom "Hey Bantz" we need a .ppn file trained on Picovoice Console).
# If no custom keyword file exists, fall back to "Computer" (built-in).
_FALLBACK_KEYWORD = "computer"


class WakeWordListener:
    """Always-on Porcupine wake word listener on a dedicated daemon thread.

    Usage:
        listener = WakeWordListener()
        listener.start(on_wake=my_callback)
        ...
        listener.stop()

    The ``on_wake`` callback is called from the audio thread.  If you need
    to post a Textual message, use ``app.call_from_thread(...)`` inside the
    callback.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._on_wake: Optional[Callable[[], None]] = None
        self._running = False
        self._paused = False
        self._last_trigger: float = 0.0
        self._total_detections: int = 0

        # Lazy-initialized
        self._porcupine = None
        self._audio_stream = None
        self._pa = None  # PyAudio instance

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def total_detections(self) -> int:
        return self._total_detections

    def start(self, on_wake: Optional[Callable[[], None]] = None) -> bool:
        """Start listening for the wake word in a daemon thread.

        Args:
            on_wake: Callback fired (on the audio thread) when the wake
                     word is detected.  Use ``app.call_from_thread()``
                     to safely relay to the Textual main thread.

        Returns:
            True if the listener started successfully.
        """
        if self._running:
            log.warning("Wake word listener already running")
            return True

        self._on_wake = on_wake

        if not self._init_porcupine():
            return False

        if not self._init_audio():
            self._cleanup_porcupine()
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="bantz-wake-word",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        log.info("Wake word listener started (sensitivity=%.2f)",
                 self._get_sensitivity())
        return True

    def stop(self) -> None:
        """Stop the listener and release audio resources."""
        if not self._running:
            return
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._cleanup_audio()
        self._cleanup_porcupine()
        self._running = False
        log.info("Wake word listener stopped (total detections: %d)",
                 self._total_detections)

    # ── Pause / Resume (for Ghost Loop mic sharing) ─────────────────────

    def pause(self) -> None:
        """Temporarily release the microphone so another module can use it.

        The wake word thread keeps running but ``stream.read()`` will
        raise after we close the stream — the loop catches that and
        sleeps until ``resume()`` re-opens the stream.
        """
        if not self._running:
            return
        self._paused = True
        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        log.info("Wake word: paused (mic released for voice capture)")

    def resume(self) -> None:
        """Re-open the microphone after a pause."""
        if not self._running:
            return
        self._paused = False
        if self._init_audio():
            log.info("Wake word: resumed (mic re-acquired)")
        else:
            log.error("Wake word: could not re-acquire mic after pause")

    def diagnose(self) -> dict:
        """Return diagnostic info for --doctor."""
        result: dict = {
            "porcupine_available": False,
            "pyaudio_available": False,
            "running": self._running,
            "total_detections": self._total_detections,
        }
        try:
            import pvporcupine  # noqa: F401
            result["porcupine_available"] = True
            result["porcupine_version"] = getattr(pvporcupine, "__version__", "unknown")
        except ImportError:
            pass

        try:
            import pyaudio  # noqa: F401
            result["pyaudio_available"] = True
        except ImportError:
            pass

        return result

    def stats(self) -> dict:
        return {
            "running": self._running,
            "total_detections": self._total_detections,
            "last_trigger": self._last_trigger,
        }

    def status_line(self) -> str:
        if not self._running:
            return "wake_word=stopped"
        return f"wake_word=listening detections={self._total_detections}"

    # ── Audio thread loop ───────────────────────────────────────────────

    def _listen_loop(self) -> None:
        """Blocking audio read loop — runs on the daemon thread."""
        porcupine = self._porcupine
        stream = self._audio_stream

        if not porcupine or not stream:
            log.error("Wake word: porcupine or audio stream not initialized")
            self._running = False
            return

        frame_length = porcupine.frame_length
        log.debug("Wake word: entering listen loop (frame_length=%d)", frame_length)

        # Lazy-load ambient analyser (piggyback, #166)
        _ambient = None
        try:
            from bantz.agent.ambient import ambient_analyzer
            from bantz.config import config
            if getattr(config, "ambient_enabled", False):
                _ambient = ambient_analyzer
                log.debug("Wake word: ambient analyser piggybacking enabled")
        except Exception:
            pass

        while not self._stop.is_set():
            # While paused, sleep and skip audio reads
            if self._paused or self._audio_stream is None:
                time.sleep(0.05)
                # Re-acquire stream reference after resume()
                if not self._paused and self._audio_stream is not None:
                    stream = self._audio_stream
                continue

            try:
                pcm_bytes = stream.read(frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from(f"{frame_length}h", pcm_bytes)

                # Feed ambient analyser before Porcupine (zero-cost if disabled)
                if _ambient is not None:
                    try:
                        _ambient.feed_frames(pcm)
                    except Exception:
                        pass  # never crash the wake word loop

                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    now = time.monotonic()
                    if now - self._last_trigger < _COOLDOWN_SECONDS:
                        continue
                    self._last_trigger = now
                    self._total_detections += 1

                    log.info("Wake word detected! (count=%d)", self._total_detections)

                    # Interrupt TTS immediately
                    self._interrupt_tts()

                    # Play acknowledgment sound
                    self._play_ack()

                    # Publish to EventBus (Sprint 3 Part 2)
                    try:
                        bus.emit_threadsafe(
                            "wake_word_detected",
                            count=self._total_detections,
                        )
                    except Exception:
                        pass  # never crash the audio thread

                    # Fire legacy callback (backward compat)
                    if self._on_wake:
                        try:
                            self._on_wake()
                        except Exception as exc:
                            log.error("Wake word callback error: %s", exc)

            except Exception as exc:
                if self._stop.is_set():
                    break
                if self._paused:
                    # Stream was closed by pause() — just loop back
                    continue
                log.warning("Wake word: audio read error — %s", exc)
                time.sleep(0.1)

        log.debug("Wake word: exited listen loop")

    # ── TTS interrupt ───────────────────────────────────────────────────

    @staticmethod
    def _interrupt_tts() -> None:
        """Stop any active TTS playback — JARVIS-style interrupt."""
        try:
            from bantz.agent.tts import tts_engine
            if tts_engine.is_speaking:
                tts_engine.stop()
                log.info("Wake word: interrupted TTS playback")
        except Exception:
            pass

    # ── Acknowledgment sound ────────────────────────────────────────────

    @staticmethod
    def _play_ack() -> None:
        """Play a short acknowledgment beep/chime.

        Uses aplay with a simple sine tone generated in-memory.
        Falls back to terminal bell if aplay is unavailable.
        """
        import shutil
        import subprocess

        aplay = shutil.which("aplay")
        if not aplay:
            # Terminal bell fallback
            print("\a", end="", flush=True)
            return

        try:
            # Generate a short 200ms 880Hz sine tone (A5)
            import math
            sample_rate = 22050
            duration = 0.2  # seconds
            n_samples = int(sample_rate * duration)
            pcm = bytearray()
            for i in range(n_samples):
                # Fade envelope: quick attack, gradual decay
                t = i / sample_rate
                envelope = min(1.0, t * 20) * max(0.0, 1.0 - t * 3)
                sample = int(16000 * envelope * math.sin(2 * math.pi * 880 * t))
                pcm.extend(struct.pack("<h", max(-32768, min(32767, sample))))

            subprocess.run(
                [aplay, "-r", str(sample_rate), "-f", "S16_LE", "-t", "raw", "-c", "1"],
                input=bytes(pcm),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception as exc:
            log.debug("Wake word: ack sound failed — %s", exc)
            print("\a", end="", flush=True)

    # ── Initialization ──────────────────────────────────────────────────

    def _get_sensitivity(self) -> float:
        try:
            from bantz.config import config
            return config.wake_word_sensitivity
        except Exception:
            return 0.5

    def _get_access_key(self) -> str:
        try:
            from bantz.config import config
            return config.picovoice_access_key
        except Exception:
            return ""

    def _find_keyword_path(self) -> Optional[str]:
        """Look for a custom Bantz .ppn wake word model.

        Search order:
          1. Data dir (~/.local/share/bantz/ or BANTZ_DATA_DIR)
          2. Project / working directory

        Patterns: hey-bantz*.ppn, hey_bantz*.ppn, Bantz*.ppn, bantz*.ppn

        Returns None to use built-in fallback keyword.
        """
        _PATTERNS = (
            "hey-bantz*.ppn", "hey_bantz*.ppn", "Hey-Bantz*.ppn",
            "Bantz*.ppn", "bantz*.ppn",
        )
        try:
            from bantz.config import config
            data_dir = (
                Path(config.data_dir) if config.data_dir
                else Path.home() / ".local" / "share" / "bantz"
            )
            # Search data dir first, then project root (cwd)
            search_dirs = [data_dir, Path.cwd()]
            # Also check the package root (two levels up from this file)
            pkg_root = Path(__file__).resolve().parent.parent.parent.parent
            if pkg_root not in search_dirs:
                search_dirs.append(pkg_root)

            for d in search_dirs:
                if not d.is_dir():
                    continue
                for pattern in _PATTERNS:
                    matches = list(d.glob(pattern))
                    if matches:
                        log.info("Wake word: using custom keyword file %s", matches[0])
                        return str(matches[0])
        except Exception:
            pass
        return None

    def _init_porcupine(self) -> bool:
        """Initialize Porcupine engine. Returns True on success."""
        access_key = self._get_access_key()
        if not access_key:
            log.warning("Wake word: BANTZ_PICOVOICE_ACCESS_KEY not set")
            return False

        try:
            import pvporcupine
        except ImportError:
            log.warning("Wake word: pvporcupine not installed — pip install pvporcupine")
            return False

        sensitivity = self._get_sensitivity()
        keyword_path = self._find_keyword_path()

        try:
            if keyword_path:
                self._porcupine = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=[keyword_path],
                    sensitivities=[sensitivity],
                )
            else:
                log.info("Wake word: no custom .ppn found, using built-in '%s'",
                         _FALLBACK_KEYWORD)
                self._porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=[_FALLBACK_KEYWORD],
                    sensitivities=[sensitivity],
                )
            return True
        except Exception as exc:
            log.error("Wake word: Porcupine init failed — %s", exc)
            return False

    def _init_audio(self) -> bool:
        """Open the microphone stream. Returns True on success."""
        try:
            import pyaudio
        except ImportError:
            log.warning("Wake word: pyaudio not installed — pip install pyaudio")
            return False

        if not self._porcupine:
            return False

        try:
            self._pa = pyaudio.PyAudio()
            self._audio_stream = self._pa.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length,
            )
            return True
        except Exception as exc:
            log.error("Wake word: microphone init failed — %s", exc)
            self._cleanup_audio()
            return False

    # ── Cleanup ─────────────────────────────────────────────────────────

    def _cleanup_audio(self) -> None:
        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _cleanup_porcupine(self) -> None:
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

wake_listener = WakeWordListener()
