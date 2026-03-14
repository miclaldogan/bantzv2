"""
Bantz — The Ghost Loop (#36)

Hands-free voice interaction loop that ties together:
  1. Wake word detection (Porcupine via ``wake_word.py``)
  2. Voice capture with VAD (``voice_capture.py``)
  3. Speech-to-text (``stt.py``)
  4. Brain dispatch via EventBus

Architecture (The Ghost Loop):
    ┌───────────────────────────────────────────────────────────────┐
    │  EventBus: "wake_word_detected"                              │
    │       │                                                      │
    │       ▼                                                      │
    │  GhostLoop._on_wake_word()                                   │
    │       │                                                      │
    │       ├─ 1. Play earcon (acknowledge)                        │
    │       ├─ 2. VoiceCapture.capture() [blocking, on thread]     │
    │       ├─ 3. STTEngine.transcribe(pcm_bytes)                  │
    │       ├─ 4. Emit "voice_input" to EventBus                   │
    │       │     (TUI picks this up → displays + brain.process)   │
    │       └─ 5. Loop back to idle (wake word listener continues) │
    └───────────────────────────────────────────────────────────────┘

The Ghost Loop runs as a daemon thread.  It subscribes to
``wake_word_detected`` on the EventBus and chains the capture →
transcribe → dispatch pipeline each time the wake word fires.

No manual keybinds needed.  Fully hands-free.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from bantz.core.event_bus import bus, Event

log = logging.getLogger("bantz.ghost_loop")


class GhostLoop:
    """Orchestrates the wake → capture → STT → dispatch cycle."""

    def __init__(self) -> None:
        self._running = False
        self._busy = False  # True while capture+STT is in progress
        self._total_transcriptions: int = 0
        self._last_text: str = ""
        self._thread_pool: Optional[threading.Thread] = None

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the Ghost Loop — subscribe to wake_word_detected.

        Returns True if started successfully.
        """
        if self._running:
            return True

        try:
            from bantz.config import config
            if not config.ghost_loop_enabled:
                log.debug("Ghost Loop: disabled (BANTZ_GHOST_LOOP_ENABLED=false)")
                return False
            if not config.stt_enabled:
                log.debug("Ghost Loop: STT disabled (BANTZ_STT_ENABLED=false)")
                return False
        except Exception:
            return False

        # Subscribe to wake word events
        bus.on("wake_word_detected", self._on_wake_event)
        self._running = True
        log.info("Ghost Loop: active — waiting for wake word")

        # Pre-load STT model in background so first voice command is fast
        threading.Thread(
            target=self._preload_stt,
            name="bantz-stt-preload",
            daemon=True,
        ).start()

        return True

    def stop(self) -> None:
        """Stop the Ghost Loop."""
        if not self._running:
            return
        bus.off("wake_word_detected", self._on_wake_event)
        self._running = False
        log.info("Ghost Loop: stopped (total transcriptions: %d)",
                 self._total_transcriptions)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def busy(self) -> bool:
        return self._busy

    def stats(self) -> dict:
        return {
            "running": self._running,
            "busy": self._busy,
            "total_transcriptions": self._total_transcriptions,
            "last_text": self._last_text[:80] if self._last_text else "",
        }

    # ── Event handler ───────────────────────────────────────────────────

    @staticmethod
    def _preload_stt() -> None:
        """Load the STT model in the background at startup."""
        try:
            from bantz.agent.stt import stt_engine
            stt_engine._ensure_model()
        except Exception as exc:
            log.debug("Ghost Loop: STT preload failed — %s", exc)

    def _on_wake_event(self, event: Event) -> None:
        """Called (on the bus dispatcher) when wake word is detected.

        Spawns the capture → STT → dispatch pipeline on a background
        thread to avoid blocking the event bus dispatcher.
        """
        if self._busy:
            log.debug("Ghost Loop: already busy, ignoring wake word")
            return

        # Run the blocking pipeline on a separate thread
        thread = threading.Thread(
            target=self._capture_and_transcribe,
            name="bantz-ghost-loop",
            daemon=True,
        )
        thread.start()

    # ── Pipeline (runs on background thread) ────────────────────────────

    def _capture_and_transcribe(self) -> None:
        """Blocking pipeline: capture audio → STT → emit to bus."""
        self._busy = True

        try:
            # 0. Pause wake word listener to release the mic
            try:
                from bantz.agent.wake_word import wake_listener
                wake_listener.pause()
                import time
                time.sleep(0.15)  # give ALSA a moment to release the device
            except Exception as exc:
                log.debug("Ghost Loop: could not pause wake word — %s", exc)

            # 1. Emit "listening" event so TUI can show indicator
            bus.emit_threadsafe("ghost_loop_listening")

            # 2. Capture audio with VAD
            from bantz.agent.voice_capture import voice_capture
            pcm_bytes = voice_capture.capture()

            if not pcm_bytes:
                log.info("Ghost Loop: no audio captured")
                bus.emit_threadsafe("ghost_loop_idle")
                return

            # 3. Emit "transcribing" event
            bus.emit_threadsafe("ghost_loop_transcribing")

            # 4. Transcribe via faster-whisper
            from bantz.agent.stt import stt_engine
            text = stt_engine.transcribe(pcm_bytes)

            if not text:
                log.info("Ghost Loop: STT returned empty")
                bus.emit_threadsafe("ghost_loop_idle")
                return

            self._total_transcriptions += 1
            self._last_text = text
            log.info("Ghost Loop: transcribed → \"%s\"", text[:80])

            # 5. Dispatch as a voice input event
            #    The TUI subscribes to this and feeds it to brain.process()
            bus.emit_threadsafe("voice_input", text=text)

        except Exception as exc:
            log.error("Ghost Loop: pipeline error — %s", exc)
        finally:
            self._busy = False
            # Resume wake word listener so it re-acquires the mic
            try:
                from bantz.agent.wake_word import wake_listener
                wake_listener.resume()
            except Exception:
                pass
            try:
                bus.emit_threadsafe("ghost_loop_idle")
            except Exception:
                pass

    # ── Diagnostics ─────────────────────────────────────────────────────

    def diagnose(self) -> dict:
        """Diagnostic info for --doctor."""
        from bantz.agent.voice_capture import voice_capture
        from bantz.agent.stt import stt_engine

        return {
            "running": self._running,
            "busy": self._busy,
            "total_transcriptions": self._total_transcriptions,
            "voice_capture": voice_capture.diagnose(),
            "stt": stt_engine.diagnose(),
        }


# Module singleton
ghost_loop = GhostLoop()
