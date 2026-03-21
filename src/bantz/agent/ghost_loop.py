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
    │       └─ 5. Conversation window: listen for 60s follow-ups   │
    └───────────────────────────────────────────────────────────────┘

After a successful transcription the Ghost Loop stays in *conversation
mode* for ``CONVERSATION_WINDOW_SEC`` seconds.  During this window a
follow-up command can be spoken without saying the wake word again.
If no speech is detected within 30 s (VAD capture timeout) the window
expires and the wake-word listener is resumed.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from bantz.core.event_bus import bus, Event

log = logging.getLogger("bantz.ghost_loop")

# After a successful transcription, keep listening for this many seconds
# without requiring another wake word.
CONVERSATION_WINDOW_SEC = 60


class GhostLoop:
    """Orchestrates the wake → capture → STT → dispatch cycle."""

    def __init__(self) -> None:
        self._running = False
        self._busy = False  # True while capture+STT is in progress
        self._total_transcriptions: int = 0
        self._last_text: str = ""
        self._thread_pool: Optional[threading.Thread] = None
        # Monotonic deadline for conversation-mode follow-ups (0 = not active)
        self._conversation_end: float = 0.0

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
        self._conversation_end = 0.0
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
            "conversation_mode": self._conversation_end > time.monotonic(),
        }

    # ── Event handler ───────────────────────────────────────────────────

    @staticmethod
    def _preload_stt() -> None:
        """Load the STT model in the background at startup."""
        try:
            bus.emit_threadsafe("stt_model_loading")
            log.info("Ghost Loop: pre-loading STT model …")
            from bantz.agent.stt import stt_engine
            stt_engine._ensure_model()
            bus.emit_threadsafe("stt_model_ready")
            log.info("Ghost Loop: STT model ready")
        except Exception as exc:
            log.debug("Ghost Loop: STT preload failed — %s", exc)
            bus.emit_threadsafe("stt_model_failed", error=str(exc))

    def _on_wake_event(self, event: Event) -> None:
        """Called (on the bus dispatcher) when wake word is detected.

        Spawns the capture → STT → dispatch pipeline on a background
        thread to avoid blocking the event bus dispatcher.
        """
        if self._busy:
            log.debug("Ghost Loop: already busy, ignoring wake word")
            return

        thread = threading.Thread(
            target=self._capture_and_transcribe,
            kwargs={"release_mic": True},
            name="bantz-ghost-loop",
            daemon=True,
        )
        thread.start()

    # ── Pipeline (runs on background thread) ────────────────────────────

    def _capture_and_transcribe(self, *, release_mic: bool = True) -> None:
        """Blocking pipeline: capture audio → STT → emit to bus.

        Args:
            release_mic: When True, pause the wake word listener to free
                the mic before opening the capture stream.  Set to False
                for conversation-mode continuations where the wake word
                listener is already paused and the mic is available.
        """
        self._busy = True
        should_chain = False  # set True on successful transcription

        try:
            # 0. Release mic from wake word listener (first call only)
            if release_mic:
                try:
                    from bantz.agent.wake_word import wake_listener
                    wake_listener.pause()
                    # 500ms for ALSA/PulseAudio to fully release the device
                    time.sleep(0.50)
                    log.info("Ghost Loop: [1/4] mic released, starting capture")
                except Exception as exc:
                    log.warning("Ghost Loop: could not pause wake word — %s", exc)

            # 1. Emit "listening" event so TUI shows indicator
            bus.emit_threadsafe("ghost_loop_listening")

            # 2. Capture audio with VAD
            from bantz.agent.voice_capture import voice_capture
            log.info("Ghost Loop: [2/4] recording with VAD…")
            pcm_bytes = voice_capture.capture()

            if not pcm_bytes:
                log.warning("Ghost Loop: [2/4] FAILED — no audio captured (mic busy or VAD timeout)")
                bus.emit_threadsafe("voice_no_speech", reason="no_audio")
                return

            log.info("Ghost Loop: [2/4] captured %d bytes of PCM audio", len(pcm_bytes))

            # 3. Emit "transcribing" event
            bus.emit_threadsafe("ghost_loop_transcribing")

            # 4. Transcribe via faster-whisper
            from bantz.agent.stt import stt_engine
            log.info("Ghost Loop: [3/4] sending to STT (faster-whisper)…")
            text = stt_engine.transcribe(pcm_bytes)

            if not text:
                log.warning("Ghost Loop: [3/4] FAILED — STT returned empty")
                bus.emit_threadsafe("voice_no_speech", reason="empty_transcription")
                return

            self._total_transcriptions += 1
            self._last_text = text
            log.info("Ghost Loop: [4/4] ✅ TRANSCRIBED → \"%s\"", text[:120])

            # 5. Dispatch as voice input
            bus.emit_threadsafe("voice_input", text=text)
            log.info("Ghost Loop: voice_input emitted to EventBus")

            # Refresh conversation window — user can speak again without wake word
            self._conversation_end = time.monotonic() + CONVERSATION_WINDOW_SEC
            should_chain = True

        except Exception as exc:
            log.error("Ghost Loop: pipeline EXCEPTION at step — %s: %s",
                      type(exc).__name__, exc, exc_info=True)
            bus.emit_threadsafe("voice_no_speech", reason=f"pipeline_error: {exc}")
        finally:
            self._busy = False

            if should_chain and self._running and time.monotonic() < self._conversation_end:
                # ── Conversation mode: chain another capture cycle ───────
                remaining = int(self._conversation_end - time.monotonic())
                log.info("Ghost Loop: conversation mode active (%ds left) — listening again", remaining)
                # Update TUI header to show we're still listening
                bus.emit_threadsafe("ghost_loop_listening")
                thread = threading.Thread(
                    target=self._capture_and_transcribe,
                    # Mic is already free — skip the pause step
                    kwargs={"release_mic": False},
                    name="bantz-ghost-loop-conv",
                    daemon=True,
                )
                thread.start()
            else:
                # ── Return to idle: resume wake word listener ─────────
                try:
                    from bantz.agent.wake_word import wake_listener
                    wake_listener.resume()
                    log.info("Ghost Loop: wake word listener resumed")
                except Exception as exc:
                    log.warning("Ghost Loop: could not resume wake word — %s", exc)
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
            "conversation_mode": self._conversation_end > time.monotonic(),
            "voice_capture": voice_capture.diagnose(),
            "stt": stt_engine.diagnose(),
        }


# Module singleton
ghost_loop = GhostLoop()
