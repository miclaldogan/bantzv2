"""
Bantz — Textual TUI Application (#431, Ghost Loop integration)

Main Textual App class for the Bantz assistant.  Bridges the EventBus
(background threads) with the Textual UI thread via a ``BantzEventMessage``
Textual Message, keeping all widget mutations on the correct thread.

Event flow:
    Background thread
        └─ bus.emit_threadsafe("voice_input", text=...)
             └─ _bus_handler(event) — registered in _subscribe_event_bus
                  └─ app.call_from_thread(app.post_message, BantzEventMessage(event))
                       └─ on_bantz_event_message(self, msg)   ← Textual main thread
                            └─ _on_bus_voice_input / _on_bus_ghost_listening / …
"""
from __future__ import annotations

import logging

from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Footer, Header, Label

from bantz.core.event_bus import bus, Event

log = logging.getLogger("bantz.tui")


# ═══════════════════════════════════════════════════════════════════════════
# Message carrier — bridges bus events to the Textual main thread
# ═══════════════════════════════════════════════════════════════════════════

class BantzEventMessage(Message):
    """Carries an EventBus Event from a background thread to the Textual thread."""

    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event


# ═══════════════════════════════════════════════════════════════════════════
# BantzApp — main Textual application
# ═══════════════════════════════════════════════════════════════════════════

class BantzApp(App):
    """Main Textual application for the Bantz assistant."""

    CSS = """
    Screen {
        background: $surface;
    }
    #status {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._busy: bool = False
        self._ghost_loop = None
        self._bus_handler = None  # shared handler registered in _subscribe_event_bus

    # ── Compose ─────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Ready.", id="status")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to bus events and start the Ghost Loop."""
        self._subscribe_event_bus()
        self._start_ghost_loop()

    def action_quit(self) -> None:
        """Stop the ghost loop then quit."""
        try:
            from bantz.agent.ghost_loop import ghost_loop
            ghost_loop.stop()
        except Exception:
            pass
        self._unsubscribe_event_bus()
        self.exit()

    # ── Ghost Loop startup ───────────────────────────────────────────────

    def _start_ghost_loop(self) -> None:
        """Start the Ghost Loop if enabled."""
        try:
            from bantz.agent.ghost_loop import ghost_loop
            ghost_loop.start()
            self._ghost_loop = ghost_loop
        except Exception as exc:
            log.debug("Ghost Loop: could not start — %s", exc)

    # ── EventBus bridge ──────────────────────────────────────────────────

    def _subscribe_event_bus(self) -> None:
        """Subscribe to all Ghost Loop / STT bus events."""

        def _handler(event: Event) -> None:
            try:
                self.call_from_thread(self.post_message, BantzEventMessage(event))
            except Exception:
                pass

        self._bus_handler = _handler
        bus.on("voice_input", _handler)
        bus.on("ghost_loop_listening", _handler)
        bus.on("ghost_loop_transcribing", _handler)
        bus.on("ghost_loop_idle", _handler)
        bus.on("voice_no_speech", _handler)
        bus.on("stt_model_loading", _handler)
        bus.on("stt_model_ready", _handler)
        bus.on("stt_model_failed", _handler)

    def _unsubscribe_event_bus(self) -> None:
        """Unsubscribe from all Ghost Loop / STT bus events."""
        _handler = self._bus_handler
        if _handler is None:
            return
        bus.off("voice_input", _handler)
        bus.off("ghost_loop_listening", _handler)
        bus.off("ghost_loop_transcribing", _handler)
        bus.off("ghost_loop_idle", _handler)
        bus.off("voice_no_speech", _handler)
        bus.off("stt_model_loading", _handler)
        bus.off("stt_model_ready", _handler)
        bus.off("stt_model_failed", _handler)

    # ── Textual message handler (runs on Textual main thread) ────────────

    def on_bantz_event_message(self, message: BantzEventMessage) -> None:
        """Dispatch incoming bus events to the appropriate handler."""
        name = message.event.name
        if name == "voice_input":
            self._on_bus_voice_input(message.event)
        elif name == "ghost_loop_listening":
            self._on_bus_ghost_listening(message.event)
        elif name == "ghost_loop_transcribing":
            self._on_bus_ghost_transcribing(message.event)
        elif name == "voice_no_speech":
            self._on_bus_voice_no_speech(message.event)
        elif name == "stt_model_loading":
            self._on_bus_stt_model_loading(message.event)
        elif name == "stt_model_ready":
            self._on_bus_stt_model_ready(message.event)
        elif name == "stt_model_failed":
            self._on_bus_stt_model_failed(message.event)

    # ── Per-event handlers ───────────────────────────────────────────────

    def _on_bus_voice_input(self, event: Event) -> None:
        """Handle a transcribed voice command from the Ghost Loop."""
        if self._busy:
            log.debug("TUI: dropping voice_input — already busy")
            return
        text = event.data.get("text", "").strip()
        if not text:
            return
        self._busy = True
        try:
            status = self.query_one("#status", Label)
            status.update(f"You said: {text}")
        except Exception:
            pass
        # Relay to brain via asyncio (fire-and-forget)
        import asyncio
        async def _dispatch():
            try:
                from bantz.core.brain import brain
                await brain.handle_message(text)
            except Exception as exc:
                log.error("TUI: brain dispatch error — %s", exc)
            finally:
                self._busy = False
        asyncio.get_event_loop().create_task(_dispatch())

    def _on_bus_ghost_listening(self, event: Event) -> None:
        """Update status bar to show the listening indicator."""
        try:
            status = self.query_one("#status", Label)
            status.update("Listening…")
        except Exception:
            pass

    def _on_bus_ghost_transcribing(self, event: Event) -> None:
        """Update status bar to show transcription in progress."""
        try:
            status = self.query_one("#status", Label)
            status.update("Transcribing…")
        except Exception:
            pass

    def _on_bus_voice_no_speech(self, event: Event) -> None:
        """Reset status bar when no speech was detected."""
        reason = event.data.get("reason", "")
        log.debug("TUI: voice_no_speech — %s", reason)
        try:
            status = self.query_one("#status", Label)
            status.update("Ready.")
        except Exception:
            pass

    def _on_bus_stt_model_loading(self, event: Event) -> None:
        """Show that the STT model is loading."""
        try:
            status = self.query_one("#status", Label)
            status.update("Loading STT model…")
        except Exception:
            pass

    def _on_bus_stt_model_ready(self, event: Event) -> None:
        """Confirm the STT model is ready."""
        try:
            status = self.query_one("#status", Label)
            status.update("STT model ready.")
        except Exception:
            pass

    def _on_bus_stt_model_failed(self, event: Event) -> None:
        """Warn the user that the STT model failed to load."""
        error = event.data.get("error", "unknown error")
        log.warning("TUI: STT model failed — %s", error)
        try:
            status = self.query_one("#status", Label)
            status.update(f"⚠ Voice unavailable: {error}")
        except Exception:
            pass
