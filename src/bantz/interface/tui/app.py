"""
Bantz v3 — Textual TUI Application

Entry point for the terminal user interface.
Left panel: chat.  Right panel: system status + clock.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, Static
from textual import work

from bantz.core.brain import brain
from bantz.core.types import BrainResult
from bantz.core.event_bus import bus, Event
from bantz.config import config
from bantz.interface.tui.panels.system import SystemStatus
from bantz.interface.tui.panels.chat import ChatLog, ThinkingLabel
from bantz.interface.tui.panels.header import (
    OperationsHeader,
    ServiceHealthChanged,
    ServiceStatus,
    MemoryCountUpdated,
)
from bantz.interface.tui.widgets.toast import (
    ToastContainer,
    ToastType,
    ToastData,
    ToastAccepted,
    ToastDismissed,
    ToastExpired,
)
from textual.message import Message

log = logging.getLogger("bantz.tui")
_STYLES_PATH = Path(__file__).parent / "styles.tcss"


# ── Messages ────────────────────────────────────────────────────────

class WakeWordDetected(Message):
    """Fired (from the audio thread via call_from_thread) when the user says the wake word."""


class BantzEventMessage(Message):
    """Bridges EventBus → Textual main thread (#220, Sprint 3 Part 3).

    Bus subscribers call ``app.call_from_thread(app.post_message,
    BantzEventMessage(event))`` so the Textual event loop picks it up
    safely — no ThreadError.
    """

    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event


class BantzApp(App):
    TITLE = "BANTZ v3"
    SUB_TITLE = "your terminal host"

    CSS_PATH = _STYLES_PATH

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+c", "copy_selection", "Copy"),
        Binding("escape", "focus_input", "Focus"),
        Binding("ctrl+s", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+u", "toggle_quiet", "Quiet", show=False),
        Binding("ctrl+f", "toggle_focus", "Focus mode", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._pending: BrainResult | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield OperationsHeader(id="ops-header")
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield ChatLog(id="chat-log", highlight=True, markup=True)
                yield Static("", id="thinking-area")
                with Horizontal(id="input-row"):
                    yield Input(
                        placeholder="Tell Bantz something... (Ctrl+Q to quit, Ctrl+L to clear)",
                        id="chat-input",
                    )
            with Vertical(id="right-panel"):
                yield SystemStatus()
        yield ToastContainer(id="toast-container")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_system("Bantz v3 started.")
        chat.add_system(f"Model: {config.ollama_model}")
        chat.add_system("─" * 38)

        from bantz.core.time_context import time_ctx
        chat.add_bantz(time_ctx.greeting_line())

        self._check_ollama()
        self._warm_up_ollama()
        self._enrich_butler_greeting()
        self._start_gps_server()
        self._start_stationary_checker()
        self._start_morning_briefing_timer()
        self._start_reminder_checker()
        self._start_digest_checker()
        self._start_observer()
        self._start_intervention_processor()
        self._wire_brain_toast_hook()
        self._subscribe_event_bus()
        self._start_wake_word_listener()
        self.query_one("#chat-input", Input).focus()

    async def action_quit(self) -> None:
        try:
            from bantz.core.gps_server import gps_server
            await gps_server.stop()
        except Exception:
            pass
        # Stop observer daemon (#124)
        try:
            from bantz.agent.observer import observer
            observer.stop()
        except Exception:
            pass
        # Stop wake word listener (#165)
        try:
            from bantz.agent.wake_word import wake_listener
            wake_listener.stop()
        except Exception:
            pass
        # Tear down EventBus subscriptions (#220)
        self._unsubscribe_event_bus()
        try:
            await bus.shutdown()
        except Exception:
            pass
        self.exit()

    @work(exclusive=False)
    async def _start_gps_server(self) -> None:
        try:
            from bantz.core.gps_server import gps_server
            ok = await gps_server.start()
            if ok:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_system(f"GPS: {gps_server.url}")
                chat.add_system(f"Relay: {gps_server.relay_topic}")
        except Exception:
            pass

    def _start_stationary_checker(self) -> None:
        self.set_interval(300, self._check_stationary)

    @work(exclusive=False)
    async def _check_stationary(self) -> None:
        try:
            from bantz.core.places import places
            notice = places.check_stationary()
            if notice:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(notice)
                chat.scroll_end()
        except Exception:
            pass

    @work(exclusive=False)
    async def _warm_up_ollama(self) -> None:
        try:
            from bantz.llm.ollama import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass

    @work(exclusive=False)
    async def _enrich_butler_greeting(self) -> None:
        from bantz.core.memory import memory
        try:
            from bantz.core.session import session_tracker
            from bantz.core.butler import butler

            session_info = session_tracker.on_launch()

            def _run_greet():
                return asyncio.run(butler.greet(session_info))

            loop = asyncio.get_event_loop()
            text = await asyncio.wait_for(
                loop.run_in_executor(None, _run_greet), timeout=8
            )
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="startup")
                except Exception:
                    pass
        except (asyncio.CancelledError, Exception):
            pass

    def _start_morning_briefing_timer(self) -> None:
        self.set_interval(60, self._check_morning_briefing)

    def _start_reminder_checker(self) -> None:
        self.set_interval(config.reminder_check_interval, self._check_reminders)

    @work(exclusive=False)
    async def _check_reminders(self) -> None:
        try:
            from bantz.core.scheduler import scheduler
            from bantz.core.memory import memory

            due = scheduler.check_due()
            if not due:
                return

            # Route through intervention queue if available
            try:
                from bantz.agent.interventions import (
                    intervention_queue,
                    intervention_from_reminder,
                )
                if intervention_queue.initialized:
                    for r in due:
                        repeat_tag = r.get("repeat", "none")
                        iv = intervention_from_reminder(
                            title=r["title"],
                            repeat=repeat_tag,
                            ttl=config.intervention_toast_ttl,
                        )
                        intervention_queue.push(iv)
                    return
            except Exception:
                pass

            # Fallback: render directly
            chat = self.query_one("#chat-log", ChatLog)
            for r in due:
                repeat_tag = f" (repeats {r['repeat']})" if r['repeat'] != 'none' else ''
                text = f"\u23f0 Reminder: {r['title']}{repeat_tag}"
                chat.add_bantz(text)
                try:
                    memory.add("assistant", text, tool_used="reminder")
                except Exception:
                    pass
                # TTS: speak reminders aloud (#171)
                try:
                    from bantz.agent.tts import tts_engine
                    if tts_engine.available():
                        await tts_engine.speak_background(text)
                except Exception:
                    pass
            chat.scroll_end()
        except Exception:
            pass

    @work(exclusive=False)
    async def _check_morning_briefing(self) -> None:
        try:
            from bantz.personality.greeting import greeting_manager
            from bantz.core.memory import memory

            text = await greeting_manager.morning_briefing_if_due()
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="briefing")
                except Exception:
                    pass
                # TTS: speak the briefing if available (#131/#171)
                try:
                    from bantz.agent.tts import tts_engine
                    if config.tts_auto_briefing and tts_engine.available():
                        await tts_engine.speak_background(text)
                except Exception:
                    pass
        except Exception:
            pass

    def _start_digest_checker(self) -> None:
        self.set_interval(60, self._check_digest)

    def _start_observer(self) -> None:
        """Start the background stderr observer daemon (#124).

        Routes error events through the intervention queue (#126) when
        available, otherwise falls back to direct chat rendering.
        """
        if not config.observer_enabled:
            return
        try:
            from bantz.agent.observer import observer, ErrorEvent, Severity

            def _on_error(event: ErrorEvent) -> None:
                """Deliver observer notifications via intervention queue."""
                # Try intervention queue first
                try:
                    from bantz.agent.interventions import (
                        intervention_queue,
                        intervention_from_observer,
                    )
                    if intervention_queue.initialized:
                        iv = intervention_from_observer(
                            raw_text=event.raw_text,
                            severity=event.severity.value,
                            analysis=event.analysis or "",
                            ttl=config.intervention_toast_ttl,
                        )
                        intervention_queue.push(iv)
                        return
                except Exception:
                    pass

                # Fallback: render directly to chat
                try:
                    chat = self.query_one("#chat-log", ChatLog)
                    if event.severity == Severity.CRITICAL:
                        icon = "\U0001f6a8"  # 🚨
                        msg = f"{icon} **Terminal Error Detected**\n```\n{event.raw_text[:500]}\n```"
                        if event.analysis:
                            msg += f"\n\n**Analysis:** {event.analysis}"
                        chat.add_error(msg)
                    elif event.severity == Severity.WARNING:
                        icon = "\u26a0\ufe0f"  # ⚠️
                        chat.add_system(f"{icon} stderr: {event.raw_text[:200]}")
                    else:
                        chat.add_system(f"\u2139\ufe0f stderr: {event.raw_text[:120]}")
                    chat.scroll_end()
                except Exception:
                    pass

            observer.on_error = _on_error
            observer.threshold = Severity(config.observer_severity_threshold)
            observer.buffer._batch_sec = config.observer_batch_seconds
            observer.buffer._dedup_window = config.observer_dedup_window
            observer.classifier._model = config.observer_analysis_model
            observer.classifier._enable_llm = config.observer_enable_llm
            observer.start()

            chat = self.query_one("#chat-log", ChatLog)
            chat.add_system(f"Observer: monitoring stderr (threshold={config.observer_severity_threshold})")
        except Exception as exc:
            log.debug("Observer start failed: %s", exc)

    def _start_intervention_processor(self) -> None:
        """Start the unified intervention processor (#126).

        Combines RL suggestions, observer alerts, and reminders into a
        single priority queue with rate limiting, TTL, and focus/quiet modes.
        """
        if not config.rl_enabled:
            return
        try:
            from bantz.agent.interventions import intervention_queue
            if not intervention_queue.initialized:
                return

            # Apply config modes
            if config.intervention_quiet_mode:
                intervention_queue.set_quiet(True)
            if config.intervention_focus_mode:
                intervention_queue.set_focus(True)

            # RL suggestion feeder — runs on the RL interval
            self.set_interval(config.rl_suggestion_interval, self._feed_rl_suggestions)
            # Intervention display processor — checks every 2 seconds
            self.set_interval(2, self._process_interventions)

            # Auto-focus mode checker (#127) — polls app detector
            if config.app_detector_enabled and config.app_detector_auto_focus:
                self.set_interval(config.app_detector_polling_interval, self._check_auto_focus)

            chat = self.query_one("#chat-log", ChatLog)
            chat.add_system(
                f"Interventions: rate={config.intervention_rate_limit}/h "
                f"ttl={config.intervention_toast_ttl:.0f}s"
            )
        except Exception as exc:
            log.debug("Intervention processor start failed: %s", exc)

    @work(exclusive=False)
    async def _feed_rl_suggestions(self) -> None:
        """Periodically ask the RL engine for a suggestion and push to queue."""
        try:
            from bantz.agent.rl_engine import rl_engine, encode_state
            from bantz.agent.interventions import (
                intervention_queue,
                intervention_from_rl,
            )
            from bantz.core.time_context import time_ctx

            if not rl_engine.initialized:
                return

            snap = time_ctx.snapshot()
            import datetime
            day = datetime.datetime.now().strftime("%A").lower()

            location = "home"
            try:
                from bantz.core.places import places
                label = places.current_place_label()
                if label:
                    location = label.lower()
            except Exception:
                pass

            recent_tool = ""
            try:
                from bantz.core.memory import memory
                recent = memory.recent(1)
                if recent:
                    recent_tool = recent[0].get("tool_used", "") or ""
            except Exception:
                pass

            state = encode_state(
                time_segment=snap["segment_en"],
                day=day,
                location=location,
                recent_tool=recent_tool,
            )
            action = rl_engine.suggest(state)
            if action:
                # Build explainability reason with app context (#127)
                reason = f"{snap['segment_en'].title()} {day.title()} routine"
                if location != "home":
                    reason += f" at {location}"
                try:
                    from bantz.agent.app_detector import app_detector
                    if app_detector.initialized:
                        activity = app_detector.get_activity_category()
                        reason += f" ({activity.value})"
                        win = app_detector.get_active_window()
                        if win and win.name:
                            reason += f" — {win.name}"
                except Exception:
                    pass

                iv = intervention_from_rl(
                    action_value=action.value,
                    state_key=state.key,
                    reason=reason,
                    ttl=config.intervention_toast_ttl,
                )
                intervention_queue.push(iv)
        except Exception as exc:
            log.debug("RL suggestion feed failed: %s", exc)

    @work(exclusive=False)
    async def _process_interventions(self) -> None:
        """Pop from intervention queue and display as toast (#137).

        Handles TTL auto-dismiss with mild RL penalty, and renders
        source/reason labels for explainability.
        """
        try:
            from bantz.agent.interventions import (
                intervention_queue,
                Outcome,
                SOURCE_LABELS,
            )

            if not intervention_queue.initialized:
                return

            # Check if active intervention has expired (auto-dismiss)
            if intervention_queue.has_active:
                active = intervention_queue.active
                if active and active.expired:
                    iv = intervention_queue.expire_active()
                    if iv:
                        self._send_rl_feedback(iv, Outcome.AUTO_DISMISSED)
                return

            # Pop next intervention
            iv = intervention_queue.pop()
            if not iv:
                return

            # Push to toast container (#137) instead of ChatLog
            try:
                container = self.query_one("#toast-container", ToastContainer)
                container.push_toast(iv)
            except Exception:
                # Fallback: render directly in chat
                chat = self.query_one("#chat-log", ChatLog)
                source_tag = SOURCE_LABELS.get(iv.source, iv.source)
                chat.add_bantz(f"{iv.title}\n[{source_tag}: {iv.reason}]")
                chat.scroll_end()

            # Desktop notification (#153) — fire if TUI is not active
            try:
                from bantz.agent.notifier import notifier
                if notifier.initialized:
                    notifier.dispatch(iv)
            except Exception as exc:
                log.debug("Desktop notification dispatch failed: %s", exc)

        except Exception as exc:
            log.debug("Intervention processing failed: %s", exc)

    def _send_rl_feedback(self, iv: "Intervention", outcome: "Outcome") -> None:
        """Send reward feedback to the RL engine based on intervention outcome."""
        if not iv.action or not iv.state_key:
            return
        try:
            from bantz.agent.rl_engine import rl_engine, Reward
            from bantz.agent.interventions import Outcome as Oc

            reward_map = {
                Oc.ACCEPTED: Reward.ACCEPT,
                Oc.DISMISSED: Reward.DISMISS,
                Oc.NEVER: Reward.BLACKLIST,
                Oc.AUTO_DISMISSED: None,  # special: mild penalty
            }
            reward = reward_map.get(outcome)
            if outcome == Oc.AUTO_DISMISSED:
                # Mild penalty: user didn't see or ignored — not as harsh as dismiss
                rl_engine.reward(-0.1)
            elif reward is not None:
                rl_engine.reward(reward.value)
        except Exception as exc:
            log.debug("RL feedback failed: %s", exc)

    @work(exclusive=False)
    async def _check_auto_focus(self) -> None:
        """Auto-enable/disable focus mode based on app detector (#127).

        When the user is in a flow state (coding, media, video call),
        focus mode is enabled which drops LOW/MEDIUM interventions.
        """
        try:
            from bantz.agent.app_detector import app_detector
            from bantz.agent.interventions import intervention_queue

            if not app_detector.initialized or not intervention_queue.initialized:
                return

            should_focus = app_detector.should_enable_focus()
            if should_focus != intervention_queue.focus:
                intervention_queue.set_focus(should_focus)
                if should_focus:
                    activity = app_detector.get_activity_category()
                    log.debug("Auto-focus ON: user is %s", activity.value)
                else:
                    log.debug("Auto-focus OFF: user appears idle/browsing")
        except Exception as exc:
            log.debug("Auto-focus check failed: %s", exc)

    @work(exclusive=False)
    async def _check_digest(self) -> None:
        try:
            from bantz.core.digest import digest_manager
            from bantz.core.memory import memory

            # Check daily digest
            text = await digest_manager.daily_if_due()
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="digest")
                except Exception:
                    pass

            # Check weekly digest
            text = await digest_manager.weekly_if_due()
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="digest")
                except Exception:
                    pass
        except Exception:
            pass

    @work(exclusive=False)
    async def _check_ollama(self) -> None:
        from bantz.llm.ollama import ollama
        chat = self.query_one("#chat-log", ChatLog)
        ok = await ollama.is_available()
        if ok:
            chat.add_system(f"✓ Ollama connected → {config.ollama_model}")
        else:
            chat.add_error(f"Ollama unreachable: {config.ollama_base_url}")
            chat.add_system("  → Is `ollama serve` running?")

    def notify_service_health(
        self, service: str, status: ServiceStatus, detail: str = "",
    ) -> None:
        """Fire a ServiceHealthChanged event for the OperationsHeader.

        Call from anywhere: ``app.notify_service_health('ollama', ServiceStatus.UP)``
        This is the event-driven health hook — no periodic pinging needed.
        """
        try:
            header = self.query_one("#ops-header", OperationsHeader)
            header.post_message(ServiceHealthChanged(service, status, detail))
        except Exception:
            pass

    def notify_memory_counts(self, messages: int, sessions: int) -> None:
        """Fire a MemoryCountUpdated event (call after adding a message)."""
        try:
            header = self.query_one("#ops-header", OperationsHeader)
            header.post_message(MemoryCountUpdated(messages, sessions))
        except Exception:
            pass

    # ── Input handler ──────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return

        self.query_one("#chat-input", Input).value = ""
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_user(text)

        # Check if this is a response to an active intervention (#126)
        if self._handle_intervention_response(text, chat):
            return

        if self._pending is not None:
            await self._handle_confirm(text, chat)
            return

        self._start_processing(text, chat)

    def _handle_intervention_response(self, text: str, chat: "ChatLog") -> bool:
        """Check if user input is a response to an active intervention.

        Returns True if the input was consumed as an intervention response.
        Also removes the corresponding toast widget (#137).
        """
        try:
            from bantz.agent.interventions import intervention_queue, Outcome
        except Exception:
            return False

        if not intervention_queue.has_active:
            return False

        low = text.lower().strip()
        _ACCEPT = {"accept", "yes", "y", "ok", "sure", "go", "do it"}
        _DISMISS = {"dismiss", "no", "n", "skip", "not now", "later", "nah"}
        _NEVER = {"never", "block", "stop", "never again", "don't"}

        if low in _ACCEPT:
            iv = intervention_queue.respond(Outcome.ACCEPTED)
            if iv:
                chat.add_system("✓ Accepted.")
                self._send_rl_feedback(iv, Outcome.ACCEPTED)
                self._remove_intervention_toast(iv)
            return True
        elif low in _DISMISS:
            iv = intervention_queue.respond(Outcome.DISMISSED)
            if iv:
                chat.add_system("✗ Dismissed.")
                self._send_rl_feedback(iv, Outcome.DISMISSED)
                self._remove_intervention_toast(iv)
            return True
        elif low in _NEVER:
            iv = intervention_queue.respond(Outcome.NEVER)
            if iv:
                chat.add_system("⊘ Blocked — will never suggest this again.")
                self._send_rl_feedback(iv, Outcome.NEVER)
                self._remove_intervention_toast(iv)
            return True

        # Not an intervention response — auto-dismiss and pass through
        iv = intervention_queue.expire_active()
        if iv:
            self._send_rl_feedback(iv, Outcome.AUTO_DISMISSED)
            self._remove_intervention_toast(iv)
        return False

    def _remove_intervention_toast(self, iv: "Intervention") -> None:
        """Remove the toast widget showing this intervention (#137)."""
        try:
            container = self.query_one("#toast-container", ToastContainer)
            container.remove_by_intervention(iv)
        except Exception:
            pass

    @work(exclusive=False)
    async def _start_processing(self, text: str, chat: ChatLog) -> None:
        self._busy = True
        self._show_thinking(True)

        try:
            result = await brain.process(text)
        except Exception as exc:
            self._show_thinking(False)
            self._busy = False
            err_name = type(exc).__name__
            chat.add_error(f"Network/tool error: {err_name} — {exc}")
            chat.add_system("  Check your internet connection and try again.")
            return

        # ── Streaming response ──
        if result.stream is not None:
            self._show_thinking(False)
            if result.tool_used:
                chat.add_tool(result.tool_used)
            chat.stream_start()

            accumulated = ""
            try:
                async for token in result.stream:
                    accumulated += token
                    chat.stream_token(token)
            except Exception as exc:
                chat.stream_end()
                self._busy = False
                chat.add_error(f"Stream error: {exc}")
                return

            full_text = chat.stream_end()
            self._busy = False

            from bantz.core.finalizer import strip_markdown
            cleaned = strip_markdown(full_text)

            from bantz.core.memory import memory as _mem
            _mem.add("assistant", cleaned, tool_used=result.tool_used)
            try:
                from bantz.core.brain import brain as _brain
                await _brain._graph_store(text, cleaned, result.tool_used)
            except Exception:
                pass

            self._update_header_counts()
            return

        # ── Non-streaming response ──
        self._show_thinking(False)
        self._busy = False

        if result.needs_confirm:
            self._pending = result
            chat.add_bantz(result.response)
        else:
            if result.tool_used:
                chat.add_tool(result.tool_used)
            chat.add_bantz(result.response)
        self._update_header_counts()

    async def _handle_confirm(self, text: str, chat: ChatLog) -> None:
        pending = self._pending
        self._pending = None
        confirmed = text.lower().strip() in ("yes", "y", "ok", "evet", "e", "tamam")
        if confirmed:
            self._busy = True
            self._show_thinking(True)
            try:
                if pending.pending_tool and pending.pending_args:
                    from bantz.tools import registry as _reg
                    tool = _reg.get(pending.pending_tool)
                    if tool:
                        tr = await tool.execute(**pending.pending_args)
                        self._show_thinking(False)
                        self._busy = False
                        chat.add_tool(pending.pending_tool)
                        chat.add_bantz(tr.output if tr.success else f"Error: {tr.error}")
                        return
                result = await brain.process(
                    pending.pending_command,
                    confirmed=True,
                )
                self._show_thinking(False)
                self._busy = False
                if result.tool_used:
                    chat.add_tool(result.tool_used)
                chat.add_bantz(result.response)
            except Exception as exc:
                self._show_thinking(False)
                self._busy = False
                err_name = type(exc).__name__
                chat.add_error(f"Network/tool error: {err_name} — {exc}")
                chat.add_system("  Check your internet connection and try again.")
        else:
            chat.add_system("Cancelled.")

    # ── Thinking indicator ─────────────────────────────────────────────────

    def _show_thinking(self, show: bool) -> None:
        area = self.query_one("#thinking-area", Static)
        if show:
            area.update("[dim cyan]  ⟳ thinking...[/]")
        else:
            area.update("")

    # ── Actions ────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", ChatLog).clear()

    def action_copy_selection(self) -> None:
        import pyperclip
        chat = self.query_one("#chat-log", ChatLog)
        try:
            lines = chat.lines
            for line in reversed(lines):
                text = line.text if hasattr(line, 'text') else str(line)
                if 'Bantz' in text or '◆' in text:
                    clean = text.replace('◆ Bantz', '').strip()
                    pyperclip.copy(clean)
                    chat.add_system('Copied ✓')
                    return
        except Exception:
            pass

    def action_focus_input(self) -> None:
        """Escape: dismiss top toast if any, otherwise focus input (#137)."""
        try:
            container = self.query_one("#toast-container", ToastContainer)
            if container.has_toasts:
                container.dismiss_top()
                return
        except Exception:
            pass
        self.query_one("#chat-input", Input).focus()

    def action_toggle_quiet(self) -> None:
        """Toggle quiet mode — suppress non-critical interventions."""
        try:
            from bantz.agent.interventions import intervention_queue
            if not intervention_queue.initialized:
                return
            intervention_queue.set_quiet(not intervention_queue.quiet)
            state = "ON" if intervention_queue.quiet else "OFF"
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_system(f"Quiet mode: {state}")
        except Exception:
            pass

    def action_toggle_focus(self) -> None:
        """Toggle focus mode — only HIGH/CRITICAL interventions pass."""
        try:
            from bantz.agent.interventions import intervention_queue
            if not intervention_queue.initialized:
                return
            intervention_queue.set_focus(not intervention_queue.focus)
            state = "ON" if intervention_queue.focus else "OFF"
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_system(f"Focus mode: {state}")
        except Exception:
            pass

    def action_toggle_sidebar(self) -> None:
        """Toggle the right system-status panel visibility (#134)."""
        try:
            panel = self.query_one("#right-panel")
            panel.display = not panel.display
        except Exception:
            pass

    # ── Toast message handlers (#137) ──────────────────────────────────

    def on_toast_accepted(self, event: ToastAccepted) -> None:
        """User accepted an action toast → RL reward + queue response."""
        try:
            from bantz.agent.interventions import intervention_queue, Outcome
            if intervention_queue.has_active and intervention_queue.active is event.intervention:
                iv = intervention_queue.respond(Outcome.ACCEPTED)
                if iv:
                    self._send_rl_feedback(iv, Outcome.ACCEPTED)
                    chat = self.query_one("#chat-log", ChatLog)
                    chat.add_system("✓ Accepted.")
        except Exception:
            pass

    def on_toast_dismissed(self, event: ToastDismissed) -> None:
        """User dismissed a toast → RL penalty + queue response."""
        try:
            from bantz.agent.interventions import intervention_queue, Outcome
            if intervention_queue.has_active and intervention_queue.active is event.intervention:
                iv = intervention_queue.respond(Outcome.DISMISSED)
                if iv:
                    self._send_rl_feedback(iv, Outcome.DISMISSED)
        except Exception:
            pass

    def on_toast_expired(self, event: ToastExpired) -> None:
        """Toast auto-expired after TTL → mild RL penalty."""
        try:
            from bantz.agent.interventions import intervention_queue, Outcome
            if intervention_queue.has_active and intervention_queue.active is event.intervention:
                iv = intervention_queue.expire_active()
                if iv:
                    self._send_rl_feedback(iv, Outcome.AUTO_DISMISSED)
        except Exception:
            pass

    # ── Public toast API (#137) ────────────────────────────────────────

    def push_toast(
        self, title: str, reason: str = "", toast_type: str = "info",
    ) -> None:
        """Push a simple (non-intervention) toast.  Thread-safe.

        Called from brain, observer, or any background context via
        ``app.call_from_thread(app.push_toast, ...)``.
        """
        _TYPE_MAP = {
            "info": ToastType.INFO,
            "success": ToastType.SUCCESS,
            "warning": ToastType.WARNING,
            "error": ToastType.ERROR,
            "action": ToastType.ACTION,
        }
        try:
            tt = _TYPE_MAP.get(toast_type, ToastType.INFO)
            data = ToastData(title=title, reason=reason)
            container = self.query_one("#toast-container", ToastContainer)
            container.push_toast(data, tt)
        except Exception:
            pass

    def _wire_brain_toast_hook(self) -> None:
        """Connect notification system to this app's push_toast (#137, #225)."""
        try:
            # Canonical: set on notification_manager directly
            from bantz.core import notification_manager as _notif_mod
            _notif_mod.toast_callback = self._on_brain_toast
        except Exception:
            pass
        try:
            # Backward compat: old brain_mod._toast_callback
            from bantz.core import brain as brain_mod
            brain_mod._toast_callback = self._on_brain_toast
        except Exception:
            pass

    def _on_brain_toast(
        self, title: str, reason: str = "", toast_type: str = "info",
    ) -> None:
        """Receive toast events from brain context (possibly threaded)."""
        try:
            self.call_from_thread(self.push_toast, title, reason, toast_type)
        except Exception:
            pass

    def _update_header_counts(self) -> None:
        """Push latest memory/session counts to OperationsHeader (event-driven)."""
        try:
            from bantz.data import data_layer
            if data_layer.conversations:
                stats = data_layer.conversations.stats()
                self.notify_memory_counts(
                    stats.get("total_messages", 0),
                    stats.get("total_conversations", 0),
                )
        except Exception:
            pass

    # ── EventBus → TUI Bridge (#220, Sprint 3 Part 3) ──────────────────

    def _subscribe_event_bus(self) -> None:
        """Subscribe to relevant EventBus events and relay to the TUI.

        The bus dispatcher runs as a separate asyncio task.  We must
        NOT touch Textual widgets directly from a bus callback — that
        would trigger ``ThreadError``.  Instead each callback uses
        ``call_from_thread(post_message, BantzEventMessage(...))`` to
        safely enqueue a Textual ``Message`` processed on the main
        thread by ``on_bantz_event_message()``.

        Events subscribed:
          - ``wake_word_detected``  → focus input + "Yes boss? 🎤"
          - ``ambient_change``      → update system status ambient label
          - ``health_alert``        → push warning toast
        """
        bus.bind_loop()  # idempotent — ensures dispatcher task exists

        # Keep refs so we can bus.off() on quit
        self._bus_relay = self._relay_bus_event  # prevent GC

        bus.on("wake_word_detected", self._bus_relay)
        bus.on("ambient_change", self._bus_relay)
        bus.on("health_alert", self._bus_relay)

        log.debug("EventBus → TUI bridge active (3 subscriptions)")

    def _relay_bus_event(self, event: Event) -> None:
        """Relay a single bus Event into the Textual message loop.

        Called by the EventBus dispatcher task — NOT the main thread.
        ``call_from_thread`` schedules ``post_message`` on Textual's
        own event loop so we never hit a ``ThreadError``.
        """
        try:
            self.call_from_thread(self.post_message, BantzEventMessage(event))
        except Exception:
            # App shutting down or not yet mounted — silently discard
            pass

    def _unsubscribe_event_bus(self) -> None:
        """Remove all bus subscriptions (called on quit)."""
        relay = getattr(self, "_bus_relay", None)
        if relay is not None:
            bus.off("wake_word_detected", relay)
            bus.off("ambient_change", relay)
            bus.off("health_alert", relay)

    def on_bantz_event_message(self, msg: BantzEventMessage) -> None:
        """Dispatch bus events to the appropriate TUI handler.

        Runs on the Textual main thread — safe to touch any widget.
        """
        event = msg.event
        name = event.name
        try:
            if name == "wake_word_detected":
                self._on_bus_wake_word(event)
            elif name == "ambient_change":
                self._on_bus_ambient_change(event)
            elif name == "health_alert":
                self._on_bus_health_alert(event)
        except Exception:
            log.debug("on_bantz_event_message(%s) error", name, exc_info=True)

    # ── per-event handlers (main thread safe) ─────────────────────────

    def _on_bus_wake_word(self, event: Event) -> None:
        """Wake word detected via bus → focus input + greet."""
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_bantz("Yes boss? 🎤")
        chat.scroll_end()
        self.query_one("#chat-input", Input).focus()

    def _on_bus_ambient_change(self, event: Event) -> None:
        """Ambient noise classification changed → update system status."""
        label = event.data.get("label", "")
        rms = event.data.get("rms")
        try:
            panel = self.query_one(SystemStatus)
            # SystemStatus can optionally display ambient info
            if hasattr(panel, "update_ambient"):
                panel.update_ambient(label, rms)
        except Exception:
            pass

    def _on_bus_health_alert(self, event: Event) -> None:
        """Health rule fired → push a warning toast."""
        title = event.data.get("title", "Health Alert")
        reason = event.data.get("reason", "")
        self.push_toast(title, reason, "warning")

    # ── Wake Word Listener (#165) ─────────────────────────────────────

    def _start_wake_word_listener(self) -> None:
        """Start the always-on wake word listener in a daemon thread.

        The legacy ``_on_wake`` closure is no longer needed — the sensor
        now emits ``wake_word_detected`` on the EventBus and the TUI
        bridge picks it up via ``_subscribe_event_bus()`` (#220 Part 3).
        """
        if not config.wake_word_enabled:
            return
        if not config.picovoice_access_key:
            log.warning("Wake word enabled but BANTZ_PICOVOICE_ACCESS_KEY not set")
            return

        try:
            from bantz.agent.wake_word import wake_listener

            ok = wake_listener.start()  # no on_wake callback — bus handles it
            if ok:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_system("Wake Word: listening for \"Hey Bantz\"")
        except Exception as exc:
            log.debug("Wake word start failed: %s", exc)

    def on_wake_word_detected(self, _msg: WakeWordDetected) -> None:
        """Handle wake word detection — focus input + notify user.

        .. deprecated:: Sprint 3 Part 3
           Kept for backward compat.  Primary path is now
           ``on_bantz_event_message`` → ``_on_bus_wake_word``.
        """
        try:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_bantz("Yes boss? 🎤")
            chat.scroll_end()
            self.query_one("#chat-input", Input).focus()
        except Exception as exc:
            log.debug("Wake word handler error: %s", exc)


def run() -> None:
    """Launch the Bantz TUI."""
    BantzApp().run()
