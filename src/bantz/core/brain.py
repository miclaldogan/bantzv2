"""Bantz v3 — Brain (Orchestrator).

Pure orchestrator: delegates to routing_engine, finalizer, memory_injector,
prompt_builder, translation_layer, rl_hooks, notification_manager, and
location_handler.  See each module’s docstring for details.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass, field as _dc_field
from typing import Any, AsyncIterator, Callable, Optional

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.data import data_layer
from bantz.core.profile import profile
from bantz.core.intent import cot_route, _ANAPHORA
from bantz.core.finalizer import (
    finalize as _finalize_fn,
    finalize_stream as _finalize_stream_fn,
    hallucination_check as _hallucination_check_fn,
    strip_markdown,
    strip_internal,
)
from bantz.tools import registry, ToolResult
from bantz.core.context import BantzContext  # noqa: F401  — re-export for compat
from bantz.core.types import BrainResult, Attachment  # noqa: F401  — canonical def in types.py
from bantz.core.routing_engine import (
    quick_route as _quick_route_fn,
    dispatch_internal as _dispatch_internal,
    execute_plan as _execute_plan_fn,
    handle_maintenance as _handle_maintenance_fn,
    handle_list_reflections as _handle_list_reflections_fn,
    handle_run_reflection as _handle_run_reflection_fn,
)
from bantz.core.memory_injector import (
    inject as _inject_memory,
    style_hint as _style_hint,
    formality_hint as _formality_hint,
    graph_context as _graph_ctx_fn,
    deep_memory_context as _deep_memory_ctx_fn,
)
from bantz.core.prompt_builder import (
    CHAT_SYSTEM,      # noqa: F401  — re-export for backward compat
    COMMAND_SYSTEM,    # noqa: F401  — re-export + _generate_command
    build_chat_system as _build_chat_system,
    is_refusal as _is_refusal,
)
# Re-exported for backward compat — canonical impl in translation_layer.py
from bantz.core.translation_layer import (  # noqa: F401
    POSITIVE_FEEDBACK_KWS,
    NEGATIVE_FEEDBACK_KWS,
    detect_feedback as _detect_feedback,
    get_bridge,  # exposed at module level so tests can patch bantz.core.brain.get_bridge (#435)
)
# Toast compat shim — canonical impl in notification_manager.py (#225)
import bantz.core.notification_manager as _notif_mod

log = logging.getLogger("bantz.brain")

try:
    from bantz.memory.bridge import palace_bridge
except ImportError:
    palace_bridge = None  # mempalace not installed
_toast_callback = None  # written by app.py / tests

# ── Butler-voice error messages (#442) ────────────────────────────────────────
_BUTLER_NET_ERROR = (
    "I'm afraid I cannot reach the service at present, ma'am. "
    "The wires appear to be indisposed."
)
_BUTLER_TOOL_ERROR = (
    "I was unable to complete that task, ma'am. "
    "The mechanism has encountered a difficulty."
)
_BUTLER_LLM_ERROR = (
    "I'm afraid I encountered a slight mechanical difficulty, ma'am. "
    "Please do try again presently."
)

# Topic-discipline fence appended to the chat system prompt (live re-run
# tests 19/20: chat answers inherited the previous turn's topic).
_TOPIC_DISCIPLINE = (
    "\n\nAnswer the user's CURRENT message. Earlier conversation is context "
    "for tone and references only — do NOT continue a previous topic unless "
    "the current message explicitly refers back to it."
)


def _mood_suffix() -> str:
    """Personality 'mood bias' dial → an instruction appended to the chat
    system prompt. Read live from config so Settings changes apply at once."""
    try:
        from bantz.config import config
        mood = (config.mood_bias or "tolerant").lower()
    except Exception:
        mood = "tolerant"
    if mood == "impatient":
        return "\n\nBe concise. Skip pleasantries."
    if mood == "resigned":
        return "\n\nKeep responses minimal and dry."
    return ""  # tolerant = default disposition


def _exc_to_butler(exc: Exception) -> str:
    """Map an exception to a butler-voice error message (#442)."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if any(x in msg or x in name for x in (
        "connection", "timeout", "network", "unreachable", "refused",
        "timed out", "ssl", "socket",
    )):
        return _BUTLER_NET_ERROR
    return _BUTLER_TOOL_ERROR


# Immediate "working on it" lines emitted before a tool blocks, so the user
# sees intent instead of dead silence (speed illusion). Module-level so the
# C1 recovery loop (#503) can announce on every executed iteration.
_PRE_TOOL_LINES: dict[str, str] = {
    "calendar":      "Let me check our schedule...",
    "gmail":         "Let me check your inbox...",
    "weather":       "Let me check the weather...",
    "web_search":    "Let me look that up for you...",
    "shell":         "Running that for you...",
    "system":        "Checking system status...",
    "reminder":      "Checking your reminders...",
    "filesystem":    "Accessing the file...",
    "document":      "Let me read that document...",
    "news":          "Fetching the latest news...",
    "read_url":      "Fetching that page...",
    "visual_click":     "Clicking that for you...",
    "input_control":    "Performing that action...",
    "accessibility":    "Analysing the screen...",
    "browser_control":  "Operating the browser for you...",
    "vision_execute":   "On it — watching the screen and working through this...",
    "screen_query":     "Taking a look at your screen...",
    "web_research":     "Diving deep into that — this can take a few minutes...",
    "classroom":     "Checking your assignments...",
    "delegate_task":    "Delegating to a specialist agent...",
    "run_workflow":     "Running your workflow...",
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used only to *bound* the C1
    recovery loop's re-decide cost against ``tool_loop_token_budget`` (#501).
    Deterministic and provider-agnostic; not reported as real usage (the eval
    runner measures actual tokens)."""
    return max(1, len(text) // 4)


@dataclass
class _RecoveryOutcome:
    """Result of the C1 observe→re-decide loop (#503).

    Exactly one of ``terminal`` / ``result`` carries the outcome:

    - ``terminal`` set → ``process()`` returns it directly (needs-confirm,
      tool-not-found, or exhausted-on-exception butler reply).
    - ``result`` set → a ToolResult (success or honest failure) that flows
      into the normal post-processing / finalize path, with ``tool_name`` /
      ``tool_args`` reflecting the action that actually ran last.

    ``iterations`` is always the full per-iteration trace (contract #500).
    """

    iterations: list
    terminal: "BrainResult | None" = None
    result: "ToolResult | None" = None
    tool_name: str = ""
    tool_args: dict = _dc_field(default_factory=dict)


def _notify_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Compat shim → ``notification_manager.notify_toast``."""
    _notif_mod.toast_callback = _toast_callback
    _notif_mod.notify_toast(title, reason, toast_type)


def _load_optional_tool(modname: str, reason: str = "") -> None:
    """Import an optional tool module so it self-registers.

    Logs a ``warning`` (not silence) when it can't load, so a tool vanishing
    from the registry is visible at startup instead of only surfacing later as
    an opaque "tool not found" at routing time (audit T2).
    """
    try:
        importlib.import_module(modname)
    except (ImportError, ModuleNotFoundError) as exc:
        log.warning("Optional tool %r not registered (%s): %s",
                    modname, reason or "optional dependency missing", exc)


class Brain:
    def __init__(self) -> None:
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
        import bantz.tools.weather      # noqa: F401
        _load_optional_tool("bantz.tools.news", "defusedxml")
        import bantz.tools.web_search   # noqa: F401
        import bantz.tools.web_reader   # noqa: F401
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401
        import bantz.tools.classroom    # noqa: F401
        import bantz.tools.reminder     # noqa: F401
        _load_optional_tool("bantz.tools.document", "PDF/DOCX deps")
        _load_optional_tool("bantz.tools.accessibility", "AT-SPI2/gi")
        # gui_action removed (#185) — superseded by visual_click
        _load_optional_tool("bantz.tools.visual_click")  # (#185)
        # input_control must be imported here explicitly (#122): otherwise it
        # only registers as a side effect of a lazy import elsewhere, so the
        # router advertised "input_control" but the registry never held it.
        _load_optional_tool("bantz.tools.input_control")  # (#122)
        _load_optional_tool("bantz.tools.browser_control")
        _load_optional_tool("bantz.tools.screenshot_tool")
        _load_optional_tool("bantz.tools.desktop")         # (#322)
        _load_optional_tool("bantz.tools.delegate_task")   # (#321)
        _load_optional_tool("bantz.tools.workflow_tool")   # (#323)
        import bantz.tools.summarizer    # noqa: F401  (Architect's Revision)
        _load_optional_tool("bantz.tools.screen_query_tool", "PIL/vision deps")
        _load_optional_tool("bantz.tools.vision_execute", "PIL/vision deps")
        self._memory_ready = False
        self._graph_ready = False
        # Session state: stores last tool results for contextual follow-ups
        self._last_messages: list[dict] = []   # last listed emails [{id, from, subject, ...}]
        self._last_events: list[dict] = []     # last listed calendar events
        self._last_draft: dict | None = None   # last email draft {to, subject, body}
        self._last_tool_output: str = ""     # last tool output snippet (generic)
        self._last_tool_name: str = ""       # which tool produced it
        self._feedback_ctx: str = ""  # one-shot RLHF context (#180)
        # Which caller/session currently owns the follow-up state above. The
        # Brain is a process-global singleton shared across the desktop and
        # every Telegram user; when the active session changes we wipe the
        # follow-up context so one caller's emails/output/screen can never be
        # served as another's context (audit C2).
        self._state_owner: str = ""
        # Turn-based TTL (#276): context expires after N process() calls
        self._turn_counter: int = 0
        self._context_turn: int = 0  # turn when context was last stored
        self._CONTEXT_TTL: int = 3   # expire after 3 turns of no related queries
        # Per-turn memory-recall memo (audit M4): _finalize_stream is tried
        # first and recalls, returns None for short output, then _finalize
        # recalls again — same query, twice. Cache by en_input within a turn.
        self._recall_cache: tuple[str, Any] | None = None
        # Screen vision context (#189+): VLM description of last screenshot
        self._last_screen_description: str = ""
        self._screen_description_turn: int = -1
        self._pending_vlm_task: object = None  # asyncio.Task | None
        # Continuous Awareness (#325): background collector task
        self._awareness_task: object = None  # asyncio.Task | None

    def _ensure_memory(self) -> None:
        if not self._memory_ready:
            data_layer.init(config)
            self._memory_ready = True
            # Initialise multi-agent subsystem (#321)
            if config.multi_agent_enabled:
                try:
                    from bantz.agent.agent_manager import agent_manager
                    agent_manager.init()
                except Exception as exc:
                    log.warning("Failed to initialise multi-agent: %s", exc)
            # Start continuous awareness collector on first process() call (#325)
            if config.awareness_enabled:
                self._start_awareness()

    def _start_awareness(self) -> None:
        """Launch the AwarenessCollector background task (idempotent)."""
        if self._awareness_task is not None:
            return
        try:
            from bantz.agent.awareness import awareness_collector
            awareness_collector.interval_s = config.awareness_interval_s
            self._awareness_task = asyncio.create_task(
                awareness_collector.run(),
                name="bantz-awareness",
            )
            log.info("Awareness collector started (interval=%.0fs)", config.awareness_interval_s)
        except Exception as exc:
            log.warning("Failed to start awareness collector: %s", exc)

    def _desktop_context(self) -> str:
        """Compat shim → ``memory_injector.desktop_context`` (#227)."""
        from bantz.core.memory_injector import desktop_context
        return desktop_context()

    async def _ensure_graph(self) -> None:
        if not self._graph_ready and palace_bridge:
            await palace_bridge.init()
            self._graph_ready = True

    async def _graph_context(self, user_msg: str) -> str:
        """Compat shim → ``memory_injector.graph_context`` (#227)."""
        return await _graph_ctx_fn(user_msg)

    async def _vector_context(self, user_msg: str, limit: int = 3) -> str:
        """Compat shim → ``memory_injector.vector_context`` (#227)."""
        from bantz.core.memory_injector import vector_context
        return await vector_context(user_msg, limit=limit)

    async def _deep_memory_context(self, user_msg: str) -> str:
        """Compat shim → ``memory_injector.deep_memory_context`` (#227)."""
        return await _deep_memory_ctx_fn(user_msg)

    def _fire_embeddings(self) -> None:
        """No-op: embeddings now handled by MemPalace (ChromaDB built-in)."""
        pass

    async def _graph_store(self, user_msg: str, assistant_msg: str,
                           tool_used: str | None = None,
                           tool_data: dict | None = None) -> None:
        """Store entities from exchange in palace (fire-and-forget)."""
        if palace_bridge and palace_bridge.enabled:
            try:
                await palace_bridge.store_exchange(
                    user_msg, assistant_msg, tool_used, tool_data)
            except Exception:
                pass

    def _get_bridge(self):
        """Delegate to translation_layer (#226)."""
        from bantz.core.translation_layer import get_bridge
        return get_bridge()

    async def _to_en(self, text: str) -> str:
        """Delegate to translation_layer (#226)."""
        from bantz.core.translation_layer import to_en
        return await to_en(text)

    def _resolve_message_ref(self, text: str) -> str | None:
        """Delegate to translation_layer (#226)."""
        from bantz.core.translation_layer import resolve_message_ref
        return resolve_message_ref(text, self._last_messages)

    # ── Dynamic context injection (#275) ─────────────────────────────

    _EMAIL_HINTS = frozenset({
        "email", "mail", "gmail", "inbox", "message", "reply", "forward",
        "send", "compose", "draft", "thread", "unread",
    })
    _CALENDAR_HINTS = frozenset({
        "calendar", "event", "meeting", "schedule", "appointment",
        "today", "tomorrow", "week", "upcoming",
    })

    def _is_context_expired(self) -> bool:
        """Check if stored tool context has expired (turn-based TTL #276)."""
        return (self._turn_counter - self._context_turn) > self._CONTEXT_TTL

    def _clear_stale_context(self) -> None:
        """Wipe tool context that has exceeded its TTL (#276)."""
        if self._is_context_expired():
            if self._last_messages or self._last_events or self._last_tool_output:
                log.debug("TTL expired (turn %d vs stored %d) — clearing tool context",
                          self._turn_counter, self._context_turn)
            self._last_messages = []
            self._last_events = []
            self._last_tool_output = ""
            self._last_tool_name = ""

    def _switch_session(self, session_key: str) -> None:
        """Isolate per-conversation follow-up state across callers (audit C2).

        When the active session changes, wipe the singleton's follow-up
        context (last emails/events/tool output/screen/draft/recall) and drop
        any in-flight VLM task, so one caller's data cannot surface as
        another's context. Single-user desktop use keeps the same key every
        turn, so nothing is cleared and behaviour is unchanged.

        Residual: this is a clear-on-switch boundary safe for *serialized*
        turns; truly concurrent interleaving of two sessions would still race
        the shared active state — the complete fix is per-session state
        objects (tracked as C2b). Also note conversation *history* in the data
        layer is still a single shared session (C2b).
        """
        # Read the owner defensively: tests build half-constructed Brains via
        # Brain.__new__ (no __init__), and the default-session fast path must
        # return before touching any other possibly-missing attribute.
        if session_key == getattr(self, "_state_owner", ""):
            return
        if (self._last_messages or self._last_events
                or self._last_tool_output or self._last_draft):
            log.info("Session switch %r → %r: clearing follow-up context (C2)",
                     self._state_owner, session_key)
        self._last_messages = []
        self._last_events = []
        self._last_tool_output = ""
        self._last_tool_name = ""
        self._last_draft = None
        self._last_screen_description = ""
        self._recall_cache = None
        # An in-flight VLM task belongs to the previous session's screenshot;
        # cancel it so its description can't land in the new session's state.
        task = self._pending_vlm_task
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
            self._pending_vlm_task = None
        self._state_owner = session_key

    def _store_tool_context(self, tool_name: str, result: ToolResult) -> None:
        """Persist tool results for contextual follow-ups (#276).

        Stores structured data (email IDs, event IDs) plus a truncated
        generic output snippet.  Resets the TTL turn counter.
        """
        self._context_turn = self._turn_counter  # reset TTL
        if result.data:
            if result.data.get("messages"):
                self._last_messages = result.data["messages"]
            if result.data.get("events"):
                self._last_events = result.data["events"]
        # Always store a generic snippet for non-email/calendar tools
        self._last_tool_output = (result.output or "")[:500]
        self._last_tool_name = tool_name

    @staticmethod
    def _embed_metadata(output: str, tool_name: str, data: dict | None) -> str:
        """Embed essential IDs into tool output text (#276).

        Appends a [CONTEXT: ...] block with IDs so the LLM can see them
        in conversation history without schema changes to the DB.
        """
        if not data:
            return output
        import json as _json
        meta: dict = {}
        # Email: embed message IDs
        if data.get("messages"):
            meta["message_ids"] = [
                {"id": m.get("id", ""), "from": m.get("from", ""), "subject": m.get("subject", "")}
                for m in data["messages"][:5]
            ]
        # Calendar: embed event IDs
        if data.get("events"):
            meta["event_ids"] = [
                {"id": e.get("id", ""), "summary": e.get("summary", "")}
                for e in data["events"][:5]
            ]
        # Single-item results (read_message, create_event, etc.)
        if data.get("message_id"):
            meta["message_id"] = data["message_id"]
        if data.get("event_id"):
            meta["event_id"] = data["event_id"]
        if data.get("thread_id"):
            meta["thread_id"] = data["thread_id"]
        if data.get("path"):
            meta["path"] = data["path"]
        if not meta:
            return output
        return f"{output}\n[CONTEXT: {_json.dumps(meta, ensure_ascii=False)}]"

    def _build_tool_context(self, en_input: str) -> str:
        """Build dynamic tool context — injected only when relevant.

        Avoids bloating the CoT prompt with stale email/event data
        when the user asks about weather, shell commands, etc.
        Expires after ``_CONTEXT_TTL`` turns of unrelated queries (#276).
        """
        self._clear_stale_context()
        parts: list[str] = []
        lower = en_input.lower()

        # Inject recent email IDs only if query relates to email
        if self._last_messages and any(h in lower for h in self._EMAIL_HINTS):
            lines = ["RECENT EMAIL RESULTS (use these IDs for follow-ups):"]
            for m in self._last_messages[:5]:
                mid = m.get("id", "?")
                frm = m.get("from", "?")
                subj = m.get("subject", "(no subject)")
                lines.append(f"  - ID: {mid} | From: {frm} | Subject: \"{subj}\"")
            parts.append("\n".join(lines))
            self._context_turn = self._turn_counter  # refresh TTL on use

        # Inject recent calendar events only if query relates to calendar
        if self._last_events and any(h in lower for h in self._CALENDAR_HINTS):
            lines = ["RECENT CALENDAR EVENTS (use these for follow-ups):"]
            for ev in self._last_events[:5]:
                eid = ev.get("id", "?")
                title = ev.get("summary", ev.get("title", "?"))
                when = ev.get("start_local", ev.get("start", ev.get("date", "?")))
                lines.append(f"  - ID: {eid} | Title: \"{title}\" | When: {when}")
            parts.append("\n".join(lines))
            self._context_turn = self._turn_counter  # refresh TTL on use

        # Screen vision context: inject VLM description of last screenshot
        # Expires after _CONTEXT_TTL turns so stale screen state doesn't confuse routing
        if self._last_screen_description and (
            self._turn_counter - self._screen_description_turn <= self._CONTEXT_TTL
        ):
            parts.append(
                f"CURRENT SCREEN STATE (from last screenshot — use this to resolve "
                f"'there', 'here', 'it', 'that element', 'the page'):\n"
                f"{self._last_screen_description[:700]}"
            )

        # Generic: inject last tool output for any follow-up referencing it
        if self._last_tool_output and not parts:
            # Check if user is asking a follow-up about the previous result
            followup_hints = {
                "that", "this", "it", "those", "these", "the result",
                "the output", "what about", "more", "details", "explain",
                "tell me more", "summarize", "summary", "again",
                "read", "open", "reply", "forward", "delete",
                "first", "second", "third", "last", "next",
                "play", "click", "watch", "launch",
                # Turkish follow-up words
                "aç", "tıkla", "onu", "şunu", "bunu", "oynat",
            }
            if any(h in lower for h in followup_hints):
                parts.append(
                    f"PREVIOUS TOOL RESULT (from {self._last_tool_name}):\n"
                    f"{self._last_tool_output[:300]}"
                )
                self._context_turn = self._turn_counter  # refresh TTL on use

        # Continuous Awareness context (#325): inject desktop state when enabled
        if config.awareness_enabled:
            try:
                from bantz.agent.awareness import awareness_collector
                awareness_ctx = awareness_collector.get_current_context()
                if awareness_ctx:
                    parts.append(awareness_ctx)
            except Exception as exc:
                log.debug("Awareness context injection failed: %s", exc)

        return "\n\n".join(parts)

    async def _vlm_describe_screen(self, img_bytes: bytes) -> None:
        """Run VLM screen description in background and store for next CoT call."""
        try:
            import base64 as _b64
            from bantz.config import config as _cfg
            if not _cfg.vlm_enabled:
                return
            from bantz.vision.remote_vlm import describe_screen
            b64 = _b64.b64encode(img_bytes).decode()
            result = await describe_screen(b64, timeout=12)
            if result.success and result.raw_text:
                self._last_screen_description = result.raw_text
                self._screen_description_turn = self._turn_counter
                log.debug("VLM screen description stored (%d chars)", len(result.raw_text))
        except Exception as exc:
            log.debug("VLM screen describe failed: %s", exc)

    # ── Continuous Awareness helpers (#325) ───────────────────────────

    # Deictic / visual-reference keywords (English + Turkish)
    _AWARENESS_SCREENSHOT_TRIGGERS: frozenset[str] = frozenset({
        # English
        "this", "here", "what is this", "fix this", "what's this",
        "what do you see", "what's on", "look at", "analyze this",
        "check this", "read this", "explain this",
        # Turkish
        "bu", "bak", "buna", "bunu", "ne var", "ne görüyorsun",
        "ekran", "burası", "burada", "şuna", "şu",
    })

    def _maybe_inject_awareness_screenshot(
        self, en_input: str, orig_input: str,
    ) -> None:
        """If the message is a deictic reference, pre-load the awareness screenshot
        into the VLM pipeline so the next LLM call has visual context.

        This is a fire-and-forget background task — any failure is silent.
        Uses word-boundary matching so short tokens like "bu" don't falsely
        trigger on substrings (e.g. "Istanbul").
        """
        import re as _re
        lower = (en_input + " " + orig_input).lower()
        triggered = any(
            _re.search(r"(?<!\w)" + _re.escape(kw) + r"(?!\w)", lower)
            for kw in self._AWARENESS_SCREENSHOT_TRIGGERS
        )
        if not triggered:
            return
        try:
            from bantz.agent.awareness import awareness_collector
            screenshot_path = awareness_collector.get_screenshot_for_vlm()
            if not screenshot_path:
                return
            import pathlib
            img_bytes = pathlib.Path(screenshot_path).read_bytes()
            if img_bytes and self._pending_vlm_task is None:
                self._pending_vlm_task = asyncio.create_task(
                    self._vlm_describe_screen(img_bytes),
                    name="bantz-awareness-vlm",
                )
                log.debug(
                    "Awareness: deictic trigger detected — loading screenshot for VLM (%d bytes)",
                    len(img_bytes),
                )
        except Exception as exc:
            log.debug("Awareness: screenshot VLM injection failed: %s", exc)

    @staticmethod
    def _quick_route(orig: str, en: str) -> dict | None:
        """Compat shim → ``routing_engine.quick_route`` (#228)."""
        return _quick_route_fn(orig, en)

    # ── C1 observe→re-decide loop (#503, audit C1) ───────────────────
    @staticmethod
    def _iter_record(
        index: int, tool_name: str, tool_args: dict, decision_source: str, *,
        result: "ToolResult | None", exc: "Exception | None", gated: str | None,
        wall_ms: int, error: str | None = None,
    ) -> dict:
        """One per-iteration trace entry in the loop_eval contract shape (#500).

        Brain does not measure LLM tokens (the eval runner does), so tokens_in/
        tokens_out are reported as 0 here; only ``wall_ms`` is real.
        """
        if result is not None:
            res = {
                "success": bool(result.success),
                "error": result.error or None,
                "output_excerpt": (result.output or "")[:500],
            }
        elif exc is not None:
            res = {"success": False, "error": str(exc), "output_excerpt": ""}
        else:
            res = {"success": False, "error": error, "output_excerpt": ""}
        return {
            "index": index,
            "route": "tool",
            "tool_name": tool_name,
            "tool_args": dict(tool_args) if isinstance(tool_args, dict) else {},
            "decision_source": decision_source,
            "result": res,
            "exception": f"{type(exc).__name__}: {exc}" if exc is not None else None,
            "gated": gated,
            "tokens_in": 0,
            "tokens_out": 0,
            "wall_ms": int(wall_ms),
        }

    @staticmethod
    def _observation_block(
        tool_name: str, tool_args: dict,
        result: "ToolResult | None", exc: "Exception | None",
    ) -> str:
        """Carry THIS turn's tool failure into the next cot_route call as an
        observation (extends the existing PREVIOUS-TOOL-RESULT carrier, #276),
        so the re-decision can see what went wrong and pick a DIFFERENT action.
        """
        import json as _json
        try:
            args_str = _json.dumps(tool_args, ensure_ascii=False)[:300]
        except Exception:
            args_str = str(tool_args)[:300]
        if exc is not None:
            detail = f"raised {type(exc).__name__}: {exc}"
        elif result is not None:
            detail = f"failed: {(result.error or result.output or 'no error message')[:300]}"
        else:
            detail = "is not an available tool"
        return (
            "PREVIOUS TOOL ATTEMPT FAILED — observe and choose a DIFFERENT "
            "action (or answer honestly that it can't be done). Do NOT repeat "
            "the same failing call:\n"
            f"  tool: {tool_name}\n"
            f"  args: {args_str}\n"
            f"  outcome: {detail}"
        )

    async def _execute_with_recovery(
        self, *,
        tool_name: str, tool_args: dict, risk: str, requires_confirm: Any,
        confirmed: bool, en_input: str, recent_history: list, tool_ctx: str,
    ) -> _RecoveryOutcome:
        """Bounded observe→re-decide loop around tool execution (audit C1, #503).

        Default (``config.tool_loop_max_steps == 1``): the body runs exactly
        once — byte-identical to the historical single-shot path, with ZERO
        extra LLM calls. When ``max_steps > 1``, a tool failure (``success=
        False`` OR a raised exception) is fed back as an observation and
        ``cot_route`` is re-invoked with ``skip_fastpath=True`` (#502) to pick a
        different action. The destructive + autonomy gates are re-checked every
        iteration; a re-decided action that needs confirmation STOPS the loop
        and returns the prompt — it is never auto-approved (the user's earlier
        confirmation authorises only the first decision).
        """
        from bantz.tools.shell import is_destructive

        max_steps = max(1, config.tool_loop_max_steps)
        token_budget = config.tool_loop_token_budget
        autonomy = (config.autonomy or "high").lower()

        iterations: list = []
        overhead_tokens = 0
        decision_source = "initial"
        result: ToolResult | None = None
        last_exc: Exception | None = None

        for index in range(1, max_steps + 1):
            # ── Gates (re-checked every iteration) ──────────────────────
            # Deterministic destructive detection for shell (audit S2): don't
            # trust the model's self-reported risk on shell commands.
            if tool_name == "shell" and is_destructive(tool_args.get("command", "")):
                if risk != "destructive":
                    log.info("shell.is_destructive promoted risk safe/moderate → destructive")
                risk = "destructive"

            # Autonomy dial (audit S1). The user's confirmation authorises ONLY
            # the first decision; a re-decided (substituted) action must earn
            # its own confirmation — never auto-approve it.
            iter_confirmed = confirmed if index == 1 else False
            rc = requires_confirm
            if rc is None:  # older verdict without the field
                rc = risk == "destructive"
            if autonomy == "absolute":
                need_confirm = False
            elif autonomy == "low":
                need_confirm = rc and not iter_confirmed
            else:  # medium / high (default): destructive only, legacy toggle
                need_confirm = (
                    risk == "destructive"
                    and config.shell_confirm_destructive
                    and not iter_confirmed
                )

            if need_confirm:
                cmd_str = tool_args.get("command", tool_name)
                warn = (
                    f"⚠️  Destructive operation: [{tool_name}] `{cmd_str}`\n"
                    f"Confirm? (yes/no)"
                )
                data_layer.conversations.add("assistant", warn)
                iterations.append(self._iter_record(
                    index, tool_name, tool_args, decision_source,
                    result=None, exc=None, gated="needs_confirm", wall_ms=0,
                ))
                return _RecoveryOutcome(
                    iterations=iterations,
                    terminal=BrainResult(
                        response=warn, tool_used=tool_name, needs_confirm=True,
                        pending_command=cmd_str, pending_tool=tool_name,
                        pending_args=tool_args, iterations=iterations,
                    ),
                )

            # ── Resolve tool ────────────────────────────────────────────
            tool = registry.get(tool_name)
            if tool is None:
                iterations.append(self._iter_record(
                    index, tool_name, tool_args, decision_source,
                    result=None, exc=None, gated=None, wall_ms=0,
                    error=f"Tool not found: {tool_name}",
                ))
                result = None
                if index >= max_steps:
                    err = f"Tool not found: {tool_name}"
                    data_layer.conversations.add("assistant", err)
                    return _RecoveryOutcome(
                        iterations=iterations,
                        terminal=BrainResult(response=err, tool_used=None,
                                             iterations=iterations),
                    )
                # else: treat as a failed observation and re-decide below.
            else:
                # ── Pre-tool announcement (speed illusion) ──────────────
                pre_line = _PRE_TOOL_LINES.get(tool_name)
                if pre_line:
                    try:
                        from bantz.core.event_bus import bus as _bus
                        await _bus.emit("pre_tool_message", message=pre_line, tool=tool_name)
                        await asyncio.sleep(0)  # yield so the line renders first
                    except Exception:
                        pass

                # ── Execute ─────────────────────────────────────────────
                t0 = time.monotonic()
                exc: Exception | None = None
                try:
                    result = await tool.execute(**tool_args)
                except Exception as e:  # noqa: BLE001 — recoverable failure class
                    log.exception("Tool %r raised unexpectedly: %s", tool_name, e)
                    exc = e
                    last_exc = e
                    result = None
                wall_ms = int((time.monotonic() - t0) * 1000)

                iterations.append(self._iter_record(
                    index, tool_name, tool_args, decision_source,
                    result=result, exc=exc, gated=None, wall_ms=wall_ms,
                ))

                if result is not None and result.success:
                    return _RecoveryOutcome(
                        iterations=iterations, result=result,
                        tool_name=tool_name, tool_args=tool_args,
                    )
                if exc is None:
                    # A clean success=False failure (not an exception): keep the
                    # ToolResult so, if the loop stops, the finalizer narrates
                    # the real error exactly as the single-shot path does today.
                    last_exc = None

            # ── Failure: observe → re-decide (budget & steps permitting) ─
            if index >= max_steps:
                break
            observation = self._observation_block(tool_name, tool_args, result, last_exc)
            est = _estimate_tokens(observation + en_input)
            if overhead_tokens + est > token_budget:
                log.info("C1 loop: token budget %d exhausted (+%d) — stop re-decide",
                         token_budget, est)
                break
            overhead_tokens += est
            redecide_ctx = (tool_ctx + "\n\n" + observation).strip()
            plan2, _err2 = await cot_route(
                en_input, registry.all_schemas(),
                recent_history=recent_history,
                tool_context=redecide_ctx,
                skip_fastpath=True,   # SAFETY (#502): never re-enter fast-path
            )
            if not plan2 or plan2.get("route") != "tool" or not plan2.get("tool_name"):
                # Model chose to stop / answer in chat → honest give-up. Keep
                # the last failed result for the finalizer to narrate.
                break
            tool_name = plan2.get("tool_name") or tool_name
            tool_args = plan2.get("tool_args") or {}
            risk = plan2.get("risk_level", "safe")
            requires_confirm = plan2.get("requires_confirm")
            decision_source = "llm"

        # ── Loop exhausted / gave up ────────────────────────────────────
        if result is None and last_exc is not None:
            # Ended on a raised exception with nothing recovered → historical
            # butler reply (identical to the pre-#503 single-shot behaviour).
            return _RecoveryOutcome(
                iterations=iterations,
                terminal=BrainResult(
                    response=_exc_to_butler(last_exc), tool_used=tool_name,
                    iterations=iterations,
                ),
            )
        if result is None:
            err = f"Tool not found: {tool_name}"
            data_layer.conversations.add("assistant", err)
            return _RecoveryOutcome(
                iterations=iterations,
                terminal=BrainResult(response=err, tool_used=None,
                                     iterations=iterations),
            )
        return _RecoveryOutcome(
            iterations=iterations, result=result,
            tool_name=tool_name, tool_args=tool_args,
        )

    # ── RL & Intervention hooks (#125, #126) ─────────────────────────

    def _rl_reward_hook(self, tool_name: str, result: ToolResult) -> None:
        """Delegate to rl_hooks — offloaded via AsyncDBExecutor (#226)."""
        try:
            from bantz.core.rl_hooks import rl_reward_hook
            asyncio.get_event_loop().create_task(rl_reward_hook(tool_name, result))
        except Exception:
            pass  # never crash the pipeline

    def _push_toast(
        self, title: str, reason: str = "", toast_type: str = "info",
    ) -> None:
        """Push a toast notification (#137 → #225 notification_manager)."""
        from bantz.core.notification_manager import push_toast
        push_toast(title, reason, toast_type)

    # ── Workflow handlers → routing_engine (#228) ──────────────────────

    async def _handle_maintenance(self, dry_run: bool = False) -> str:
        """Compat shim → ``routing_engine.handle_maintenance`` (#228)."""
        return await _handle_maintenance_fn(dry_run)

    def _handle_list_reflections(self, limit: int = 5) -> str:
        """Compat shim → ``routing_engine.handle_list_reflections`` (#228)."""
        return _handle_list_reflections_fn(limit)

    async def _handle_run_reflection(self, dry_run: bool = False) -> str:
        """Compat shim → ``routing_engine.handle_run_reflection`` (#228)."""
        return await _handle_run_reflection_fn(dry_run)

    async def _handle_location(self) -> str:
        """Delegate to location_handler (#225)."""
        from bantz.core.location_handler import handle_location
        return await handle_location()

    async def process(
        self,
        user_input: str,
        confirmed: bool = False,
        is_remote: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
        session_key: str = "",
    ) -> BrainResult:
        """Process *user_input* through the full pipeline.

        Args:
            progress_cb: Optional callback invoked at key pipeline stages
                         (translation, routing, …) so callers can display
                         progress to the user (#435).
            session_key: Identifies the calling conversation (e.g. a Telegram
                         user). Switching keys wipes the follow-up context so
                         callers cannot see each other's data (audit C2).
                         Empty string is the default single-user desktop
                         session.
        """
        # Isolate follow-up state per caller before anything reads it (C2).
        self._switch_session(session_key)
        self._is_remote = is_remote
        self._ensure_memory()
        await self._ensure_graph()
        # Show translation progress before the (potentially slow) MarianMT load
        if progress_cb:
            _b = get_bridge()
            if _b and _b.is_enabled():
                progress_cb("Translating\u2026")
        en_input = await self._to_en(user_input)
        tc = time_ctx.snapshot()
        self._turn_counter += 1  # (#276) advance TTL clock

        # ── Sentiment RLHF intercept (#180) ──────────────────────────
        # Uses RAW user_input (not en_input) to catch Turkish keywords
        # before the translation layer converts them to English.
        feedback = _detect_feedback(user_input)
        if feedback:
            # Offload RL write to AsyncDBExecutor thread-pool (#226)
            try:
                from bantz.core.rl_hooks import rl_feedback_reward
                asyncio.get_event_loop().create_task(
                    rl_feedback_reward(feedback, tc)
                )
            except Exception:
                pass  # never crash the pipeline
            if feedback == "positive":
                self._feedback_ctx = (
                    "\n[The user just praised you. Show humble butler gratitude — "
                    "a brief, dignified acknowledgement. Do not be excessive.]"
                )
            else:
                self._feedback_ctx = (
                    "\n[The user just scolded you. Show a brief moment of butler "
                    "composure under pressure. Apologise sincerely, ask how to "
                    "correct yourself. Do NOT grovel.]"
                )

        # NOTE: Intervention queue is now consumed by the TUI's toast
        # system (#137) instead of being popped here.  Brain no longer
        # prepends intervention text to chat responses.

        # Save user message ONCE — before any branching
        data_layer.conversations.add("user", user_input)
        self._recall_cache = None  # new turn → invalidate memoized recall (M4)

        # Await pending VLM screen description (started after last screenshot).
        # User typically takes 2-5s to read the photo before typing — VLM usually done.
        if self._pending_vlm_task is not None:
            import asyncio as _asyncio
            if not self._pending_vlm_task.done():
                try:
                    await _asyncio.wait_for(
                        _asyncio.shield(self._pending_vlm_task), timeout=3.0,
                    )
                except (_asyncio.TimeoutError, Exception):
                    pass
            self._pending_vlm_task = None

        recent_history = data_layer.conversations.context(n=6)

        # ═══════════════════════════════════════════════════════════════
        # NEW PIPELINE (#272): quick_route → cot_route → branch
        # ═══════════════════════════════════════════════════════════════

        # ── Step 1: Quick-route — hardware/UI controls + GUI fast-path ─
        quick = self._quick_route(user_input, en_input)
        if quick:
            q_tool = quick["tool"]
            q_args  = quick["args"]

            if q_tool.startswith("_"):
                # Internal tool (TTS, wake-word, etc.) — existing path
                internal = await _dispatch_internal(
                    q_tool, q_args,
                    user_input, en_input, tc,
                    is_remote=is_remote,
                )
                if internal is not None:
                    return internal
                # dispatch_internal returned None — fall through to cot_route
            else:
                # External tool fast-path (browser_control, visual_click)
                # bypasses the LLM entirely for reliable GUI command dispatch
                q_reg = registry.get(q_tool)
                if q_reg:
                    log.info("quick_route fast-dispatch: %s %s", q_tool, q_args)
                    q_result = await q_reg.execute(**q_args)
                    self._rl_reward_hook(q_tool, q_result)
                    if q_result.success:
                        self._store_tool_context(q_tool, q_result)
                    q_resp = await self._finalize(en_input, q_result, tc)
                    data_layer.conversations.add("assistant", q_resp, tool_used=q_tool)
                    # Keep the [CONTEXT:...] block in history (carries IDs for
                    # follow-ups) but never show it to the user (#276 leak).
                    q_resp = strip_internal(q_resp)
                    await self._graph_store(user_input, q_resp, q_tool,
                                            q_result.data if q_result else None)
                    self._fire_embeddings()
                    q_attachments: list[Attachment] = []
                    if q_result.success and q_result.data and q_result.data.get("screenshot"):
                        _img = q_result.data["screenshot"]
                        q_attachments.append(Attachment(
                            type="image", data=_img, caption="",
                            mime_type=q_result.data.get("mime_type", "image/jpeg"),
                        ))
                        try:
                            import asyncio as _ai
                            self._pending_vlm_task = _ai.create_task(
                                self._vlm_describe_screen(_img)
                            )
                        except Exception:
                            pass
                    return BrainResult(
                        response=q_resp, tool_used=q_tool,
                        tool_result=q_result, attachments=q_attachments,
                    )
                log.warning("quick_route: external tool '%s' not in registry — falling through", q_tool)

        # ── Awareness screenshot injection (#325) ─────────────────────
        # If the message contains a deictic reference word and awareness is
        # enabled, eagerly attach the latest screenshot to the VLM pipeline
        # so the LLM has visual context for words like "this", "here", etc.
        if config.awareness_enabled:
            self._maybe_inject_awareness_screenshot(en_input, user_input)

        # ── Step 2: cot_route — LLM decides everything ───────────────
        if progress_cb:  # #435: tell the user we're about to call the LLM
            progress_cb("Thinking\u2026")
        tool_ctx = self._build_tool_context(en_input)
        plan, routing_error = await cot_route(
            en_input, registry.all_schemas(),
            recent_history=recent_history,
            tool_context=tool_ctx,
        )

        # ── Step 3: Safety net — if cot_route fails, chat gracefully ─
        if plan is None:
            log.warning("cot_route returned None (error=%s) for: %.80s", routing_error, en_input)
            stream = self._chat_stream(
                en_input, tc, system_alert=routing_error,
            )
            return BrainResult(
                response="", tool_used=None, stream=stream,
            )

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        log.info("cot_route decision: route=%s tool=%s confidence=%.2f for: %.80s",
                 route, tool_name or "(none)",
                 plan.get("confidence", 0), en_input)

        # ── Step 3b: Normalise sloppy routes from small LLMs ─────────
        #    llama3.1:8b often returns route="gmail" instead of "tool",
        #    or route="Web Search" instead of "tool".
        #    Fix: if route looks like a tool name → normalise to "tool".
        if route not in ("tool", "planner", "chat"):
            # Fuzzy-match route against registered tool names
            if registry.get(route):
                if not tool_name:
                    tool_name = route
                route = "tool"
                log.info("Normalised sloppy route=%r → tool=%s", plan.get("route"), tool_name)
            elif tool_name and registry.get(tool_name):
                route = "tool"
                log.info("Normalised unknown route=%r → tool (tool_name=%s)", plan.get("route"), tool_name)
            else:
                log.warning("Unrecognised route=%r, tool_name=%r — falling back to chat", route, tool_name)
                route = "chat"

        # ── Step 3c: Normalise tool_name ──────────────────────────────
        #    Small LLMs return "Web Search", "email", "Visual Click", etc.
        #    Two-pass approach: (1) fuzzy registry lookup, (2) alias map.
        _TOOL_ALIASES: dict[str, str] = {
            "email": "gmail", "mail": "gmail", "inbox": "gmail",
            "search": "web_search", "google": "web_search",
            "bash": "shell", "terminal": "shell", "command": "shell",
            "file": "filesystem", "files": "filesystem",
            "click": "visual_click", "screen": "visual_click",
            "web": "web_search", "browse": "read_url",
            "remind": "reminder", "reminders": "reminder",
            "cancel_reminder": "reminder", "delete_reminder": "reminder",
            "events": "calendar", "schedule": "calendar",
            "create_event": "calendar", "delete_event": "calendar",
            "add_event": "calendar",
            "class": "classroom", "homework": "classroom",
            "firefox": "browser_control", "chrome": "browser_control",
            "chromium": "browser_control", "browser": "browser_control",
            "accessibility": "accessibility",
            "gemini": "browser_control", "chatgpt": "browser_control",
        }
        if tool_name:
            # Pass 1: fuzzy registry lookup (handles "Web Search" → "web_search")
            matched = registry.get(tool_name)
            if matched and hasattr(matched, "name") and isinstance(matched.name, str):
                if matched.name != tool_name:
                    log.info("Normalised tool_name %r → %r (fuzzy)", tool_name, matched.name)
                    tool_name = matched.name
            elif not matched:
                # Pass 2: alias map (handles "email" → "gmail", etc.)
                from bantz.tools import _normalise_tool_name
                norm = _normalise_tool_name(tool_name)
                alias = _TOOL_ALIASES.get(norm) or _TOOL_ALIASES.get(tool_name.lower())
                if alias:
                    log.info("Normalised tool alias %r → %r", tool_name, alias)
                    # Inject correct action when compound alias implies it
                    _ALIAS_ACTIONS: dict[str, dict] = {
                        "cancel_reminder": {"action": "cancel"},
                        "delete_reminder": {"action": "cancel"},
                        "create_event": {"action": "create"},
                        "add_event": {"action": "create"},
                        "delete_event": {"action": "delete"},
                    }
                    implied = _ALIAS_ACTIONS.get(norm) or _ALIAS_ACTIONS.get(tool_name.lower())
                    if implied:
                        for k, v in implied.items():
                            if k not in tool_args:
                                tool_args[k] = v
                                log.info("Injected implied arg %s=%r from alias", k, v)
                    tool_name = alias

        # ── Step 4: route == "planner" → multi-step decomposition ─────
        if route == "planner":
            try:
                plan_result = await _execute_plan_fn(
                    user_input, en_input, tc, recent_history=recent_history,
                )
                if plan_result is not None:
                    # (#276) Store planner output for contextual follow-ups
                    self._last_tool_output = (plan_result.response or "")[:500]
                    self._last_tool_name = "planner"
                    self._context_turn = self._turn_counter
                    await self._graph_store(user_input, plan_result.response, "planner")
                    self._fire_embeddings()
                    return plan_result
            except Exception as exc:
                log.warning("Planner execution failed: %s — falling back to chat", exc)
            # Planner failed or returned None → fall through to chat
            stream = self._chat_stream(
                en_input, tc,
                system_alert=f"Multi-step planner failed: {routing_error or 'decomposition returned no steps'}",
            )
            return BrainResult(response="", tool_used=None, stream=stream)

        # ── Step 5: route == "chat" → streaming chat ──────────────────
        if route != "tool" or not tool_name:
            # Anomaly Watch "Investigate" directives must analyse REAL system
            # state, not free-associate. Run live read-only diagnostics and
            # ground the LLM on the actual output (#anomaly-hallucination).
            if en_input.strip().lower().startswith("investigate:"):
                stream = self._investigate_stream(en_input, tc)
            else:
                stream = self._chat_stream(en_input, tc)
            return BrainResult(response="", tool_used=None, stream=stream)

        # ── Step 5b: Internal tools (prefixed with "_") → dispatch_internal
        #    These were previously matched by quick_route regex; now the LLM
        #    routes them via cot_route.  (#272)
        if tool_name.startswith("_"):
            internal = await _dispatch_internal(
                tool_name, tool_args,
                user_input, en_input, tc,
                is_remote=is_remote,
            )
            if internal is not None:
                return internal
            # dispatch_internal didn't handle it — fall through to registry

        # ── C1 observe→re-decide loop (audit C1, #503) ───────────────────
        # Wraps the destructive (S2) + autonomy (S1) gates and tool execution.
        # Default (config.tool_loop_max_steps=1) runs the body exactly once —
        # byte-identical to the historical single-shot path, no extra LLM
        # calls. Higher budgets feed a failed ToolResult / raised exception
        # back as an observation and re-decide via cot_route(skip_fastpath=True).
        _outcome = await self._execute_with_recovery(
            tool_name=tool_name, tool_args=tool_args, risk=risk,
            requires_confirm=plan.get("requires_confirm"),
            confirmed=confirmed, en_input=en_input,
            recent_history=recent_history, tool_ctx=tool_ctx,
        )
        _loop_iters = _outcome.iterations
        if _outcome.terminal is not None:
            return _outcome.terminal
        result = _outcome.result
        tool_name = _outcome.tool_name
        tool_args = _outcome.tool_args

        # ── RL reward: positive signal on successful tool use (#125) ──
        self._rl_reward_hook(tool_name, result)

        # ── Store tool results for contextual follow-ups (#56, #276) ──
        if result.success:
            self._store_tool_context(tool_name, result)

        # ── Compose/reply draft → confirmation flow ──
        if result.success and result.data and result.data.get("draft"):
            d = result.data
            self._last_draft = {
                "to": d["to"],
                "subject": d.get("subject", ""),
                "body": d["body"],
            }
            data_layer.conversations.add("assistant", result.output, tool_used=tool_name)
            return BrainResult(
                response=result.output,
                tool_used=tool_name,
                tool_result=result,
                needs_confirm=True,
                pending_tool="gmail",
                pending_args={
                    "action": "send",
                    "to": d["to"],
                    "subject": d.get("subject", ""),
                    "body": d["body"],
                },
                iterations=_loop_iters,
            )

        # ── Embed metadata in output for conversation history (#276) ──
        if result.success and result.data:
            result = ToolResult(
                success=result.success,
                output=self._embed_metadata(result.output, tool_name, result.data),
                data=result.data,
                error=result.error,
            )

        # Try streaming finalize for long tool output (#67)
        fin_stream = await self._finalize_stream(en_input, result, tc)
        if fin_stream is not None:
            # Screenshot bytes must be promoted even on the streaming path (#189)
            stream_attachments: list[Attachment] = []
            if result.success and result.data and result.data.get("screenshot"):
                stream_attachments.append(Attachment(
                    type="image",
                    data=result.data["screenshot"],
                    caption="",
                    mime_type=result.data.get("mime_type", "image/jpeg"),
                ))
            return BrainResult(
                response="", tool_used=tool_name,
                tool_result=result, stream=fin_stream,
                attachments=stream_attachments,
                iterations=_loop_iters,
            )

        # Short output — non-streaming finalize
        resp = await self._finalize(en_input, result, tc)
        data_layer.conversations.add("assistant", resp, tool_used=tool_name)
        # Keep the [CONTEXT:...] block in history (carries IDs for follow-ups)
        # but never show it to the user (#276 leak).
        resp = strip_internal(resp)
        await self._graph_store(user_input, resp, tool_name,
                                result.data if result else None)
        self._fire_embeddings()

        # ── Screenshot → Attachment promotion (#189) ──────────────────────
        # If the tool returned image bytes, wrap them in an Attachment so
        # Telegram (and future image-capable interfaces) can dispatch the
        # daguerreotype without writing anything to disk.
        # Also: if the user is on Telegram and their message contains a
        # screenshot trigger word, auto-append a screenshot after the action.
        attachments: list[Attachment] = []
        if result.success and result.data and result.data.get("screenshot"):
            # Caption is empty — butler response is sent as a separate text
            # message in telegram_bot so the photo renders cleanly (#189).
            img_bytes = result.data["screenshot"]
            attachments.append(Attachment(
                type="image",
                data=img_bytes,
                caption="",
                mime_type=result.data.get("mime_type", "image/jpeg"),
            ))
            # Kick off VLM screen description in background so the NEXT
            # command has screen-aware context ("there" = visible search box etc.)
            try:
                import asyncio as _asyncio
                self._pending_vlm_task = _asyncio.create_task(
                    self._vlm_describe_screen(img_bytes)
                )
            except Exception:
                pass
        elif self._is_remote and result.success:
            from bantz.tools.screenshot_tool import SCREENSHOT_TRIGGERS
            from bantz.config import config as _cfg
            if _cfg.screenshot_auto_after_action and any(
                t in user_input.lower() for t in SCREENSHOT_TRIGGERS
            ):
                try:
                    import asyncio as _asyncio
                    await _asyncio.sleep(2)  # let the action settle before capture
                    from bantz.vision import screenshot as _ss
                    shot = await _ss.capture()
                    if shot and shot.data:
                        attachments.append(Attachment(
                            type="image",
                            data=shot.data,
                            caption=resp,
                            mime_type="image/jpeg",
                        ))
                except Exception as _exc:
                    log.debug("Auto-screenshot after action failed: %s", _exc)

        return BrainResult(
            response=resp, tool_used=tool_name,
            tool_result=result, attachments=attachments,
            iterations=_loop_iters,
        )

    @staticmethod
    def _gate_history(prior: list[dict], en_input: str) -> list[dict]:
        """Anaphora-gate the chat history window (live re-run tests 19/20).

        Unconditional 12-message history made chat answers inherit the
        previous turn's topic (both memory-check tests answered about Alan
        Turing). When the current input contains a back-reference
        (``_ANAPHORA``), keep the full window; otherwise keep only the last
        two turns for tone continuity and let ``_TOPIC_DISCIPLINE`` mark the
        message as a fresh topic.
        """
        if not prior or _ANAPHORA.search(en_input):
            return prior
        return prior[-4:]

    @staticmethod
    def _dedup_history(history: list[dict]) -> list[dict]:
        """Collapse repeated assistant responses to prevent context-window echo loops (#184).

        If the same assistant message appears ≥ LOOP_THRESHOLD times in the
        supplied history, all duplicates are removed (keeping the first copy)
        and a system-level anti-loop warning is appended so the LLM knows to
        say something new.

        User messages are never touched.
        Comparison is case-insensitive and whitespace-normalised.
        """
        from collections import Counter

        LOOP_THRESHOLD = 3

        assistant_msgs = [
            m["content"].strip().lower()
            for m in history if m["role"] == "assistant"
        ]
        counts = Counter(assistant_msgs)
        repeated = {msg for msg, count in counts.items() if count >= LOOP_THRESHOLD}

        if not repeated:
            return history

        seen_repeated: set[str] = set()
        deduped: list[dict] = []
        for m in history:
            key = m["content"].strip().lower() if m["role"] == "assistant" else None
            if key and key in repeated:
                if key in seen_repeated:
                    continue  # drop duplicate
                seen_repeated.add(key)
            deduped.append(m)

        deduped.append({
            "role": "system",
            "content": (
                "[WARNING: You have been repeating the same response. "
                "Say something NEW and relevant. Do NOT repeat previous messages.]"
            ),
        })
        log.warning("_dedup_history: collapsed %d repeated assistant message(s)", len(repeated))
        return deduped

    async def _chat(self, en_input: str, tc: dict) -> str:
        """
        Chat mode with conversation history.
        history[-1] = the user message we just saved → exclude to avoid duplication.
        Context gathering is concurrent via memory_injector.inject (#227).
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history
        prior = self._dedup_history(prior)
        prior = self._gate_history(prior, en_input)

        # Concurrent context injection (#227)
        ctx = BantzContext(en_input=en_input)
        ctx.feedback_hint = getattr(self, "_feedback_ctx", "")
        self._feedback_ctx = ""  # clear after consumption
        await _inject_memory(ctx, en_input)

        messages = [
            {"role": "system", "content": _build_chat_system(ctx, tc) + _TOPIC_DISCIPLINE + _mood_suffix()},
            *prior,
            {"role": "user", "content": en_input},
        ]

        from bantz.llm.router import get_provider
        try:
            provider = get_provider()
            raw = await provider.chat(messages)
            if _is_refusal(raw):
                return "I'm afraid that's outside my service area, sir."
            return strip_markdown(raw)
        except Exception as exc:
            log.error("LLM chat error: %s", exc)
            return _BUTLER_LLM_ERROR

    async def _chat_stream(
        self, en_input: str, tc: dict,
        *, system_alert: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming chat — yields tokens as they arrive from LLM.
        Post-processing (strip_markdown) runs on accumulated text at consumer side.
        Context gathering is concurrent via memory_injector.inject (#227).

        Args:
            system_alert: (#253) When a tool routing / planner attempt
                failed, this carries the error description.  It is
                injected into the system prompt so the LLM honestly
                reports the failure instead of hallucinating success.
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history
        prior = self._dedup_history(prior)
        prior = self._gate_history(prior, en_input)

        # Concurrent context injection (#227)
        ctx = BantzContext(en_input=en_input)
        ctx.feedback_hint = getattr(self, "_feedback_ctx", "")
        self._feedback_ctx = ""  # clear after consumption
        await _inject_memory(ctx, en_input)

        system_content = _build_chat_system(ctx, tc) + _TOPIC_DISCIPLINE + _mood_suffix()

        # (#253) People-Pleaser guard: inject routing failure context
        if system_alert:
            system_content += (
                "\n\n[SYSTEM ALERT: You just attempted to use a tool to "
                "fulfill the user's request, but your internal routing/JSON "
                f"generation failed with error: {system_alert}. "
                "DO NOT pretend you completed the task. DO NOT fabricate "
                "data, emails, search results, or file contents. "
                "Apologize to the user in your butler persona and explain "
                "that a technical difficulty prevented you from completing "
                "the request. Suggest they try again or rephrase.]"
            )
            log.warning("People-Pleaser guard activated: %s", system_alert)

        messages = [
            {"role": "system", "content": system_content},
            *prior,
            {"role": "user", "content": en_input},
        ]

        from bantz.llm.router import get_provider
        try:
            provider = get_provider()
            async for token in provider.chat_stream(messages):
                yield token
        except Exception as exc:
            log.error("LLM stream error: %s", exc)
            yield _BUTLER_LLM_ERROR

    # ── System-anomaly investigation (grounded in live diagnostics) ───────
    # Read-only command sets, selected by keywords in the anomaly directive.
    # Hardcoded (no user input is interpolated) so there is no injection risk.
    _DIAG_MEMORY = (
        "free -h",
        "swapon --show",
        "ps -eo pid,comm,rss,%mem --sort=-%mem | head -n 12",
    )
    _DIAG_CPU = (
        "uptime",
        "ps -eo pid,comm,%cpu --sort=-%cpu | head -n 12",
    )
    _DIAG_DISK = (
        "df -h -x tmpfs -x devtmpfs",
        "du -xh --max-depth=1 \"$HOME\" 2>/dev/null | sort -rh | head -n 10",
    )

    _INVESTIGATE_SYSTEM = (
        "You are Bantz, a 1920s English butler who also keeps a sharp eye on "
        "the household's mechanical contraptions (this computer). An anomaly "
        "was flagged. Below is REAL diagnostic output captured just now by "
        "running live system commands. Report to your employer:\n"
        "- State what the data actually shows (cite the real numbers and "
        "process names).\n"
        "- Identify the most likely cause (e.g. which process is consuming "
        "the memory or CPU).\n"
        "- Recommend ONE concrete next action.\n"
        "RULES:\n"
        "- Use ONLY the diagnostic data shown. NEVER invent process names, "
        "PIDs, or numbers that are not present in it.\n"
        "- If the data is inconclusive, say so honestly rather than guessing.\n"
        "- Be concise: 3-6 sentences. Plain text, no markdown. Address the "
        "user as 'ma'am'.{persona_state}"
    )

    @staticmethod
    async def _run_diag(cmd: str) -> str:
        """Run a single read-only diagnostic command, capturing combined output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            return out.decode("utf-8", errors="replace").strip()
        except Exception as exc:  # never let a probe crash the investigation
            return f"(failed: {exc})"

    async def _gather_diagnostics(self, directive: str) -> str:
        """Run the read-only diagnostics relevant to an anomaly directive."""
        low = directive.lower()
        cmds: list[str] = []
        if any(k in low for k in (
            "swap", "memory", "ram", "paging", "pressure", "oom",
        )):
            cmds += self._DIAG_MEMORY
        if any(k in low for k in ("cpu", "saturat", "load")):
            cmds += self._DIAG_CPU
        if any(k in low for k in ("disk", "full", "storage", "space")):
            cmds += self._DIAG_DISK
        if not cmds:  # unknown anomaly → broad system snapshot
            cmds = [
                "free -h", "uptime",
                "ps -eo pid,comm,rss,%mem,%cpu --sort=-%mem | head -n 10",
            ]
        # De-dup while preserving order (memory+CPU sets share nothing today,
        # but a combined "memory pressure" directive could request both).
        seen: set[str] = set()
        ordered = [c for c in cmds if not (c in seen or seen.add(c))]
        results = await asyncio.gather(*(self._run_diag(c) for c in ordered))
        return "\n\n".join(f"$ {c}\n{r}" for c, r in zip(ordered, results))

    async def _investigate_stream(
        self, en_input: str, tc: dict,
    ) -> AsyncIterator[str]:
        """Stream a grounded analysis of a system anomaly.

        Unlike ``_chat_stream``, this runs live read-only diagnostics and
        feeds the real output to the LLM as ground truth, so Bantz reports on
        actual swap/memory/CPU/disk state instead of hallucinating one.
        """
        directive = en_input.split(":", 1)[1].strip() if ":" in en_input else en_input
        diagnostics = await self._gather_diagnostics(directive)

        # Persona flavour (best-effort; never block the analysis on it).
        persona_state = ""
        try:
            from bantz.personality.persona import persona_builder
            persona_state = persona_builder.build()
        except Exception:
            pass

        messages = [
            {"role": "system", "content": self._INVESTIGATE_SYSTEM.format(
                persona_state=persona_state,
            )},
            {"role": "user", "content": (
                f"ANOMALY: {directive}\n\n"
                f"LIVE DIAGNOSTICS (ground truth — captured just now):\n"
                f"{diagnostics[:4000]}\n\n"
                "Analyse the anomaly using ONLY the diagnostics above and advise."
            )},
        ]

        from bantz.llm.router import get_provider
        try:
            provider = get_provider()
            async for token in provider.chat_stream(messages):
                yield token
        except Exception as exc:
            log.error("investigate stream error: %s", exc)
            yield _BUTLER_LLM_ERROR

    async def _recall_cached(self, en_input: str) -> Any:
        """Per-turn memoized memory recall (audit M4).

        ``_finalize_stream`` is attempted before ``_finalize`` on the tool
        path; for short output the stream variant recalls, returns None, and
        the non-stream variant then recalls the identical query again. Cache
        the result for the duration of one turn so recall runs once.
        """
        from bantz.memory.omni_memory import omni_memory
        cached = self._recall_cache
        if cached is not None and cached[0] == en_input:
            return cached[1]
        recall = await omni_memory.recall(en_input)
        self._recall_cache = (en_input, recall)
        return recall

    async def _finalize(self, en_input: str, result: ToolResult, tc: dict) -> str:
        """Delegate to core.finalizer module (#227, #211: use OmniMemory)."""
        recall = await self._recall_cached(en_input)
        return await _finalize_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            memory_context=recall.combined,
            formality_hint=_formality_hint(),
        )

    async def _finalize_stream(
        self, en_input: str, result: ToolResult, tc: dict,
    ) -> AsyncIterator[str] | None:
        """Delegate to core.finalizer module (#227, #211: use OmniMemory)."""
        recall = await self._recall_cached(en_input)
        return await _finalize_stream_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            memory_context=recall.combined,
            formality_hint=_formality_hint(),
        )

    @staticmethod
    def _hallucination_check(response: str, tool_output: str) -> tuple[str, float]:
        """Delegate to core.finalizer module."""
        return _hallucination_check_fn(response, tool_output)


brain = Brain()