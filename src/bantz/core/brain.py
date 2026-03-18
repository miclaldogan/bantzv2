"""Bantz v3 — Brain (Orchestrator).

Pure orchestrator: delegates to routing_engine, finalizer, memory_injector,
prompt_builder, translation_layer, rl_hooks, notification_manager, and
location_handler.  See each module’s docstring for details.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.data import data_layer
from bantz.core.profile import profile
from bantz.core.intent import cot_route
from bantz.core.finalizer import (
    finalize as _finalize_fn,
    finalize_stream as _finalize_stream_fn,
    hallucination_check as _hallucination_check_fn,
    strip_markdown,
)
from bantz.llm.ollama import ollama
from bantz.tools import registry, ToolResult
from bantz.core.context import BantzContext  # noqa: F401  — re-export for compat
from bantz.core.types import BrainResult, Attachment  # noqa: F401  — canonical def in types.py
from bantz.core.routing_engine import (
    quick_route as _quick_route_fn,
    dispatch_internal as _dispatch_internal,
    generate_command as _generate_command_fn,
    execute_plan as _execute_plan_fn,
    handle_maintenance as _handle_maintenance_fn,
    handle_list_reflections as _handle_list_reflections_fn,
    handle_run_reflection as _handle_run_reflection_fn,
)
from bantz.core.memory_injector import (
    inject as _inject_memory,
    style_hint as _style_hint,
    persona_hint as _persona_hint,
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

log = logging.getLogger("bantz.brain")

try:
    from bantz.memory.graph import graph_memory
except ImportError:
    graph_memory = None  # neo4j driver not installed


# Re-exported for backward compat — canonical impl in translation_layer.py
from bantz.core.translation_layer import (  # noqa: F401
    POSITIVE_FEEDBACK_KWS,
    NEGATIVE_FEEDBACK_KWS,
    detect_feedback as _detect_feedback,
)


# Toast compat shim — canonical impl in notification_manager.py (#225)
import bantz.core.notification_manager as _notif_mod
_toast_callback = None  # written by app.py / tests


def _notify_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Compat shim → ``notification_manager.notify_toast``."""
    _notif_mod.toast_callback = _toast_callback
    _notif_mod.notify_toast(title, reason, toast_type)


class Brain:
    def __init__(self) -> None:
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
        import bantz.tools.weather      # noqa: F401
        import bantz.tools.news         # noqa: F401
        import bantz.tools.web_search   # noqa: F401
        import bantz.tools.web_reader   # noqa: F401
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401
        import bantz.tools.classroom    # noqa: F401
        import bantz.tools.reminder     # noqa: F401
        try:
            import bantz.tools.document     # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # PDF/DOCX deps may not be installed
        try:
            import bantz.tools.accessibility  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # AT-SPI2/gi deps may not be installed
        # gui_action removed (#185) — superseded by visual_click
        try:
            import bantz.tools.visual_click  # noqa: F401  (#185)
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import bantz.tools.browser_control  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass
        import bantz.tools.summarizer    # noqa: F401  (Architect's Revision)
        self._memory_ready = False
        self._graph_ready = False
        # Session state: stores last tool results for contextual follow-ups
        self._last_messages: list[dict] = []   # last listed emails [{id, from, subject, ...}]
        self._last_events: list[dict] = []     # last listed calendar events
        self._last_draft: dict | None = None   # last email draft {to, subject, body}
        self._last_tool_output: str = ""     # last tool output snippet (generic)
        self._last_tool_name: str = ""       # which tool produced it
        self._feedback_ctx: str = ""  # one-shot RLHF context (#180)
        # Turn-based TTL (#276): context expires after N process() calls
        self._turn_counter: int = 0
        self._context_turn: int = 0  # turn when context was last stored
        self._CONTEXT_TTL: int = 3   # expire after 3 turns of no related queries

    def _ensure_memory(self) -> None:
        if not self._memory_ready:
            data_layer.init(config)
            self._memory_ready = True

    def _desktop_context(self) -> str:
        """Compat shim → ``memory_injector.desktop_context`` (#227)."""
        from bantz.core.memory_injector import desktop_context
        return desktop_context()

    async def _ensure_graph(self) -> None:
        if not self._graph_ready and graph_memory:
            await graph_memory.init()
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
        """Fire-and-forget: embed any queued messages from this exchange."""
        try:
            from bantz.core.memory import memory
            if memory._embed_queue:
                asyncio.ensure_future(memory.embed_pending())
        except Exception:
            pass

    async def _graph_store(self, user_msg: str, assistant_msg: str,
                           tool_used: str | None = None,
                           tool_data: dict | None = None) -> None:
        """Store entities from exchange in graph (fire-and-forget)."""
        if graph_memory and graph_memory.enabled:
            try:
                await graph_memory.extract_and_store(
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

        return "\n\n".join(parts)

    @staticmethod
    def _quick_route(orig: str, en: str) -> dict | None:
        """Compat shim → ``routing_engine.quick_route`` (#228)."""
        return _quick_route_fn(orig, en)

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

    async def process(self, user_input: str, confirmed: bool = False,
                      is_remote: bool = False) -> BrainResult:
        self._is_remote = is_remote
        self._ensure_memory()
        await self._ensure_graph()
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

        recent_history = data_layer.conversations.context(n=6)

        # ═══════════════════════════════════════════════════════════════
        # NEW PIPELINE (#272): quick_route → cot_route → branch
        # ═══════════════════════════════════════════════════════════════

        # ── Step 1: Quick-route — hardware/UI controls ONLY ──────────
        quick = self._quick_route(user_input, en_input)
        if quick:
            internal = await _dispatch_internal(
                quick["tool"], quick["args"],
                user_input, en_input, tc,
                is_remote=is_remote,
            )
            if internal is not None:
                return internal
            # If dispatch_internal returns None, the tool isn't internal.
            # This shouldn't happen with the stripped quick_route, but
            # fall through to cot_route as a safety net.

        # ── Step 2: cot_route — LLM decides everything ───────────────
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
            "events": "calendar", "schedule": "calendar",
            "class": "classroom", "homework": "classroom",
            "firefox": "browser_control", "chrome": "browser_control",
            "chromium": "browser_control", "browser": "browser_control",
            "screenshot": "browser_control", "accessibility": "accessibility",
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

        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd_str = tool_args.get("command", tool_name)
            warn = (
                f"⚠️  Destructive operation: [{tool_name}] `{cmd_str}`\n"
                f"Confirm? (yes/no)"
            )
            data_layer.conversations.add("assistant", warn)
            return BrainResult(
                response=warn,
                tool_used=tool_name,
                needs_confirm=True,
                pending_command=cmd_str,
                pending_tool=tool_name,
                pending_args=tool_args,
            )

        tool = registry.get(tool_name)
        if not tool:
            err = f"Tool not found: {tool_name}"
            data_layer.conversations.add("assistant", err)
            return BrainResult(response=err, tool_used=None)

        # ── Pre-tool announcement (speed illusion) ────────────────────
        # Emit an immediate chat message before the tool runs so the user
        # sees "Let me check your calendar..." instead of dead silence.
        _PRE_TOOL_LINES = {
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
            "classroom":     "Checking your assignments...",
        }
        pre_line = _PRE_TOOL_LINES.get(tool_name)
        if pre_line:
            try:
                from bantz.core.event_bus import bus as _bus
                await _bus.emit("pre_tool_message", message=pre_line, tool=tool_name)
                await asyncio.sleep(0)  # yield so TUI renders the line before tool blocks
            except Exception:
                pass

        result = await tool.execute(**tool_args)

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
            return BrainResult(
                response="", tool_used=tool_name,
                tool_result=result, stream=fin_stream,
            )

        # Short output — non-streaming finalize
        resp = await self._finalize(en_input, result, tc)
        data_layer.conversations.add("assistant", resp, tool_used=tool_name)
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
            attachments.append(Attachment(
                type="image",
                data=result.data["screenshot"],
                caption=resp,
                mime_type=result.data.get("mime_type", "image/jpeg"),
            ))
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
        )

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

        # Concurrent context injection (#227)
        ctx = BantzContext(en_input=en_input)
        ctx.feedback_hint = getattr(self, "_feedback_ctx", "")
        self._feedback_ctx = ""  # clear after consumption
        await _inject_memory(ctx, en_input)

        messages = [
            {"role": "system", "content": _build_chat_system(ctx, tc)},
            *prior,
            {"role": "user", "content": en_input},
        ]

        # Prefer Gemini for chat if available (#58)
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages)
                if not _is_refusal(raw):
                    return strip_markdown(raw)
        except Exception:
            pass  # fall through to Ollama

        try:
            raw = await ollama.chat(messages)
            if _is_refusal(raw):
                return "Sorry, I can't help with that. Try something else."
            return strip_markdown(raw)
        except Exception as exc:
            return f"(Ollama error: {exc})"

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

        # Concurrent context injection (#227)
        ctx = BantzContext(en_input=en_input)
        ctx.feedback_hint = getattr(self, "_feedback_ctx", "")
        self._feedback_ctx = ""  # clear after consumption
        await _inject_memory(ctx, en_input)

        system_content = _build_chat_system(ctx, tc)

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

        # Try Gemini streaming first
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                async for token in gemini.chat_stream(messages):
                    yield token
                return
        except Exception:
            pass  # fall through to Ollama

        # Ollama streaming fallback
        try:
            async for token in ollama.chat_stream(messages):
                yield token
        except Exception as exc:
            yield f"(Ollama error: {exc})"

    async def _finalize(self, en_input: str, result: ToolResult, tc: dict) -> str:
        """Delegate to core.finalizer module (#227, #211: use OmniMemory)."""
        from bantz.memory.omni_memory import omni_memory
        recall = await omni_memory.recall(en_input)
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
        from bantz.memory.omni_memory import omni_memory
        recall = await omni_memory.recall(en_input)
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