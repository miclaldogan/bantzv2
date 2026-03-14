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
from bantz.core.types import BrainResult  # noqa: F401  — canonical def in types.py
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
        try:
            import bantz.tools.gui_action  # noqa: F401  (#123)
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            import bantz.tools.visual_click  # noqa: F401  (#185)
        except (ImportError, ModuleNotFoundError):
            pass
        self._memory_ready = False
        self._graph_ready = False
        # Session state: stores last tool results for contextual follow-ups
        self._last_messages: list[dict] = []   # last listed emails [{id, from, subject, ...}]
        self._last_events: list[dict] = []     # last listed calendar events
        self._last_draft: dict | None = None   # last email draft {to, subject, body}
        self._feedback_ctx: str = ""  # one-shot RLHF context (#180)

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

        # ── Plan-and-Solve: LLM-based multi-step decomposition (#187) ────
        # Planner runs FIRST — it handles complex multi-tool requests via
        # LLM heuristics.  Must be above workflow_engine to prevent the
        # old regex-based engine from eagerly stealing autonomous commands.
        #
        # recent_history feeds coreference resolution (#212) so the
        # planner can resolve pronouns like "him", "it", "that file".
        recent_history = data_layer.conversations.context(n=6)
        planner_error: str | None = None
        try:
            from bantz.agent.planner import planner_agent
            if planner_agent.is_complex(en_input, recent_history=recent_history):
                plan_result = await _execute_plan_fn(
                    user_input, en_input, tc, recent_history=recent_history,
                )
                if plan_result is not None:
                    # Orchestrator owns persistence (hotfix #228)
                    await self._graph_store(user_input, plan_result.response, "planner")
                    self._fire_embeddings()
                    return plan_result
        except Exception as exc:
            log.warning("Planner check failed: %s — propagating to LLM", exc)
            planner_error = f"Multi-step planner failed: {exc}"

        # ── Quick-route → internal dispatch (#228) ──────────────────────
        quick = self._quick_route(user_input, en_input)

        if quick:
            internal = await _dispatch_internal(
                quick["tool"], quick["args"],
                user_input, en_input, tc,
                is_remote=is_remote,
            )
            if internal is not None:
                return internal

            # Not an internal tool → build a plan for the registry
            if quick["tool"] == "_generate":
                cmd = await _generate_command_fn(user_input, en_input)
                plan = {"route": "tool", "tool_name": "shell",
                        "tool_args": {"command": cmd}, "risk_level": "moderate"}
            else:
                plan = {"route": "tool", "tool_name": quick["tool"],
                        "tool_args": quick["args"], "risk_level": "safe"}
        else:
            plan, routing_error = await cot_route(
                en_input, registry.all_schemas(),
                recent_history=recent_history,
            )

            # Merge planner_error if cot_route had no error of its own
            if routing_error is None and planner_error is not None:
                routing_error = planner_error

            if plan is None:
                # (#253) If routing_error is set, a tool was attempted
                # but failed.  Inject a system alert so the LLM does NOT
                # hallucinate tool success.
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

        if route != "tool" or not tool_name:
            stream = self._chat_stream(en_input, tc)
            return BrainResult(response="", tool_used=None, stream=stream)

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

        result = await tool.execute(**tool_args)

        # ── RL reward: positive signal on successful tool use (#125) ──
        self._rl_reward_hook(tool_name, result)

        # ── Store tool results for contextual follow-ups (#56) ──
        if result.success and result.data:
            if result.data.get("messages"):
                self._last_messages = result.data["messages"]
            if result.data.get("events"):
                self._last_events = result.data["events"]

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
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    async def _chat(self, en_input: str, tc: dict) -> str:
        """
        Chat mode with conversation history.
        history[-1] = the user message we just saved → exclude to avoid duplication.
        Context gathering is concurrent via memory_injector.inject (#227).
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history

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
        """Delegate to core.finalizer module (#227: use memory_injector)."""
        return await _finalize_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            graph_hint=await _graph_ctx_fn(en_input),
            deep_memory=await _deep_memory_ctx_fn(en_input),
            formality_hint=_formality_hint(),
        )

    async def _finalize_stream(
        self, en_input: str, result: ToolResult, tc: dict,
    ) -> AsyncIterator[str] | None:
        """Delegate to core.finalizer module (#227: use memory_injector)."""
        return await _finalize_stream_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            graph_hint=await _graph_ctx_fn(en_input),
            deep_memory=await _deep_memory_ctx_fn(en_input),
            formality_hint=_formality_hint(),
        )

    @staticmethod
    def _hallucination_check(response: str, tool_output: str) -> tuple[str, float]:
        """Delegate to core.finalizer module."""
        return _hallucination_check_fn(response, tool_output)


brain = Brain()