"""
Bantz v3 — Brain (Orchestrator)

Pipeline:
  user input → [bridge: optional TR→EN] → quick_route OR intent (Ollama CoT) → tool → finalizer → output

Extracted modules:
  - core/finalizer.py    — LLM post-processing + hallucination check
  - core/intent.py       — Qwen CoT intent parser
  - core/router.py       — simpler one-shot routing
  - core/notification_manager.py — toast/notification routing (#225)
  - core/location_handler.py     — GPS/place management (#225)
  - core/translation_layer.py    — i18n bridge, feedback detection (#226)
  - core/rl_hooks.py             — RL reward signals via AsyncDBExecutor (#226)
  - core/memory_injector.py      — context gathering + concurrent inject (#227)
  - core/prompt_builder.py       — CHAT_SYSTEM / COMMAND_SYSTEM templates (#227)
  - memory/nodes.py      — graph schema + entity extraction
  - memory/context_builder.py — graph → LLM context string
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import logging

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.data import data_layer
from bantz.core.profile import profile
from bantz.core.intent import cot_route
from bantz.core.date_parser import resolve_date
from bantz.core.finalizer import (
    finalize as _finalize_fn,
    finalize_stream as _finalize_stream_fn,
    hallucination_check as _hallucination_check_fn,
    log_hallucination as _log_hallucination_fn,
    strip_markdown,
    FINALIZER_SYSTEM,
)
from bantz.llm.ollama import ollama
from bantz.tools import registry, ToolResult
from bantz.core.context import BantzContext  # noqa: F401  — re-export for compat
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


# ── RLHF keywords & feedback detection (#180 → #226 translation_layer) ──
# Canonical implementation now in ``translation_layer.py``.
# Re-exported here for backward compat (test imports, etc.).
from bantz.core.translation_layer import (  # noqa: F401  — public re-exports
    POSITIVE_FEEDBACK_KWS,
    NEGATIVE_FEEDBACK_KWS,
    detect_feedback as _detect_feedback,
)


# ── Toast notification hook (#137 → #225 notification_manager) ───────
# Canonical implementation now lives in ``notification_manager.py``.
# We keep ``_toast_callback`` and ``_notify_toast`` here as **thin
# backward-compat shims** so that ``brain_mod._toast_callback = cb``
# and ``brain_mod._notify_toast(…)`` still work for existing callers
# (app.py, tests).  New code should import from notification_manager.
import bantz.core.notification_manager as _notif_mod

_toast_callback = None  # ← compat: written by app.py / tests


def _notify_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Compat shim → ``notification_manager.notify_toast``.

    Syncs the local ``_toast_callback`` into the canonical module before
    delegating, so old ``brain_mod._toast_callback = cb`` callers work.
    """
    _notif_mod.toast_callback = _toast_callback
    _notif_mod.notify_toast(title, reason, toast_type)


# ── Hints, templates, refusal detection ──────────────────────────────
# Canonical implementations moved to:
#   memory_injector.py  — style_hint, persona_hint, formality_hint (#227)
#   prompt_builder.py   — CHAT_SYSTEM, COMMAND_SYSTEM, is_refusal (#227)
# All symbols re-imported above for full backward compatibility.


@dataclass
class BrainResult:
    response: str
    tool_used: str | None
    tool_result: ToolResult | None = None
    needs_confirm: bool = False
    pending_command: str = ""
    pending_tool: str = ""
    pending_args: dict = field(default_factory=dict)
    stream: AsyncIterator[str] | None = None


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
        o = orig.lower().strip()
        e = en.lower().strip()
        both = o + " " + e

        # Direct shell commands typed literally
        _DIRECT = ("ls", "cd ", "df", "free", "ps ", "cat ", "grep ",
                   "find ", "pwd", "uname", "whoami", "du ", "mount",
                   "ip ", "ping ", "top", "htop", "mkdir", "touch",
                   "echo ", "head ", "tail ", "chmod ", "cp ", "mv ")
        for p in _DIRECT:
            if o == p.rstrip() or o.startswith(p if p.endswith(" ") else p + " "):
                # Guard: 'find' is both a bash command and a natural word.
                # Only treat as shell if followed by a path-like token
                # (/, ~, ., -) — otherwise fall through to web_search.
                if p == "find ":
                    _after_find = o[len("find "):].lstrip()
                    if not _after_find or not _after_find[0] in "/~.-":
                        continue
                return {"tool": "shell", "args": {"command": orig.strip()}}

        # System metrics — bypass router completely
        if any(k in both for k in ("disk", "df -", "storage", "disk space")):
            return {"tool": "shell", "args": {"command": "df -h"}}
        if any(k in both for k in ("memory", "free -", "ram usage", "how much ram")) or \
           re.search(r"\bram\b", both):
            return {"tool": "system", "args": {"metric": "ram"}}
        if any(k in both for k in ("cpu", "processor", "uptime", "load average")):
            return {"tool": "system", "args": {"metric": "all"}}
        if re.search(r"system\s*(status|info|check)|check\s*(my\s*)?system", both):
            return {"tool": "system", "args": {"metric": "all"}}

        # Folder/directory sizes — route to shell du, NEVER to chat
        # Requires BOTH a size keyword AND a disk-context keyword to avoid
        # false positives (e.g. "how big is EDITH" → no disk context → skip).
        _SIZE_KW = re.search(
            r"\b(big|large|size|bigger|largest|biggest|heaviest)\b", both,
        )
        _DISK_CTX = re.search(
            r"\b(folder|directory|dir|file|disk|storage|path|home|~/)\b", both,
        )
        if _SIZE_KW and _DISK_CTX:
            # Extract path if mentioned, default to home
            path_match = re.search(r"(?:in|under|of|check)\s+(~/?\S+|/\S+|home)", both)
            target = path_match.group(1) if path_match else "~"
            if target == "home":
                target = "~"
            return {"tool": "shell", "args": {"command": f"du -sh {target}/*/ 2>/dev/null | sort -rh | head -10"}}

        # TTS stop (#131) — "shut up" / "stop talking"
        if re.search(
            r"shut\s*up|be\s+quiet|stop\s+talk(?:ing)?",
            both,
        ):
            return {"tool": "_tts_stop", "args": {}}

        # Wake word control (#165)
        if re.search(
            r"start\s+listen(?:ing)?|resume\s+listen(?:ing)?|wake\s*word\s+on|"
            r"enable\s+wake|listen\s+for\s+me",
            both,
        ):
            return {"tool": "_wake_word_on", "args": {}}
        if re.search(
            r"stop\s+listen(?:ing)?|pause\s+(?:wake|listen)|wake\s*word\s+off|"
            r"disable\s+wake|don'?t\s+listen",
            both,
        ):
            return {"tool": "_wake_word_off", "args": {}}

        # Audio Ducking control (#171)
        if re.search(
            r"enable\s+duck|duck(?:ing)?\s+on|turn\s+on\s+duck",
            both,
        ):
            return {"tool": "_audio_duck_on", "args": {}}
        if re.search(
            r"disable\s+duck|duck(?:ing)?\s+off|turn\s+off\s+duck|no\s+duck",
            both,
        ):
            return {"tool": "_audio_duck_off", "args": {}}

        # Ambient status (#166)
        if re.search(
            r"ambient\s+(?:noise|sound|status|level|info)|environment\s+noise|"
            r"how(?:'s|\s+is)\s+(?:the\s+)?(?:noise|environment|ambient)",
            both,
        ):
            return {"tool": "_ambient_status", "args": {}}

        # Proactive engagement status (#167)
        if re.search(
            r"proactive\s+(?:status|info|count|stats)|"
            r"how\s+many\s+proactive|"
            r"engagement\s+status|check.?in\s+(?:status|count)",
            both,
        ):
            return {"tool": "_proactive_status", "args": {}}

        # Health & break status (#168)
        if re.search(
            r"health\s+(?:status|info|stats|check)|"
            r"break\s+(?:status|timer|count)|"
            r"session\s+(?:time|timer|hours)",
            both,
        ):
            return {"tool": "_health_status", "args": {}}

        # Briefing
        if any(k in both for k in ("good morning", "morning briefing", "daily briefing",
                                    "what's today", "what do i have today")):
            return {"tool": "_briefing", "args": {}}

        # Maintenance (#129) — manual trigger
        if re.search(
            r"run\s+maintenance|system\s+cleanup|"
            r"clean\s+(?:up\s+)?(?:the\s+)?system|maintenance\s+run",
            both,
        ):
            return {"tool": "_maintenance", "args": {"dry_run": "dry" in both}}

        # Reflection (#130) — run reflection now
        if re.search(
            r"run\s+reflect|generate\s+reflect|"
            r"reflect\s+(?:on\s+)?today",
            both,
        ):
            return {"tool": "_run_reflection", "args": {"dry_run": "dry" in both}}

        # Reflection (#130) — show past reflections
        if re.search(
            r"show\s+reflect|list\s+reflect|past\s+reflect",
            both,
        ):
            return {"tool": "_list_reflections", "args": {}}
            
        # Clear memory
        if re.search(r"clear\s+memory", both):
            return {"tool": "_clear_memory", "args": {}}

        return None

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

    # ── Maintenance & Reflection handlers (#129, #130) ────────────────

    async def _handle_maintenance(self, dry_run: bool = False) -> str:
        """Run the maintenance workflow and return its summary."""
        try:
            from bantz.agent.workflows.maintenance import run_maintenance
            report = await run_maintenance(dry_run=dry_run)
            return report.summary()
        except Exception as exc:
            return f"❌ Maintenance failed: {exc}"

    def _handle_list_reflections(self, limit: int = 5) -> str:
        """List recent reflections from the KV store."""
        try:
            from bantz.agent.workflows.reflection import list_reflections
            items = list_reflections(limit=limit)
            if not items:
                return "No reflections stored yet. They are generated nightly."
            lines = ["🤔 Recent reflections:"]
            for item in items:
                date = item.get("date", "?")
                summary = item.get("summary", "")[:120]
                sessions = item.get("sessions", 0)
                lines.append(f"  • {date} ({sessions} sessions): {summary}")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Could not load reflections: {exc}"

    async def _handle_run_reflection(self, dry_run: bool = False) -> str:
        """Run the reflection workflow and return its summary."""
        try:
            from bantz.agent.workflows.reflection import run_reflection
            result = await run_reflection(dry_run=dry_run)
            return result.summary_line()
        except Exception as exc:
            return f"❌ Reflection failed: {exc}"

    async def _generate_command(self, orig: str, en: str) -> str:
        raw = await ollama.chat([
            {"role": "system", "content": COMMAND_SYSTEM},
            {"role": "user", "content": en or orig},
        ])
        return raw.strip().strip("`")

    async def _execute_plan(
        self, user_input: str, en_input: str, tc: dict,
    ) -> BrainResult | None:
        """Decompose a complex request into steps, then execute them.

        Returns BrainResult on success, or None if decomposition fails
        (so caller can fall through to normal routing).
        """
        from bantz.agent.planner import planner_agent
        from bantz.agent.executor import plan_executor

        tool_names = registry.names() + ["process_text"]
        steps = await planner_agent.decompose(en_input, tool_names)
        if not steps or len(steps) < 2:
            # Not genuinely multi-step — fall through to normal routing
            return None

        # Announce the itinerary to the user
        itinerary = planner_agent.format_itinerary(steps)
        log.info("Plan-and-Solve itinerary:\n%s", itinerary)

        # Execute all steps
        exec_result = await plan_executor.run(steps, llm_fn=ollama.chat)

        # Combine itinerary + execution summary
        resp = itinerary + "\n\n" + exec_result.summary()

        data_layer.conversations.add("assistant", resp, tool_used="planner")
        await self._graph_store(user_input, resp, "planner")
        self._fire_embeddings()

        return BrainResult(response=resp, tool_used="planner")

    async def _handle_location(self) -> str:
        """Delegate to location_handler (#225)."""
        from bantz.core.location_handler import handle_location
        return await handle_location()

    async def _handle_save_place(self, name: str) -> str:
        """Delegate to location_handler (#225)."""
        from bantz.core.location_handler import handle_save_place
        return await handle_save_place(name)

    async def _handle_list_places(self) -> str:
        """Delegate to location_handler (#225)."""
        from bantz.core.location_handler import handle_list_places
        return await handle_list_places()

    async def _handle_delete_place(self, name: str) -> str:
        """Delegate to location_handler (#225)."""
        from bantz.core.location_handler import handle_delete_place
        return await handle_delete_place(name)

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
        try:
            from bantz.agent.planner import planner_agent
            if planner_agent.is_complex(en_input):
                plan_result = await self._execute_plan(user_input, en_input, tc)
                if plan_result is not None:
                    return plan_result
        except Exception as exc:
            log.debug("Planner check failed: %s — falling through", exc)

        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_tts_stop":
            from bantz.agent.tts import tts_engine
            if tts_engine.is_speaking:
                tts_engine.stop()
                text = "🔇 Stopped."
            else:
                text = "I'm not speaking right now."
            data_layer.conversations.add("assistant", text, tool_used="tts")
            return BrainResult(response=text, tool_used="tts")

        if quick and quick["tool"] == "_wake_word_off":
            try:
                from bantz.agent.wake_word import wake_listener
                if wake_listener.running:
                    wake_listener.stop()
                    text = "🔇 Wake word listener stopped."
                else:
                    text = "Wake word listener is not running."
            except Exception:
                text = "Wake word listener is not available."
            data_layer.conversations.add("assistant", text, tool_used="wake_word")
            return BrainResult(response=text, tool_used="wake_word")

        if quick and quick["tool"] == "_wake_word_on":
            try:
                from bantz.agent.wake_word import wake_listener
                if wake_listener.running:
                    text = "Wake word listener is already running."
                else:
                    ok = wake_listener.start()
                    text = "🎤 Wake word listener started." if ok else "❌ Could not start wake word listener."
            except Exception:
                text = "Wake word listener is not available."
            data_layer.conversations.add("assistant", text, tool_used="wake_word")
            return BrainResult(response=text, tool_used="wake_word")

        if quick and quick["tool"] == "_audio_duck_on":
            try:
                from bantz.agent.audio_ducker import audio_ducker
                if audio_ducker.available():
                    audio_ducker.enabled = True
                    text = "🔉 Audio ducking enabled."
                else:
                    text = "❌ Audio ducking not available (pactl not found)."
            except Exception:
                text = "Audio ducking module is not available."
            data_layer.conversations.add("assistant", text, tool_used="audio_ducker")
            return BrainResult(response=text, tool_used="audio_ducker")

        if quick and quick["tool"] == "_audio_duck_off":
            try:
                from bantz.agent.audio_ducker import audio_ducker
                audio_ducker.enabled = False
                text = "🔇 Audio ducking disabled."
            except Exception:
                text = "Audio ducking module is not available."
            data_layer.conversations.add("assistant", text, tool_used="audio_ducker")
            return BrainResult(response=text, tool_used="audio_ducker")

        if quick and quick["tool"] == "_ambient_status":
            try:
                from bantz.agent.ambient import ambient_analyzer
                snap = ambient_analyzer.latest()
                if snap:
                    text = (
                        f"🎤 Ambient: **{snap.label.value.upper()}** "
                        f"(RMS={snap.rms:.0f}, ZCR={snap.zcr:.3f})\n"
                        f"{ambient_analyzer.day_summary()}"
                    )
                else:
                    text = "No ambient data yet — analyzer is waiting for samples."
            except Exception:
                text = "Ambient analyzer is not available."
            data_layer.conversations.add("assistant", text, tool_used="ambient")
            return BrainResult(response=text, tool_used="ambient")

        if quick and quick["tool"] == "_proactive_status":
            try:
                from bantz.agent.proactive import (
                    proactive_engine, _get_daily_count, _compute_adaptive_max,
                )
                from bantz.agent.rl_engine import rl_engine
                kv = data_layer.kv
                if kv:
                    count, date = _get_daily_count(kv)
                    avg_r = rl_engine.episodes.avg_reward(7) if rl_engine.initialized else 0.0
                    max_d = _compute_adaptive_max(config.proactive_max_daily, avg_r)
                    text = (
                        f"💬 Proactive Engagement Status\n"
                        f"  Enabled: {'✅' if config.proactive_enabled else '❌'}\n"
                        f"  Today: {count}/{max_d} messages\n"
                        f"  RL avg reward (7d): {avg_r:.2f}\n"
                        f"  Interval: {config.proactive_interval_hours}h ±{config.proactive_jitter_minutes}m"
                    )
                else:
                    text = "Proactive engine: KV store not available."
            except Exception:
                text = "Proactive engagement module is not available."
            data_layer.conversations.add("assistant", text, tool_used="proactive")
            return BrainResult(response=text, tool_used="proactive")

        if quick and quick["tool"] == "_health_status":
            try:
                from bantz.agent.health import health_engine
                s = health_engine.status()
                cooldown_lines = "\n".join(
                    f"    {rid}: {mins:.0f}m left" for rid, mins in s["cooldowns"].items() if mins > 0
                )
                text = (
                    f"🏥 Health & Break Status\n"
                    f"  Enabled: {'✅' if config.health_enabled else '❌'}\n"
                    f"  Active session: {s['active_hours']:.1f}h\n"
                    f"  Break taken: {'✅' if s['had_break'] else '❌'}\n"
                    f"  Since last break: {s['minutes_since_break']:.0f}m\n"
                    f"  Thermal streak: CPU={s['thermal_cpu_streak']} GPU={s['thermal_gpu_streak']}\n"
                    f"  Check interval: {config.health_check_interval}s"
                )
                if cooldown_lines:
                    text += f"\n  Active cooldowns:\n{cooldown_lines}"
            except Exception:
                text = "Health & break module is not available."
            data_layer.conversations.add("assistant", text, tool_used="health")
            return BrainResult(response=text, tool_used="health")

        if quick and quick["tool"] == "_briefing":
            from bantz.core.briefing import briefing as _briefing
            text = await _briefing.generate()
            data_layer.conversations.add("assistant", text, tool_used="briefing")
            # Speak via TTS if available (#131) — suppress for remote (#178)
            if not getattr(self, '_is_remote', False):
                try:
                    from bantz.agent.tts import tts_engine
                    if tts_engine.available():
                        await tts_engine.speak_background(text)
                except Exception:
                    pass
            return BrainResult(response=text, tool_used="briefing")

        if quick and quick["tool"] == "_maintenance":
            text = await self._handle_maintenance(quick["args"].get("dry_run", False))
            data_layer.conversations.add("assistant", text, tool_used="maintenance")
            return BrainResult(response=text, tool_used="maintenance")

        if quick and quick["tool"] == "_list_reflections":
            text = self._handle_list_reflections()
            data_layer.conversations.add("assistant", text, tool_used="reflection")
            return BrainResult(response=text, tool_used="reflection")

        if quick and quick["tool"] == "_run_reflection":
            text = await self._handle_run_reflection(quick["args"].get("dry_run", False))
            data_layer.conversations.add("assistant", text, tool_used="reflection")
            return BrainResult(response=text, tool_used="reflection")

        if quick and quick["tool"] == "_location":
            text = await self._handle_location()
            data_layer.conversations.add("assistant", text, tool_used="location")
            return BrainResult(response=text, tool_used="location")

        if quick and quick["tool"] == "_save_place":
            text = await self._handle_save_place(quick["args"]["name"])
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_list_places":
            text = await self._handle_list_places()
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_delete_place":
            text = await self._handle_delete_place(quick["args"]["name"])
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_schedule_today":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_today()
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_next":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_next()
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_date":
            from bantz.core.schedule import schedule as _sched
            from datetime import datetime as _dt
            target = _dt.fromisoformat(quick["args"]["date_iso"])
            text = _sched.format_for_date(target)
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_week":
            from bantz.core.schedule import schedule as _sched
            resolved = resolve_date(user_input)
            text = _sched.format_week(resolved)
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_generate":
            cmd = await self._generate_command(user_input, en_input)
            plan = {"route": "tool", "tool_name": "shell",
                    "tool_args": {"command": cmd}, "risk_level": "moderate"}

        elif quick:
            plan = {"route": "tool", "tool_name": quick["tool"],
                    "tool_args": quick["args"], "risk_level": "safe"}

        else:
            plan = await cot_route(en_input, registry.all_schemas())
            if plan is None:
                # Stream chat responses for lower perceived latency (#67)
                stream = self._chat_stream(en_input, tc)
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

    async def _chat_stream(self, en_input: str, tc: dict) -> AsyncIterator[str]:
        """
        Streaming chat — yields tokens as they arrive from LLM.
        Post-processing (strip_markdown) runs on accumulated text at consumer side.
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