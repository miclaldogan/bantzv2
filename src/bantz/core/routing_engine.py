"""
Bantz — Routing Engine (#228, #272)

Owns routing logic that decides *what* the brain should do for a given
user utterance.

1. **quick_route(orig, en)**
   Regex fast-path for **hardware controls ONLY**: TTS stop, wake-word
   on/off, audio ducking, and clear-memory.  GUI app launches, browser
   navigation, and desktop clicks are now routed through ``cot_route()``
   in ``intent.py`` so the LLM can reason about context (#340).

2. **dispatch_internal(tool, args, user_input, en_input, tc, *, is_remote)**
   Handles every "internal" tool (``_tts_stop``, ``_briefing``, …) that
   ``quick_route`` may return.  Returns ``BrainResult | None``.
   Completely decoupled from Brain — uses ``is_remote`` kwarg instead.

3. **generate_command(orig, en)**
   LLM-based bash-command generation via ``COMMAND_SYSTEM``.

4. **execute_plan(user_input, en_input, tc)**
   Plan-and-Solve multi-step execution (#187).  Returns the result
   to the caller — does NOT persist to graph/embeddings itself.

5. **handle_maintenance / handle_list_reflections / handle_run_reflection**
   Thin wrappers around workflow modules.

Extracted from ``brain.py`` in Part 5 of epic #218.
"""
from __future__ import annotations

import re
import logging

from bantz.core.types import BrainResult
from bantz.config import config
from bantz.core.date_parser import resolve_date
from bantz.core.prompt_builder import COMMAND_SYSTEM
from bantz.data import data_layer
from bantz.llm.ollama import ollama
from bantz.tools import registry

log = logging.getLogger("bantz.routing_engine")


# ═══════════════════════════════════════════════════════════════════════════
# 1. quick_route — regex fast-path
# ═══════════════════════════════════════════════════════════════════════════

def quick_route(orig: str, en: str) -> dict | None:
    """Hardware/UI controls ONLY — instant routing, no LLM needed (#272).

    Covers only true hardware switches (TTS, wake word, audio ducking,
    clear memory).  All other routing — including GUI app launches,
    browser navigation, and desktop clicks — goes through ``cot_route()``
    so the LLM can reason about context and pick the right tool (#340).
    """
    o = orig.lower().strip()
    e = en.lower().strip()
    both = o + " " + e

    # ── TTS stop (#131) ───────────────────────────────────────────────
    if re.search(r"shut\s*up|be\s+quiet|stop\s+talk(?:ing)?", both):
        return {"tool": "_tts_stop", "args": {}}

    # ── Wake word control (#165) ──────────────────────────────────────
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

    # ── Audio ducking (#171) ──────────────────────────────────────────
    if re.search(r"enable\s+duck|duck(?:ing)?\s+on|turn\s+on\s+duck", both):
        return {"tool": "_audio_duck_on", "args": {}}
    if re.search(r"disable\s+duck|duck(?:ing)?\s+off|turn\s+off\s+duck|no\s+duck", both):
        return {"tool": "_audio_duck_off", "args": {}}

    # ── Clear memory ──────────────────────────────────────────────────
    if re.search(r"clear\s+memory", both):
        return {"tool": "_clear_memory", "args": {}}

    # ── App launches (unambiguous desktop apps) ───────────────────────
    # These bypass LLM entirely for instant response (#340 speed fix).
    m = re.search(
        r"(?:open|launch|start|run)\s+"
        r"(firefox|chrome|chromium|terminal|files|vscode|gedit)",
        both,
    )
    if m:
        return {"tool": "browser_control", "args": {"action": "open", "app": m.group(1)}}

    # ── URL navigation (explicit URL in input) ────────────────────────
    m = re.search(
        r"(?:go\s+to|open|navigate\s+(?:to)?)\s+"
        r"(https?://\S+|(?:www\.)\S+)",
        both,
    )
    if m:
        url = m.group(1)
        if not url.startswith("http"):
            url = "https://" + url
        return {"tool": "browser_control", "args": {"action": "navigate", "url": url}}

    # ── Well-known web apps (open Gemini, open ChatGPT, etc.) ─────────
    _WEB_APP_URLS: dict[str, str] = {
        "gemini": "https://gemini.google.com",
        "chatgpt": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "perplexity": "https://perplexity.ai",
        "github": "https://github.com",
        "reddit": "https://reddit.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "youtube": "https://youtube.com",
        "spotify": "https://open.spotify.com",
        "netflix": "https://netflix.com",
        "google": "https://google.com",
        "gmail": "https://mail.google.com",
        "whatsapp": "https://web.whatsapp.com",
        "discord": "https://discord.com/app",
        "telegram": "https://web.telegram.org",
        "linkedin": "https://linkedin.com",
        "instagram": "https://instagram.com",
        "wikipedia": "https://wikipedia.org",
        "stackoverflow": "https://stackoverflow.com",
    }
    m = re.search(
        r"(?:open|launch|go\s+to|navigate\s+to)\s+(\w+)"
        r"(?:\s+(?:in\s+(?:the\s+)?)?(?:web\s*)?browser|\s+web(?:site)?)?",
        both,
    )
    if m:
        app_name = m.group(1).lower()
        web_url = _WEB_APP_URLS.get(app_name)
        if web_url:
            return {"tool": "browser_control", "args": {"action": "navigate", "url": web_url}}

    # Everything else → cot_route (LLM-based reasoning)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. dispatch_internal — execute internal (underscore-prefixed) tools
# ═══════════════════════════════════════════════════════════════════════════

async def dispatch_internal(
    tool: str,
    args: dict,
    user_input: str,
    en_input: str,
    tc: dict,
    *,
    is_remote: bool = False,
) -> BrainResult | None:
    """Execute an internal tool returned by ``quick_route``.

    Returns a ``BrainResult`` for internal tools (``_tts_stop``, etc.)
    or ``None`` if ``tool`` is not an internal dispatch target.
    """
    text: str | None = None
    tool_label: str = tool.lstrip("_") or tool

    if tool == "_tts_stop":
        from bantz.agent.tts import tts_engine
        if tts_engine.is_speaking:
            tts_engine.stop()
            text = "🔇 Stopped."
        else:
            text = "I'm not speaking right now."
        tool_label = "tts"

    elif tool == "_wake_word_off":
        try:
            from bantz.agent.wake_word import wake_listener
            if wake_listener.running:
                wake_listener.stop()
                text = "🔇 Wake word listener stopped."
            else:
                text = "Wake word listener is not running."
        except Exception:
            text = "Wake word listener is not available."
        tool_label = "wake_word"

    elif tool == "_wake_word_on":
        try:
            from bantz.agent.wake_word import wake_listener
            if wake_listener.running:
                text = "Wake word listener is already running."
            else:
                ok = wake_listener.start()
                text = "🎤 Wake word listener started." if ok else "❌ Could not start wake word listener."
        except Exception:
            text = "Wake word listener is not available."
        tool_label = "wake_word"

    elif tool == "_audio_duck_on":
        try:
            from bantz.agent.audio_ducker import audio_ducker
            if audio_ducker.available():
                audio_ducker.enabled = True
                text = "🔉 Audio ducking enabled."
            else:
                text = "❌ Audio ducking not available (pactl not found)."
        except Exception:
            text = "Audio ducking module is not available."
        tool_label = "audio_ducker"

    elif tool == "_audio_duck_off":
        try:
            from bantz.agent.audio_ducker import audio_ducker
            audio_ducker.enabled = False
            text = "🔇 Audio ducking disabled."
        except Exception:
            text = "Audio ducking module is not available."
        tool_label = "audio_ducker"

    elif tool == "_ambient_status":
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
        tool_label = "ambient"

    elif tool == "_proactive_status":
        try:
            from bantz.agent.proactive import (
                _get_daily_count, _compute_adaptive_max,
            )
            from bantz.agent.affinity_engine import affinity_engine
            kv = data_layer.kv
            if kv:
                count, date = _get_daily_count(kv)
                avg_r = affinity_engine.get_score() if affinity_engine.initialized else 0.0
                max_d = _compute_adaptive_max(config.proactive_max_daily, avg_r)
                text = (
                    f"💬 Proactive Engagement Status\n"
                    f"  Enabled: {'✅' if config.proactive_enabled else '❌'}\n"
                    f"  Today: {count}/{max_d} messages\n"
                    f"  Affinity score: {avg_r:.1f}\n"
                    f"  Interval: {config.proactive_interval_hours}h ±{config.proactive_jitter_minutes}m"
                )
            else:
                text = "Proactive engine: KV store not available."
        except Exception:
            text = "Proactive engagement module is not available."
        tool_label = "proactive"

    elif tool == "_health_status":
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
        tool_label = "health"

    elif tool == "_briefing":
        from bantz.core.briefing import briefing as _briefing
        text = await _briefing.generate()
        tool_label = "briefing"
        # Speak via TTS if available — suppress for remote (#178)
        if not is_remote:
            try:
                from bantz.agent.tts import tts_engine
                if tts_engine.available():
                    await tts_engine.speak_background(text)
            except Exception:
                pass

    elif tool == "_maintenance":
        text = await handle_maintenance(args.get("dry_run", False))
        tool_label = "maintenance"

    elif tool == "_list_reflections":
        text = handle_list_reflections()
        tool_label = "reflection"

    elif tool == "_run_reflection":
        text = await handle_run_reflection(args.get("dry_run", False))
        tool_label = "reflection"

    elif tool == "_location":
        from bantz.core.location_handler import handle_location
        text = await handle_location()
        tool_label = "location"

    elif tool == "_save_place":
        from bantz.core.location_handler import handle_save_place
        text = await handle_save_place(args["name"])
        tool_label = "places"

    elif tool == "_list_places":
        from bantz.core.location_handler import handle_list_places
        text = await handle_list_places()
        tool_label = "places"

    elif tool == "_delete_place":
        from bantz.core.location_handler import handle_delete_place
        text = await handle_delete_place(args["name"])
        tool_label = "places"

    elif tool == "_schedule_today":
        from bantz.core.schedule import schedule as _sched
        text = _sched.format_today()
        tool_label = "schedule"

    elif tool == "_schedule_next":
        from bantz.core.schedule import schedule as _sched
        text = _sched.format_next()
        tool_label = "schedule"

    elif tool == "_schedule_date":
        from bantz.core.schedule import schedule as _sched
        from datetime import datetime as _dt
        target = _dt.fromisoformat(args["date_iso"])
        text = _sched.format_for_date(target)
        tool_label = "schedule"

    elif tool == "_schedule_week":
        from bantz.core.schedule import schedule as _sched
        resolved = resolve_date(user_input)
        text = _sched.format_week(resolved)
        tool_label = "schedule"

    else:
        return None  # not an internal tool — caller should continue routing

    data_layer.conversations.add("assistant", text, tool_used=tool_label)
    return BrainResult(response=text, tool_used=tool_label)


# ═══════════════════════════════════════════════════════════════════════════
# 3. generate_command — LLM shell-command generation
# ═══════════════════════════════════════════════════════════════════════════

async def generate_command(orig: str, en: str) -> str:
    """Ask the LLM to produce a single bash command from natural language."""
    raw = await ollama.chat([
        {"role": "system", "content": COMMAND_SYSTEM},
        {"role": "user", "content": en or orig},
    ])
    return raw.strip().strip("`")


# ═══════════════════════════════════════════════════════════════════════════
# 4. execute_plan — Plan-and-Solve multi-step (#187)
# ═══════════════════════════════════════════════════════════════════════════

async def execute_plan(
    user_input: str,
    en_input: str,
    tc: dict,
    *,
    recent_history: list[dict] | None = None,
) -> BrainResult | None:
    """Decompose a complex request into steps, then execute them.

    Returns ``BrainResult`` on success, or ``None`` if decomposition
    fails (so the caller can fall through to normal routing).

    Note: the caller (brain.py) is responsible for persisting the
    response via ``_graph_store`` / ``_fire_embeddings``.
    """
    from bantz.agent.planner import planner_agent
    from bantz.agent.executor import plan_executor

    tool_names = registry.names() + ["process_text", "summarizer"]
    steps = await planner_agent.decompose(
        en_input, tool_names, recent_history=recent_history,
    )
    if not steps or len(steps) < 2:
        return None

    itinerary = planner_agent.format_itinerary(steps)
    log.info("Plan-and-Solve itinerary:\n%s", itinerary)

    # Butler Lore toast — plan start (Architect's Revision)
    try:
        from bantz.core.notification_manager import notify_toast
        notify_toast(
            "📋 Drafting an itinerary...",
            f"{len(steps)} steps planned",
        )
    except Exception:
        pass

    exec_result = await plan_executor.run(
        steps, llm_fn=ollama.chat, original_request=en_input,
    )

    # Butler Lore toast — plan complete
    try:
        from bantz.core.notification_manager import notify_toast as _toast
        if exec_result.all_success:
            _toast("✓ Itinerary complete", f"All {exec_result.total} tasks finished.")
        else:
            _toast("⚠ Itinerary concluded", f"{exec_result.succeeded}/{exec_result.total} succeeded.")
    except Exception:
        pass

    resp = itinerary + "\n\n" + exec_result.summary()

    data_layer.conversations.add("assistant", resp, tool_used="planner")

    return BrainResult(response=resp, tool_used="planner")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Workflow handlers
# ═══════════════════════════════════════════════════════════════════════════

async def handle_maintenance(dry_run: bool = False) -> str:
    """Run the maintenance workflow and return its summary."""
    try:
        from bantz.agent.workflows.maintenance import run_maintenance
        report = await run_maintenance(dry_run=dry_run)
        return report.summary()
    except Exception as exc:
        return f"❌ Maintenance failed: {exc}"


def handle_list_reflections(limit: int = 5) -> str:
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


async def handle_run_reflection(dry_run: bool = False) -> str:
    """Run the reflection workflow and return its summary."""
    try:
        from bantz.agent.workflows.reflection import run_reflection
        result = await run_reflection(dry_run=dry_run)
        return result.summary_line()
    except Exception as exc:
        return f"❌ Reflection failed: {exc}"
