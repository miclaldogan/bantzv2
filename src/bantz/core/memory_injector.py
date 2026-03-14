"""
Bantz — Memory Injector (#227)

Gathers all **context signals** (vector search, graph memory, deep-memory
recall, desktop snapshot, persona, bonding, style) into a single
``BantzContext`` object ready for ``prompt_builder``.

Every DB/memory read is wrapped in ``asyncio.to_thread`` so the
event-loop is never blocked by vector-DB or Neo4j I/O.

Public API
----------
- ``inject(ctx, en_input)`` → populate ``BantzContext`` memory fields
- Individual helpers also exported for fine-grained use.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bantz.core.context import BantzContext

log = logging.getLogger("bantz.memory_injector")


# ═══════════════════════════════════════════════════════════════════════════
# Style / persona / formality hints (pure-CPU, no I/O)
# ═══════════════════════════════════════════════════════════════════════════


def style_hint() -> str:
    """Return a style instruction based on profile response_style and pronoun."""
    from bantz.core.profile import profile
    style = profile.response_style
    pronoun = profile.get("pronoun", "casual")
    address = profile.get("preferred_address", "")
    if not address:
        if pronoun in ("siz", "formal", "ma'am", "madam"):
            address = "ma'am"
        else:
            address = "boss"
    if style == "formal" or pronoun in ("siz", "formal"):
        return f"Tone: professional, respectful. Address the user as '{address}'."
    return f"Tone: casual, friendly. Address the user as '{address}'."


def persona_hint() -> str:
    """Return dynamic persona state instruction (#169)."""
    try:
        from bantz.personality.persona import persona_builder
        return persona_builder.build()
    except Exception:
        return ""


def formality_hint() -> str:
    """Return bonding-meter formality instruction (#172)."""
    try:
        from bantz.personality.bonding import bonding_meter
        hint = bonding_meter.get_formality_hint()
        return f"\n[Bonding level] {hint}" if hint else ""
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Desktop context (AppDetector — no DB, but may be slow on dbus)
# ═══════════════════════════════════════════════════════════════════════════


def desktop_context() -> str:
    """Build desktop context from AppDetector for the system prompt."""
    try:
        from bantz.agent.app_detector import app_detector
        if not app_detector.initialized:
            return ""
        ctx = app_detector.get_workspace_context()
        if not ctx:
            return ""

        lines = ["Desktop Context (live data from AppDetector):"]

        # Active window
        win_info = ctx.get("active_window")
        if win_info:
            lines.append(
                f"  Active window: {win_info.get('name', '?')} "
                f"— {win_info.get('title', '')}"
            )

        # Activity
        activity = ctx.get("activity", "idle")
        lines.append(f"  Activity: {activity}")

        # Running apps
        apps = ctx.get("apps", [])
        if apps:
            lines.append(f"  Running apps ({len(apps)}): {', '.join(apps[:15])}")

        # IDE context
        ide = ctx.get("ide")
        if ide and ide.get("ide"):
            lines.append(
                f"  IDE: {ide['ide']} — file: {ide.get('file', '?')} "
                f"project: {ide.get('project', '?')}"
            )

        # Docker containers
        docker = ctx.get("docker")
        if docker:
            running = [c for c in docker if c.get("state") == "running"]
            if running:
                names = [c.get("name", c.get("image", "?")) for c in running]
                lines.append(
                    f"  Docker ({len(running)} running): {', '.join(names[:10])}"
                )

        return "\n".join(lines)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Async memory fetchers (offloaded via asyncio.to_thread where needed)
# ═══════════════════════════════════════════════════════════════════════════


async def graph_context(user_msg: str) -> str:
    """Get graph memory context string (empty if disabled)."""
    try:
        from bantz.memory.graph import graph_memory
    except ImportError:
        return ""
    if graph_memory and graph_memory.enabled:
        try:
            return await graph_memory.context_for(user_msg)
        except Exception:
            pass
    return ""


async def vector_context(user_msg: str, limit: int = 3) -> str:
    """Get relevant past messages via semantic search (#116)."""
    try:
        from bantz.core.memory import memory
        results = await memory.hybrid_search(user_msg, limit=limit)
        if not results:
            return ""
        lines = []
        for r in results:
            src = r.get("source", "?")
            score = r.get("hybrid_score", 0)
            lines.append(f"[{src} {score:.2f}] {r['role']}: {r['content'][:200]}")

        # Append distillation context (#118)
        try:
            distills = await memory.search_distillations(user_msg, limit=2)
            for d in distills:
                lines.append(
                    f"[session-summary {d['score']:.2f}] {d['summary'][:200]}"
                )
        except Exception:
            pass

        return "Relevant past context:\n" + "\n".join(lines)
    except Exception:
        return ""


async def deep_memory_context(user_msg: str) -> str:
    """Spontaneous deep memory recall (#170)."""
    try:
        from bantz.memory.deep_probe import deep_probe
        return await deep_probe.probe(user_msg)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# inject() — one-call context enrichment
# ═══════════════════════════════════════════════════════════════════════════


async def inject(ctx: "BantzContext", en_input: str) -> None:
    """Populate all memory/persona fields on *ctx* concurrently.

    Runs async memory fetchers in parallel via ``asyncio.gather``,
    and fills pure-CPU hints synchronously.
    """
    # Fire all async fetchers concurrently
    graph_coro = graph_context(en_input)
    vector_coro = vector_context(en_input)
    deep_coro = deep_memory_context(en_input)

    graph_res, vector_res, deep_res = await asyncio.gather(
        graph_coro, vector_coro, deep_coro,
        return_exceptions=True,
    )

    # Assign results (empty string on exception)
    ctx.graph_context = graph_res if isinstance(graph_res, str) else ""
    ctx.vector_context = vector_res if isinstance(vector_res, str) else ""
    ctx.deep_memory = deep_res if isinstance(deep_res, str) else ""

    # Sync helpers (cheap, no I/O)
    ctx.desktop_context = desktop_context()
    ctx.persona_state = persona_hint()
    ctx.formality_hint = formality_hint()
    ctx.style_hint = style_hint()

    from bantz.core.profile import profile
    ctx.profile_hint = profile.prompt_hint()
