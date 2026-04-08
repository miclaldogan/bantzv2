"""
Bantz — Memory Injector (#227, refactored for #211)

Gathers all **context signals** into a single ``BantzContext`` object
ready for ``prompt_builder``.

Architecture split (Issue #211 — Architect's Revision):
  ┌─────────────────────────────────────────────────────────────────┐
  │  OmniMemoryManager.recall()  [HISTORICAL — budget-trimmed]     │
  │    ├─ Graph search   ─┐                                        │
  │    ├─ Vector search   ├─ Parallel → Merge → Re-rank → Trim    │
  │    └─ Deep memory    ─┘                                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  Real-Time Status    [ALWAYS INCLUDED — never filtered]         │
  │    ├─ desktop_context()  (AppDetector snapshot)                 │
  │    ├─ persona_hint()     (dynamic persona)                     │
  │    ├─ formality_hint()   (bonding meter)                       │
  │    ├─ style_hint()       (response style)                      │
  │    └─ profile.prompt_hint()                                    │
  └─────────────────────────────────────────────────────────────────┘

Key design decisions:
  💡1: Real-time context is NEVER filtered.  Desktop + persona are
       short, ephemeral, and represent the "present moment".  Routing
       them through GraphRAG would cause "blind assistant" syndrome.

  💡2: Historical memory uses async parallel hybrid search via
       OmniMemoryManager.  Graph + Vector fire simultaneously.
       If Graph finds entities → vector results are re-ranked.
       If Graph is empty → pure vector results are returned.
       This prevents "Amnesia by Over-Filtering".

  💡3: Total memory budget is ~400 tokens (~1600 chars), enforced
       by OmniMemoryManager.  Real-time hints are excluded from
       the budget (they're short and essential).

Public API (unchanged)
----------------------
- ``inject(ctx, en_input)`` → populate ``BantzContext`` memory fields
- Individual helpers also exported for fine-grained use.
"""
from __future__ import annotations

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
    """Return dynamic persona state instruction (#169).

    Merges the persona builder output with the affinity engine's
    mood directive (#221) so the LLM has a complete picture.
    """
    parts: list[str] = []
    try:
        from bantz.personality.persona import persona_builder
        base = persona_builder.build()
        if base:
            parts.append(base)
    except Exception:
        pass
    try:
        from bantz.agent.affinity_engine import affinity_engine
        if affinity_engine.initialized:
            parts.append(f"[Affinity mood] {affinity_engine.get_persona_state()}")
    except Exception:
        pass
    return "\n".join(parts)


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
        from bantz.memory.bridge import palace_bridge
    except ImportError:
        return ""
    if palace_bridge and palace_bridge.enabled:
        try:
            return palace_bridge.graph_context(user_msg)
        except Exception:
            pass
    return ""


async def vector_context(user_msg: str, limit: int = 3) -> str:
    """Get relevant past memories via MemPalace semantic search."""
    try:
        from bantz.memory.bridge import palace_bridge
        if palace_bridge and palace_bridge.enabled:
            return palace_bridge.vector_context(user_msg, limit=limit)
    except Exception:
        pass
    return ""


async def deep_memory_context(user_msg: str) -> str:
    """Spontaneous deep memory recall (#170) via MemPalace."""
    try:
        from bantz.memory.bridge import palace_bridge
        if palace_bridge and palace_bridge.enabled:
            return palace_bridge.deep_memory(user_msg)
    except Exception:
        return ""
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# inject() — one-call context enrichment
# ═══════════════════════════════════════════════════════════════════════════


async def inject(ctx: "BantzContext", en_input: str) -> None:
    """Populate all memory/persona fields on *ctx* concurrently.

    Historical memory (graph, vector, deep) is gathered via
    ``OmniMemoryManager.recall()`` which enforces a token budget
    and runs all searches in parallel (#211).

    Real-time context (desktop, persona, style, formality, profile)
    is injected directly — never filtered or budget-trimmed.
    """
    # ── Historical memory via OmniMemoryManager (#211) ────────────
    from bantz.memory.omni_memory import omni_memory
    recall_result = await omni_memory.recall(en_input)

    # Populate individual fields (for backward compat / logging)
    ctx.graph_context = recall_result.graph_context
    ctx.vector_context = recall_result.vector_context
    ctx.deep_memory = recall_result.deep_memory
    ctx.memory_combined = recall_result.combined

    # ── Real-time context (always included, never filtered) ───────
    ctx.desktop_context = desktop_context()
    ctx.persona_state = persona_hint()
    ctx.formality_hint = formality_hint()
    ctx.style_hint = style_hint()

    from bantz.core.profile import profile
    ctx.profile_hint = profile.prompt_hint()
