"""
Bantz v3 — Proactive Engagement Engine (#167)

Initiates context-aware conversations during idle periods.  Uses APScheduler
cron job with jitter, context gate (AppDetector), vector DB with time-decay
recency, KV store fresh data, ambient-aware LLM prompt, and RL-adaptive
daily limits.

Architecture:
    APScheduler (every 3–4h, jitter=30min)
        │
        ▼
    _job_proactive_engagement
        1. Guard: proactive_enabled?
        2. Guard: daily limit (RL-adaptive: 1→3 based on avg reward)
        3. Guard: activity gate — abort if CODING/PRODUCTIVITY/COMMUNICATION
        4. Guard: focus_mode / quiet_mode on InterventionQueue
        5. Gather context: vector DB (time-decay), KV store, calendar
        6. Build ambient-aware LLM prompt
        7. Generate casual message via LLM
        8. Push to InterventionQueue (PROACTIVE / LOW)
        9. Telegram fallback if user away >30min
       10. Track daily count in KV store

Usage:
    from bantz.agent.proactive import proactive_engine

    # Called by job_scheduler — not directly by user code
    await proactive_engine.run()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bantz.config import config

log = logging.getLogger("bantz.proactive")

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Activities that block proactive engagement
_BLOCKED_ACTIVITIES = frozenset({"coding", "productivity", "communication"})

# Vector DB queries to find user interests
_INTEREST_QUERIES = [
    "user interests and hobbies",
    "ongoing projects and tasks",
    "recent conversation topics",
]

# KV keys for tracking
KV_PROACTIVE_COUNT = "proactive_daily_count"
KV_PROACTIVE_DATE = "proactive_daily_date"
KV_PROACTIVE_LAST = "proactive_last_timestamp"

# RL adaptive limit thresholds
_RL_REWARD_THRESHOLD_2 = 0.3   # avg reward > 0.3 → allow 2/day
_RL_REWARD_THRESHOLD_3 = 0.6   # avg reward > 0.6 → allow 3/day


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ProactiveContext:
    """Gathered context for generating a proactive message."""
    interests: list[dict] = field(default_factory=list)
    fresh_data: dict = field(default_factory=dict)  # KV overnight_poll cache
    ambient_label: str = "unknown"
    activity: str = "idle"
    time_of_day: str = "afternoon"
    daily_count: int = 0
    max_daily: int = 1


@dataclass
class ProactiveResult:
    """Result of a proactive engagement attempt."""
    success: bool = False
    message: str = ""
    reason: str = ""  # why it was sent or aborted
    context: Optional[ProactiveContext] = None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _time_of_day() -> str:
    """Return current time segment."""
    h = datetime.now().hour
    if h < 6:
        return "night"
    elif h < 12:
        return "morning"
    elif h < 18:
        return "afternoon"
    return "evening"


def _compute_adaptive_max(base_max: int, avg_reward: float) -> int:
    """Compute RL-adaptive daily limit.

    Starts at base_max (default 1), scales up to 3 as avg reward increases.
    This prevents Clippy-style spam — RL must earn the right to be chatty.

    Args:
        base_max: Configured base maximum (usually 1).
        avg_reward: Average RL reward over last 7 days.

    Returns:
        Allowed daily proactive messages (1–3).
    """
    if avg_reward >= _RL_REWARD_THRESHOLD_3:
        return max(base_max, 3)
    elif avg_reward >= _RL_REWARD_THRESHOLD_2:
        return max(base_max, 2)
    return base_max


def _get_daily_count(kv) -> tuple[int, str]:
    """Read today's proactive count from KV store.

    Returns (count, date_str).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    stored_date = kv.get(KV_PROACTIVE_DATE, "")
    if stored_date != today:
        # New day — reset counter
        kv.set(KV_PROACTIVE_DATE, today)
        kv.set(KV_PROACTIVE_COUNT, "0")
        return 0, today
    count = int(kv.get(KV_PROACTIVE_COUNT, "0"))
    return count, today


def _increment_daily_count(kv) -> None:
    """Bump today's proactive count."""
    count, _ = _get_daily_count(kv)
    kv.set(KV_PROACTIVE_COUNT, str(count + 1))
    kv.set(KV_PROACTIVE_LAST, str(time.time()))


def _build_prompt(ctx: ProactiveContext) -> list[dict]:
    """Build the LLM prompt for generating a proactive message.

    Ambient-aware: if NOISY → extremely brief and text-only.
    """
    # Interests summary
    interest_lines = []
    for item in ctx.interests[:3]:
        content = item.get("content", "")[:200]
        interest_lines.append(f"- {content}")
    interests_text = "\n".join(interest_lines) if interest_lines else "No specific interests found."

    # Fresh data summary
    fresh_parts = []
    if ctx.fresh_data:
        for key, val in ctx.fresh_data.items():
            if val and key in ("overnight_emails", "overnight_calendar", "overnight_news"):
                fresh_parts.append(f"- {key}: {str(val)[:200]}")
    fresh_text = "\n".join(fresh_parts) if fresh_parts else "No overnight data."

    # Ambient-aware tone instruction
    if ctx.ambient_label.lower() in ("noisy", "speech"):
        tone = (
            "The environment is NOISY. Keep the message EXTREMELY brief "
            "(one short sentence max) and text-only. No long explanations."
        )
    else:
        tone = (
            "The environment is quiet. You can be a bit more conversational, "
            "but still keep it to 1-2 sentences max."
        )

    system = (
        "You are Bantz, a witty and sharp digital companion. "
        "Generate a casual, one-line conversation starter for the user. "
        "Be natural and context-aware, not robotic or generic. "
        "Never say 'How can I help you?' — instead reference something specific.\n\n"
        f"Time: {ctx.time_of_day}\n"
        f"Activity: {ctx.activity}\n"
        f"Ambient: {ctx.ambient_label}\n"
        f"Tone instruction: {tone}\n\n"
        f"User interests (from memory):\n{interests_text}\n\n"
        f"Fresh data (overnight poll):\n{fresh_text}\n\n"
        "Generate ONLY the message text. No quotes, no labels, no preamble."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Generate a proactive check-in message."},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Proactive Engine
# ═══════════════════════════════════════════════════════════════════════════


class ProactiveEngine:
    """Orchestrates proactive engagement attempts.

    Designed to be called by ``job_scheduler`` as a cron job with jitter.
    All guards (daily limit, activity gate, focus mode) are checked here.
    """

    def __init__(self) -> None:
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init(self) -> None:
        """Mark as ready (data layer must already be initialized)."""
        self._initialized = True
        log.info("ProactiveEngine initialized")

    async def run(self) -> ProactiveResult:
        """Execute one proactive engagement attempt.

        Returns ProactiveResult describing what happened.
        """
        # ── Guard: enabled? ───────────────────────────────────────────
        if not config.proactive_enabled:
            return ProactiveResult(reason="proactive disabled in config")

        if not self._initialized:
            return ProactiveResult(reason="engine not initialized")

        # ── Guard: daily limit (RL-adaptive) ──────────────────────────
        from bantz.data import data_layer
        from bantz.agent.rl_engine import rl_engine

        kv = data_layer.kv
        if kv is None:
            return ProactiveResult(reason="KV store not available")

        avg_reward = 0.0
        if rl_engine.initialized:
            avg_reward = rl_engine.episodes.avg_reward(days=7)

        max_daily = _compute_adaptive_max(config.proactive_max_daily, avg_reward)
        daily_count, _ = _get_daily_count(kv)

        if daily_count >= max_daily:
            return ProactiveResult(
                reason=f"daily limit reached ({daily_count}/{max_daily})"
            )

        # ── Guard: activity gate ──────────────────────────────────────
        activity = "idle"
        try:
            from bantz.agent.app_detector import app_detector
            if app_detector.initialized:
                cat = app_detector.get_activity_category()
                activity = cat.value if cat else "idle"
                if activity in _BLOCKED_ACTIVITIES:
                    return ProactiveResult(
                        reason=f"blocked activity: {activity}"
                    )
        except Exception as exc:
            log.debug("AppDetector check failed: %s", exc)

        # ── Guard: intervention queue modes ───────────────────────────
        from bantz.agent.interventions import intervention_queue

        if intervention_queue.initialized:
            if intervention_queue.focus:
                return ProactiveResult(reason="focus mode active")
            if intervention_queue.quiet:
                return ProactiveResult(reason="quiet mode active")

        # ── Gather context ────────────────────────────────────────────
        ctx = ProactiveContext(
            activity=activity,
            time_of_day=_time_of_day(),
            daily_count=daily_count,
            max_daily=max_daily,
        )

        # Ambient state
        try:
            from bantz.agent.ambient import ambient_analyzer
            snap = ambient_analyzer.latest()
            if snap:
                ctx.ambient_label = snap.label.value
        except Exception:
            pass

        # Vector DB interests (with time-decay recency)
        try:
            from bantz.memory.embeddings import embedder
            from bantz.core.memory import memory

            if hasattr(memory, '_vector_store') and memory._vector_store:
                vs = memory._vector_store
                for query_text in _INTEREST_QUERIES:
                    vec = await embedder.embed(query_text)
                    if vec:
                        results = vs.search(
                            vec, limit=2, min_score=0.25,
                            recency_weight=0.3,
                        )
                        ctx.interests.extend(results)
                # Deduplicate by message_id
                seen = set()
                unique = []
                for item in ctx.interests:
                    mid = item.get("message_id")
                    if mid not in seen:
                        seen.add(mid)
                        unique.append(item)
                ctx.interests = unique[:5]
        except Exception as exc:
            log.debug("Vector search failed: %s", exc)

        # KV store fresh data (overnight poll cache)
        try:
            for key in ("overnight_emails", "overnight_calendar", "overnight_news"):
                val = kv.get(key, "")
                if val:
                    ctx.fresh_data[key] = val[:500]
        except Exception:
            pass

        # ── Generate message via LLM ──────────────────────────────────
        messages = _build_prompt(ctx)
        generated = ""
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                generated = await gemini.chat(messages, temperature=0.7)
        except Exception:
            pass

        if not generated:
            try:
                from bantz.llm.ollama import ollama
                generated = await ollama.chat(messages)
            except Exception as exc:
                return ProactiveResult(
                    reason=f"LLM generation failed: {exc}",
                    context=ctx,
                )

        generated = generated.strip().strip('"').strip("'")
        if not generated:
            return ProactiveResult(reason="LLM returned empty", context=ctx)

        # ── Push to intervention queue ────────────────────────────────
        from bantz.agent.interventions import (
            Intervention, InterventionType, Priority,
        )

        iv = Intervention(
            type=InterventionType.PROACTIVE,
            priority=Priority.LOW,
            title=f"💬 {generated[:100]}",
            reason="Proactive check-in based on context and interests",
            source="proactive",
            action="proactive_chat",
            state_key=None,
            ttl=60.0,  # longer TTL for casual messages
        )

        pushed = False
        if intervention_queue.initialized:
            pushed = intervention_queue.push(iv)

        # Track daily count
        _increment_daily_count(kv)

        # ── Telegram fallback if user away ────────────────────────────
        telegram_sent = False
        try:
            away_timeout = config.proactive_away_timeout
            tui_active = _is_tui_active()
            if not tui_active and away_timeout > 0:
                from bantz.agent.notifier import notifier
                if notifier.available:
                    notifier.send(f"💬 Bantz: {generated[:200]}")
                    telegram_sent = True
        except Exception:
            pass

        log.info(
            "Proactive message sent (count=%d/%d, pushed=%s, telegram=%s): %s",
            daily_count + 1, max_daily, pushed, telegram_sent,
            generated[:80],
        )

        return ProactiveResult(
            success=True,
            message=generated,
            reason="delivered",
            context=ctx,
        )


def _is_tui_active() -> bool:
    """Check if the TUI is currently running."""
    try:
        from textual.app import App
        return App.current is not None
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

proactive_engine = ProactiveEngine()
