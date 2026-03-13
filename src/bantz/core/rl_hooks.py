"""
Bantz — RL Hooks (#226)

Extracted from ``brain.py``: reinforcement-learning reward signals
that fire after tool execution and sentiment feedback.

Key improvement: ``rl_reward_hook`` now offloads the RL engine's
synchronous SQLite writes (Q-table updates, episode logs) to the
``AsyncDBExecutor`` thread-pool (#224), so the async event-loop is
**never blocked** by RL bookkeeping.

Public API
----------
- ``rl_reward_hook(tool_name, result)``          → fire-and-forget async
- ``rl_feedback_reward(raw_input, time_ctx)``    → sentiment RLHF signal
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bantz.tools import ToolResult

log = logging.getLogger("bantz.rl_hooks")


# ── Reward hook (tool success / failure) ─────────────────────────────


async def rl_reward_hook(tool_name: str, result: "ToolResult") -> None:
    """Give the RL engine a reward signal after tool execution.

    Offloads the blocking ``rl_engine.reward()`` call (which writes to
    SQLite) onto ``AsyncDBExecutor`` so the event-loop stays free.

    Silently swallows all errors — this must never crash the pipeline.
    """
    try:
        from bantz.agent.rl_engine import rl_engine, encode_state
        if not rl_engine.initialized:
            return

        from bantz.core.time_context import time_ctx
        tc = time_ctx.snapshot()
        state = encode_state(
            time_segment=tc.get("time_segment", "morning"),
            day=tc.get("day_name", "monday").lower(),
            location=tc.get("location", "home"),
            recent_tool=tool_name,
        )
        reward_val = 1.0 if result.success else -0.5

        # Offload the synchronous rl_engine.reward() to the DB thread-pool
        from bantz.data.async_executor import run_in_db
        await run_in_db(
            lambda conn: rl_engine.reward(reward_val, next_state=state),
            write=True,
        )
    except Exception:
        pass  # never crash the pipeline


# ── Sentiment RLHF feedback (#180) ──────────────────────────────────


async def rl_feedback_reward(
    feedback: str,
    time_snapshot: dict,
) -> None:
    """Send a strong sentiment reward to the RL engine.

    Parameters
    ----------
    feedback : str
        ``'positive'`` or ``'negative'``.
    time_snapshot : dict
        Result of ``time_ctx.snapshot()`` — provides state context.

    Offloads ``rl_engine.force_reward()`` to the DB thread-pool via
    ``run_in_db``.
    """
    try:
        from bantz.agent.rl_engine import rl_engine, Action, encode_state
        if not rl_engine.initialized:
            return

        state = encode_state(
            time_segment=time_snapshot.get("time_segment", "morning"),
            day=time_snapshot.get("day_name", "monday").lower(),
            location=time_snapshot.get("location", "home"),
            recent_tool="feedback_chat",
        )
        reward_val = 2.0 if feedback == "positive" else -2.0

        from bantz.data.async_executor import run_in_db
        await run_in_db(
            lambda conn: rl_engine.force_reward(state, Action.FEEDBACK_CHAT, reward_val),
            write=True,
        )
        log.info("RLHF sentiment: %s → reward %.1f", feedback, reward_val)
    except Exception:
        pass  # never crash the pipeline
