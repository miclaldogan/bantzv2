"""
Bantz — RL Hooks (#226, updated for #221 Affinity Engine)

Extracted from ``brain.py``: reward signals that fire after tool
execution and sentiment feedback.

Since #221 the heavy Q-learning engine is replaced by a simple
cumulative ``AffinityEngine`` score.  Writes go through the engine's
built-in ``add_reward()`` which is KV-store backed and thread-safe,
so we no longer need ``AsyncDBExecutor`` for these hooks.

Public API
----------
- ``rl_reward_hook(tool_name, result)``          → fire-and-forget async
- ``rl_feedback_reward(feedback, time_snapshot)`` → sentiment signal
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bantz.tools import ToolResult

log = logging.getLogger("bantz.rl_hooks")


# ── Reward hook (tool success / failure) ─────────────────────────────


async def rl_reward_hook(tool_name: str, result: "ToolResult") -> None:
    """Give the affinity engine a reward after tool execution.

    +1.0 on success, -0.5 on failure.
    Silently swallows all errors — this must never crash the pipeline.
    """
    try:
        from bantz.agent.affinity_engine import affinity_engine
        if not affinity_engine.initialized:
            return

        reward_val = 1.0 if result.success else -0.5
        affinity_engine.add_reward(reward_val)
        log.debug("affinity reward: tool=%s val=%.1f", tool_name, reward_val)
    except Exception:
        pass  # never crash the pipeline


# ── Sentiment RLHF feedback (#180) ──────────────────────────────────


async def rl_feedback_reward(
    feedback: str,
    time_snapshot: dict,
) -> None:
    """Send a strong sentiment reward to the affinity engine.

    Parameters
    ----------
    feedback : str
        ``'positive'`` or ``'negative'``.
    time_snapshot : dict
        Kept for API compatibility (unused by affinity engine).
    """
    try:
        from bantz.agent.affinity_engine import affinity_engine
        if not affinity_engine.initialized:
            return

        reward_val = 2.0 if feedback == "positive" else -2.0
        affinity_engine.add_reward(reward_val)
        log.info("RLHF sentiment: %s → reward %.1f", feedback, reward_val)
    except Exception:
        pass  # never crash the pipeline
