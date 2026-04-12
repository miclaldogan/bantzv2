"""
Bantz — Agent Manager (Orchestrator for Sub-Agents) (#321)

The AgentManager coordinates delegation to specialised sub-agents.
It tracks active delegations, enforces rate limits, and provides
the interface that the ``delegate_task`` tool uses.

Thread safety: AgentManager is a singleton, but sub-agents are
ephemeral — each ``delegate()`` call creates a fresh instance.
No shared mutable state between sub-agents.

Rate limiting: max ``MAX_CONCURRENT`` delegations at once (configurable
via config), max ``MAX_PER_CONVERSATION`` per conversation session to
prevent runaway loops.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bantz.agent.sub_agent import (
    SubAgent,
    SubAgentResult,
    create_agent,
    resolve_role,
    available_roles,
    AGENT_ROLES,
)

log = logging.getLogger("bantz.agent_manager")


# ═══════════════════════════════════════════════════════════════════════════
# Delegation record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DelegationRecord:
    """Tracks a single delegation for auditing and rate-limiting."""
    role: str
    task: str
    started_at: float
    finished_at: float = 0.0
    success: bool = False
    summary: str = ""
    tools_used: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def duration_s(self) -> float:
        if self.finished_at > 0:
            return self.finished_at - self.started_at
        return time.monotonic() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "task": self.task[:200],
            "duration_s": round(self.duration_s, 2),
            "success": self.success,
            "summary": self.summary[:500],
            "tools_used": self.tools_used,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Agent Manager
# ═══════════════════════════════════════════════════════════════════════════

class AgentManager:
    """Orchestrator for the multi-agent hierarchy.

    Usage::

        result = await agent_manager.delegate("researcher", "Find the GDP of Turkey", {"year": 2025})
        print(result.summary)

    Rate limits:
      • MAX_CONCURRENT — max simultaneous delegations (default 3)
      • MAX_PER_CONVERSATION — max total delegations per session (default 20)
      • DELEGATION_TIMEOUT — max seconds per delegation (default 120)
    """

    MAX_CONCURRENT: int = 3
    MAX_PER_CONVERSATION: int = 20
    DELEGATION_TIMEOUT: float = 120.0

    def __init__(self) -> None:
        self._active: int = 0
        self._history: list[DelegationRecord] = []
        self._lock = asyncio.Lock()
        self._enabled: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def init(self) -> None:
        """Enable the agent manager (called from brain on startup)."""
        from bantz.config import config
        self._enabled = getattr(config, "multi_agent_enabled", False)
        if self._enabled:
            log.info("AgentManager initialised — %d roles available", len(AGENT_ROLES))

    @property
    def total_delegations(self) -> int:
        return len(self._history)

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def history(self) -> list[DelegationRecord]:
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        """Return manager statistics for diagnostics."""
        return {
            "enabled": self._enabled,
            "active": self._active,
            "total_delegations": len(self._history),
            "successful": sum(1 for r in self._history if r.success),
            "failed": sum(1 for r in self._history if not r.success),
            "available_roles": [r for r in AGENT_ROLES],
        }

    async def delegate(
        self,
        role: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Delegate a task to a sub-agent.

        This is the main entry point for the multi-agent system.

        Args:
            role: Agent role name or alias (e.g. "researcher", "dev").
            task: Natural language task description.
            context: Optional context dict from the orchestrator (memory
                     snippets, prior results, user preferences, etc.).

        Returns:
            SubAgentResult with the agent's findings/output.

        Raises:
            ValueError: If the role is invalid or rate limits exceeded.
            TimeoutError: If the agent takes too long.
        """
        if not self._enabled:
            return SubAgentResult.failure("Multi-agent system is disabled.")

        # Resolve role
        canonical = resolve_role(role)
        if canonical is None:
            roles_str = ", ".join(AGENT_ROLES.keys())
            return SubAgentResult.failure(
                f"Unknown agent role: '{role}'. Available: {roles_str}"
            )

        # Rate limit checks
        if self._active >= self.MAX_CONCURRENT:
            return SubAgentResult.failure(
                f"Too many concurrent delegations ({self._active}/{self.MAX_CONCURRENT}). "
                "Wait for current tasks to finish."
            )
        if len(self._history) >= self.MAX_PER_CONVERSATION:
            return SubAgentResult.failure(
                f"Delegation limit reached ({self.MAX_PER_CONVERSATION} per session). "
                "Start a new conversation."
            )

        # Create ephemeral agent
        agent = create_agent(canonical)
        if agent is None:
            return SubAgentResult.failure(f"Failed to create agent for role: {canonical}")

        record = DelegationRecord(
            role=canonical,
            task=task,
            started_at=time.monotonic(),
        )

        async with self._lock:
            self._active += 1

        log.info(
            "Delegating to %s agent: %s (active: %d)",
            agent.display_name, task[:80], self._active,
        )

        # Toast notification
        try:
            from bantz.core.notification_manager import notify_toast
            notify_toast(
                f"🤖 Delegating to {agent.display_name}",
                task[:60],
                "info",
            )
        except Exception:
            pass

        try:
            result = await asyncio.wait_for(
                agent.run(task, context),
                timeout=self.DELEGATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            result = SubAgentResult.failure(
                f"{agent.display_name} agent timed out after {self.DELEGATION_TIMEOUT}s."
            )
        except Exception as exc:
            result = SubAgentResult.failure(
                f"{agent.display_name} agent crashed: {type(exc).__name__}: {exc}"
            )
        finally:
            async with self._lock:
                self._active -= 1

        # Record result
        record.finished_at = time.monotonic()
        record.success = result.success
        record.summary = result.summary[:500]
        record.tools_used = result.tools_used
        record.error = result.error
        self._history.append(record)

        log.info(
            "Delegation complete: %s → %s (%.1fs, tools: %s)",
            canonical,
            "success" if result.success else "failed",
            record.duration_s,
            result.tools_used,
        )

        return result

    def list_roles(self) -> list[dict[str, str]]:
        """Return available roles (for the delegate_task tool schema)."""
        return available_roles()

    def reset(self) -> None:
        """Reset delegation history (for testing / new session)."""
        self._history.clear()
        self._active = 0


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

agent_manager = AgentManager()
