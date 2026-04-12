"""Tests for bantz.agent.agent_manager — Issue #321 AgentManager.

Covers:
  1. init() / enabled flag — reads config, toggles manager on/off
  2. delegate() — happy path, unknown role, disabled state
  3. Rate limiting — concurrent limit, per-session limit
  4. Timeout handling — delegation_timeout enforcement
  5. DelegationRecord — creation, duration, to_dict
  6. Stats / reset — statistics tracking and session reset
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.agent.agent_manager import AgentManager, DelegationRecord
from bantz.agent.sub_agent import SubAgentResult


# ═══════════════════════════════════════════════════════════════════════════
# DelegationRecord
# ═══════════════════════════════════════════════════════════════════════════

class TestDelegationRecord:
    """DelegationRecord dataclass and helpers."""

    def test_duration_completed(self):
        """Duration when both started and finished are set."""
        rec = DelegationRecord(
            role="researcher",
            task="Find GDP",
            started_at=100.0,
            finished_at=103.5,
        )
        assert rec.duration_s == pytest.approx(3.5)

    def test_duration_in_progress(self):
        """Duration when still in progress uses monotonic clock."""
        rec = DelegationRecord(
            role="researcher",
            task="Find GDP",
            started_at=time.monotonic() - 2.0,
        )
        assert rec.duration_s >= 1.5  # at least ~2 secs

    def test_to_dict(self):
        """to_dict returns JSON-serialisable dict."""
        rec = DelegationRecord(
            role="developer",
            task="Write script",
            started_at=100.0,
            finished_at=105.0,
            success=True,
            summary="Script written",
            tools_used=["shell", "filesystem"],
        )
        d = rec.to_dict()
        assert d["role"] == "developer"
        assert d["success"] is True
        assert d["duration_s"] == 5.0
        assert d["tools_used"] == ["shell", "filesystem"]

    def test_to_dict_truncates_long_fields(self):
        """Long task and summary are truncated in to_dict."""
        rec = DelegationRecord(
            role="researcher",
            task="x" * 500,
            started_at=0,
            finished_at=1,
            summary="y" * 1000,
        )
        d = rec.to_dict()
        assert len(d["task"]) == 200
        assert len(d["summary"]) == 500


# ═══════════════════════════════════════════════════════════════════════════
# AgentManager — Init & Enable
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentManagerInit:
    """Initialisation and enable/disable logic."""

    def test_disabled_by_default(self):
        mgr = AgentManager()
        assert mgr.enabled is False

    def test_init_enables(self):
        """init() reads config and enables the manager."""
        mgr = AgentManager()
        mock_cfg = MagicMock()
        mock_cfg.multi_agent_enabled = True
        with patch("bantz.agent.agent_manager.config", mock_cfg, create=True), \
             patch("bantz.config.config", mock_cfg):
            mgr.init()
        assert mgr.enabled is True

    def test_init_disabled_stays_disabled(self):
        """init() with config.multi_agent_enabled=False keeps disabled."""
        mgr = AgentManager()
        mock_cfg = MagicMock()
        mock_cfg.multi_agent_enabled = False
        with patch("bantz.agent.agent_manager.config", mock_cfg, create=True), \
             patch("bantz.config.config", mock_cfg):
            mgr.init()
        assert mgr.enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# AgentManager — Delegation
# ═══════════════════════════════════════════════════════════════════════════

class TestDelegate:
    """delegate() — the core delegation method."""

    @pytest.mark.asyncio
    async def test_delegate_success(self):
        """Happy path: delegate to researcher, get result."""
        mgr = AgentManager()
        mgr._enabled = True

        mock_result = SubAgentResult(
            success=True,
            summary="Found the data",
            tools_used=["web_search"],
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.display_name = "Researcher"

        with patch("bantz.agent.agent_manager.create_agent", return_value=mock_agent), \
             patch("bantz.agent.agent_manager.resolve_role", return_value="researcher"):
            result = await mgr.delegate("researcher", "Find GDP")

        assert result.success is True
        assert result.summary == "Found the data"
        assert mgr.total_delegations == 1
        assert mgr.active_count == 0  # finished

    @pytest.mark.asyncio
    async def test_delegate_disabled(self):
        """Delegation when manager is disabled returns failure."""
        mgr = AgentManager()
        mgr._enabled = False

        result = await mgr.delegate("researcher", "Find GDP")
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delegate_unknown_role(self):
        """Unknown role returns failure with available roles."""
        mgr = AgentManager()
        mgr._enabled = True

        with patch("bantz.agent.agent_manager.resolve_role", return_value=None):
            result = await mgr.delegate("plumber", "Fix pipes")

        assert result.success is False
        assert "Unknown" in result.error or "plumber" in result.error

    @pytest.mark.asyncio
    async def test_delegate_records_history(self):
        """Delegation is recorded in history."""
        mgr = AgentManager()
        mgr._enabled = True

        mock_result = SubAgentResult(success=True, summary="done")
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.display_name = "Developer"

        with patch("bantz.agent.agent_manager.create_agent", return_value=mock_agent), \
             patch("bantz.agent.agent_manager.resolve_role", return_value="developer"):
            await mgr.delegate("developer", "Write code")

        assert len(mgr.history) == 1
        rec = mgr.history[0]
        assert rec.role == "developer"
        assert rec.success is True
        assert rec.finished_at > 0

    @pytest.mark.asyncio
    async def test_delegate_agent_crash(self):
        """Agent crash is caught and recorded."""
        mgr = AgentManager()
        mgr._enabled = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Boom"))
        mock_agent.display_name = "Researcher"

        with patch("bantz.agent.agent_manager.create_agent", return_value=mock_agent), \
             patch("bantz.agent.agent_manager.resolve_role", return_value="researcher"):
            result = await mgr.delegate("researcher", "Find data")

        assert result.success is False
        assert "crashed" in result.error.lower() or "Boom" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

class TestRateLimiting:
    """Concurrent and per-session rate limits."""

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """When MAX_CONCURRENT is reached, new delegations are refused."""
        mgr = AgentManager()
        mgr._enabled = True
        mgr.MAX_CONCURRENT = 2
        mgr._active = 2  # simulate 2 active

        result = await mgr.delegate("researcher", "Another task")
        assert result.success is False
        assert "concurrent" in result.error.lower() or "Too many" in result.error

    @pytest.mark.asyncio
    async def test_session_limit(self):
        """When MAX_PER_CONVERSATION is reached, delegation is refused."""
        mgr = AgentManager()
        mgr._enabled = True
        mgr.MAX_PER_CONVERSATION = 2
        # Simulate 2 completed delegations
        mgr._history = [
            DelegationRecord("r", "t1", 0, 1, True, "ok"),
            DelegationRecord("r", "t2", 1, 2, True, "ok"),
        ]

        result = await mgr.delegate("researcher", "Third task")
        assert result.success is False
        assert "limit" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Timeout
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeout:
    """Delegation timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_enforced(self):
        """Agent that takes too long is cancelled."""
        mgr = AgentManager()
        mgr._enabled = True
        mgr.DELEGATION_TIMEOUT = 0.1  # 100ms

        async def slow_run(task, context):
            await asyncio.sleep(10)
            return SubAgentResult(success=True, summary="too late")

        mock_agent = MagicMock()
        mock_agent.run = slow_run
        mock_agent.display_name = "Researcher"

        with patch("bantz.agent.agent_manager.create_agent", return_value=mock_agent), \
             patch("bantz.agent.agent_manager.resolve_role", return_value="researcher"):
            result = await mgr.delegate("researcher", "Slow task")

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert mgr.active_count == 0  # cleaned up


# ═══════════════════════════════════════════════════════════════════════════
# Stats & Reset
# ═══════════════════════════════════════════════════════════════════════════

class TestStatsAndReset:
    """stats() and reset() methods."""

    def test_stats_empty(self):
        mgr = AgentManager()
        s = mgr.stats()
        assert s["total_delegations"] == 0
        assert s["active"] == 0
        assert s["successful"] == 0
        assert "available_roles" in s

    def test_stats_after_delegations(self):
        mgr = AgentManager()
        mgr._history = [
            DelegationRecord("researcher", "t1", 0, 1, True, "ok"),
            DelegationRecord("developer", "t2", 1, 2, False, "", error="fail"),
        ]
        s = mgr.stats()
        assert s["total_delegations"] == 2
        assert s["successful"] == 1
        assert s["failed"] == 1

    def test_reset_clears_state(self):
        mgr = AgentManager()
        mgr._history = [
            DelegationRecord("researcher", "t1", 0, 1, True, "ok"),
        ]
        mgr._active = 1
        mgr.reset()
        assert mgr.total_delegations == 0
        assert mgr.active_count == 0

    def test_list_roles(self):
        mgr = AgentManager()
        roles = mgr.list_roles()
        assert len(roles) == 3
        role_ids = {r["role"] for r in roles}
        assert "researcher" in role_ids
