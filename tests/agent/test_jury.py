"""Tests for the event-driven Jury (#557)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.agent.jury import Jury
from bantz.agent.sub_agent import SubAgentResult
from bantz.config import config
from bantz.core.event_bus import Event


class FakeKV:
    def __init__(self):
        self.d = {}

    def get(self, k, default=""):
        return self.d.get(k, default)

    def set(self, k, v):
        self.d[k] = v


@pytest.fixture
def jury():
    j = Jury()
    j._enabled = True
    return j


def _signal(jury, kind, **data):
    jury._make_handler(kind)(Event(name=kind, data=data))


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0.01)


async def test_three_job_failures_escalate_once_with_one_llm_call(jury):
    delegate = AsyncMock(return_value=SubAgentResult(
        success=True, summary="judged",
        data={"verdict": "degraded", "cause": "cron broke", "suggestion": "check"},
    ))
    pushes = []
    with patch("bantz.agent.jury._get_kv", return_value=FakeKV()), \
         patch("bantz.agent.agent_manager.agent_manager") as mgr, \
         patch("bantz.agent.interventions.intervention_queue") as q:
        mgr.delegate = delegate
        q.push = lambda i: pushes.append(i)
        for i in range(3):
            _signal(jury, "job_failed", job_id=f"j{i}", exception="boom")
        await _settle()

    assert delegate.await_count == 1, "expected exactly one LLM escalation"
    assert len(pushes) == 1
    assert "degraded" in pushes[0].title
    # internal delegations bypass the per-conversation cap
    assert delegate.call_args.kwargs.get("internal") is True


async def test_budget_exhausted_still_produces_rule_verdict(jury):
    delegate = AsyncMock()
    verdicts = []
    jury._llm_calls = [time.time()] * 10  # budget gone
    with patch("bantz.agent.jury._get_kv", return_value=FakeKV()), \
         patch("bantz.agent.agent_manager.agent_manager") as mgr, \
         patch("bantz.agent.interventions.intervention_queue"), \
         patch.object(config, "jury_llm_budget_per_hour", 4), \
         patch("bantz.core.event_bus.bus") as mock_bus:
        mgr.delegate = delegate
        mock_bus.emit_threadsafe = lambda name, **kw: verdicts.append((name, kw))
        for i in range(3):
            _signal(jury, "job_failed", job_id=f"j{i}", exception="boom")
        await _settle()

    delegate.assert_not_awaited()
    jury_events = [kw for name, kw in verdicts if name == "jury_verdict"]
    assert len(jury_events) == 1
    assert jury_events[0]["llm"] is False


async def test_rule_cooldown_prevents_refire(jury):
    pushes = []
    with patch("bantz.agent.jury._get_kv", return_value=FakeKV()), \
         patch("bantz.agent.agent_manager.agent_manager") as mgr, \
         patch("bantz.agent.interventions.intervention_queue") as q:
        mgr.delegate = AsyncMock(return_value=SubAgentResult.failure("x"))
        q.push = lambda i: pushes.append(i)
        for i in range(3):
            _signal(jury, "job_failed", job_id=f"j{i}", exception="boom")
        await _settle()
        # More failures inside the cooldown window → same rule must not refire.
        for i in range(3, 6):
            _signal(jury, "job_failed", job_id=f"j{i}", exception="boom")
        await _settle()

    assert len(pushes) == 1


async def test_steady_state_zero_llm_calls(jury):
    delegate = AsyncMock()
    with patch("bantz.agent.jury._get_kv", return_value=FakeKV()), \
         patch("bantz.agent.agent_manager.agent_manager") as mgr:
        mgr.delegate = delegate
        # Benign signals only: successful delegations, one job failure.
        _signal(jury, "delegation_done", role="web", success=True)
        _signal(jury, "job_failed", job_id="one_off", exception="x")
        await _settle()
    delegate.assert_not_awaited()


async def test_critical_anomaly_persistence_rule(jury):
    pushes = []
    with patch("bantz.agent.jury._get_kv", return_value=FakeKV()), \
         patch("bantz.agent.agent_manager.agent_manager") as mgr, \
         patch("bantz.agent.interventions.intervention_queue") as q:
        mgr.delegate = AsyncMock(return_value=SubAgentResult.failure("x"))
        q.push = lambda i: pushes.append(i)
        jury._on_anomalies(Event(name="anomaly_detected", data={
            "anomalies": [{"id": "swap", "severity": "critical"}]}))
        await _settle()
        assert not pushes  # too fresh
        jury._anomaly_seen["swap"] -= 6 * 60  # age it past 5 minutes
        jury._on_anomalies(Event(name="anomaly_detected", data={
            "anomalies": [{"id": "swap", "severity": "critical"}]}))
        await _settle()
    assert len(pushes) == 1


async def test_selfcheck_produces_summary_and_cache(jury):
    kv = FakeKV()
    pushes = []
    with patch("bantz.agent.jury._get_kv", return_value=kv), \
         patch("bantz.llm.lane.llm_call", new=AsyncMock(return_value="All fine.")), \
         patch("bantz.agent.interventions.intervention_queue") as q:
        q.push = lambda i: pushes.append(i)
        summary = await jury.selfcheck()
    assert summary == "All fine."
    assert "jury:selfcheck" in kv.d
    assert len(pushes) == 1
