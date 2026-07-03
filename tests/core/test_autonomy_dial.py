"""Autonomy dial enforcement across all four tiers (issue #492).

The dial is a monotonic caution ladder — each lower tier confirms a superset
of the tier above:

    absolute → confirm nothing (run everything, even destructive)
    high     → confirm destructive only (default; legacy shell toggle applies)
    medium   → confirm moderate + destructive
    low      → confirm every tool action (even safe)

Two layers must agree: the routing layer (`intent._extract_json` sets the
`requires_confirm` flag) and the executor gate (`Brain._execute_with_recovery`
actually stops for confirmation). Both are pinned here per (tier × risk).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import bantz.config as bantz_config
import bantz.core.brain as brain_mod
from bantz.core.brain import Brain
from bantz.core.intent import _extract_json
from bantz.tools import ToolResult


def _run(coro):
    return asyncio.run(coro)


# Expected confirmation per (autonomy, risk). True = must stop and ask.
EXPECT = {
    ("absolute", "safe"): False, ("absolute", "moderate"): False, ("absolute", "destructive"): False,
    ("high", "safe"): False,     ("high", "moderate"): False,     ("high", "destructive"): True,
    ("medium", "safe"): False,   ("medium", "moderate"): True,    ("medium", "destructive"): True,
    ("low", "safe"): True,       ("low", "moderate"): True,       ("low", "destructive"): True,
}


def _intent_requires_confirm(autonomy: str, risk: str) -> bool:
    """What intent._extract_json sets requires_confirm to for a tool verdict."""
    return {
        "absolute": False,
        "high": risk == "destructive",
        "medium": risk in ("moderate", "destructive"),
        "low": True,
    }[autonomy]


# ── routing layer: intent._extract_json sets requires_confirm per the ladder ──

@pytest.mark.parametrize("autonomy,risk", list(EXPECT))
def test_intent_sets_requires_confirm(autonomy, risk, monkeypatch):
    monkeypatch.setattr(bantz_config.config, "autonomy", autonomy)
    verdict = json.dumps({
        "route": "tool", "tool_name": "shell", "tool_args": {},
        "risk_level": risk, "confidence": 0.9,
    })
    plan = _extract_json(verdict)
    assert plan["requires_confirm"] is EXPECT[(autonomy, risk)]


def test_intent_chat_route_never_confirms(monkeypatch):
    monkeypatch.setattr(bantz_config.config, "autonomy", "low")
    verdict = json.dumps({"route": "chat", "tool_name": None, "tool_args": {},
                          "risk_level": "safe", "confidence": 0.9})
    assert _extract_json(verdict)["requires_confirm"] is False


# ── executor gate: Brain._execute_with_recovery honours the dial ─────────────

class _FakeTool:
    def __init__(self):
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(success=True, output="ran")


class _FakeRegistry:
    def __init__(self, tool):
        self._tool = tool

    def get(self, name):
        return self._tool

    def all_schemas(self):
        return []


@pytest.fixture
def brain():
    return Brain.__new__(Brain)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(brain_mod, "data_layer", MagicMock())
    monkeypatch.setattr(brain_mod, "cot_route", AsyncMock())
    # single-shot: the dial decision happens on iteration 1
    monkeypatch.setattr(brain_mod.config, "tool_loop_max_steps", 1)
    monkeypatch.setattr(brain_mod.config, "shell_confirm_destructive", True)


@pytest.mark.parametrize("autonomy,risk", list(EXPECT))
def test_gate_confirms_per_autonomy(autonomy, risk, brain, monkeypatch):
    monkeypatch.setattr(brain_mod.config, "autonomy", autonomy)
    tool = _FakeTool()
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry(tool))
    # a non-shell tool so is_destructive promotion doesn't override the risk
    outcome = _run(brain._execute_with_recovery(
        tool_name="demo", tool_args={}, risk=risk,
        requires_confirm=_intent_requires_confirm(autonomy, risk),
        confirmed=False, en_input="do it", recent_history=[], tool_ctx="",
    ))

    should_confirm = EXPECT[(autonomy, risk)]
    if should_confirm:
        assert outcome.terminal is not None
        assert outcome.terminal.needs_confirm is True
        assert tool.calls == []                    # gated → never executed
        assert outcome.iterations[-1]["gated"] == "needs_confirm"
    else:
        assert outcome.terminal is None
        assert outcome.result is not None and outcome.result.success is True
        assert len(tool.calls) == 1                 # ran without confirmation


def test_absolute_runs_destructive_without_confirm(brain, monkeypatch):
    monkeypatch.setattr(brain_mod.config, "autonomy", "absolute")
    tool = _FakeTool()
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry(tool))
    outcome = _run(brain._execute_with_recovery(
        tool_name="demo", tool_args={}, risk="destructive",
        requires_confirm=False, confirmed=False,
        en_input="wipe it", recent_history=[], tool_ctx="",
    ))
    assert outcome.terminal is None
    assert len(tool.calls) == 1


def test_confirmed_bypasses_first_iteration(brain, monkeypatch):
    # A user who already said "yes" (confirmed=True) is not re-prompted for the
    # first decision, even at low autonomy.
    monkeypatch.setattr(brain_mod.config, "autonomy", "low")
    tool = _FakeTool()
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry(tool))
    outcome = _run(brain._execute_with_recovery(
        tool_name="demo", tool_args={}, risk="moderate",
        requires_confirm=True, confirmed=True,
        en_input="do it", recent_history=[], tool_ctx="",
    ))
    assert outcome.terminal is None
    assert len(tool.calls) == 1
