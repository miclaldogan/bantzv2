"""C1 observe→re-decide loop (audit C1, issue #503).

Exercises ``Brain._execute_with_recovery`` directly — the self-contained
recovery loop — with fake tools and a mocked ``cot_route``, so no LLM or DB is
touched. Covers the issue's safety-critical acceptance criteria:

- max_steps=1 (default) is single-shot: one iteration, NO re-decide call.
- re-decide calls use skip_fastpath=True (#502) — never re-enter the fast-path.
- the exception path enters the loop as an observation when steps>1.
- destructive + autonomy gates are re-checked every iteration; a re-decided
  action needing confirmation STOPS the loop and never executes.
- the per-iteration trace on BrainResult matches the loop_eval contract (#500).
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import bantz.core.brain as brain_mod
from bantz.core.brain import Brain
from bantz.tools import ToolResult


# ── load the #500 contract validator straight from the eval tree ─────────────
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "eval" / "loop_eval" / "schema.py"
_spec = importlib.util.spec_from_file_location("loop_eval_schema", _SCHEMA_PATH)
loop_schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop_schema)


def _run(coro):
    return asyncio.run(coro)


class _FakeTool:
    """Returns/raises a scripted sequence of outcomes, recording each call."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeRegistry:
    def __init__(self, tools: dict):
        self.tools = tools

    def get(self, name):
        return self.tools.get(name)

    def all_schemas(self):
        return []


@pytest.fixture
def brain():
    # Bypass the heavy __init__ — the loop only uses module globals + statics.
    return Brain.__new__(Brain)


@pytest.fixture(autouse=True)
def _quiet_conversation(monkeypatch):
    # The loop appends confirm / not-found messages to the conversation log;
    # stub the whole data_layer so no DB is required (conversations is lazily
    # None until the app boots).
    monkeypatch.setattr(brain_mod, "data_layer", MagicMock())


def _cfg(monkeypatch, *, max_steps=1, token_budget=4096,
         autonomy="high", confirm_destructive=True):
    monkeypatch.setattr(brain_mod.config, "tool_loop_max_steps", max_steps)
    monkeypatch.setattr(brain_mod.config, "tool_loop_token_budget", token_budget)
    monkeypatch.setattr(brain_mod.config, "autonomy", autonomy)
    monkeypatch.setattr(brain_mod.config, "shell_confirm_destructive",
                        confirm_destructive)


async def _call(brain, **over):
    kw = dict(
        tool_name="alpha", tool_args={}, risk="safe", requires_confirm=None,
        confirmed=False, en_input="do the thing", recent_history=[], tool_ctx="",
    )
    kw.update(over)
    return await brain._execute_with_recovery(**kw)


# ── max_steps=1: single-shot, no re-decide ───────────────────────────────────

def test_single_shot_failure_no_redecide(brain, monkeypatch):
    _cfg(monkeypatch, max_steps=1)
    tool = _FakeTool([ToolResult(success=False, output="", error="boom")])
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry({"alpha": tool}))
    cot = AsyncMock()
    monkeypatch.setattr(brain_mod, "cot_route", cot)

    outcome = _run(_call(brain))

    assert outcome.terminal is None
    assert outcome.result is not None and outcome.result.success is False
    assert len(outcome.iterations) == 1
    assert outcome.iterations[0]["decision_source"] == "initial"
    assert len(tool.calls) == 1
    cot.assert_not_called()          # the default path makes NO extra LLM call


def test_single_shot_success(brain, monkeypatch):
    _cfg(monkeypatch, max_steps=1)
    tool = _FakeTool([ToolResult(success=True, output="ok")])
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry({"alpha": tool}))
    monkeypatch.setattr(brain_mod, "cot_route", AsyncMock())

    outcome = _run(_call(brain))

    assert outcome.terminal is None
    assert outcome.result.success is True
    assert outcome.tool_name == "alpha"
    assert len(outcome.iterations) == 1


def test_single_shot_exception_is_terminal_butler(brain, monkeypatch):
    # Byte-identical to the pre-#503 behaviour: a raised exception at
    # max_steps=1 returns the butler reply, not a recovered result.
    _cfg(monkeypatch, max_steps=1)
    tool = _FakeTool([RuntimeError("kaboom")])
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry({"alpha": tool}))
    cot = AsyncMock()
    monkeypatch.setattr(brain_mod, "cot_route", cot)

    outcome = _run(_call(brain))

    assert outcome.result is None
    assert outcome.terminal is not None
    assert outcome.terminal.tool_used == "alpha"
    assert outcome.terminal.needs_confirm is False
    assert len(outcome.iterations) == 1
    assert "RuntimeError" in (outcome.iterations[0]["exception"] or "")
    cot.assert_not_called()


# ── re-decide behaviour (steps>1) ────────────────────────────────────────────

def test_redecide_uses_skip_fastpath(brain, monkeypatch):
    _cfg(monkeypatch, max_steps=2)
    first = _FakeTool([ToolResult(success=False, output="", error="boom")])
    second = _FakeTool([ToolResult(success=True, output="recovered")])
    monkeypatch.setattr(brain_mod, "registry",
                        _FakeRegistry({"alpha": first, "beta": second}))
    cot = AsyncMock(return_value=(
        {"route": "tool", "tool_name": "beta", "tool_args": {"x": 1},
         "risk_level": "safe"}, None))
    monkeypatch.setattr(brain_mod, "cot_route", cot)

    outcome = _run(_call(brain))

    assert outcome.result.success is True
    assert outcome.tool_name == "beta"
    assert len(outcome.iterations) == 2
    assert outcome.iterations[1]["tool_name"] == "beta"
    assert outcome.iterations[1]["decision_source"] == "llm"
    # SAFETY (#502): re-decide MUST bypass the fast-path.
    cot.assert_awaited_once()
    assert cot.await_args.kwargs["skip_fastpath"] is True


def test_exception_enters_loop_as_observation(brain, monkeypatch):
    _cfg(monkeypatch, max_steps=2)
    first = _FakeTool([RuntimeError("network down")])
    second = _FakeTool([ToolResult(success=True, output="recovered")])
    monkeypatch.setattr(brain_mod, "registry",
                        _FakeRegistry({"alpha": first, "beta": second}))
    monkeypatch.setattr(brain_mod, "cot_route", AsyncMock(return_value=(
        {"route": "tool", "tool_name": "beta", "tool_args": {},
         "risk_level": "safe"}, None)))

    outcome = _run(_call(brain))

    assert outcome.result.success is True
    assert len(outcome.iterations) == 2
    assert outcome.iterations[0]["result"]["success"] is False
    assert "RuntimeError" in (outcome.iterations[0]["exception"] or "")
    assert outcome.iterations[1]["result"]["success"] is True


def test_redecide_honest_giveup_keeps_failed_result(brain, monkeypatch):
    # If the model answers in chat instead of picking a tool, the loop stops
    # and the last failed ToolResult flows on to the finalizer (honest error).
    _cfg(monkeypatch, max_steps=3)
    first = _FakeTool([ToolResult(success=False, output="", error="nope")])
    monkeypatch.setattr(brain_mod, "registry", _FakeRegistry({"alpha": first}))
    monkeypatch.setattr(brain_mod, "cot_route",
                        AsyncMock(return_value=(None, None)))  # chat / give-up

    outcome = _run(_call(brain))

    assert outcome.terminal is None
    assert outcome.result is not None and outcome.result.success is False
    assert len(outcome.iterations) == 1


# ── safety: re-decided destructive action is gated, never executed ───────────

def test_redecided_destructive_shell_gates_and_never_runs(brain, monkeypatch):
    _cfg(monkeypatch, max_steps=2, autonomy="high", confirm_destructive=True)
    first = _FakeTool([ToolResult(success=False, output="", error="boom")])
    shell = _FakeTool([ToolResult(success=True, output="deleted")])  # must NOT run
    monkeypatch.setattr(brain_mod, "registry",
                        _FakeRegistry({"alpha": first, "shell": shell}))
    # Router mislabels the destructive command "safe"; is_destructive must
    # promote it and the gate must stop the loop.
    monkeypatch.setattr(brain_mod, "cot_route", AsyncMock(return_value=(
        {"route": "tool", "tool_name": "shell",
         "tool_args": {"command": "rm -rf /tmp/x"}, "risk_level": "safe"}, None)))

    outcome = _run(_call(brain))

    assert outcome.terminal is not None
    assert outcome.terminal.needs_confirm is True
    assert outcome.terminal.pending_tool == "shell"
    assert shell.calls == []                      # gated action never executed
    assert outcome.iterations[-1]["gated"] == "needs_confirm"


def test_low_autonomy_confirms_redecided_nonsafe(brain, monkeypatch):
    # autonomy=low → confirm any action the router flags requires_confirm.
    _cfg(monkeypatch, max_steps=2, autonomy="low")
    first = _FakeTool([ToolResult(success=False, output="", error="boom")])
    beta = _FakeTool([ToolResult(success=True, output="done")])       # must NOT run
    monkeypatch.setattr(brain_mod, "registry",
                        _FakeRegistry({"alpha": first, "beta": beta}))
    monkeypatch.setattr(brain_mod, "cot_route", AsyncMock(return_value=(
        {"route": "tool", "tool_name": "beta", "tool_args": {},
         "risk_level": "moderate", "requires_confirm": True}, None)))

    outcome = _run(_call(brain))

    assert outcome.terminal is not None
    assert outcome.terminal.needs_confirm is True
    assert beta.calls == []


# ── per-iteration trace conforms to the #500 contract ────────────────────────

def test_iteration_trace_matches_contract(brain, monkeypatch):
    _cfg(monkeypatch, max_steps=2)
    first = _FakeTool([ToolResult(success=False, output="", error="boom")])
    second = _FakeTool([ToolResult(success=True, output="recovered")])
    monkeypatch.setattr(brain_mod, "registry",
                        _FakeRegistry({"alpha": first, "beta": second}))
    monkeypatch.setattr(brain_mod, "cot_route", AsyncMock(return_value=(
        {"route": "tool", "tool_name": "beta", "tool_args": {},
         "risk_level": "safe"}, None)))

    iters = _run(_call(brain)).iterations
    record = {
        "schema_version": loop_schema.SCHEMA_VERSION,
        "batch_id": "b_test", "task_id": "shell_t.transient",
        "base_id": "shell_t", "category": "shell",
        "failure_class": "transient_error", "recoverable": True,
        "condition": {"model": "m", "tool_loop_max_steps": 2,
                      "provider": "ollama"},
        "success": True, "selection_correct_first": False,
        "loop_triggered": True, "iterations_used": len(iters),
        "recovery": True, "outcome_class": "completed",
        "iterations": iters,
        "cost": {"llm_calls": 2, "tokens_in": 0, "tokens_out": 0,
                 "wall_ms": 0, "loop_overhead_tokens": 0, "finalize_tokens": 0},
        "transcript_path": "x", "ts": "2026-07-03T00:00:00Z",
    }
    assert loop_schema.validate_result(record) == []
