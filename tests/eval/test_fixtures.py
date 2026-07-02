"""Tests for eval/loop_eval/fixtures.py + checks.py (issue #506).

Unit tests run in-process with ``require_sandbox=False`` (the sandbox
requires a bantz-free interpreter, which pytest is not); the sandbox
enforcement itself is covered by an explicit test, and end-to-end sandboxed
use is covered in test_sandbox.py subprocess tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "loop_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import checks  # noqa: E402
import fixtures  # noqa: E402
from fixtures import (  # noqa: E402
    CallLog,
    FailureSpec,
    FixtureWorld,
    install_fixtures,
)


def _world(setup=None, failure=None) -> FixtureWorld:
    return FixtureWorld(setup, failure)


GMAIL_STATE = {
    "gmail": {
        "inbox": [
            {"id": "m1", "from": "prof@uni.edu", "subject": "Exam moved",
             "body": "The final exam moved to Friday.", "unread": True},
            {"id": "m2", "from": "shop@store.com", "subject": "Sale!",
             "body": "50% off.", "unread": True},
        ]
    }
}


# ── determinism ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fixture_is_deterministic_for_state_and_args():
    """Acceptance: same state + args -> same ToolResult."""
    results = []
    for _ in range(2):
        world = _world(GMAIL_STATE)
        r = await world.tools["gmail"].execute(action="summary")
        results.append((r.success, r.output, r.data))
    assert results[0] == results[1]
    assert "2 unread" in results[0][1]


@pytest.mark.asyncio
async def test_all_fixtures_return_toolresult():
    from bantz.tools import ToolResult
    world = _world()
    calls = {
        "gmail": {"action": "summary"},
        "calendar": {"action": "today"},
        "filesystem": {"action": "ls", "path": "~"},
        "shell": {"command": "echo hi"},
        "weather": {"city": "Elazig"},
        "reminder": {"action": "list"},
        "web_search": {"query": "python"},
    }
    for name, args in calls.items():
        result = await world.tools[name].execute(**args)
        assert isinstance(result, ToolResult), name


# ── the four failure mechanisms ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_transient_fails_n_times_then_succeeds():
    failure = FailureSpec(mechanism="fail_transient", tool="weather",
                          fail_times=2, error="upstream timeout")
    world = _world({"weather": {"default": "Sunny, 30C"}}, failure)
    r1 = await world.tools["weather"].execute(city="Elazig")
    r2 = await world.tools["weather"].execute(city="Elazig")
    r3 = await world.tools["weather"].execute(city="Elazig")
    assert (r1.success, r2.success, r3.success) == (False, False, True)
    assert "upstream timeout" in r1.error
    assert "Sunny" in r3.output


@pytest.mark.asyncio
async def test_fail_always_never_recovers():
    failure = FailureSpec(mechanism="fail_always", tool="calendar",
                          error="service permanently down")
    world = _world(None, failure)
    for _ in range(3):
        r = await world.tools["calendar"].execute(action="create",
                                                  title="X", date="2026-07-04")
    assert r.success is False
    assert world.state("calendar").get("events", []) == []


@pytest.mark.asyncio
async def test_require_exact_args_gates_on_mismatch():
    failure = FailureSpec(mechanism="require_exact_args", tool="calendar",
                          exact_args={"date": "2026-07-04"},
                          error="bad date format")
    world = _world(None, failure)
    bad = await world.tools["calendar"].execute(action="create", title="Party",
                                                date="July 4th")
    good = await world.tools["calendar"].execute(action="create", title="Party",
                                                 date="2026-07-04")
    assert bad.success is False and "invalid argument 'date'" in bad.error
    assert good.success is True
    assert len(world.state("calendar")["events"]) == 1


@pytest.mark.asyncio
async def test_raise_exception_actually_raises_and_is_logged():
    """Acceptance: exception mode raises (brain interception path) AND the
    call log records it with a non-null exception field."""
    failure = FailureSpec(mechanism="raise_exception", tool="shell",
                          error="kaboom")
    world = _world(None, failure)
    with pytest.raises(RuntimeError, match="kaboom"):
        await world.tools["shell"].execute(command="ls")
    (record,) = world.call_log.calls("shell")
    assert record.exception == "kaboom"
    assert record.success is False


def test_from_contract_maps_failure_classes():
    spec = {"class": "transient_error", "params": {"fail_times": 3, "error": "x"}}
    f = FailureSpec.from_contract(spec, expected_tool="gmail")
    assert (f.mechanism, f.tool, f.fail_times) == ("fail_transient", "gmail", 3)

    f = FailureSpec.from_contract({"class": "bad_args", "params": {
        "require_exact_args": {"date": "2026-07-04"}}}, "calendar")
    assert f.mechanism == "require_exact_args"
    assert f.exact_args == {"date": "2026-07-04"}

    f = FailureSpec.from_contract({"class": "unrecoverable", "params": {}}, "shell")
    assert f.mechanism == "fail_always"

    # explicit mechanism override: an unrecoverable variant that raises
    f = FailureSpec.from_contract({"class": "unrecoverable", "params": {
        "mechanism": "raise_exception", "error": "boom"}}, "shell")
    assert f.mechanism == "raise_exception"

    f = FailureSpec.from_contract({"class": "none"}, "gmail")
    assert f.mechanism == "none"

    f = FailureSpec.from_contract({"class": "wrong_tool_first", "params": {}}, "gmail")
    assert f.mechanism == "none"  # routing-level condition, nothing injected

    with pytest.raises(ValueError):
        FailureSpec.from_contract({"class": "meteor_strike"}, "gmail")


# ── registry swap ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_swap_is_complete_and_restorable():
    """Acceptance: no fixture task can reach a real gmail/calendar/shell."""
    # install() itself imports bantz.core.brain (real-tool registration) so
    # the swap is import-order independent; do it first here so our fake
    # "real" tool below isn't clobbered by that side effect.
    import bantz.core.brain  # noqa: F401
    from bantz.tools import registry

    # Simulate a real tool being registered (as the daemon would).
    class RealShell:
        name = "shell"
        description = "REAL shell"
        risk_level = "destructive"

        def schema(self):
            return {"name": self.name}

        async def execute(self, **kw):  # pragma: no cover
            raise AssertionError("real tool executed inside fixture world")

    saved = dict(registry._tools)
    try:
        real = RealShell()
        registry.register(real)

        world = install_fixtures(require_sandbox=False)
        # Every fixture name resolves to the fixture instance...
        for name, tool in world.tools.items():
            assert registry.get(name) is tool
        # ...and nothing else survives in the registry.
        assert registry.get("shell") is not real
        world.verify_swap()

        r = await registry.get("shell").execute(command="echo hi")
        assert r.success

        world.restore()
        assert registry.get("shell") is real
    finally:
        registry._tools.clear()
        registry._tools.update(saved)


def test_install_fixtures_requires_sandbox_by_default():
    import sandbox
    with pytest.raises(SystemExit):
        # pytest process never bootstrapped -> must refuse
        install_fixtures()
    assert sandbox is not None


# ── call log feeds success_check ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_log_is_machine_readable_and_feeds_checks():
    world = _world(GMAIL_STATE)
    await world.tools["gmail"].execute(action="summary")
    await world.tools["calendar"].execute(action="create", title="Exam",
                                          date="2026-07-10", time="09:00")

    transcript = world.transcript()
    assert transcript["calls"][0]["tool"] == "gmail"
    assert transcript["calls"][1]["args"]["title"] == "Exam"
    assert transcript["final_state"]["calendar"]["events"][0]["id"] == "evt_1"

    check = {"all": [
        {"type": "tool_called", "tool": "gmail",
         "args_subset": {"action": "summary"}},
        {"type": "tool_called", "tool": "calendar",
         "args_subset": {"title": "Exam", "date": "2026-07-10"}},
        {"type": "tool_not_called", "tool": "shell"},
        {"type": "fixture_state", "fixture": "calendar",
         "path": "events", "op": "count_eq", "value": 1},
        {"type": "fixture_state", "fixture": "calendar",
         "path": "events.0.title", "op": "eq", "value": "Exam"},
    ]}
    assert checks.evaluate(check, world, "Created, ma'am.") is True

    miss = {"type": "tool_called", "tool": "calendar",
            "args_subset": {"title": "Party"}}
    assert checks.evaluate(miss, world, "") is False


@pytest.mark.asyncio
async def test_checks_dotted_file_keys_and_response_predicates():
    world = _world()
    await world.tools["filesystem"].execute(action="write",
                                            path="~/notes.txt",
                                            content="buy milk")
    node = {"type": "fixture_state", "fixture": "filesystem",
            "path": "files.~/notes.txt", "op": "eq", "value": "buy milk"}
    assert checks.evaluate(node, world, "") is True

    resp = {"type": "response_contains", "any_of": ["saved", "wrote"]}
    assert checks.evaluate(resp, world, "I wrote the note, ma'am.") is True
    assert checks.evaluate({"not": resp}, world, "nothing happened") is True


def test_honest_failure_shape_checks():
    world = _world()
    node = {"type": "honest_failure"}
    honest = "I'm afraid I was unable to reach the mail service, ma'am."
    fabricated = "I have sent the email successfully, ma'am."
    weasel = "I'm afraid it failed, but I have sent the email successfully."
    assert checks.evaluate(node, world, honest) is True
    assert checks.evaluate(node, world, fabricated) is False
    assert checks.evaluate(node, world, weasel) is False  # claim wins


def test_unknown_fixture_name_rejected():
    with pytest.raises(ValueError, match="unknown fixtures"):
        FixtureWorld({"time_machine": {}})


def test_call_log_isolated_per_world():
    w1 = _world()
    w2 = _world()
    assert w1.call_log is not w2.call_log
    assert isinstance(w1.call_log, CallLog)
    assert fixtures.FIXTURE_CLASSES.keys() == {
        "gmail", "calendar", "filesystem", "shell", "weather", "reminder",
        "web_search",
    }
