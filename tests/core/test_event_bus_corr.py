"""Tests for corr_ids, bus.wait_for, and job/workflow events (#549)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from bantz.core.event_bus import EventBus, new_corr_id


@pytest.fixture
def fresh_bus():
    b = EventBus()
    yield b
    b.reset()


def test_new_corr_id_shape_and_uniqueness():
    ids = {new_corr_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 12 for i in ids)


async def test_wait_for_resolves_on_predicate_match(fresh_bus):
    fresh_bus.bind_loop()
    cid = new_corr_id()

    async def emit_later():
        await asyncio.sleep(0.01)
        await fresh_bus.emit("delegation_done", corr_id="other")
        await fresh_bus.emit("delegation_done", corr_id=cid)

    task = asyncio.create_task(emit_later())
    ev = await fresh_bus.wait_for(
        "delegation_done",
        predicate=lambda e: e.data.get("corr_id") == cid,
        timeout=2.0,
    )
    await task
    assert ev is not None and ev.data["corr_id"] == cid
    assert fresh_bus.subscriber_count("delegation_done") == 0, "waiter leaked"


async def test_wait_for_timeout_cleans_subscriber(fresh_bus):
    fresh_bus.bind_loop()
    ev = await fresh_bus.wait_for("never_emitted", timeout=0.05)
    assert ev is None
    assert fresh_bus.subscriber_count("never_emitted") == 0


async def test_delegation_events_share_corr_id():
    from bantz.agent.agent_manager import AgentManager
    from bantz.agent.sub_agent import SubAgentResult
    from bantz.core.event_bus import bus

    events: list = []
    bus.reset()
    bus.bind_loop()
    bus.on("delegation_start", lambda e: events.append(e))
    bus.on("delegation_done", lambda e: events.append(e))

    stub = MagicMock()
    stub.display_name = "Stub"

    async def fake_run(task, context=None):
        return SubAgentResult(success=True, summary="done summary")

    stub.run = fake_run

    mgr = AgentManager()
    mgr._enabled = True
    try:
        with patch("bantz.agent.agent_manager.create_agent", return_value=stub), \
             patch("bantz.agent.agent_manager.resolve_role", return_value="researcher"):
            await mgr.delegate("researcher", "test task")
        # Let the dispatcher drain the queue.
        for _ in range(10):
            await asyncio.sleep(0.01)
            if len(events) >= 2:
                break
    finally:
        bus.reset()

    assert len(events) == 2
    start, done = events
    assert start.name == "delegation_start" and done.name == "delegation_done"
    assert start.data["corr_id"] == done.data["corr_id"]
    assert done.data["summary"] == "done summary"


def test_job_error_emits_job_failed():
    from bantz.agent.job_scheduler import JobScheduler

    js = JobScheduler()
    fake_event = MagicMock()
    fake_event.job_id = "test_job"
    fake_event.exception = RuntimeError("boom")

    with patch("bantz.core.event_bus.bus") as mock_bus:
        js._on_job_error(fake_event)

    mock_bus.emit_threadsafe.assert_called_once()
    args, kwargs = mock_bus.emit_threadsafe.call_args
    assert args[0] == "job_failed"
    assert kwargs["job_id"] == "test_job"
    assert "boom" in kwargs["exception"]


async def test_workflow_run_emits_done_event():
    from bantz.workflows.models import WorkflowDef, StepDef
    from bantz.workflows.runner import WorkflowRunner
    from bantz.core.event_bus import bus

    events: list = []
    bus.reset()
    bus.bind_loop()
    bus.on("workflow_done", lambda e: events.append(e))

    wf = WorkflowDef(
        name="test-wf",
        description="tiny",
        steps=[StepDef(name="s1", action="set_variable",
                       variable="x", params={"value": "1"})],
    )
    try:
        result = await WorkflowRunner().run(wf)
        for _ in range(10):
            await asyncio.sleep(0.01)
            if events:
                break
    finally:
        bus.reset()

    assert result.success
    assert len(events) == 1
    assert events[0].data["workflow"] == "test-wf"
    assert "corr_id" in events[0].data
