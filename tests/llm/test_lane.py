"""Tests for the serialized LLM lane (#547)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from bantz.config import config
from bantz.llm.lane import LLMLane
from bantz.llm.ollama import OllamaClient


class FakeProvider:
    """Records (start, end, kwargs) per chat call."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls: list[tuple[float, float, dict]] = []

    async def chat(self, messages, **kwargs):
        start = time.monotonic()
        await asyncio.sleep(self.delay)
        self.calls.append((start, time.monotonic(), kwargs))
        return "ok"


@pytest.fixture
def lane():
    return LLMLane()


@pytest.fixture
def enabled():
    with patch.object(config, "llm_lane_enabled", True):
        yield


@pytest.fixture
def fake_provider():
    provider = FakeProvider()
    with patch("bantz.llm.router.get_provider", return_value=provider):
        yield provider


async def test_concurrent_calls_serialize(lane, enabled, fake_provider):
    await asyncio.gather(
        lane.call([{"role": "user", "content": "a"}]),
        lane.call([{"role": "user", "content": "b"}]),
        lane.call([{"role": "user", "content": "c"}]),
    )
    intervals = sorted((s, e) for s, e, _ in fake_provider.calls)
    for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
        assert next_start >= prev_end, "lane allowed overlapping LLM calls"


async def test_interactive_priority(lane, enabled, fake_provider):
    order: list[str] = []

    async def run(tag: str, interactive: bool):
        await lane.call([{"role": "user", "content": tag}], interactive=interactive)
        order.append(tag)

    bg1 = asyncio.create_task(run("bg1", False))
    await asyncio.sleep(0.01)  # bg1 holds the lane
    bg2 = asyncio.create_task(run("bg2", False))
    await asyncio.sleep(0.01)  # bg2 queued
    inter = asyncio.create_task(run("interactive", True))
    await asyncio.sleep(0.01)  # interactive queued behind bg2
    await asyncio.gather(bg1, bg2, inter)

    assert order[0] == "bg1"
    assert order[1] == "interactive", f"background jumped the queue: {order}"
    assert order[2] == "bg2"


async def test_reentrant_call_does_not_deadlock(lane, enabled):
    class NestingProvider:
        def __init__(self):
            self.depth = 0

        async def chat(self, messages, **kwargs):
            self.depth += 1
            if self.depth == 1:
                # A call made while the lane is held (workflow self-heal
                # pattern) must run inline, not queue behind itself.
                return await lane.call([{"role": "user", "content": "nested"}])
            return "inner"

    provider = NestingProvider()
    with patch("bantz.llm.router.get_provider", return_value=provider):
        result = await asyncio.wait_for(
            lane.call([{"role": "user", "content": "outer"}]), timeout=2.0
        )
    assert result == "inner"
    assert not lane.busy


async def test_disabled_lane_bypasses_serialization(lane, fake_provider):
    with patch.object(config, "llm_lane_enabled", False):
        await asyncio.gather(
            lane.call([{"role": "user", "content": "a"}]),
            lane.call([{"role": "user", "content": "b"}]),
        )
    (s1, e1, _), (s2, e2, _) = sorted(fake_provider.calls)
    assert s2 < e1, "calls did not overlap — lane serialized despite being disabled"
    assert not lane.busy


class RecordingOllama(OllamaClient):
    """Real OllamaClient (so isinstance passes) with chat stubbed out."""

    def __init__(self):
        super().__init__()
        self.kwargs: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return "ok"


@pytest.fixture
def recording_ollama():
    provider = RecordingOllama()
    with patch("bantz.llm.router.get_provider", return_value=provider):
        yield provider


async def test_keep_alive_policy(lane, enabled, recording_ollama):
    with patch.object(config, "ollama_keep_alive", "30m"), \
         patch.object(config, "ollama_bg_keep_alive", "5m"):
        await lane.call([{"role": "user", "content": "main"}])
        await lane.call([{"role": "user", "content": "bg"}], model="gemma3:4b")
        await lane.call([{"role": "user", "content": "once"}], one_shot=True)
        await lane.call([{"role": "user", "content": "explicit"}], keep_alive="2m")

    main, bg, once, explicit = recording_ollama.kwargs
    assert main["keep_alive"] == "30m" and "model_override" not in main
    assert bg["keep_alive"] == "5m" and bg["model_override"] == "gemma3:4b"
    assert once["keep_alive"] == 0
    assert explicit["keep_alive"] == "2m"


async def test_non_ollama_provider_gets_no_ollama_kwargs(lane, enabled, fake_provider):
    await lane.call([{"role": "user", "content": "x"}], model="gemma3:4b", one_shot=True)
    (_, _, kwargs), = fake_provider.calls
    assert kwargs == {}, f"Ollama-only knobs leaked to a non-Ollama provider: {kwargs}"


async def test_stats(lane, enabled, recording_ollama):
    await lane.call([{"role": "user", "content": "a"}])
    await lane.call([{"role": "user", "content": "b"}], model="gemma3:4b")
    s = lane.stats()
    assert s["calls_total"] == 2
    assert s["calls_last_hour"] == 2
    assert s["by_model"]["gemma3:4b"] == 1
    assert s["busy"] is False and s["waiting"] == 0


async def test_ollama_chat_payload_keep_alive():
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "hi"}}

    class FakeHTTP:
        def __init__(self):
            self.payloads = []

        async def post(self, url, json=None, timeout=None):
            self.payloads.append(json)
            return FakeResp()

    client = OllamaClient()
    client._client = FakeHTTP()

    await client.chat([{"role": "user", "content": "x"}], keep_alive="30m")
    await client.chat([{"role": "user", "content": "x"}])

    with_ka, without_ka = client._client.payloads
    assert with_ka["keep_alive"] == "30m"
    assert "keep_alive" not in without_ka
