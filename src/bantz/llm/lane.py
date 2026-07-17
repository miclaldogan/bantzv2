"""
Bantz — Serialized LLM lane (#547).

The single choke point for every LLM call made by agents, workflows, the
jury, and background jobs. On a machine that can hold one local model
comfortably, parallel Ollama calls mean memory thrash and heat — so the
lane makes concurrent local inference structurally impossible instead of
relying on caller discipline:

  - ONE call computes at a time (semaphore-of-one via a Condition).
  - Interactive callers (the user waiting on a chat reply) are served
    before any queued background caller.
  - The Ollama keep_alive policy is applied here: the main conversation
    model stays resident (config.ollama_keep_alive, default 30m), agent
    side-models unload quickly (config.ollama_bg_keep_alive, default 5m),
    and one-shot heavy calls (VLM, nightly reflection) evict immediately.
  - Re-entrancy: a call issued while the caller already holds the lane
    (e.g. a workflow self-heal inside a lane-held job) runs inline in the
    held slot instead of deadlocking.

Interactive brain traffic does NOT go through the lane yet — call-site
rollout is #560. Until then the lane serializes agent/background traffic
against itself, which is where the parallelism risk lives.

Usage:
    from bantz.llm.lane import llm_call
    reply = await llm_call(messages, model="gemma3:4b", interactive=False)
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections import deque
from typing import Any

log = logging.getLogger("bantz.llm.lane")

# True while the current asyncio task (or its children via context copy)
# holds the lane — the re-entrancy guard.
_in_lane: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "bantz_llm_in_lane", default=False
)


class LLMLane:
    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._busy = False
        self._interactive_waiting = 0
        self._background_waiting = 0
        self._calls_total = 0
        self._calls_by_model: dict[str, int] = {}
        self._call_times: deque[float] = deque(maxlen=1000)

    # ── acquire / release ────────────────────────────────────────────────

    async def _acquire(self, interactive: bool) -> None:
        async with self._cond:
            if interactive:
                self._interactive_waiting += 1
                try:
                    await self._cond.wait_for(lambda: not self._busy)
                finally:
                    self._interactive_waiting -= 1
            else:
                # Background waiters also yield to any queued interactive
                # caller — this is the whole priority mechanism.
                self._background_waiting += 1
                try:
                    await self._cond.wait_for(
                        lambda: not self._busy and self._interactive_waiting == 0
                    )
                finally:
                    self._background_waiting -= 1
            self._busy = True

    async def _release(self) -> None:
        async with self._cond:
            self._busy = False
            self._cond.notify_all()

    # ── keep_alive policy ────────────────────────────────────────────────

    @staticmethod
    def _resolve_keep_alive(
        model: str, keep_alive: str | int | None, one_shot: bool
    ) -> str | int | None:
        from bantz.config import config
        if one_shot:
            return 0
        if keep_alive is not None:
            return keep_alive
        if not model or model == config.ollama_model:
            return config.ollama_keep_alive
        return config.ollama_bg_keep_alive

    # ── the call ─────────────────────────────────────────────────────────

    async def call(
        self,
        messages: list[dict],
        *,
        model: str = "",
        keep_alive: str | int | None = None,
        interactive: bool = False,
        options: dict | None = None,
        one_shot: bool = False,
    ) -> str:
        from bantz.config import config
        from bantz.llm.router import get_provider

        provider = get_provider()

        # model_override / options / keep_alive are Ollama-only knobs;
        # cloud providers take plain (messages) and ignore the policy.
        kwargs: dict[str, Any] = {}
        from bantz.llm.ollama import OllamaClient
        if isinstance(provider, OllamaClient):
            if model:
                kwargs["model_override"] = model
            if options:
                kwargs["options"] = options
            ka = self._resolve_keep_alive(model, keep_alive, one_shot)
            if ka is not None:
                kwargs["keep_alive"] = ka

        if not config.llm_lane_enabled or _in_lane.get():
            return await provider.chat(messages, **kwargs)

        await self._acquire(interactive)
        token = _in_lane.set(True)
        try:
            return await provider.chat(messages, **kwargs)
        finally:
            _in_lane.reset(token)
            self._record(model or getattr(provider, "model", "") or "default")
            await self._release()

    # ── introspection (consumed by ws_server/UI, #561) ───────────────────

    def _record(self, model: str) -> None:
        self._calls_total += 1
        self._call_times.append(time.time())
        self._calls_by_model[model] = self._calls_by_model.get(model, 0) + 1

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def waiting(self) -> int:
        return self._interactive_waiting + self._background_waiting

    def stats(self) -> dict:
        now = time.time()
        return {
            "busy": self._busy,
            "waiting": self.waiting,
            "waiting_interactive": self._interactive_waiting,
            "calls_total": self._calls_total,
            "calls_last_hour": sum(1 for t in self._call_times if now - t < 3600),
            "by_model": dict(self._calls_by_model),
        }


lane = LLMLane()


async def llm_call(
    messages: list[dict],
    *,
    model: str = "",
    keep_alive: str | int | None = None,
    interactive: bool = False,
    options: dict | None = None,
    one_shot: bool = False,
) -> str:
    """The canonical way for agents, workflows, and background jobs to talk
    to the LLM. See module docstring."""
    return await lane.call(
        messages,
        model=model,
        keep_alive=keep_alive,
        interactive=interactive,
        options=options,
        one_shot=one_shot,
    )
