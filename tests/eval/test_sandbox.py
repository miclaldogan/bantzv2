"""Tests for eval/loop_eval/sandbox.py (issue #505).

The happy-path tests run in SUBPROCESSES on purpose: bootstrap() requires
that no bantz module is imported yet, and the pytest process imports bantz
all over the place. Subprocess-per-task is also the runner's default mode,
so these tests exercise the real deployment shape.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"


def _run_child(code: str, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter with sandbox importable."""
    env = os.environ.copy()
    # A stale redirect from the parent must not leak into the child —
    # bootstrap() has to do the redirecting itself.
    for key in (
        "BANTZ_DATA_DIR",
        "BANTZ_PALACE_PATH",
        "BANTZ_MEMPALACE_KG_PATH",
        "BANTZ_MEMPALACE_IDENTITY_PATH",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EVAL_DIR), str(REPO / "src"), env.get("PYTHONPATH", "")]
    )
    env["PYTHONUTF8"] = "1"
    # Keep the child lean and network-free.
    env["BANTZ_LANGUAGE"] = "en"
    env["BANTZ_MEMPALACE_ENABLED"] = "false"
    env["BANTZ_VOICE_ENABLED"] = "false"
    env["BANTZ_OBSERVER_ENABLED"] = "false"
    env["BANTZ_RL_ENABLED"] = "false"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO),
    )


# ── guard ─────────────────────────────────────────────────────────────────────


def test_guard_trips_if_bantz_imported_first():
    """Acceptance: import bantz first -> bootstrap() exits (SystemExit)."""
    proc = _run_child(
        """
        import bantz.config  # freeze live paths — the mistake
        import sandbox
        try:
            sandbox.bootstrap()
        except SystemExit as e:
            print("TRIPPED:", e)
            raise
        """
    )
    assert proc.returncode != 0
    assert "SANDBOX VIOLATION" in (proc.stdout + proc.stderr)
    assert "bantz was imported before" in (proc.stdout + proc.stderr)


def test_guard_trips_on_env_escape():
    """Paths pointing outside the sandbox root must refuse to run."""
    proc = _run_child(
        """
        import os, tempfile
        import sandbox
        root = tempfile.mkdtemp(prefix="bantz_eval_guard_")
        os.environ["BANTZ_PALACE_PATH"] = os.path.expanduser("~/.mempalace/palace")

        # bootstrap() would overwrite the escape — call the internals the way
        # a buggy runner might: pre-set env, then only guard.
        import pathlib
        try:
            sandbox._assert_sandboxed(pathlib.Path(root))
        except SystemExit as e:
            print("TRIPPED:", e)
            raise
        """
    )
    assert proc.returncode != 0
    assert "SANDBOX VIOLATION" in (proc.stdout + proc.stderr)


# ── happy path ────────────────────────────────────────────────────────────────


def test_bootstrap_redirects_all_paths_and_writes_nothing_live():
    """Acceptance: resolved paths under temp root; ZERO live-root writes."""
    proc = _run_child(
        """
        import json
        import sandbox

        marker = sandbox.live_marker()
        root = sandbox.bootstrap()
        assert sandbox.bootstrap() == root  # idempotent

        from bantz.config import config
        paths = {
            "db": str(config.db_path),
            "palace": config.resolved_palace_path,
            "kg": config.resolved_kg_path,
            "identity": config.resolved_identity_path,
        }

        # Exercise real persistence inside the sandbox.
        key_a = sandbox.new_task("t1")
        key_b = sandbox.new_task("t2")
        assert key_a != key_b
        from bantz.data.layer import data_layer
        data_layer.conversations.add("user", "sandboxed hello")

        dirty = sandbox.live_writes_since(marker)
        print(json.dumps({"root": str(root), "paths": paths, "dirty": dirty}))
        """
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    root = Path(payload["root"]).resolve()
    for name, p in payload["paths"].items():
        assert Path(p).resolve().is_relative_to(root), (name, p)
    assert payload["dirty"] == [], f"live writes detected: {payload['dirty']}"


def test_new_session_isolates_context_between_tasks():
    """Acceptance: task B's prompt context has none of task A's messages.

    Captured at the LLM boundary (mocked provider); also runs a CONTROL
    within one session proving the capture would detect leakage.
    """
    proc = _run_child(
        """
        import asyncio, json
        from unittest.mock import patch
        import sandbox

        sandbox.bootstrap()

        from bantz.core.brain import brain

        class CaptureProvider:
            def __init__(self):
                self.calls = []
            async def chat(self, messages, **kw):
                self.calls.append(messages)
                return "Understood, ma'am."
            async def chat_stream(self, messages, **kw):
                self.calls.append(messages)
                yield "Understood, ma'am."

        provider = CaptureProvider()
        CHAT_TC = (
            {"route": "chat", "tool_name": None, "tool_args": {},
             "risk_level": "safe", "confidence": 0.95, "reasoning": "chat"},
            None,
        )

        async def ask(text, key):
            result = await brain.process(text, session_key=key)
            if result.stream is not None:
                async for _tok in result.stream:
                    pass
            return result

        async def main():
            with patch("bantz.core.brain.cot_route", return_value=CHAT_TC), \\
                 patch("bantz.llm.router.get_provider", return_value=provider):
                # Task A plants a unique marker in its conversation.
                key_a = sandbox.new_task("task_a")
                await ask("My secret code is ZEBRA_ALPHA_77.", key_a)

                # Task B (fresh session) must not see task A's history.
                key_b = sandbox.new_task("task_b")
                n_before = len(provider.calls)
                await ask("What is the CTRL_TOKEN_456 status?", key_b)
                task_b_calls = provider.calls[n_before:]
                b_text = json.dumps(task_b_calls)

                # CONTROL: same session, no reset -> history IS visible,
                # proving the capture detects leakage when it exists.
                n_before = len(provider.calls)
                await ask("And one more question.", key_b)
                control_text = json.dumps(provider.calls[n_before:])

            print(json.dumps({
                "b_sees_a": "ZEBRA_ALPHA_77" in b_text,
                "control_sees_prev": "CTRL_TOKEN_456" in control_text,
            }))

        asyncio.run(main())
        """
    )
    assert proc.returncode == 0, proc.stderr
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["control_sees_prev"] is True, (
        "control failed — capture cannot detect history, test is void"
    )
    assert verdict["b_sees_a"] is False, (
        "C2b contamination: task B's LLM prompt contained task A's marker"
    )


def test_new_task_requires_bootstrap():
    proc = _run_child(
        """
        import sandbox
        try:
            sandbox.new_task("t")
        except SystemExit as e:
            print("TRIPPED:", e)
            raise
        """
    )
    assert proc.returncode != 0
    assert "before bootstrap()" in (proc.stdout + proc.stderr)
