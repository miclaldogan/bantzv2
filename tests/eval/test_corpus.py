"""Tests for the loop_eval task corpus (issue #507).

Acceptance criteria covered:
- every task validates against schema.py, ids unique, ≥5 categories, ~40 tasks
- every success_check EXECUTES (no judge model) — and is solvable: the
  task's reference_call against a fresh FixtureWorld makes it pass, while an
  untouched world makes it fail (non-trivial)
- 10-task pilot runs end-to-end single-shot in the sandbox (subprocess),
  through the real brain.process pipeline with routing pinned
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"
TASKS_DIR = EVAL_DIR / "tasks"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import checks  # noqa: E402
import schema  # noqa: E402
from fixtures import FixtureWorld  # noqa: E402


def load_corpus() -> list[dict]:
    tasks = []
    for path in sorted(TASKS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


CORPUS = load_corpus()


# ── shape ─────────────────────────────────────────────────────────────────────


def test_corpus_size_and_categories():
    assert len(CORPUS) >= 40, f"corpus has only {len(CORPUS)} tasks"
    categories = {t["category"] for t in CORPUS}
    assert len(categories) >= 5, f"only {len(categories)} categories"
    pilots = [t for t in CORPUS if "pilot" in t.get("tags", [])]
    assert len(pilots) >= 10, f"only {len(pilots)} pilot tasks"


def test_every_task_validates_against_contract():
    errors = []
    for t in CORPUS:
        errors.extend(schema.validate_task(t))
    assert errors == []


def test_task_ids_unique_and_base_variant_convention():
    ids = [t["id"] for t in CORPUS]
    assert len(ids) == len(set(ids))
    for t in CORPUS:
        assert t["id"] == f"{t['base_id']}.base"
        assert t["failure_injection"]["class"] == "none"
        assert t["recoverable"] is None


def test_expected_tool_is_a_known_fixture():
    from fixtures import FIXTURE_CLASSES
    for t in CORPUS:
        assert t["expected_tool"] in FIXTURE_CLASSES, t["id"]
        assert t["reference_call"]["tool"] == t["expected_tool"], t["id"]


# ── solvability: reference_call -> success_check passes ──────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("task", CORPUS, ids=lambda t: t["id"])
async def test_task_is_solvable_and_nontrivial(task):
    """The golden call must satisfy success_check; an untouched world must
    not — so every predicate both EXECUTES and discriminates."""
    # Non-trivial: nothing happened -> check must fail.
    empty_world = FixtureWorld(task["fixture_setup"])
    assert checks.evaluate(task["success_check"], empty_world, "") is False, (
        f"{task['id']}: success_check passes on an untouched world"
    )

    # Solvable: execute the reference call, use its output as the response
    # (fixture outputs carry the data tokens a faithful answer must contain).
    world = FixtureWorld(task["fixture_setup"])
    call = task["reference_call"]
    result = await world.tools[call["tool"]].execute(**call["args"])
    assert result.success, f"{task['id']}: reference call failed: {result.error}"
    verdict = checks.evaluate(task["success_check"], world, result.output)
    assert verdict is True, f"{task['id']}: success_check rejects the solution"


# ── pilot: end-to-end single-shot in the sandbox ─────────────────────────────


def test_pilot_tasks_run_end_to_end_in_sandbox():
    """10 pilot tasks through the REAL brain.process inside a bootstrapped
    sandbox (subprocess), with routing pinned to the reference call and an
    echo LLM (returns its prompt) so fixture data tokens flow to the final
    response deterministically."""
    pilots = [t for t in CORPUS if "pilot" in t.get("tags", [])][:10]
    assert len(pilots) == 10

    child = textwrap.dedent("""
        import asyncio, json, sys
        from unittest.mock import patch
        import sandbox

        sandbox.bootstrap()

        import checks
        from fixtures import FailureSpec, FixtureWorld, install_fixtures
        from bantz.core.brain import brain

        tasks = json.loads(sys.stdin.read())

        class EchoProvider:
            # Returns every message's content concatenated — whatever data
            # the pipeline grounded the LLM on appears in the "response".
            async def chat(self, messages, **kw):
                return " ".join(m.get("content", "") for m in messages)[-4000:]
            async def chat_stream(self, messages, **kw):
                yield " ".join(m.get("content", "") for m in messages)[-4000:]

        async def run_task(t):
            world = install_fixtures(t["fixture_setup"],
                                     t["failure_injection"],
                                     t["expected_tool"])
            call = t["reference_call"]
            tc = ({"route": "tool", "tool_name": call["tool"],
                   "tool_args": call["args"], "risk_level": "safe",
                   "confidence": 0.95, "reasoning": "pilot pinned route"},
                  None)
            key = sandbox.new_task(t["id"])
            with patch("bantz.core.brain.cot_route", return_value=tc), \\
                 patch("bantz.llm.router.get_provider",
                       return_value=EchoProvider()):
                result = await brain.process(t["prompt"], session_key=key)
                text = result.response or ""
                if result.stream is not None:
                    async for tok in result.stream:
                        text += tok
            called = [r.tool for r in world.call_log.calls()]
            verdict = checks.evaluate(t["success_check"], world, text)
            world.restore()
            return {"id": t["id"], "ok": bool(verdict), "called": called,
                    "response_head": text[:120]}

        async def main():
            out = []
            for t in tasks:
                out.append(await run_task(t))
            print("PILOT_JSON:" + json.dumps(out))

        asyncio.run(main())
    """)

    env = os.environ.copy()
    for key in ("BANTZ_DATA_DIR", "BANTZ_PALACE_PATH",
                "BANTZ_MEMPALACE_KG_PATH", "BANTZ_MEMPALACE_IDENTITY_PATH"):
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EVAL_DIR), str(REPO / "src"), env.get("PYTHONPATH", "")])
    env["PYTHONUTF8"] = "1"
    env["BANTZ_LANGUAGE"] = "en"
    env["BANTZ_MEMPALACE_ENABLED"] = "false"
    env["BANTZ_VOICE_ENABLED"] = "false"
    env["BANTZ_OBSERVER_ENABLED"] = "false"
    env["BANTZ_RL_ENABLED"] = "false"

    proc = subprocess.run(
        [sys.executable, "-c", child],
        input=json.dumps(pilots),
        capture_output=True, text=True, timeout=300, env=env, cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines()
                if ln.startswith("PILOT_JSON:"))
    results = json.loads(line[len("PILOT_JSON:"):])
    failures = [r for r in results if not r["ok"]]
    assert failures == [], f"pilot failures: {failures}"
