"""Eval runner — condition matrix, subprocess-per-task, watchdog,
checkpoint/resume (issues #510 + #511).

Executes the task corpus across conditions and writes result records per the
frozen contract (#500, ``schema.py``). One OS process per task is the
strongest defense against C2b and singleton state bleed: the Brain is a
process-global singleton, so the subprocess dies and takes every bit of
per-task state with it.

Batch:

    python eval/loop_eval/runner.py --batch-id b1 --steps 1 --limit 5
    python eval/loop_eval/runner.py --batch-id b1 --resume            # continue
    python eval/loop_eval/runner.py --batch-id b1 --resume --retry-errors

Layout (under eval/loop_eval/):

    manifests/{batch_id}.json               reproducibility manifest
    results/{batch_id}/{condition}.jsonl    append-only, fsynced per task
    results/{batch_id}/transcripts/*.json   full per-task transcripts

Key properties:
- **Watchdog**: per-task hard timeout kills the subprocess; the record says
  ``outcome_class=timeout`` and the batch moves on — an Ollama stall costs
  one task, not the night.
- **Checkpoint/resume**: every record is fsynced as it lands; ``--resume``
  dedupes on ``(task_id, condition)``; ``--retry-errors`` additionally
  re-runs ``timeout``/``runner_crash`` records.
- **Conditions via env only**: the child gets ``BANTZ_OLLAMA_MODEL`` and
  ``BANTZ_TOOL_LOOP_MAX_STEPS`` (consumed by the C1 loop, #501) — no code
  edits between conditions.
- **Run queue**: takes the #509 lock for the whole batch.
- **--mock-llm** pins routing to each task's ``reference_call`` and stubs
  the LLM with an echo provider — the full runner machinery (subprocess,
  watchdog, resume, manifest, schema validation) exercised with zero Ollama
  load. Real batches omit it.

The child re-verifies the #505 sandbox guarantee: any write under the live
roots turns the record into ``outcome_class=sandbox_violation``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
RESULTS_DIR = HERE / "results"
MANIFESTS_DIR = HERE / "manifests"
DEFAULT_TIMEOUT_S = 180
SMOKE_CAP = 5  # RUNBOOK rule: anything bigger than this is a batch

_STRIP_ENV = ("BANTZ_DATA_DIR", "BANTZ_PALACE_PATH",
              "BANTZ_MEMPALACE_KG_PATH", "BANTZ_MEMPALACE_IDENTITY_PATH")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def condition_slug(cond: dict) -> str:
    model = "".join(c if c.isalnum() else "-" for c in cond["model"])
    slug = f"{model}_steps{cond['tool_loop_max_steps']}"
    # Default mode keeps historical slugs stable so --resume still dedupes
    # against pre-ablation batches.
    if cond.get("tool_loop_mode", "redecide") != "redecide":
        slug += f"_{cond['tool_loop_mode']}"
    return slug


# ── task loading ─────────────────────────────────────────────────────────────

def load_tasks(tasks_dir: Path, include_variants: bool = True,
               tags: list[str] | None = None,
               only: list[str] | None = None) -> list[dict]:
    tasks = []
    for path in sorted(tasks_dir.glob("*.jsonl")):
        if not include_variants and ".variants" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(json.loads(line))
    if tags:
        tasks = [t for t in tasks if set(tags) & set(t.get("tags", []))]
    if only:
        tasks = [t for t in tasks if t["id"] in set(only)]
    return sorted(tasks, key=lambda t: t["id"])


# ── manifest (#511: reproduce the batch bit-for-bit) ─────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(HERE),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _model_digest(model: str) -> str:
    try:
        proc = subprocess.run(["ollama", "show", model, "--modelfile"],
                              capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            return hashlib.sha256(proc.stdout.encode()).hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


def write_manifest(batch_id: str, conditions: list[dict],
                   tasks: list[dict], mock: bool,
                   manifests_dir: Path = MANIFESTS_DIR) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    task_blob = "\n".join(json.dumps(t, sort_keys=True) for t in tasks)
    manifest = {
        "batch_id": batch_id,
        "git_sha": _git_sha(),
        "mock_llm": mock,
        "conditions": conditions,
        "model_digests": {c["model"]: ("mock" if mock else
                                       _model_digest(c["model"]))
                          for c in conditions},
        "task_ids": [t["id"] for t in tasks],
        "task_file_sha256": hashlib.sha256(task_blob.encode()).hexdigest(),
        "env_snapshot": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith("BANTZ_")},
        "started_ts": _now_iso(),
        "finished_ts": None,
    }
    path = manifests_dir / f"{batch_id}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def finish_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["finished_ts"] = _now_iso()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")


# ── resume bookkeeping (#511) ────────────────────────────────────────────────

def load_done(results_file: Path, retry_errors: bool) -> set[str]:
    """task_ids already recorded for this condition (dedupe key half —
    the file IS the condition). With retry_errors, timeout/runner_crash
    records don't count as done."""
    done: set[str] = set()
    if not results_file.exists():
        return done
    for line in results_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn write from a kill -9 — that task re-runs
        if retry_errors and rec.get("outcome_class") in ("timeout",
                                                         "runner_crash"):
            continue
        done.add(rec["task_id"])
    return done


def append_record(results_file: Path, record: dict) -> None:
    with open(results_file, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── synthesized records for tasks the child never reported ──────────────────

def _skeleton_record(task: dict, cond: dict, batch_id: str,
                     outcome: str, error_detail: str,
                     transcript_path: str, wall_ms: int) -> dict:
    return {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "task_id": task["id"],
        "base_id": task["base_id"],
        "category": task["category"],
        "failure_class": task["failure_injection"]["class"],
        "recoverable": task["recoverable"],
        "condition": cond,
        "success": False,
        "selection_correct_first": False,
        "loop_triggered": False,
        "iterations_used": 1,
        "recovery": False,
        "outcome_class": outcome,
        "iterations": [{
            "index": 1, "route": "tool",
            "tool_name": task["expected_tool"], "tool_args": {},
            "decision_source": "initial",
            "result": {"success": False, "error": error_detail,
                       "output_excerpt": ""},
            "exception": None, "gated": None,
            "tokens_in": 0, "tokens_out": 0, "wall_ms": wall_ms,
        }],
        "cost": {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0,
                 "wall_ms": wall_ms, "loop_overhead_tokens": 0,
                 "finalize_tokens": 0},
        "transcript_path": transcript_path,
        "ts": _now_iso(),
        "error_detail": error_detail[:500],
    }


# ── child process (one task, fully sandboxed) ────────────────────────────────

CHILD_RESULT_PREFIX = "RESULT_JSON:"


def run_child() -> int:
    payload = json.loads(sys.stdin.read())
    task, cond = payload["task"], payload["condition"]
    transcript_path = Path(payload["transcript_path"])
    mock = payload.get("mock", False)

    if os.environ.get("BANTZ_EVAL_TEST_HANG") == task["id"]:
        time.sleep(3600)  # watchdog test hook — hang exactly this task

    import sandbox
    marker = sandbox.live_marker()
    sandbox.bootstrap()

    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    import checks
    from fixtures import install_fixtures

    world = install_fixtures(task["fixture_setup"],
                             task["failure_injection"],
                             task["expected_tool"])
    from bantz.core.brain import brain

    t0 = time.perf_counter()

    async def drive() -> tuple[str, bool]:
        with ExitStack() as stack:
            if mock:
                call = task["reference_call"]
                tc = ({"route": "tool", "tool_name": call["tool"],
                       "tool_args": call["args"], "risk_level": "safe",
                       "confidence": 0.95, "reasoning": "mock pinned route"},
                      None)
                stack.enter_context(
                    patch("bantz.core.brain.cot_route", return_value=tc))

                class EchoProvider:
                    async def chat(self, messages, **kw):
                        return " ".join(m.get("content", "")
                                        for m in messages)[-4000:]

                    async def chat_stream(self, messages, **kw):
                        yield " ".join(m.get("content", "")
                                       for m in messages)[-4000:]

                stack.enter_context(
                    patch("bantz.llm.router.get_provider",
                          return_value=EchoProvider()))

            key = sandbox.new_task(task["id"])
            result = await brain.process(task["prompt"], session_key=key)
            text = result.response or ""
            if result.stream is not None:
                async for token in result.stream:
                    text += token
            # The C1 loop's per-iteration trace (#503) is authoritative for
            # what the loop actually did; carry it out for the record.
            trace = [dict(it) for it in (getattr(result, "iterations", None) or [])]
            return text, bool(result.needs_confirm), trace

    response, needs_confirm, brain_iters = asyncio.run(drive())
    wall_ms = int((time.perf_counter() - t0) * 1000)

    live_writes = sandbox.live_writes_since(marker)
    success = checks.evaluate(task["success_check"], world, response)

    # outcome (frozen decision #1: the runner never auto-confirms)
    if needs_confirm:
        outcome = "gated_confirm"
    elif live_writes:
        outcome = "sandbox_violation"
    elif task["recoverable"] is False and success:
        outcome = "honest_giveup"
    else:
        outcome = "completed"

    calls = world.call_log.calls()
    # Prefer the authoritative C1 loop trace from BrainResult (#503): it
    # directly encodes each iteration's decision + result, so loop_triggered /
    # recovery reflect what the loop actually did. Fall back to the fixture
    # call log for no-tool (chat) paths that never enter the loop.
    if brain_iters:
        iterations = brain_iters
    else:
        iterations = []
        for i, rec in enumerate(calls):
            iterations.append({
                "index": i + 1,
                "route": "tool",
                "tool_name": rec.tool,
                "tool_args": rec.args,
                "decision_source": "initial" if i == 0 else "llm",
                "result": {"success": rec.success, "error": rec.error,
                           "output_excerpt": rec.output_excerpt},
                "exception": rec.exception,
                "gated": ("needs_confirm"
                          if needs_confirm and i == len(calls) - 1 else None),
                "tokens_in": 0, "tokens_out": 0, "wall_ms": 0,
            })
        if not iterations:  # routed to chat / never reached a tool
            iterations.append({
                "index": 1, "route": "chat", "tool_name": "", "tool_args": {},
                "decision_source": "initial",
                "result": {"success": False, "error": None,
                           "output_excerpt": response[:500]},
                "exception": None,
                "gated": "needs_confirm" if needs_confirm else None,
                "tokens_in": 0, "tokens_out": 0, "wall_ms": 0,
            })

    # The loop re-decided iff more than one iteration was recorded (steps=1
    # baseline always has exactly one → loop_triggered stays False).
    loop_triggered = len(iterations) > 1
    record = {
        "schema_version": "1.0",
        "batch_id": payload["batch_id"],
        "task_id": task["id"],
        "base_id": task["base_id"],
        "category": task["category"],
        "failure_class": task["failure_injection"]["class"],
        "recoverable": task["recoverable"],
        "condition": cond,
        "success": bool(success),
        "selection_correct_first": bool(
            calls and calls[0].tool == task["expected_tool"]),
        "loop_triggered": loop_triggered,
        "iterations_used": len(iterations),
        "recovery": bool(success) and loop_triggered,
        "outcome_class": outcome,
        "iterations": iterations,
        "cost": {"llm_calls": 0 if mock else 2, "tokens_in": 0,
                 "tokens_out": 0, "wall_ms": wall_ms,
                 "loop_overhead_tokens": 0, "finalize_tokens": 0},
        "transcript_path": str(transcript_path),
        "ts": _now_iso(),
        "final_response_excerpt": response[:500],
    }
    if live_writes:
        record["error_detail"] = f"live writes: {live_writes[:5]}"[:500]

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps({
        "task": task, "condition": cond, "response": response,
        "needs_confirm": needs_confirm, "world": world.transcript(),
        "live_writes": live_writes, "record": record,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(CHILD_RESULT_PREFIX + json.dumps(record, ensure_ascii=False))
    return 0


# ── parent orchestration ─────────────────────────────────────────────────────

def _child_env(cond: dict, mock: bool) -> dict:
    env = os.environ.copy()
    for key in _STRIP_ENV:  # bootstrap must do its own redirecting
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(HERE.parent.parent / "src"),
         env.get("PYTHONPATH", "")])
    env["PYTHONUTF8"] = "1"
    env["BANTZ_LANGUAGE"] = "en"          # frozen decision #3
    env["BANTZ_OLLAMA_MODEL"] = cond["model"]
    env["BANTZ_LLM_PROVIDER"] = cond["provider"]
    env["BANTZ_TOOL_LOOP_MAX_STEPS"] = str(cond["tool_loop_max_steps"])
    env["BANTZ_TOOL_LOOP_MODE"] = cond.get("tool_loop_mode", "redecide")
    # Disable eval-irrelevant, leak-prone subsystems for EVERY child (mock and
    # real), not just mock. MemPalace writes lock files under the live
    # ~/.mempalace/locks regardless of BANTZ_PALACE_PATH, and voice/observer/RL
    # touch live paths too — any of which trips the sandbox guard (#505) on a
    # real run. None affect the measured routing/recovery on fresh sandboxed
    # tasks (no stored memory to recall); only the model call stays real when
    # mock is off.
    env["BANTZ_MEMPALACE_ENABLED"] = "false"
    env["BANTZ_VOICE_ENABLED"] = "false"
    env["BANTZ_OBSERVER_ENABLED"] = "false"
    env["BANTZ_RL_ENABLED"] = "false"
    return env


def run_task_subprocess(task: dict, cond: dict, batch_id: str,
                        batch_dir: Path, timeout_s: float,
                        mock: bool) -> dict:
    """One task = one OS process. Always returns a contract-valid record."""
    slug = condition_slug(cond)
    transcript_path = batch_dir / "transcripts" / f"{task['id']}__{slug}.json"
    payload = json.dumps({"task": task, "condition": cond,
                          "batch_id": batch_id,
                          "transcript_path": str(transcript_path),
                          "mock": mock})
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / "runner.py"), "--child"],
            input=payload, capture_output=True, text=True,
            timeout=timeout_s, env=_child_env(cond, mock),
            cwd=str(HERE.parent.parent), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        wall_ms = int((time.perf_counter() - t0) * 1000)
        return _skeleton_record(
            task, cond, batch_id, "timeout",
            f"watchdog killed the task after {timeout_s:.0f}s",
            str(transcript_path), wall_ms)

    wall_ms = int((time.perf_counter() - t0) * 1000)
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith(CHILD_RESULT_PREFIX)), None)
    if proc.returncode != 0 or line is None:
        detail = (proc.stderr or proc.stdout or "no output").strip()[-500:]
        return _skeleton_record(task, cond, batch_id, "runner_crash",
                                f"child exit {proc.returncode}: {detail}",
                                str(transcript_path), wall_ms)
    return json.loads(line[len(CHILD_RESULT_PREFIX):])


def run_batch(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import schema
    from runlock import RunLock

    out_root = Path(args.out_dir) if args.out_dir else HERE
    results_root = out_root / "results"
    manifests_root = out_root / "manifests"

    tasks = load_tasks(Path(args.tasks_dir),
                       include_variants=not args.base_only,
                       tags=args.tags, only=args.only)
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 1
    if len(tasks) > SMOKE_CAP and args.smoke:
        print(f"--smoke caps runs at {SMOKE_CAP} tasks (RUNBOOK rule)",
              file=sys.stderr)
        return 1

    conditions = [
        {"model": model, "tool_loop_max_steps": steps,
         "provider": args.provider, "tool_loop_mode": mode}
        for model in args.model for steps in args.steps
        for mode in args.mode
    ]

    batch_dir = results_root / args.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    lock_kwargs = {"batch_id": args.batch_id}
    if args.lock_path:
        lock_kwargs["lock_path"] = Path(args.lock_path)
    with RunLock(**lock_kwargs):
        manifest_path = write_manifest(args.batch_id, conditions, tasks,
                                       args.mock_llm, manifests_root)
        print(f"batch {args.batch_id}: {len(tasks)} task(s) × "
              f"{len(conditions)} condition(s)  manifest={manifest_path}")

        totals: dict[str, int] = {}
        for cond in conditions:
            slug = condition_slug(cond)
            results_file = batch_dir / f"{slug}.jsonl"
            done = (load_done(results_file, args.retry_errors)
                    if args.resume else set())
            pending = [t for t in tasks if t["id"] not in done]
            print(f"[{slug}] done={len(done)} pending={len(pending)}")

            for i, task in enumerate(pending, 1):
                timeout_s = float(task.get("timeout_s", args.timeout))
                record = run_task_subprocess(task, cond, args.batch_id,
                                             batch_dir, timeout_s,
                                             args.mock_llm)
                errors = schema.validate_result(record)
                if errors:
                    # A contract bug must be loud, but one bad record must
                    # not kill the batch: mark it and keep going.
                    record["outcome_class"] = "runner_crash"
                    record["error_detail"] = ("schema violations: "
                                              + "; ".join(errors))[:500]
                append_record(results_file, record)
                outcome = record["outcome_class"]
                totals[outcome] = totals.get(outcome, 0) + 1
                print(f"[{slug}] {i}/{len(pending)} {task['id']}: "
                      f"{outcome} success={record['success']} "
                      f"({record['cost']['wall_ms']}ms)")

        finish_manifest(manifest_path)
        print(f"batch {args.batch_id} finished: {json.dumps(totals)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true",
                        help="internal: run one task from stdin payload")
    parser.add_argument("--batch-id")
    parser.add_argument("--tasks-dir", default=str(TASKS_DIR))
    parser.add_argument("--base-only", action="store_true",
                        help="exclude *.variants.jsonl")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="only tasks carrying one of these tags")
    parser.add_argument("--only", nargs="*", default=None,
                        help="explicit task ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help=f"enforce the {SMOKE_CAP}-task dev cap")
    parser.add_argument("--model", nargs="*",
                        default=[os.environ.get("BANTZ_OLLAMA_MODEL",
                                                "qwen2.5:7b")])
    parser.add_argument("--steps", type=lambda s: [int(x) for x in
                                                   s.split(",")],
                        default=[1],
                        help="comma-separated tool_loop_max_steps, e.g. 1,3")
    parser.add_argument("--mode", type=lambda s: s.split(","),
                        default=["redecide"],
                        help="comma-separated tool_loop_mode values "
                             "(redecide,retry) — retry is the ablation "
                             "baseline: same call re-executed, no re-decide")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--mock-llm", action="store_true",
                        help="pin routing + echo LLM (tests/pre-flight)")
    parser.add_argument("--lock-path", default=None,
                        help="override run-lock location (tests)")
    parser.add_argument("--out-dir", default=None,
                        help="override results/manifests root (tests)")
    args = parser.parse_args(argv)

    if args.child:
        return run_child()
    if not args.batch_id:
        parser.error("--batch-id is required")
    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
