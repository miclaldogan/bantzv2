"""Pre-flight gate — rehearse every batch failure mode cheaply (issue #512).

A full batch is a GPU-night; losing one to a broken checkpoint or a leaky
sandbox costs a calendar day. This script runs five checks in a few minutes
and writes ``preflight_report.md``. Full batches REQUIRE a green pre-flight
on the current git SHA (RUNBOOK rule).

    python eval/loop_eval/preflight.py --mock-llm        # dev rehearsal
    python eval/loop_eval/preflight.py                   # real conditions
    python eval/loop_eval/preflight.py --allow-no-loop   # baseline-only gate

Checks:
1. mini-batch: 3 tasks × conditions steps={1,3}, including one
   transient-failure task (synthesized from the corpus, so this gate does
   not depend on the #508 variant files being present)
2. hard-kill mid-batch → resume → exact-remainder completion, no duplicates
3. zero writes under ~/.mempalace and ~/.local/share/bantz throughout
4. the steps=3 condition shows a REAL recovery on the transient task
   (the C1 loop signal exists end-to-end; #503). Until C1 lands this check
   fails honestly — pass ``--allow-no-loop`` to gate a baseline-only batch
5. the per-task watchdog fires on a hung task and the batch continues
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORT_PATH = HERE / "preflight_report.md"

# Small, fast, category-diverse trio (2 pilots + transient donor).
BASE_TASKS = ["gmail_send_01.base", "fs_write_01.base"]
TRANSIENT_DONOR = "weather_city_01.base"


def _load_task(task_id: str) -> dict:
    for path in sorted((HERE / "tasks").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t["id"] == task_id:
                    return t
    raise SystemExit(f"preflight: task {task_id!r} not found in corpus")


def _make_transient(base: dict) -> dict:
    v = copy.deepcopy(base)
    v["id"] = f"{base['base_id']}.transient"
    v["failure_injection"] = {
        "class": "transient_error",
        "params": {"fail_times": 1,
                   "error": "transient service error (503) — retry may succeed"},
    }
    v["recoverable"] = True
    return v


def _runner(*args: str, extra_env: dict | None = None, timeout: int = 900,
            wait: bool = True):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env.update(extra_env or {})
    cmd = [sys.executable, str(HERE / "runner.py"), *args]
    if wait:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=str(REPO))
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env,
                            cwd=str(REPO))


def _records(out_dir: Path, batch_id: str) -> list[dict]:
    records = []
    for f in (out_dir / "results" / batch_id).glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--allow-no-loop", action="store_true",
                        help="waive check 4 for a baseline-only (steps=1) "
                             "batch — the report says so loudly")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--timeout", type=float, default=120,
                        help="per-task watchdog for the mini-batches")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(HERE))
    import sandbox
    import schema

    t_start = time.perf_counter()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True).stdout.strip()

    work = Path(tempfile.mkdtemp(prefix="bantz_preflight_"))
    tasks_dir = work / "tasks"
    tasks_dir.mkdir()
    trio = [_load_task(BASE_TASKS[0]), _load_task(BASE_TASKS[1]),
            _make_transient(_load_task(TRANSIENT_DONOR))]
    with open(tasks_dir / "preflight.jsonl", "w", encoding="utf-8",
              newline="\n") as f:
        for t in trio:
            errs = schema.validate_task(t)
            if errs:
                raise SystemExit(f"preflight: invalid task: {errs}")
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    trio_ids = [t["id"] for t in trio]
    transient_id = trio[2]["id"]

    mock = ["--mock-llm"] if args.mock_llm else []
    common_base = [*mock, "--tasks-dir", str(tasks_dir),
                   "--out-dir", str(work),
                   "--lock-path", str(work / ".run.lock")]
    common = [*common_base, "--timeout", str(args.timeout)]

    checks: list[tuple[str, bool, str]] = []
    live_marker = sandbox.live_marker()

    # ── check 1: mini-batch across both conditions ────────────────────────
    proc = _runner("--batch-id", "pf1", "--steps", "1,3", *common)
    recs = _records(work, "pf1")
    schema_errs = [e for r in recs for e in schema.validate_result(r)]
    ok1 = (proc.returncode == 0 and len(recs) == len(trio_ids) * 2
           and not schema_errs)
    checks.append((
        "1. mini-batch: 3 tasks × steps={1,3}, records contract-valid",
        ok1,
        f"{len(recs)} records, rc={proc.returncode}, "
        f"schema errors: {len(schema_errs)}"
        + (f" — {proc.stderr.strip()[-300:]}" if proc.returncode else ""),
    ))

    # ── check 2: hard-kill mid-batch → resume, exact remainder, no dupes ──
    killer = _runner("--batch-id", "pf2", "--steps", "1", *common, wait=False)
    results_dir = work / "results" / "pf2"
    deadline = time.time() + 600
    first_record_seen = False
    while time.time() < deadline:
        files = list(results_dir.glob("*.jsonl")) if results_dir.exists() else []
        if files and files[0].stat().st_size > 0:
            first_record_seen = True
            break
        if killer.poll() is not None:
            break  # finished before we could kill — still fine, resume no-ops
        time.sleep(0.3)
    killer.kill()  # SIGKILL-equivalent: no cleanup, lock left behind
    killer.wait(timeout=60)

    resume = _runner("--batch-id", "pf2", "--steps", "1", "--resume", *common)
    recs2 = _records(work, "pf2")
    ids2 = [r["task_id"] for r in recs2]
    ok2 = (resume.returncode == 0 and sorted(set(ids2)) == sorted(trio_ids)
           and len(ids2) == len(set(ids2)))
    checks.append((
        "2. hard-kill mid-batch → resume: exact remainder, no duplicates",
        ok2,
        f"killed after first record: {first_record_seen}; "
        f"final ids={sorted(ids2)}, resume rc={resume.returncode}"
        + (f" — {resume.stderr.strip()[-300:]}" if resume.returncode else ""),
    ))

    # ── check 3: zero live writes across everything so far ────────────────
    dirty = sandbox.live_writes_since(live_marker)
    checks.append((
        "3. zero writes under ~/.mempalace and ~/.local/share/bantz",
        not dirty,
        "clean" if not dirty else f"DIRTY: {dirty[:5]}",
    ))

    # ── check 4: steps=3 shows a real recovery on the transient task ──────
    steps3 = [r for r in _records(work, "pf1")
              if r["condition"]["tool_loop_max_steps"] == 3
              and r["task_id"] == transient_id]
    recovered = bool(steps3) and steps3[0]["recovery"] is True
    if recovered:
        checks.append((
            "4. steps=3 recovers the transient task (loop signal end-to-end)",
            True, f"recovery=True, iterations={steps3[0]['iterations_used']}"))
    elif args.allow_no_loop:
        checks.append((
            "4. steps=3 recovery — WAIVED (--allow-no-loop)",
            True,
            "C1 loop signal absent; this pre-flight gates a BASELINE-ONLY "
            "batch. Do NOT run a steps>1 batch on this SHA.",
        ))
    else:
        detail = ("no steps=3 record for the transient task" if not steps3
                  else f"recovery={steps3[0]['recovery']}, "
                       f"loop_triggered={steps3[0]['loop_triggered']} — "
                       "C1 loop (#503) not landed or not firing")
        checks.append((
            "4. steps=3 recovers the transient task (loop signal end-to-end)",
            False, detail))

    # ── check 5: watchdog fires on a hung task, batch continues ───────────
    proc5 = _runner("--batch-id", "pf5", "--steps", "1", *common_base,
                    "--timeout", "15",
                    extra_env={"BANTZ_EVAL_TEST_HANG": trio_ids[0]})
    recs5 = {r["task_id"]: r for r in _records(work, "pf5")}
    hung = recs5.get(trio_ids[0], {})
    others = [recs5.get(t) for t in trio_ids[1:]]
    ok5 = (proc5.returncode == 0
           and hung.get("outcome_class") == "timeout"
           and all(o and o["outcome_class"] != "timeout" for o in others))
    checks.append((
        "5. watchdog kills a hung task; the batch continues",
        ok5,
        f"hung task outcome={hung.get('outcome_class')!r}, "
        f"remaining outcomes={[o.get('outcome_class') if o else None for o in others]}",
    ))

    # ── report ─────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    all_green = all(ok for _, ok, _ in checks)
    verdict = "GREEN — full batch may proceed" if all_green else \
              "RED — do NOT start a full batch"
    if all_green and args.allow_no_loop and not recovered:
        verdict = ("GREEN (BASELINE-ONLY) — steps=1 batch may proceed; "
                   "steps>1 batches are NOT cleared")

    lines = [
        "# Pre-flight report (issue #512)",
        "",
        f"- **verdict:** {verdict}",
        f"- **git SHA:** `{git_sha}`",
        f"- **started:** {started}",
        f"- **elapsed:** {elapsed:.0f}s (budget: 900s)",
        f"- **mode:** {'mock-llm' if args.mock_llm else 'REAL conditions'}",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'✅ PASS' if ok else '❌ FAIL'} | "
                     f"{detail.replace('|', '/')} |")
    lines += [
        "",
        "Full batches REQUIRE a green pre-flight on the current git SHA "
        "(RUNBOOK). Re-run: `python eval/loop_eval/preflight.py`.",
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"report written to {args.report}")
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
