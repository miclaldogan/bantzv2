"""Tests for eval/loop_eval/runner.py (issues #510 + #511).

All runs use --mock-llm (pinned routing + echo provider): the full runner
machinery — subprocess-per-task, watchdog, fsynced results, resume dedupe,
manifest — is exercised deterministically with zero Ollama load.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import schema  # noqa: E402

PILOT_IDS = ["gmail_send_01.base", "calendar_create_01.base",
             "fs_write_01.base", "reminder_add_01.base"]


def run_runner(*args, tmp_path: Path, env_extra: dict | None = None,
               timeout: int = 600) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(EVAL_DIR / "runner.py"),
         "--mock-llm",
         "--lock-path", str(tmp_path / ".run.lock"),
         "--out-dir", str(tmp_path), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
        cwd=str(REPO),
    )


def read_records(tmp_path: Path, batch_id: str) -> list[dict]:
    records = []
    for f in (tmp_path / "results" / batch_id).glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn write — the runner treats it the same way
    return records


# ── #510: batch, schema-valid records, transcripts, manifest ─────────────────


def test_mock_batch_produces_valid_records(tmp_path):
    proc = run_runner("--batch-id", "b1", "--only", *PILOT_IDS,
                      tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]

    records = read_records(tmp_path, "b1")
    assert {r["task_id"] for r in records} == set(PILOT_IDS)
    for rec in records:
        assert schema.validate_result(rec) == [], rec["task_id"]
        assert rec["outcome_class"] == "completed"
        assert rec["success"] is True, rec["task_id"]
        assert rec["selection_correct_first"] is True
        assert Path(rec["transcript_path"]).exists()
        transcript = json.loads(
            Path(rec["transcript_path"]).read_text(encoding="utf-8"))
        assert transcript["world"]["calls"], "transcript missing tool calls"

    manifest = json.loads(
        (tmp_path / "manifests" / "b1.json").read_text(encoding="utf-8"))
    assert manifest["git_sha"]
    assert manifest["task_ids"] == sorted(PILOT_IDS)
    assert manifest["conditions"][0]["tool_loop_max_steps"] == 1
    assert manifest["env_snapshot"] is not None
    assert manifest["finished_ts"], "finished_ts not patched at batch end"


def test_condition_matrix_one_file_per_condition(tmp_path):
    proc = run_runner("--batch-id", "b2", "--only", PILOT_IDS[0],
                      "--steps", "1,3", tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    files = sorted(f.name for f in (tmp_path / "results" / "b2").glob("*.jsonl"))
    assert len(files) == 2
    assert any("steps1" in f for f in files)
    assert any("steps3" in f for f in files)
    # condition travels via env into the record
    for rec in read_records(tmp_path, "b2"):
        assert rec["condition"]["tool_loop_max_steps"] in (1, 3)


# ── #510: watchdog — a hung task costs one record, not the batch ─────────────


def test_watchdog_kills_hung_task_and_batch_continues(tmp_path):
    hang_id, ok_id = PILOT_IDS[0], PILOT_IDS[1]
    proc = run_runner("--batch-id", "b3", "--only", hang_id, ok_id,
                      "--timeout", "15", tmp_path=tmp_path,
                      env_extra={"BANTZ_EVAL_TEST_HANG": hang_id})
    assert proc.returncode == 0, proc.stderr[-2000:]
    by_id = {r["task_id"]: r for r in read_records(tmp_path, "b3")}
    assert by_id[hang_id]["outcome_class"] == "timeout"
    assert "watchdog" in by_id[hang_id]["error_detail"]
    assert schema.validate_result(by_id[hang_id]) == []
    assert by_id[ok_id]["outcome_class"] == "completed"  # batch moved on


# ── #511: checkpoint/resume — no duplicates, exactly the remainder ───────────


def test_resume_skips_completed_and_adds_only_missing(tmp_path):
    first = run_runner("--batch-id", "b4", "--only", *PILOT_IDS[:2],
                       tmp_path=tmp_path)
    assert first.returncode == 0, first.stderr[-2000:]
    assert len(read_records(tmp_path, "b4")) == 2

    second = run_runner("--batch-id", "b4", "--only", *PILOT_IDS,
                        "--resume", tmp_path=tmp_path)
    assert second.returncode == 0, second.stderr[-2000:]
    records = read_records(tmp_path, "b4")
    ids = [r["task_id"] for r in records]
    assert sorted(ids) == sorted(PILOT_IDS)      # exactly the remainder ran
    assert len(ids) == len(set(ids)), "duplicate (task_id, condition) records"
    assert "pending=2" in second.stdout           # visible skip accounting


def test_resume_retry_errors_reruns_timeouts(tmp_path):
    hang_id = PILOT_IDS[0]
    first = run_runner("--batch-id", "b5", "--only", hang_id,
                       "--timeout", "15", tmp_path=tmp_path,
                       env_extra={"BANTZ_EVAL_TEST_HANG": hang_id})
    assert first.returncode == 0
    assert read_records(tmp_path, "b5")[0]["outcome_class"] == "timeout"

    # plain --resume: the timeout record counts as done
    second = run_runner("--batch-id", "b5", "--only", hang_id,
                        "--resume", tmp_path=tmp_path)
    assert second.returncode == 0
    assert len(read_records(tmp_path, "b5")) == 1

    # --retry-errors: re-runs it (no hang this time) and appends a fresh record
    third = run_runner("--batch-id", "b5", "--only", hang_id,
                       "--resume", "--retry-errors", tmp_path=tmp_path)
    assert third.returncode == 0
    records = read_records(tmp_path, "b5")
    assert len(records) == 2
    assert records[-1]["outcome_class"] == "completed"


def test_torn_last_line_rerun_not_duplicated(tmp_path):
    """kill -9 mid-write leaves a torn JSON line — that task must re-run."""
    first = run_runner("--batch-id", "b6", "--only", *PILOT_IDS[:2],
                       tmp_path=tmp_path)
    assert first.returncode == 0
    results_file = next((tmp_path / "results" / "b6").glob("*.jsonl"))
    lines = results_file.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1][: len(lines[-1]) // 2]  # tear the final record
    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    second = run_runner("--batch-id", "b6", "--only", *PILOT_IDS[:2],
                        "--resume", tmp_path=tmp_path)
    assert second.returncode == 0
    parseable = read_records(tmp_path, "b6")
    # the intact record was skipped; the torn task re-ran and appended fresh
    assert len(parseable) == 2
    assert sorted(r["task_id"] for r in parseable) == sorted(PILOT_IDS[:2])


# ── #509 integration: the batch takes the run lock ───────────────────────────


def test_batch_refuses_when_lock_held(tmp_path):
    from runlock import RunLock
    with RunLock(batch_id="occupying", lock_path=tmp_path / ".run.lock"):
        proc = run_runner("--batch-id", "b7", "--only", PILOT_IDS[0],
                          tmp_path=tmp_path)
    assert proc.returncode != 0
    assert "RUN LOCK HELD" in proc.stderr
    assert "occupying" in proc.stderr


# ── zero live writes at runner level (#505 re-verified) ──────────────────────


def test_no_records_flag_sandbox_violation_and_no_live_writes(tmp_path):
    import sandbox
    marker = sandbox.live_marker()
    proc = run_runner("--batch-id", "b8", "--only", *PILOT_IDS[:2],
                      tmp_path=tmp_path)
    assert proc.returncode == 0
    for rec in read_records(tmp_path, "b8"):
        assert rec["outcome_class"] != "sandbox_violation"
    assert sandbox.live_writes_since(marker) == []
