"""Tests for eval/loop_eval/preflight.py (issue #512).

Runs the real gate in mock mode — this IS the rehearsal the gate performs
before a batch, so the test doubles as continuous verification that the
gate itself hasn't rotted.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"


def _run(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, str(EVAL_DIR / "preflight.py"), "--mock-llm", *args],
        capture_output=True, text=True, timeout=timeout, env=env,
        cwd=str(REPO))


def test_preflight_green_with_loop(tmp_path):
    """With the C1 loop landed (#503), the UNWAIVED gate is GREEN: check #4
    shows a real steps=3 recovery end-to-end, so no --allow-no-loop is needed.
    (Before the loop landed this same call correctly returned RED.)"""
    report = tmp_path / "report.md"
    proc = _run("--report", str(report))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    text = report.read_text(encoding="utf-8")
    assert "GREEN — full batch may proceed" in text
    assert text.count("✅ PASS") == 5
    assert "recovery=True" in text
    assert "git SHA" in text
    # every one of the five checks is present in the archived report
    for marker in ("1. mini-batch", "2. hard-kill", "3. zero writes",
                   "4. steps=3", "5. watchdog"):
        assert marker in text, marker


def test_preflight_baseline_only_waiver_accepted(tmp_path):
    """--allow-no-loop remains a valid (now redundant) waiver: the gate still
    passes and archives all five checks. Because the loop recovers the
    transient task, the verdict is full GREEN rather than BASELINE-ONLY."""
    report = tmp_path / "report.md"
    proc = _run("--allow-no-loop", "--report", str(report))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    text = report.read_text(encoding="utf-8")
    assert "GREEN" in text
    assert text.count("✅ PASS") == 5
    assert "git SHA" in text
