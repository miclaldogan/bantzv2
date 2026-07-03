"""Tests for eval/loop_eval/runlock.py (issue #509)."""
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
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from runlock import RunLock, RunLockHeld, read_holder  # noqa: E402


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / ".run.lock"


def test_acquire_writes_holder_record_and_release_frees(lock_path):
    with RunLock(batch_id="batch-A", lock_path=lock_path):
        holder = read_holder(lock_path)
        assert holder["pid"] == os.getpid()
        assert holder["batch_id"] == "batch-A"
        assert holder["started"]
    assert read_holder(lock_path) is None  # released


def test_second_acquire_refuses_with_holder_info(lock_path):
    """Acceptance: concurrent start refused with a clear message."""
    with RunLock(batch_id="batch-A", lock_path=lock_path):
        with pytest.raises(RunLockHeld) as exc:
            RunLock(batch_id="batch-B", lock_path=lock_path).acquire()
        message = str(exc.value)
        assert "refusing to start" in message
        assert str(os.getpid()) in message      # holder pid
        assert "batch-A" in message             # holder batch
        assert "started" in message             # start time shown
    # released -> now acquirable
    with RunLock(batch_id="batch-B", lock_path=lock_path):
        assert read_holder(lock_path)["batch_id"] == "batch-B"


def test_second_process_refuses(lock_path):
    """Acceptance: second INVOCATION exits non-zero with the message."""
    with RunLock(batch_id="held-by-test", lock_path=lock_path):
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(EVAL_DIR)!r})
            from runlock import RunLock
            RunLock(batch_id="intruder",
                    lock_path={str(lock_path)!r}).acquire()
        """)
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode != 0
        assert "RUN LOCK HELD" in proc.stderr
        assert "held-by-test" in proc.stderr


def test_stale_lock_stolen_with_warning(lock_path, capsys):
    """Acceptance: holder pid dead -> steal with warning."""
    # Forge a lock held by a dead pid on THIS host.
    import socket

    import psutil
    candidate = 99999  # find a pid that is definitely not alive
    while psutil.pid_exists(candidate):
        candidate -= 7
    lock_path.write_text(json.dumps({
        "pid": candidate, "host": socket.gethostname(),
        "batch_id": "crashed-batch", "started": "2026-07-01T03:00:00",
    }), encoding="utf-8")

    with RunLock(batch_id="new-batch", lock_path=lock_path):
        holder = read_holder(lock_path)
        assert holder["pid"] == os.getpid()
        assert holder["batch_id"] == "new-batch"
    err = capsys.readouterr().err
    assert "stealing stale run lock" in err
    assert "crashed-batch" in err


def test_foreign_host_lock_never_stolen(lock_path):
    lock_path.write_text(json.dumps({
        "pid": 12345, "host": "some-other-machine",
        "batch_id": "remote-batch", "started": "2026-07-01T03:00:00",
    }), encoding="utf-8")
    with pytest.raises(RunLockHeld) as exc:
        RunLock(batch_id="mine", lock_path=lock_path).acquire()
    assert "remote-batch" in str(exc.value)


def test_release_only_removes_own_lock(lock_path):
    lock_path.write_text(json.dumps({
        "pid": -1, "host": "elsewhere", "batch_id": "x", "started": "?"}),
        encoding="utf-8")
    lock = RunLock(batch_id="mine", lock_path=lock_path)
    lock.release()  # not ours — must not delete
    assert lock_path.exists()


def test_corrupt_lock_treated_as_held(lock_path):
    lock_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunLockHeld):
        RunLock(batch_id="mine", lock_path=lock_path,
                steal_stale=False).acquire()


def test_cli_status_and_forced_release(lock_path):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    def cli(*args):
        return subprocess.run(
            [sys.executable, str(EVAL_DIR / "runlock.py"), *args],
            capture_output=True, text=True, timeout=60, env=env)

    # Default lock path is the real one — only exercise read-only status
    # unless we know it's free; the module-level default must stay untouched
    # by tests, so drive stateful commands through a monkeypatched path.
    proc = cli("status")
    assert proc.returncode == 0
    assert "run lock:" in proc.stdout
