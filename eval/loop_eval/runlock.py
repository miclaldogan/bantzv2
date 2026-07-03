"""Single-Ollama run-queue lock (issue #509).

One Ollama instance serves the daemon, everyone's dev smoke tests, and the
eval batches — and it stalls under concurrent load on this machine. This
module makes concurrent batch starts impossible rather than impolite:

    from runlock import RunLock
    with RunLock(batch_id="baseline_2026-07-02"):
        ...run the batch...

Or standalone:

    python eval/loop_eval/runlock.py status    # who holds the lock?
    python eval/loop_eval/runlock.py acquire --batch-id smoke  # hold until Ctrl-C
    python eval/loop_eval/runlock.py release --force           # manual cleanup

Mechanism: atomic ``O_CREAT|O_EXCL`` creation of ``.run.lock`` next to this
file (works on every OS and over the network shares people actually use),
plus a best-effort BSD ``flock`` on the open descriptor where the platform
has one (Linux prod). The lock file records holder pid, host, batch id and
start time as JSON, so a refused start can SAY who is running.

Stale locks: if the recorded pid is dead (checked with psutil on the same
host), the lock is stolen with a loud warning. A lock held by another HOST
is never stolen automatically — coordinate per the RUNBOOK.

Queue etiquette, overnight procedure and stall handling: see RUNBOOK.md in
this directory.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent / ".run.lock"


class RunLockHeld(SystemExit):
    """Raised (exits non-zero) when a batch is already running."""


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:  # pragma: no cover — psutil is a bantz core dep
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


def read_holder(lock_path: Path = LOCK_PATH) -> dict | None:
    """The current holder record, or None if the lock is free/unreadable."""
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        # Mid-write or corrupt — treat as held by parties unknown.
        return {"pid": -1, "host": "?", "batch_id": "?", "started": "?"}


class RunLock:
    """Exclusive batch lock. Context manager; re-entrant within a process
    is NOT supported (a second acquire in the same process refuses too —
    two batches in one process is still two batches)."""

    def __init__(self, batch_id: str = "", lock_path: Path = LOCK_PATH,
                 steal_stale: bool = True) -> None:
        self.batch_id = batch_id or "unnamed-batch"
        self.lock_path = Path(lock_path)
        self.steal_stale = steal_stale
        self._fd: int | None = None

    # ── acquire / release ─────────────────────────────────────────────────

    def acquire(self) -> "RunLock":
        try:
            self._create()
        except FileExistsError:
            holder = read_holder(self.lock_path) or {}
            if self._is_stale(holder):
                print(
                    f"WARNING: stealing stale run lock — holder pid "
                    f"{holder.get('pid')} ({holder.get('batch_id')!r}, "
                    f"started {holder.get('started')}) is dead.",
                    file=sys.stderr,
                )
                self.lock_path.unlink(missing_ok=True)
                try:
                    self._create()
                except FileExistsError:  # lost the race to another starter
                    holder = read_holder(self.lock_path) or {}
                    raise self._refused(holder) from None
            else:
                raise self._refused(holder) from None
        return self

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        holder = read_holder(self.lock_path)
        if holder and holder.get("pid") == os.getpid():
            self.lock_path.unlink(missing_ok=True)

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()

    # ── internals ─────────────────────────────────────────────────────────

    def _create(self) -> None:
        fd = os.open(str(self.lock_path),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            # Best-effort advisory flock on platforms that have it (Linux
            # prod box) — belt and braces on top of O_EXCL.
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:
                pass
            record = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "batch_id": self.batch_id,
                "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                "argv": sys.argv,
            }
            os.write(fd, json.dumps(record, ensure_ascii=False).encode("utf-8"))
            os.fsync(fd)
            self._fd = fd
        except BaseException:
            os.close(fd)
            self.lock_path.unlink(missing_ok=True)
            raise

    def _is_stale(self, holder: dict) -> bool:
        if not self.steal_stale:
            return False
        pid = holder.get("pid", -1)
        same_host = holder.get("host") == socket.gethostname()
        return same_host and isinstance(pid, int) and pid > 0 and not _pid_alive(pid)

    def _refused(self, holder: dict) -> RunLockHeld:
        return RunLockHeld(
            "RUN LOCK HELD: a batch is already running — refusing to start.\n"
            f"  holder : pid {holder.get('pid')} on {holder.get('host')}\n"
            f"  batch  : {holder.get('batch_id')!r}\n"
            f"  started: {holder.get('started')}\n"
            "One Ollama = one batch. See eval/loop_eval/RUNBOOK.md. If the "
            "holder is genuinely gone on another machine, release manually: "
            "python eval/loop_eval/runlock.py release --force"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"

    if cmd == "status":
        holder = read_holder()
        if holder is None:
            print("run lock: FREE")
        else:
            alive = (_pid_alive(holder.get("pid", -1))
                     if holder.get("host") == socket.gethostname() else "unknown (other host)")
            print(f"run lock: HELD by pid {holder.get('pid')} on "
                  f"{holder.get('host')} — batch {holder.get('batch_id')!r}, "
                  f"started {holder.get('started')} (pid alive: {alive})")
        return 0

    if cmd == "acquire":
        batch = ""
        if "--batch-id" in argv:
            batch = argv[argv.index("--batch-id") + 1]
        lock = RunLock(batch_id=batch).acquire()  # raises RunLockHeld if busy
        print(f"acquired run lock for {lock.batch_id!r} — Ctrl-C to release")
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            lock.release()
            print("released")
        return 0

    if cmd == "release":
        holder = read_holder()
        if holder is None:
            print("run lock already free")
            return 0
        if holder.get("pid") != os.getpid() and "--force" not in argv:
            print("refusing: lock belongs to another process — pass --force "
                  "if you are certain the holder is gone", file=sys.stderr)
            return 1
        LOCK_PATH.unlink(missing_ok=True)
        print("released")
        return 0

    print(f"unknown command {cmd!r} — use: status | acquire [--batch-id X] | "
          "release [--force]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
