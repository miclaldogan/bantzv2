"""
Bantz — Autonomous Nightly Maintenance Workflow (#129)

Runs at 3 AM daily via APScheduler.  Six idempotent steps with per-step
timeout (30 s) and a 5-minute total cap:

    Step 1  Docker cleanup        docker system prune -f, docker volume prune -f
    Step 2  Temp / cache purge    /tmp/bantz*, ~/.cache/bantz/old-logs
    Step 3  Disk health check     <10% free → alert, <5% → emergency cleanup
    Step 4  Service health        Ollama ping, DB integrity_check
    Step 5  Log rotation          compress bantz.log → .gz (keep 7)
    Step 6  Report                KV store + Telegram + desktop notification

Highlights:
  - Every step is safe to re-run (idempotent).
  - Docker-not-installed handled gracefully (skip, warn).
  - RL reward (+0.1) when maintenance frees ≥ 500 MB disk space.
  - Dry-run mode: ``bantz --maintenance --dry-run`` prints without acting.
  - Results cached in KV store → morning briefing picks them up.

Usage:
    from bantz.agent.workflows.maintenance import run_maintenance
    report = await run_maintenance(dry_run=False)
"""
from __future__ import annotations

import asyncio
import gzip
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.maintenance")

# ── Constants ────────────────────────────────────────────────────────────

_STEP_TIMEOUT = 30          # seconds per step
_TOTAL_TIMEOUT = 300        # 5 min total
_LOG_KEEP = 7               # keep N rotated logs
_TEMP_MAX_AGE_DAYS = 7      # purge temp files older than this
_DISK_WARN_PCT = 10         # warn < 10 % free
_DISK_EMERGENCY_PCT = 5     # emergency < 5 %
_RL_REWARD_THRESHOLD_MB = 500  # reward RL if freed ≥ this
_RL_REWARD_VALUE = 0.1      # small self-reward for good housekeeping


# ── Step result ──────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Outcome of a single maintenance step."""
    name: str
    ok: bool = True
    skipped: bool = False
    detail: str = ""
    bytes_freed: int = 0
    elapsed: float = 0.0


@dataclass
class MaintenanceReport:
    """Full maintenance run report."""
    started_at: str = ""
    finished_at: str = ""
    dry_run: bool = False
    steps: list[StepResult] = field(default_factory=list)
    total_freed_mb: float = 0.0
    disk_free_pct: float = 0.0
    rl_reward_given: bool = False
    errors: int = 0

    # ── helpers ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable one-paragraph summary."""
        tag = " (DRY-RUN)" if self.dry_run else ""
        freed = f"{self.total_freed_mb:.1f}" if self.total_freed_mb else "0"
        ok = sum(1 for s in self.steps if s.ok and not s.skipped)
        skip = sum(1 for s in self.steps if s.skipped)
        fail = sum(1 for s in self.steps if not s.ok)
        lines = [f"🔧 Maintenance{tag}: {ok} ok, {skip} skipped, {fail} failed"]
        lines.append(f"   Disk freed: {freed} MB  |  Free: {self.disk_free_pct:.1f}%")
        for s in self.steps:
            icon = "✓" if s.ok and not s.skipped else ("○" if s.skipped else "✗")
            detail = f" — {s.detail}" if s.detail else ""
            lines.append(f"   {icon} {s.name}{detail} ({s.elapsed:.1f}s)")
        if self.rl_reward_given:
            lines.append(f"   🎖 RL reward: +{_RL_REWARD_VALUE} (freed ≥ {_RL_REWARD_THRESHOLD_MB} MB)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dry_run": self.dry_run,
            "total_freed_mb": round(self.total_freed_mb, 1),
            "disk_free_pct": round(self.disk_free_pct, 1),
            "rl_reward_given": self.rl_reward_given,
            "errors": self.errors,
            "steps": [
                {"name": s.name, "ok": s.ok, "skipped": s.skipped,
                 "detail": s.detail, "bytes_freed": s.bytes_freed}
                for s in self.steps
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Helper — safe subprocess with timeout
# ═══════════════════════════════════════════════════════════════════════════

async def _run_cmd(
    *args: str,
    timeout: float = _STEP_TIMEOUT,
) -> tuple[int, str, str]:
    """Run a shell command with a timeout, returning (returncode, stdout, stderr).

    Returns (-1, '', error_msg) on timeout or missing binary.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode(errors="replace").strip(), stderr.decode(errors="replace").strip()
    except FileNotFoundError:
        return -1, "", f"{args[0]}: not installed"
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return -1, "", f"{args[0]}: timed out after {timeout}s"


def _disk_usage(path: str = "/") -> tuple[float, float]:
    """Return (free_bytes, free_pct) for the filesystem containing *path*."""
    st = shutil.disk_usage(path)
    free_pct = (st.free / st.total) * 100 if st.total else 0
    return st.free, free_pct


def _data_dir() -> Path:
    """Bantz data directory."""
    from bantz.config import config
    return config.db_path.parent


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Docker cleanup
# ═══════════════════════════════════════════════════════════════════════════

async def _step_docker_cleanup(dry_run: bool) -> StepResult:
    t0 = time.monotonic()
    result = StepResult(name="Docker cleanup")

    # Check if docker is available
    rc, out, err = await _run_cmd("docker", "info", timeout=10)
    if rc != 0:
        result.skipped = True
        result.detail = "Docker not installed or not running"
        result.elapsed = time.monotonic() - t0
        return result

    if dry_run:
        result.detail = "would run: docker system prune -f && docker volume prune -f"
        result.skipped = True
        result.elapsed = time.monotonic() - t0
        return result

    freed = 0

    # system prune
    rc, out, err = await _run_cmd("docker", "system", "prune", "-f")
    if rc == 0:
        # Parse "Total reclaimed space: 1.234GB"
        for line in out.split("\n"):
            if "reclaimed" in line.lower():
                freed += _parse_docker_size(line)
                result.detail = line.strip()
    else:
        result.detail = f"system prune failed: {err[:100]}"

    # volume prune (dangling only)
    rc2, out2, err2 = await _run_cmd("docker", "volume", "prune", "-f")
    if rc2 == 0:
        for line in out2.split("\n"):
            if "reclaimed" in line.lower():
                freed += _parse_docker_size(line)

    result.bytes_freed = freed
    result.ok = (rc == 0)
    result.elapsed = time.monotonic() - t0
    return result


def _parse_docker_size(line: str) -> int:
    """Parse '... 1.234GB' or '123.4MB' from docker output → bytes."""
    import re
    m = re.search(r"([\d.]+)\s*(GB|MB|KB|B)", line, re.IGNORECASE)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2).upper()
    multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return int(val * multiplier.get(unit, 1))


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Temp / cache purge
# ═══════════════════════════════════════════════════════════════════════════

async def _step_temp_cleanup(dry_run: bool) -> StepResult:
    t0 = time.monotonic()
    result = StepResult(name="Temp cleanup")
    freed = 0
    cleaned = 0
    cutoff = time.time() - _TEMP_MAX_AGE_DAYS * 86400

    targets = [
        (Path("/tmp"), "bantz*"),
        (Path("/tmp"), "bantz_*"),
        (Path.home() / ".cache" / "bantz" / "old-logs", "*"),
    ]

    for base, pattern in targets:
        if not base.exists():
            continue
        try:
            for f in base.glob(pattern):
                try:
                    stat = f.stat()
                    if stat.st_mtime < cutoff:
                        size = stat.st_size
                        if dry_run:
                            log.debug("Would delete: %s (%d bytes)", f, size)
                        else:
                            if f.is_dir():
                                shutil.rmtree(f, ignore_errors=True)
                            else:
                                f.unlink()
                            freed += size
                        cleaned += 1
                except OSError:
                    pass
        except Exception:
            pass

    result.bytes_freed = freed
    tag = "would clean" if dry_run else "cleaned"
    result.detail = f"{tag} {cleaned} items ({freed / 1024 / 1024:.1f} MB)"
    if dry_run:
        result.skipped = True
    result.elapsed = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Disk health check
# ═══════════════════════════════════════════════════════════════════════════

async def _step_disk_check(dry_run: bool) -> StepResult:
    t0 = time.monotonic()
    result = StepResult(name="Disk check")

    free_bytes, free_pct = _disk_usage("/")
    free_gb = free_bytes / (1024 ** 3)
    result.detail = f"{free_gb:.1f} GB free ({free_pct:.1f}%)"

    if free_pct < _DISK_EMERGENCY_PCT:
        result.ok = False
        result.detail += " ⚠ EMERGENCY — below 5%!"
        if not dry_run:
            # Try emergency cleanup: remove old pip caches
            try:
                pip_cache = Path.home() / ".cache" / "pip"
                if pip_cache.exists():
                    shutil.rmtree(pip_cache, ignore_errors=True)
                    result.detail += " (pip cache cleared)"
            except Exception:
                pass
    elif free_pct < _DISK_WARN_PCT:
        result.detail += " ⚠ Low disk space — below 10%"

    result.elapsed = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Service health
# ═══════════════════════════════════════════════════════════════════════════

async def _step_service_health(dry_run: bool) -> StepResult:
    t0 = time.monotonic()
    result = StepResult(name="Service health")
    checks = []

    # Ollama ping
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                checks.append("Ollama ✓")
            else:
                checks.append(f"Ollama ✗ ({resp.status_code})")
    except Exception:
        checks.append("Ollama ✗ (unreachable)")

    # DB integrity check
    try:
        db_path = _data_dir() / "bantz.db"
        if db_path.exists():
            from bantz.data.connection_pool import get_pool
            with get_pool(str(db_path)).connection() as conn:
                row = conn.execute("PRAGMA integrity_check").fetchone()
            if row and row[0] == "ok":
                checks.append("DB ✓")
            else:
                checks.append(f"DB ✗ ({row})")
                result.ok = False
        else:
            checks.append("DB ○ (not found)")
    except Exception as exc:
        checks.append(f"DB ✗ ({exc})")
        result.ok = False

    result.detail = ", ".join(checks)
    result.elapsed = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Log rotation
# ═══════════════════════════════════════════════════════════════════════════

async def _step_log_rotation(dry_run: bool) -> StepResult:
    t0 = time.monotonic()
    result = StepResult(name="Log rotation")
    freed = 0

    log_dir = _data_dir()
    log_file = log_dir / "bantz.log"

    if not log_file.exists() or log_file.stat().st_size == 0:
        result.detail = "no log to rotate"
        result.skipped = True
        result.elapsed = time.monotonic() - t0
        return result

    log_size = log_file.stat().st_size

    if dry_run:
        result.detail = f"would rotate bantz.log ({log_size / 1024:.0f} KB)"
        result.skipped = True
        result.elapsed = time.monotonic() - t0
        return result

    # Shift existing rotated logs
    for i in range(_LOG_KEEP - 1, 0, -1):
        old = log_dir / f"bantz.log.{i}.gz"
        new = log_dir / f"bantz.log.{i + 1}.gz"
        if old.exists():
            if i + 1 >= _LOG_KEEP:
                freed += old.stat().st_size
                old.unlink()
            else:
                old.rename(new)

    # Compress current log → .1.gz
    target = log_dir / "bantz.log.1.gz"
    try:
        with open(log_file, "rb") as f_in:
            with gzip.open(target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        freed += log_size
        log_file.write_text("")  # truncate
        result.detail = f"rotated ({log_size / 1024:.0f} KB → .gz)"
    except Exception as exc:
        result.ok = False
        result.detail = f"rotation failed: {exc}"

    result.bytes_freed = freed
    result.elapsed = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Report (KV + Telegram + Notification + RL)
# ═══════════════════════════════════════════════════════════════════════════

async def _step_report(report: MaintenanceReport) -> StepResult:
    t0 = time.monotonic()
    result = StepResult(name="Report")
    summary = report.summary()

    # ── Store in KV for morning briefing ──────────────────────────────
    try:
        from bantz.data.sqlite_store import SQLiteKVStore
        kv = SQLiteKVStore(_data_dir() / "bantz.db")
        import json
        kv.set("maintenance_last_run", datetime.now().isoformat())
        kv.set("maintenance_last_report", json.dumps(report.to_dict(), ensure_ascii=False))
        kv.set("maintenance_summary", summary)
    except Exception as exc:
        log.warning("Failed to store maintenance report: %s", exc)

    # ── Log to memory ────────────────────────────────────────────────
    try:
        from bantz.core.memory import memory
        if memory._initialized:
            memory.add("assistant", summary, tool_used="maintenance")
    except Exception:
        pass

    # ── Desktop notification ─────────────────────────────────────────
    try:
        from bantz.agent.notifier import notifier
        if notifier.enabled:
            tag = " (dry-run)" if report.dry_run else ""
            one_line = (
                f"🔧 Maintenance{tag}: "
                f"{report.total_freed_mb:.0f} MB freed, "
                f"{report.disk_free_pct:.0f}% free"
            )
            urgency = "critical" if report.disk_free_pct < _DISK_EMERGENCY_PCT else "normal"
            notifier.send(one_line, urgency=urgency, expire_ms=10_000)
    except Exception:
        pass

    # ── Telegram summary (if configured) ─────────────────────────────
    try:
        from bantz.config import config
        if config.telegram_bot_token and config.telegram_allowed_users:
            import httpx
            users = [u.strip() for u in config.telegram_allowed_users.split(",") if u.strip()]
            for uid in users:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(
                            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                            json={"chat_id": uid, "text": summary, "parse_mode": ""},
                        )
                except Exception:
                    pass
    except Exception:
        pass

    # ── RL reward: self-congratulate if freed significant space ───────
    if not report.dry_run and report.total_freed_mb >= _RL_REWARD_THRESHOLD_MB:
        try:
            from bantz.agent.rl_engine import rl_engine, State, Action
            if rl_engine._initialized:
                state = State(
                    time_segment="late_night",
                    day=datetime.now().strftime("%A").lower(),
                    location="home",
                    recent_tool="maintenance",
                )
                rl_engine.force_reward(
                    state, Action.RUN_MAINTENANCE, _RL_REWARD_VALUE,
                )
                report.rl_reward_given = True
                log.info(
                    "🎖 RL self-reward: +%.1f for freeing %.0f MB",
                    _RL_REWARD_VALUE, report.total_freed_mb,
                )
        except Exception as exc:
            log.debug("RL reward skipped: %s", exc)

    result.detail = "stored"
    result.elapsed = time.monotonic() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main entrypoint
# ═══════════════════════════════════════════════════════════════════════════

async def run_maintenance(*, dry_run: bool = False) -> MaintenanceReport:
    """Execute the full 6-step maintenance workflow.

    Args:
        dry_run: If True, print what would be done without actually doing it.

    Returns:
        MaintenanceReport with per-step results and totals.
    """
    from bantz.agent.job_scheduler import inhibit_sleep

    report = MaintenanceReport(
        started_at=datetime.now().isoformat(),
        dry_run=dry_run,
    )

    tag = " (DRY-RUN)" if dry_run else ""
    log.info("🔧 Maintenance starting%s...", tag)

    with inhibit_sleep("Bantz nightly maintenance"):
        # Snapshot disk before
        _, free_before = _disk_usage("/")

        steps = [
            _step_docker_cleanup,
            _step_temp_cleanup,
            _step_disk_check,
            _step_service_health,
            _step_log_rotation,
        ]

        deadline = time.monotonic() + _TOTAL_TIMEOUT

        for step_fn in steps:
            if time.monotonic() > deadline:
                sr = StepResult(name=step_fn.__name__, ok=False, detail="total timeout exceeded")
                report.steps.append(sr)
                report.errors += 1
                continue

            try:
                sr = await asyncio.wait_for(step_fn(dry_run), timeout=_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                sr = StepResult(
                    name=step_fn.__name__,
                    ok=False,
                    detail=f"timed out after {_STEP_TIMEOUT}s",
                )
            except Exception as exc:
                sr = StepResult(
                    name=step_fn.__name__,
                    ok=False,
                    detail=str(exc)[:200],
                )

            report.steps.append(sr)
            if not sr.ok:
                report.errors += 1

        # Compute totals
        total_freed = sum(s.bytes_freed for s in report.steps)
        report.total_freed_mb = total_freed / (1024 ** 2) if total_freed else 0

        _, report.disk_free_pct = _disk_usage("/")

        report.finished_at = datetime.now().isoformat()

        # Step 6: report
        try:
            report_step = await asyncio.wait_for(
                _step_report(report), timeout=_STEP_TIMEOUT
            )
        except Exception as exc:
            report_step = StepResult(name="Report", ok=False, detail=str(exc)[:200])
        report.steps.append(report_step)

    log.info("🔧 Maintenance complete%s: %s", tag, report.summary().split("\n")[0])
    return report
