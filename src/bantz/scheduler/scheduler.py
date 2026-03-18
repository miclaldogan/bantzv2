"""
Bantz — Scheduler: APScheduler-based cron / one-shot / interval task runner (#295)

Provides a clean API over APScheduler's AsyncIOScheduler:
    Scheduler.add_cron(func, cron_expr, job_id)   — recurring cron job
    Scheduler.add_once(func, run_at, job_id)       — one-shot datetime job
    Scheduler.add_interval(func, seconds, job_id)  — periodic interval job
    Scheduler.list_jobs()                          — list[ScheduledJob]
    Scheduler.cancel(job_id)                       — cancel by id

Jobs are persisted via Redis (RedisJobStore) when Redis is available, and
fall back to an in-memory store otherwise so the scheduler always starts.

Failed jobs are logged with full tracebacks to
    ~/.bantz/logs/scheduler.log

Usage:
    from bantz.scheduler.scheduler import scheduler

    scheduler.start()
    scheduler.add_cron(my_func, "0 8 * * *", "morning_briefing")
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("bantz.scheduler")

# ── Audit / error log ─────────────────────────────────────────────────────────
_SCHEDULER_LOG: Path = Path.home() / ".bantz" / "logs" / "scheduler.log"


def _ensure_log_dir() -> None:
    _SCHEDULER_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_job_error(job_id: str, exc: BaseException) -> None:
    """Append a failed-job entry with full traceback to scheduler.log."""
    try:
        _ensure_log_dir()
        ts = datetime.now().isoformat(timespec="seconds")
        tb = traceback.format_exc()
        with _SCHEDULER_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[{ts}] JOB FAILED: {job_id}\n{tb}\n")
    except Exception:
        pass


# ── ScheduledJob ──────────────────────────────────────────────────────────────

@dataclass
class ScheduledJob:
    """Lightweight snapshot of a scheduled job."""
    id: str
    next_run: datetime | None
    trigger_type: str  # "cron", "interval", "date"


# ── Wrapped job executor ──────────────────────────────────────────────────────

def _make_wrapped(func: Callable, job_id: str) -> Callable:
    """Return a wrapper that catches exceptions and logs to scheduler.log."""
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log.error("Job %s failed: %s", job_id, exc)
            _log_job_error(job_id, exc)
            raise  # re-raise so APScheduler still records the failure

    _wrapper.__name__ = getattr(func, "__name__", job_id)
    _wrapper.__qualname__ = getattr(func, "__qualname__", job_id)
    return _wrapper


# ── Scheduler ─────────────────────────────────────────────────────────────────

class Scheduler:
    """APScheduler-backed scheduler with Redis persistence and error logging.

    Always starts with a MemoryJobStore fallback if Redis is unavailable.
    """

    def __init__(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        jobstores: dict[str, Any] = {}
        jobstores["default"] = self._make_jobstore()

        self._scheduler = AsyncIOScheduler(jobstores=jobstores)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler (non-blocking; runs in the event loop)."""
        if not self._scheduler.running:
            self._scheduler.start()
            log.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            log.info("Scheduler shut down")

    @property
    def running(self) -> bool:
        return self._scheduler.running

    # ── Job registration ───────────────────────────────────────────────────

    def add_cron(
        self,
        func: Callable,
        cron_expr: str,
        job_id: str,
        *,
        replace_existing: bool = True,
    ) -> None:
        """Schedule a recurring job using a cron expression.

        Args:
            func:       Callable to execute.
            cron_expr:  Standard 5-field cron string: "min hour dom month dow"
                        e.g. "0 8 * * *" — 08:00 every day.
            job_id:     Unique job identifier.
        """
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"cron_expr must have 5 fields (got {len(parts)}): {cron_expr!r}"
            )
        minute, hour, dom, month, dow = parts
        wrapped = _make_wrapped(func, job_id)
        self._scheduler.add_job(
            wrapped,
            "cron",
            minute=minute,
            hour=hour,
            day=dom,
            month=month,
            day_of_week=dow,
            id=job_id,
            replace_existing=replace_existing,
        )
        log.info("Cron job added: %s @ %s", job_id, cron_expr)

    def add_once(
        self,
        func: Callable,
        run_at: datetime,
        job_id: str,
        *,
        replace_existing: bool = True,
    ) -> None:
        """Schedule a one-shot job to run at a specific datetime.

        Args:
            func:       Callable to execute.
            run_at:     When to execute.
            job_id:     Unique job identifier.
        """
        wrapped = _make_wrapped(func, job_id)
        self._scheduler.add_job(
            wrapped,
            "date",
            run_date=run_at,
            id=job_id,
            replace_existing=replace_existing,
        )
        log.info("One-shot job added: %s @ %s", job_id, run_at.isoformat())

    def add_interval(
        self,
        func: Callable,
        seconds: int,
        job_id: str,
        *,
        replace_existing: bool = True,
    ) -> None:
        """Schedule a recurring job on a fixed interval.

        Args:
            func:       Callable to execute.
            seconds:    Interval in seconds.
            job_id:     Unique job identifier.
        """
        if seconds <= 0:
            raise ValueError(f"seconds must be positive (got {seconds})")
        wrapped = _make_wrapped(func, job_id)
        self._scheduler.add_job(
            wrapped,
            "interval",
            seconds=seconds,
            id=job_id,
            replace_existing=replace_existing,
        )
        log.info("Interval job added: %s every %ds", job_id, seconds)

    # ── Introspection ──────────────────────────────────────────────────────

    def list_jobs(self) -> list[ScheduledJob]:
        """Return a snapshot of all scheduled jobs.

        Returns:
            list[ScheduledJob] — each with id, next_run, trigger_type.
        """
        result: list[ScheduledJob] = []
        for job in self._scheduler.get_jobs():
            trigger_type = type(job.trigger).__name__.lower().replace("trigger", "")
            result.append(
                ScheduledJob(
                    id=job.id,
                    next_run=getattr(job, "next_run_time", None),
                    trigger_type=trigger_type,
                )
            )
        return result

    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Return a single ScheduledJob by id, or None if not found."""
        job = self._scheduler.get_job(job_id)
        if job is None:
            return None
        trigger_type = type(job.trigger).__name__.lower().replace("trigger", "")
        return ScheduledJob(
            id=job.id,
            next_run=getattr(job, "next_run_time", None),
            trigger_type=trigger_type,
        )

    # ── Cancellation ───────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled job by id.

        Returns:
            True if the job was found and removed, False if not found.
        """
        from apscheduler.jobstores.base import JobLookupError as _JLE

        try:
            self._scheduler.remove_job(job_id)
            log.info("Job cancelled: %s", job_id)
            return True
        except _JLE:
            log.debug("cancel: job %r not found", job_id)
            return False

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_jobstore() -> Any:
        """Build a Redis jobstore, or fall back to MemoryJobStore."""
        try:
            from apscheduler.jobstores.redis import RedisJobStore
            from bantz.config import config as _cfg

            store = RedisJobStore(
                jobs_key="bantz:scheduler:jobs",
                run_times_key="bantz:scheduler:run_times",
                host=getattr(_cfg, "redis_host", "localhost"),
                port=getattr(_cfg, "redis_port", 6379),
            )
            log.info("Scheduler jobstore: Redis")
            return store
        except Exception as exc:
            from apscheduler.jobstores.memory import MemoryJobStore

            log.info("Scheduler jobstore: memory (Redis unavailable: %s)", exc)
            return MemoryJobStore()


# ── Module singleton ──────────────────────────────────────────────────────────

scheduler = Scheduler()
