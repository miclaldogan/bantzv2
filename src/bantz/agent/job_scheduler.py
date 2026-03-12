"""
Bantz — APScheduler-based Job Scheduler (#128)

Replaces the simple polling loop in _daemon() with a real job scheduling
engine.  Provides cron-style night workflows, one-shot/interval triggers,
persistent job store (SQLAlchemy → same bantz.db), retry with exponential
backoff, misfire grace, and sleep-inhibit for long-running night tasks.

Architecture:
    ┌──────────────────────────┐
    │      JobScheduler        │
    │  (AsyncIOScheduler)      │
    │                          │
    │  ┌─ night_maintenance    │   03:00  cron
    │  ├─ night_reflection     │   23:00  cron
    │  ├─ overnight_poll       │   every 2h (00-07)
    │  ├─ briefing_prep        │   06:00  cron
    │  ├─ reminder_check       │   every 30s interval
    │  └─ <dynamic reminders>  │   user-added via brain.py
    └──────────────────────────┘
              ↓ persists to
        bantz.db (SQLAlchemy)

Key design decisions:
  - misfire_grace_time=86400  → laptop can sleep; job runs on wake
  - coalesce=True             → missed repeats collapse into one
  - systemd-inhibit wrapper   → prevent sleep during night tasks
  - Existing Scheduler (core/scheduler.py) stays for backward compat;
    this module migrates active reminders into APScheduler on first init.

Usage:
    from bantz.agent.job_scheduler import job_scheduler
    await job_scheduler.start(db_path)
    job_scheduler.add_reminder("call dentist", fire_at=datetime(...))
    await job_scheduler.shutdown()
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("bantz.job_scheduler")

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_MAX_RETRIES = 3
_BACKOFF_BASE = 30  # seconds — retry at 30s, 60s, 120s

# Default misfire grace: 24 hours — so laptop-sleep won't discard jobs
_MISFIRE_GRACE = 86400

# Night job definitions: (job_id, hour, minute, description)
_NIGHT_JOBS = {
    "maintenance": {
        "hour": 3, "minute": 0,
        "description": "Docker prune, temp cleanup, log rotation",
    },
    "reflection": {
        "hour": 23, "minute": 0,
        "description": "Summarize today's conversations",
    },
    "briefing_prep": {
        "hour": 6, "minute": 0,
        "description": "Pre-fetch mail, calendar, weather for morning briefing",
    },
}

# Overnight poll: every 2h between 00:00-07:00
_OVERNIGHT_POLL = {
    "overnight_poll": {
        "hours": "0,2,4,6", "minute": 0,
        "description": "Check email/calendar overnight",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Sleep-inhibit context manager
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def inhibit_sleep(reason: str = "Bantz night task"):
    """Prevent system suspend using systemd-inhibit while a task runs.

    Falls back gracefully if systemd-inhibit is not available.
    """
    proc = None
    try:
        # Launch a blocking inhibitor — stays active while context is open
        proc = subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=sleep:idle",
                "--who=bantz",
                "--why=" + reason,
                "--mode=block",
                "sleep", "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.debug("Sleep inhibit acquired: %s (PID %d)", reason, proc.pid)
    except (FileNotFoundError, OSError):
        log.debug("systemd-inhibit not available — skipping sleep lock")

    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.debug("Sleep inhibit released")


# ═══════════════════════════════════════════════════════════════════════════
# Retry with exponential backoff
# ═══════════════════════════════════════════════════════════════════════════

async def _run_with_retry(
    func: Callable,
    *args: Any,
    job_name: str = "unknown",
    max_retries: int = _MAX_RETRIES,
    **kwargs: Any,
) -> Any:
    """Run an async function with exponential backoff retries.

    Backoff schedule: 30s, 60s, 120s (base * 2^attempt).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                log.info("Job '%s' succeeded on retry #%d", job_name, attempt)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = _BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Job '%s' failed (attempt %d/%d): %s — retrying in %ds",
                    job_name, attempt + 1, max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                log.error(
                    "Job '%s' failed permanently after %d attempts: %s",
                    job_name, max_retries + 1, exc,
                )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Night job implementations
# ═══════════════════════════════════════════════════════════════════════════

async def _job_maintenance() -> None:
    """Run the full 6-step maintenance workflow (#129)."""
    from bantz.agent.workflows.maintenance import run_maintenance
    report = await run_maintenance(dry_run=False)
    log.info("🔧 Maintenance finished: %d errors, %.1f MB freed",
             report.errors, report.total_freed_mb)


async def _job_reflection() -> None:
    """Run the nightly memory reflection workflow (#130)."""
    from bantz.agent.workflows.reflection import run_reflection
    result = await run_reflection(dry_run=False)
    log.info("🤔 Reflection finished: %d sessions, %d entities",
             result.sessions, result.entities_extracted)


async def _job_overnight_poll() -> None:
    """Run the overnight poll workflow (#132)."""
    from bantz.agent.workflows.overnight_poll import run_overnight_poll
    result = await run_overnight_poll(dry_run=False)
    log.info("📬 Overnight poll: %s", result.summary_line())


async def _job_briefing_prep() -> None:
    """Pre-fetch data for morning briefing and cache it.

    This runs at 06:00 so data is ready when user wakes up.
    The actual briefing is triggered by AppDetector (first active window).
    """
    log.info("📋 Briefing prep starting...")
    try:
        from bantz.core.briefing import briefing as _briefing
        text = await _briefing.generate()
        if text:
            # Store in KV for the briefing trigger to pick up
            from bantz.data.sqlite_store import SQLiteKVStore
            from bantz.config import config
            kv = SQLiteKVStore(config.db_path)
            kv.set("briefing_ready", text)
            kv.set("briefing_date", datetime.now().date().isoformat())
            log.info("Briefing prepared (%d chars)", len(text))
        else:
            log.info("Briefing prep: empty result")
    except Exception as exc:
        log.warning("Briefing prep failed: %s", exc)


async def _job_reminder_check() -> None:
    """Legacy reminder poll — bridge until all reminders are in APScheduler."""
    try:
        from bantz.core.scheduler import scheduler as _old_scheduler
        if not _old_scheduler._conn:
            return
        due = _old_scheduler.check_due()
        for r in due:
            repeat_tag = f" (repeats {r['repeat']})" if r["repeat"] != "none" else ""
            log.info("⏰ REMINDER: %s%s", r["title"], repeat_tag)

            # Send desktop notification if available
            try:
                from bantz.agent.notifier import notifier
                if notifier.enabled:
                    notifier.send(
                        f"⏰ {r['title']}",
                        repeat_tag.strip() if repeat_tag else "",
                        urgency="critical",
                        expire_ms=0,
                    )
            except Exception:
                pass

            # Store in conversations for chat context
            try:
                from bantz.core.memory import memory
                memory.add("assistant", f"⏰ Reminder: {r['title']}{repeat_tag}",
                           tool_used="reminder")
            except Exception:
                pass
    except Exception as exc:
        log.debug("Reminder check error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic reminder fire function (module-level for pickle compatibility)
# ═══════════════════════════════════════════════════════════════════════════

async def _fire_dynamic_reminder(title: str, repeat: str = "none") -> None:
    """Fire a user-set dynamic reminder. Must be module-level for APScheduler."""
    repeat_tag = f" (repeats {repeat})" if repeat != "none" else ""
    log.info("⏰ REMINDER: %s%s", title, repeat_tag)
    try:
        from bantz.agent.notifier import notifier
        if notifier.enabled:
            notifier.send(f"⏰ {title}", urgency="critical", expire_ms=0)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Briefing watcher — IDLE → active TTS trigger (#131)
# ═══════════════════════════════════════════════════════════════════════════

# Module-level state for activity transition detection
_last_activity: str = "idle"
_briefing_spoken_today: str = ""  # date string — prevents double-fire


async def _job_briefing_watcher() -> None:
    """Poll AppDetector for IDLE → active transition.

    When the user's first non-IDLE activity is detected after briefing_prep
    has cached today's briefing, speak it via TTS.

    Runs every 10s as an APScheduler interval job.
    One-shot per day: once spoken, won't trigger again.
    """
    global _last_activity, _briefing_spoken_today

    from bantz.config import config
    if not config.tts_enabled or not config.tts_auto_briefing:
        return

    today = datetime.now().date().isoformat()
    if _briefing_spoken_today == today:
        return  # Already spoken today

    # Only trigger after the briefing prep hour
    now = datetime.now()
    if now.hour < config.briefing_prep_hour:
        return

    try:
        from bantz.agent.app_detector import app_detector, Activity
        current = app_detector.get_activity_category()
        prev = _last_activity
        _last_activity = current.value

        # Detect IDLE → non-IDLE transition
        if prev == "idle" and current != Activity.IDLE:
            log.info("🔊 Activity transition: IDLE → %s — checking briefing", current.value)
            text = check_briefing_trigger()
            if text:
                _briefing_spoken_today = today
                from bantz.agent.tts import tts_engine
                await tts_engine.speak_background(text)
                log.info("🔊 Morning briefing TTS triggered (%d chars)", len(text))

                # Also send desktop notification
                try:
                    from bantz.agent.notifier import notifier
                    if notifier.enabled:
                        notifier.send("🌅 Good morning! Briefing playing…")
                except Exception:
                    pass
    except Exception as exc:
        log.debug("Briefing watcher error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Job registry — maps job_id to (async_func, description)
# ═══════════════════════════════════════════════════════════════════════════

_JOB_REGISTRY: dict[str, tuple[Callable, str]] = {
    "maintenance": (_job_maintenance, "Nightly 6-step maintenance workflow"),
    "reflection": (_job_reflection, "Nightly memory reflection workflow"),
    "overnight_poll": (_job_overnight_poll, "Overnight poll: Gmail/Calendar/Classroom → KV store"),
    "briefing_prep": (_job_briefing_prep, "Pre-fetch morning briefing data"),
    "reminder_check": (_job_reminder_check, "Check due reminders"),
    "briefing_watcher": (_job_briefing_watcher, "Watch for IDLE→active to speak briefing"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Proactive engagement job (#167)
# ═══════════════════════════════════════════════════════════════════════════

async def _job_proactive_engagement() -> None:
    """Run one proactive engagement attempt (#167).

    Called by APScheduler as an interval job with jitter.
    All guards and context gathering delegated to ProactiveEngine.
    """
    try:
        from bantz.agent.proactive import proactive_engine
        result = await proactive_engine.run()
        if result.success:
            log.info("💬 Proactive: %s", result.message[:80])
        else:
            log.debug("💬 Proactive skipped: %s", result.reason)
    except Exception as exc:
        log.warning("Proactive engagement error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Health check job (#168)
# ═══════════════════════════════════════════════════════════════════════════

async def _job_health_check() -> None:
    """Run health rule evaluation cycle (#168).

    Called by APScheduler as an interval job (default 300s / 5 min).
    Evaluates all rules; pushes an intervention on first match.
    Also checks for RL break-reward feedback.
    """
    try:
        from bantz.agent.health import health_engine
        if not health_engine.initialized:
            return

        results = health_engine.evaluate_all()
        for r in results:
            if r.fired:
                health_engine.push_intervention(r)
                log.info("🏥 Health rule fired: %s", r.rule_id)
                break  # one intervention per cycle — don't spam

        # RL reward for genuine break (Senior Fix #2)
        if health_engine.check_break_reward():
            try:
                from bantz.agent.rl_engine import rl_engine, Reward, Action
                if rl_engine.initialized:
                    rl_engine.record(Action.HEALTH_BREAK, Reward.POSITIVE)
                    log.debug("🏥 RL: break reward for HEALTH_BREAK")
            except Exception:
                pass
    except Exception as exc:
        log.warning("Health check error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# JobScheduler — main class
# ═══════════════════════════════════════════════════════════════════════════

class JobScheduler:
    """APScheduler-based job engine for Bantz daemon.

    Provides:
    - Cron-style night workflows
    - Persistent job store (SQLAlchemy → same bantz.db)
    - Retry with exponential backoff
    - Misfire grace for laptop sleep
    - Sleep-inhibit for long tasks
    - Dynamic reminder injection
    - Backward compat with core/scheduler.py
    """

    def __init__(self) -> None:
        self._scheduler = None  # AsyncIOScheduler, set in start()
        self._started = False
        self._db_url: str = ""
        self._job_history: list[dict] = []  # last N job executions

    @property
    def started(self) -> bool:
        return self._started

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self, db_path: Path, *, enable_night_jobs: bool = True) -> None:
        """Initialize and start the APScheduler.

        Uses two job stores:
        - "default" (MemoryJobStore) for built-in cron jobs — re-registered
          every startup, no need for persistence.
        - "persistent" (SQLAlchemyJobStore) for user reminders — survives
          restarts so one-shot reminders aren't lost.
        """
        if self._started:
            log.warning("JobScheduler already started")
            return

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        from apscheduler.jobstores.memory import MemoryJobStore
        from apscheduler.executors.asyncio import AsyncIOExecutor

        self._db_url = f"sqlite:///{db_path}"

        jobstores = {
            "default": MemoryJobStore(),
            "persistent": SQLAlchemyJobStore(url=self._db_url),
        }
        executors = {
            "default": AsyncIOExecutor(),
        }
        job_defaults = {
            "coalesce": True,
            "misfire_grace_time": _MISFIRE_GRACE,
            "max_instances": 1,
        }

        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
        )

        # Register built-in jobs
        if enable_night_jobs:
            self._register_night_jobs()

        # Always register reminder check (30s interval)
        self._register_reminder_check()

        # Register briefing watcher (10s interval) for TTS auto-trigger (#131)
        self._register_briefing_watcher()

        # Register proactive engagement job (#167)
        self._register_proactive()

        # Register health check job (#168)
        self._register_health_check()

        # APScheduler event listeners for history tracking
        from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
        self._scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self._scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)

        self._scheduler.start()
        self._started = True
        log.info(
            "JobScheduler started: %d jobs, store=%s",
            len(self._scheduler.get_jobs()), self._db_url,
        )

    async def shutdown(self) -> None:
        """Gracefully stop the scheduler."""
        if self._scheduler and self._started:
            self._scheduler.shutdown(wait=True)
            self._started = False
            log.info("JobScheduler shut down")

    # ── Night job registration ────────────────────────────────────────

    def _register_night_jobs(self) -> None:
        """Register cron-based night workflow jobs (in-memory store)."""
        from apscheduler.triggers.cron import CronTrigger

        for job_id, cfg in _NIGHT_JOBS.items():
            func = _JOB_REGISTRY[job_id][0]

            self._scheduler.add_job(
                func,
                CronTrigger(hour=cfg["hour"], minute=cfg["minute"]),
                id=job_id,
                name=cfg["description"],
                replace_existing=True,
                jobstore="default",
            )
            log.info("Registered cron job: %s at %02d:%02d", job_id, cfg["hour"], cfg["minute"])

        # Overnight poll: every 2h between 00-07
        for job_id, cfg in _OVERNIGHT_POLL.items():
            func = _JOB_REGISTRY[job_id][0]

            self._scheduler.add_job(
                func,
                CronTrigger(hour=cfg["hours"], minute=cfg["minute"]),
                id=job_id,
                name=cfg["description"],
                replace_existing=True,
                jobstore="default",
            )
            log.info("Registered overnight poll: %s", job_id)

    def _register_reminder_check(self) -> None:
        """Register 30s interval job for legacy reminder polling."""
        from apscheduler.triggers.interval import IntervalTrigger

        job_id = "reminder_check"

        self._scheduler.add_job(
            _job_reminder_check,
            IntervalTrigger(seconds=30),
            id=job_id,
            name="Check due reminders",
            replace_existing=True,
            jobstore="default",
        )
        log.info("Registered reminder check (30s interval)")

    def _register_briefing_watcher(self) -> None:
        """Register 10s interval job for IDLE→active TTS trigger (#131)."""
        from bantz.config import config
        if not config.tts_enabled:
            return

        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler.add_job(
            _job_briefing_watcher,
            IntervalTrigger(seconds=10),
            id="briefing_watcher",
            name="Watch for IDLE→active to speak briefing",
            replace_existing=True,
            jobstore="default",
        )
        log.info("Registered briefing watcher (10s interval, TTS)")

    def _register_proactive(self) -> None:
        """Register proactive engagement job (#167) with jitter."""
        from bantz.config import config
        if not config.proactive_enabled:
            return

        from apscheduler.triggers.interval import IntervalTrigger
        from datetime import timedelta

        hours = max(config.proactive_interval_hours, 1.0)
        jitter_sec = config.proactive_jitter_minutes * 60

        self._scheduler.add_job(
            _job_proactive_engagement,
            IntervalTrigger(
                hours=hours,
                jitter=jitter_sec,
            ),
            id="proactive_engagement",
            name="Proactive engagement (#167)",
            replace_existing=True,
            jobstore="default",
        )

        # Initialise the engine
        try:
            from bantz.agent.proactive import proactive_engine
            proactive_engine.init()
        except Exception as exc:
            log.warning("ProactiveEngine init failed: %s", exc)

        log.info(
            "Registered proactive engagement (%.1fh interval, %dm jitter, max=%d/day)",
            hours, config.proactive_jitter_minutes, config.proactive_max_daily,
        )

    def _register_health_check(self) -> None:
        """Register health check job (#168) as an interval job."""
        from bantz.config import config
        if not config.health_enabled:
            return

        from apscheduler.triggers.interval import IntervalTrigger

        interval = max(config.health_check_interval, 60)

        self._scheduler.add_job(
            _job_health_check,
            IntervalTrigger(seconds=interval),
            id="health_check",
            name="Health & break check (#168)",
            replace_existing=True,
            jobstore="default",
        )

        # Initialise the health engine
        try:
            from bantz.agent.health import health_engine
            health_engine.init()
        except Exception as exc:
            log.warning("HealthRuleEvaluator init failed: %s", exc)

        log.info(
            "Registered health check (%ds interval, thermal CPU=%.0f°C GPU=%.0f°C)",
            interval, config.health_thermal_cpu, config.health_thermal_gpu,
        )

    # ── Dynamic reminder bridge ───────────────────────────────────────

    def add_reminder(
        self,
        title: str,
        fire_at: datetime,
        repeat: str = "none",
        repeat_interval: int = 0,
    ) -> str | None:
        """Add a one-shot or repeating reminder via APScheduler.

        Returns the job ID if added, None on error.
        This bridges brain.py/reminder tool → APScheduler persistent store.
        """
        if not self._started:
            log.warning("Cannot add reminder — scheduler not started")
            return None

        job_id = f"reminder_{int(fire_at.timestamp())}_{hash(title) % 10000}"

        try:
            if repeat == "none":
                from apscheduler.triggers.date import DateTrigger
                trigger = DateTrigger(run_date=fire_at)
            elif repeat == "daily":
                from apscheduler.triggers.cron import CronTrigger
                trigger = CronTrigger(
                    hour=fire_at.hour, minute=fire_at.minute,
                )
            elif repeat == "weekly":
                from apscheduler.triggers.cron import CronTrigger
                trigger = CronTrigger(
                    day_of_week=fire_at.strftime("%a").lower()[:3],
                    hour=fire_at.hour, minute=fire_at.minute,
                )
            elif repeat == "weekdays":
                from apscheduler.triggers.cron import CronTrigger
                trigger = CronTrigger(
                    day_of_week="mon-fri",
                    hour=fire_at.hour, minute=fire_at.minute,
                )
            elif repeat == "custom" and repeat_interval > 0:
                from apscheduler.triggers.interval import IntervalTrigger
                trigger = IntervalTrigger(seconds=repeat_interval)
            else:
                from apscheduler.triggers.date import DateTrigger
                trigger = DateTrigger(run_date=fire_at)

            self._scheduler.add_job(
                _fire_dynamic_reminder,
                trigger,
                args=[title, repeat],
                id=job_id,
                name=f"Reminder: {title}",
                replace_existing=True,
                jobstore="persistent",
            )
            log.info("APScheduler reminder added: '%s' → %s", title, fire_at)
            return job_id

        except Exception as exc:
            log.error("Failed to add reminder '%s': %s", title, exc)
            return None

    # ── Job listing / CLI ─────────────────────────────────────────────

    def list_jobs(self) -> list[dict]:
        """Return all scheduled jobs with metadata."""
        if not self._started or not self._scheduler:
            return []

        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name or job.id,
                "next_run": next_run.isoformat() if next_run else "paused",
                "trigger": str(job.trigger),
            })
        return jobs

    def run_job_now(self, job_id: str) -> bool:
        """Manually trigger a job immediately."""
        if not self._started:
            return False

        # If it's a registered built-in job, run its function
        if job_id in _JOB_REGISTRY:
            func, desc = _JOB_REGISTRY[job_id]
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_run_with_retry(func, job_name=job_id))
            else:
                loop.run_until_complete(func())
            log.info("Manual trigger: %s", job_id)
            return True

        # Try modifying the job's next run time to now
        job = self._scheduler.get_job(job_id)
        if job:
            job.modify(next_run_time=datetime.now())
            log.info("Manual trigger: %s (modified next_run)", job_id)
            return True

        return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        if not self._started:
            return False
        try:
            self._scheduler.remove_job(job_id)
            return True
        except Exception:
            return False

    # ── Event listeners for history ───────────────────────────────────

    def _on_job_executed(self, event) -> None:
        self._job_history.append({
            "job_id": event.job_id,
            "time": datetime.now().isoformat(),
            "status": "ok",
        })
        # Keep last 100 entries
        if len(self._job_history) > 100:
            self._job_history = self._job_history[-100:]

    def _on_job_error(self, event) -> None:
        self._job_history.append({
            "job_id": event.job_id,
            "time": datetime.now().isoformat(),
            "status": "error",
            "exception": str(event.exception)[:200] if event.exception else "",
        })
        if len(self._job_history) > 100:
            self._job_history = self._job_history[-100:]

    # ── Stats / Doctor ────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._started:
            return {"started": False}
        jobs = self.list_jobs()
        return {
            "started": True,
            "job_count": len(jobs),
            "jobs": jobs,
            "history_count": len(self._job_history),
            "db_url": self._db_url,
        }

    def status_line(self) -> str:
        if not self._started:
            return "not started"
        jobs = self._scheduler.get_jobs() if self._scheduler else []
        ok = sum(1 for h in self._job_history[-20:] if h["status"] == "ok")
        err = sum(1 for h in self._job_history[-20:] if h["status"] == "error")
        return f"jobs={len(jobs)} ok={ok} err={err}"

    def format_jobs(self) -> str:
        """Human-readable job listing for `bantz --jobs`."""
        jobs = self.list_jobs()
        if not jobs:
            return "No scheduled jobs."

        lines = ["📅 Scheduled Jobs:"]
        for j in jobs:
            lines.append(
                f"  {j['id']:25s}  next: {j['next_run']:25s}  {j['name']}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Briefing trigger helper (for AppDetector integration)
# ═══════════════════════════════════════════════════════════════════════════

def check_briefing_trigger() -> str | None:
    """Check if a pre-fetched briefing is ready.

    Called by AppDetector/TUI when user's first activity is detected
    after 6 AM. Returns the briefing text if ready, None otherwise.
    Clears the flag after reading.
    """
    try:
        from bantz.data.sqlite_store import SQLiteKVStore
        from bantz.config import config
        kv = SQLiteKVStore(config.db_path)
        date_str = kv.get("briefing_date")
        if date_str != datetime.now().date().isoformat():
            return None
        text = kv.get("briefing_ready")
        if text:
            # Clear so it doesn't fire again
            kv.set("briefing_ready", "")
            return text
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

job_scheduler = JobScheduler()
