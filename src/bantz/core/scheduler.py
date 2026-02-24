"""
Bantz v3 — Task Scheduler (APScheduler)

Manages recurring jobs, night tasks, and reminders.
Jobs are stored persistently in SQLite (same DB as memory).

Usage:
    from bantz.core.scheduler import scheduler
    scheduler.start()
    scheduler.add_daily_briefing()
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BantzScheduler:
    """APScheduler wrapper for Bantz task queue."""

    def __init__(self) -> None:
        self._scheduler = None
        self._started = False

    def _get_scheduler(self):
        if self._scheduler is None:
            try:
                from apscheduler.schedulers.asyncio import AsyncIOScheduler
                from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
                from bantz.config import config

                config.ensure_dirs()
                db_url = f"sqlite:///{config.db_path}"

                jobstores = {
                    "default": SQLAlchemyJobStore(url=db_url, tablename="apscheduler_jobs"),
                }
                self._scheduler = AsyncIOScheduler(
                    jobstores=jobstores,
                    timezone="UTC",
                )
            except ImportError:
                logger.warning("APScheduler not installed. Task queue disabled.")
        return self._scheduler

    def start(self) -> bool:
        """Start the scheduler. Returns True if successful."""
        sched = self._get_scheduler()
        if sched and not self._started:
            try:
                sched.start()
                self._started = True
                logger.info("Scheduler started")
                return True
            except Exception as e:
                logger.error("Scheduler start failed: %s", e)
        return False

    def stop(self) -> None:
        if self._scheduler and self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False

    def add_daily_briefing(self, hour: int = 8, minute: int = 0) -> None:
        """Schedule a daily morning briefing at HH:MM."""
        sched = self._get_scheduler()
        if not sched:
            return
        sched.add_job(
            self._run_briefing,
            "cron",
            hour=hour,
            minute=minute,
            id="daily_briefing",
            replace_existing=True,
            name="Daily morning briefing",
        )
        logger.info("Daily briefing scheduled at %02d:%02d", hour, minute)

    def add_gmail_digest(self, hour: int = 8, minute: int = 0) -> None:
        """Schedule a daily Gmail digest."""
        sched = self._get_scheduler()
        if not sched:
            return
        sched.add_job(
            self._run_gmail_digest,
            "cron",
            hour=hour,
            minute=minute,
            id="gmail_digest",
            replace_existing=True,
            name="Gmail daily digest",
        )

    def add_one_shot(
        self,
        func: Callable,
        run_at: datetime,
        job_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Schedule a one-time job at a specific datetime."""
        sched = self._get_scheduler()
        if not sched:
            return ""
        job = sched.add_job(
            func,
            "date",
            run_date=run_at,
            id=job_id or f"oneshot_{run_at.timestamp():.0f}",
            replace_existing=True,
            kwargs=kwargs,
        )
        return job.id

    def list_jobs(self) -> list[dict]:
        """Return all scheduled jobs as dicts."""
        sched = self._get_scheduler()
        if not sched:
            return []
        jobs = []
        for job in sched.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name or job.id,
                "next_run": next_run.isoformat() if next_run else "paused",
            })
        return jobs

    def remove_job(self, job_id: str) -> bool:
        sched = self._get_scheduler()
        if not sched:
            return False
        try:
            sched.remove_job(job_id)
            return True
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._started

    # ── Job implementations ───────────────────────────────────────────────

    async def _run_briefing(self) -> None:
        """Run morning briefing and send via Telegram."""
        try:
            from bantz.core.briefing import briefing
            text = await briefing.generate()
            await self._send_telegram(text)
            logger.info("Daily briefing sent")
        except Exception as e:
            logger.error("Briefing job failed: %s", e)

    async def _run_gmail_digest(self) -> None:
        """Run Gmail summary and send via Telegram."""
        try:
            from bantz.tools.gmail import GmailTool
            result = await GmailTool().execute(action="summary")
            if result.success and result.output:
                await self._send_telegram(f"📬 Gmail digest:\n{result.output}")
        except Exception as e:
            logger.error("Gmail digest job failed: %s", e)

    async def _send_telegram(self, message: str) -> None:
        """Send message via Telegram bot if configured."""
        try:
            from bantz.config import config
            if not config.telegram_bot_token:
                return
            import httpx
            allowed = [u.strip() for u in config.telegram_allowed_users.split(",") if u.strip()]
            if not allowed:
                return
            for user_id in allowed:
                url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(url, json={"chat_id": user_id, "text": message})
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)


scheduler = BantzScheduler()
