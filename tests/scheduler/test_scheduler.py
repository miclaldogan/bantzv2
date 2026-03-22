"""Tests for Scheduler — APScheduler clean API wrap (#295)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _fresh_scheduler():
    """Return a new Scheduler backed by MemoryJobStore (no Redis needed)."""
    from bantz.scheduler.scheduler import Scheduler
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    s = Scheduler.__new__(Scheduler)
    s._scheduler = AsyncIOScheduler(jobstores={"default": MemoryJobStore()})
    return s


def _noop():
    pass


# ── ScheduledJob dataclass ────────────────────────────────────────────────────

class TestScheduledJob:

    def test_fields_accessible(self):
        from bantz.scheduler.scheduler import ScheduledJob
        job = ScheduledJob(id="j1", next_run=None, trigger_type="cron")
        assert job.id == "j1"
        assert job.next_run is None
        assert job.trigger_type == "cron"


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

class TestLifecycle:

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        s = _fresh_scheduler()
        try:
            s.start()
            assert s.running is True
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_shutdown_clears_running(self):
        s = _fresh_scheduler()
        s.start()
        assert s.running is True
        s.shutdown(wait=False)
        # Give the event loop a tick to process shutdown
        await asyncio.sleep(0.05)
        assert s.running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        s = _fresh_scheduler()
        try:
            s.start()
            s.start()  # idempotent — must not raise
            assert s.running
        finally:
            s.shutdown(wait=False)


# ── add_cron ──────────────────────────────────────────────────────────────────

class TestAddCron:

    def test_add_cron_appears_in_list_jobs(self):
        s = _fresh_scheduler()
        s.add_cron(_noop, "0 8 * * *", "morning")
        jobs = s.list_jobs()
        ids = [j.id for j in jobs]
        assert "morning" in ids

    def test_add_cron_trigger_type_is_cron(self):
        s = _fresh_scheduler()
        s.add_cron(_noop, "0 8 * * *", "morning")
        job = s.get_job("morning")
        assert job is not None
        assert job.trigger_type == "cron"

    def test_add_cron_invalid_expr_raises(self):
        s = _fresh_scheduler()
        with pytest.raises(ValueError):
            s.add_cron(_noop, "0 8 * *", "bad_expr")  # only 4 fields

    @pytest.mark.asyncio
    async def test_add_cron_replace_existing(self):
        """replace_existing deduplicates once the scheduler is running."""
        s = _fresh_scheduler()
        s.start()
        try:
            s.add_cron(_noop, "0 8 * * *", "j_replace")
            s.add_cron(_noop, "0 9 * * *", "j_replace", replace_existing=True)
            assert len([x for x in s.list_jobs() if x.id == "j_replace"]) == 1
        finally:
            s.shutdown(wait=False)
            await asyncio.sleep(0.05)


# ── add_once ──────────────────────────────────────────────────────────────────

class TestAddOnce:

    def test_add_once_appears_in_list_jobs(self):
        s = _fresh_scheduler()
        run_at = datetime.now() + timedelta(hours=1)
        s.add_once(_noop, run_at, "oneshot")
        ids = [j.id for j in s.list_jobs()]
        assert "oneshot" in ids

    def test_add_once_trigger_type_is_date(self):
        s = _fresh_scheduler()
        run_at = datetime.now() + timedelta(hours=1)
        s.add_once(_noop, run_at, "oneshot")
        job = s.get_job("oneshot")
        assert job is not None
        assert job.trigger_type == "date"

    def test_add_once_get_job_returns_scheduled_job(self):
        s = _fresh_scheduler()
        run_at = datetime.now() + timedelta(hours=2)
        s.add_once(_noop, run_at, "oneshot2")
        job = s.get_job("oneshot2")
        assert job is not None
        assert job.id == "oneshot2"
        assert job.trigger_type == "date"
        # next_run is None until scheduler starts — that's acceptable


# ── add_interval ──────────────────────────────────────────────────────────────

class TestAddInterval:

    def test_add_interval_appears_in_list_jobs(self):
        s = _fresh_scheduler()
        s.add_interval(_noop, 60, "heartbeat")
        ids = [j.id for j in s.list_jobs()]
        assert "heartbeat" in ids

    def test_add_interval_trigger_type_is_interval(self):
        s = _fresh_scheduler()
        s.add_interval(_noop, 60, "heartbeat")
        job = s.get_job("heartbeat")
        assert job is not None
        assert job.trigger_type == "interval"

    def test_add_interval_zero_seconds_raises(self):
        s = _fresh_scheduler()
        with pytest.raises(ValueError):
            s.add_interval(_noop, 0, "bad")

    def test_add_interval_negative_raises(self):
        s = _fresh_scheduler()
        with pytest.raises(ValueError):
            s.add_interval(_noop, -5, "bad")


# ── list_jobs ─────────────────────────────────────────────────────────────────

class TestListJobs:

    def test_empty_scheduler_returns_empty_list(self):
        s = _fresh_scheduler()
        assert s.list_jobs() == []

    def test_multiple_jobs_all_returned(self):
        s = _fresh_scheduler()
        s.add_cron(_noop, "0 8 * * *", "j1")
        s.add_interval(_noop, 60, "j2")
        s.add_once(_noop, datetime.now() + timedelta(hours=1), "j3")
        ids = {j.id for j in s.list_jobs()}
        assert ids == {"j1", "j2", "j3"}


# ── cancel ────────────────────────────────────────────────────────────────────

class TestCancel:

    def test_cancel_existing_job_returns_true(self):
        s = _fresh_scheduler()
        s.add_cron(_noop, "0 8 * * *", "to_cancel")
        result = s.cancel("to_cancel")
        assert result is True
        assert s.get_job("to_cancel") is None

    def test_cancel_nonexistent_returns_false(self):
        s = _fresh_scheduler()
        result = s.cancel("does_not_exist")
        assert result is False

    def test_cancel_removes_from_list_jobs(self):
        s = _fresh_scheduler()
        s.add_cron(_noop, "0 8 * * *", "j")
        s.cancel("j")
        ids = [job.id for job in s.list_jobs()]
        assert "j" not in ids


# ── Error logging ─────────────────────────────────────────────────────────────

class TestErrorLogging:

    def test_failed_job_writes_to_log(self, tmp_path):
        from bantz.scheduler.scheduler import _make_wrapped
        import bantz.scheduler.scheduler as mod

        original = mod._SCHEDULER_LOG
        mod._SCHEDULER_LOG = tmp_path / "scheduler.log"
        try:
            def bad_func():
                raise ValueError("intentional error")

            wrapped = _make_wrapped(bad_func, "bad_job")
            with pytest.raises(ValueError):
                wrapped()

            log_content = mod._SCHEDULER_LOG.read_text()
            assert "bad_job" in log_content
            assert "intentional error" in log_content
        finally:
            mod._SCHEDULER_LOG = original


# ── Jobstore fallback ─────────────────────────────────────────────────────────

class TestJobstoreFallback:

    def test_memory_fallback_when_redis_unavailable(self):
        """_make_jobstore() must return MemoryJobStore when Redis import fails."""
        from apscheduler.jobstores.memory import MemoryJobStore
        from bantz.scheduler.scheduler import Scheduler

        with patch.dict("sys.modules", {"apscheduler.jobstores.redis": None}):
            store = Scheduler._make_jobstore()
        assert isinstance(store, MemoryJobStore)


# ── Module singleton ──────────────────────────────────────────────────────────

class TestSingleton:

    def test_scheduler_singleton_exists(self):
        from bantz.scheduler.scheduler import scheduler, Scheduler
        assert isinstance(scheduler, Scheduler)

    def test_singleton_identity(self):
        from bantz.scheduler.scheduler import scheduler as a
        from bantz.scheduler.scheduler import scheduler as b
        assert a is b
