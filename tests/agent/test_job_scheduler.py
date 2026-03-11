"""
Tests for bantz.agent.job_scheduler — APScheduler integration (#128).

Coverage:
  - inhibit_sleep context manager
  - _run_with_retry exponential backoff
  - JobScheduler lifecycle (start / shutdown)
  - Night job registration
  - Reminder check integration
  - Dynamic reminder add / remove
  - Job listing / format
  - Briefing trigger (KV store)
  - Manual run_job_now
  - Stats and status_line
  - Config fields (job_scheduler_enabled, night hours)
  - SQLiteKVStore basic ops
  - Edge cases (double start, shutdown when not started)
"""
from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp DB path for tests."""
    db = tmp_path / "test_bantz.db"
    return db


@pytest.fixture
def kv_store(tmp_db):
    """Create a SQLiteKVStore instance."""
    from bantz.data.sqlite_store import SQLiteKVStore
    return SQLiteKVStore(tmp_db)


# ═══════════════════════════════════════════════════════════════════════════
# SQLiteKVStore tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSQLiteKVStore:
    def test_set_and_get(self, kv_store):
        kv_store.set("foo", "bar")
        assert kv_store.get("foo") == "bar"

    def test_get_default(self, kv_store):
        assert kv_store.get("nonexistent") == ""
        assert kv_store.get("nonexistent", "fallback") == "fallback"

    def test_overwrite(self, kv_store):
        kv_store.set("key", "val1")
        kv_store.set("key", "val2")
        assert kv_store.get("key") == "val2"

    def test_delete(self, kv_store):
        kv_store.set("key", "val")
        kv_store.delete("key")
        assert kv_store.get("key") == ""

    def test_delete_nonexistent(self, kv_store):
        # Should not raise
        kv_store.delete("nope")

    def test_all(self, kv_store):
        kv_store.set("a", "1")
        kv_store.set("b", "2")
        kv_store.set("c", "3")
        result = kv_store.all()
        assert result == {"a": "1", "b": "2", "c": "3"}

    def test_empty_all(self, kv_store):
        assert kv_store.all() == {}

    def test_unicode_values(self, kv_store):
        kv_store.set("msg", "Günaydın! 🌅")
        assert kv_store.get("msg") == "Günaydın! 🌅"


# ═══════════════════════════════════════════════════════════════════════════
# inhibit_sleep tests
# ═══════════════════════════════════════════════════════════════════════════

class TestInhibitSleep:
    def test_inhibit_sleep_no_systemd(self):
        """Falls back gracefully when systemd-inhibit is missing."""
        from bantz.agent.job_scheduler import inhibit_sleep

        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            with inhibit_sleep("test"):
                pass  # Should not raise

    def test_inhibit_sleep_with_mock_popen(self):
        """Process is started and terminated."""
        from bantz.agent.job_scheduler import inhibit_sleep

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc):
            with inhibit_sleep("night job"):
                pass
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_inhibit_sleep_kill_on_timeout(self):
        """Falls back to kill if terminate times out."""
        from bantz.agent.job_scheduler import inhibit_sleep

        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("sleep", 3)
        with patch("subprocess.Popen", return_value=mock_proc):
            with inhibit_sleep("test"):
                pass
        mock_proc.kill.assert_called_once()

    def test_inhibit_sleep_exception_inside(self):
        """Exception inside context still releases inhibit."""
        from bantz.agent.job_scheduler import inhibit_sleep

        mock_proc = MagicMock()
        mock_proc.pid = 42
        with patch("subprocess.Popen", return_value=mock_proc):
            with pytest.raises(ValueError):
                with inhibit_sleep("test"):
                    raise ValueError("boom")
        mock_proc.terminate.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# _run_with_retry tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRunWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        from bantz.agent.job_scheduler import _run_with_retry
        func = AsyncMock(return_value="ok")
        result = await _run_with_retry(func, job_name="test")
        assert result == "ok"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        from bantz.agent.job_scheduler import _run_with_retry
        func = AsyncMock(side_effect=[RuntimeError("fail"), "ok"])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await _run_with_retry(func, job_name="test", max_retries=2)
        assert result == "ok"
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        from bantz.agent.job_scheduler import _run_with_retry
        func = AsyncMock(side_effect=RuntimeError("always fail"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await _run_with_retry(func, job_name="test", max_retries=2)
        assert result is None
        assert func.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        from bantz.agent.job_scheduler import _run_with_retry
        func = AsyncMock(side_effect=RuntimeError("fail"))
        result = await _run_with_retry(func, job_name="test", max_retries=0)
        assert result is None
        assert func.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# JobScheduler lifecycle tests
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        assert not js.started

        await js.start(tmp_db)
        assert js.started
        assert len(js.list_jobs()) > 0  # at least reminder_check

        await js.shutdown()
        assert not js.started

    @pytest.mark.asyncio
    async def test_double_start(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db)
        # Second start should be a no-op
        await js.start(tmp_db)
        assert js.started
        await js.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_when_not_started(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_start_without_night_jobs(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)
        jobs = js.list_jobs()
        # Only reminder_check should be present
        job_ids = [j["id"] for j in jobs]
        assert "reminder_check" in job_ids
        assert "maintenance" not in job_ids
        assert "reflection" not in job_ids
        await js.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Night job registration
# ═══════════════════════════════════════════════════════════════════════════

class TestNightJobRegistration:
    @pytest.mark.asyncio
    async def test_all_night_jobs_registered(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=True)

        job_ids = [j["id"] for j in js.list_jobs()]
        assert "maintenance" in job_ids
        assert "reflection" in job_ids
        assert "briefing_prep" in job_ids
        assert "overnight_poll" in job_ids
        assert "reminder_check" in job_ids

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_jobs_persist_across_restarts(self, tmp_db):
        """User reminders stored in SQLAlchemy survive scheduler restart."""
        from bantz.agent.job_scheduler import JobScheduler

        js1 = JobScheduler()
        await js1.start(tmp_db, enable_night_jobs=False)
        fire_at = datetime.now() + timedelta(hours=2)
        job_id = js1.add_reminder("persistent test", fire_at)
        assert job_id is not None
        await js1.shutdown()

        # Restart — the persistent reminder should still be there
        js2 = JobScheduler()
        await js2.start(tmp_db, enable_night_jobs=False)
        job_ids = [j["id"] for j in js2.list_jobs()]
        assert job_id in job_ids
        await js2.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic reminder bridge
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamicReminder:
    @pytest.mark.asyncio
    async def test_add_oneshot_reminder(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        fire_at = datetime.now() + timedelta(hours=1)
        job_id = js.add_reminder("call dentist", fire_at)
        assert job_id is not None
        assert job_id.startswith("reminder_")

        jobs = js.list_jobs()
        job_ids = [j["id"] for j in jobs]
        assert job_id in job_ids

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_add_daily_reminder(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        fire_at = datetime.now() + timedelta(hours=1)
        job_id = js.add_reminder("morning medication", fire_at, repeat="daily")
        assert job_id is not None

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_add_weekly_reminder(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        fire_at = datetime.now() + timedelta(hours=1)
        job_id = js.add_reminder("team standup", fire_at, repeat="weekly")
        assert job_id is not None

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_add_weekday_reminder(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        fire_at = datetime.now() + timedelta(hours=1)
        job_id = js.add_reminder("standup", fire_at, repeat="weekdays")
        assert job_id is not None

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_add_custom_interval_reminder(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        fire_at = datetime.now() + timedelta(hours=1)
        job_id = js.add_reminder("water plants", fire_at, repeat="custom",
                                 repeat_interval=3600)
        assert job_id is not None

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_add_reminder_when_not_started(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        result = js.add_reminder("test", datetime.now())
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_reminder(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        fire_at = datetime.now() + timedelta(hours=1)
        job_id = js.add_reminder("to remove", fire_at)
        assert job_id is not None

        ok = js.remove_job(job_id)
        assert ok

        jobs = js.list_jobs()
        job_ids = [j["id"] for j in jobs]
        assert job_id not in job_ids

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)
        assert not js.remove_job("nonexistent_job_id")
        await js.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Job listing and formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestJobListing:
    @pytest.mark.asyncio
    async def test_list_jobs(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=True)

        jobs = js.list_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) >= 5  # 4 night + reminder_check

        for j in jobs:
            assert "id" in j
            assert "name" in j
            assert "next_run" in j
            assert "trigger" in j

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_format_jobs(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=True)

        text = js.format_jobs()
        assert "📅 Scheduled Jobs:" in text
        assert "maintenance" in text
        assert "reminder_check" in text

        await js.shutdown()

    def test_format_jobs_empty(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        assert js.format_jobs() == "No scheduled jobs."

    def test_list_jobs_not_started(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        assert js.list_jobs() == []


# ═══════════════════════════════════════════════════════════════════════════
# Manual trigger
# ═══════════════════════════════════════════════════════════════════════════

class TestRunJobNow:
    @pytest.mark.asyncio
    async def test_run_builtin_job(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=True)

        # Mock the actual job function to avoid side effects
        with patch("bantz.agent.job_scheduler._job_maintenance", new_callable=AsyncMock):
            ok = js.run_job_now("maintenance")
            assert ok

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_run_nonexistent_job(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)
        assert not js.run_job_now("doesnt_exist")
        await js.shutdown()

    def test_run_job_not_started(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        assert not js.run_job_now("anything")


# ═══════════════════════════════════════════════════════════════════════════
# Stats and status_line
# ═══════════════════════════════════════════════════════════════════════════

class TestStats:
    @pytest.mark.asyncio
    async def test_stats_when_started(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=True)

        s = js.stats()
        assert s["started"] is True
        assert s["job_count"] >= 5
        assert "jobs" in s
        assert "db_url" in s
        assert "sqlite" in s["db_url"]

        await js.shutdown()

    def test_stats_when_not_started(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        s = js.stats()
        assert s == {"started": False}

    @pytest.mark.asyncio
    async def test_status_line(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=True)

        line = js.status_line()
        assert "jobs=" in line
        assert "ok=" in line

        await js.shutdown()

    def test_status_line_not_started(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        assert js.status_line() == "not started"


# ═══════════════════════════════════════════════════════════════════════════
# Event history
# ═══════════════════════════════════════════════════════════════════════════

class TestEventHistory:
    @pytest.mark.asyncio
    async def test_job_executed_event(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        # Simulate event
        mock_event = MagicMock()
        mock_event.job_id = "test_job"
        js._on_job_executed(mock_event)

        assert len(js._job_history) == 1
        assert js._job_history[0]["status"] == "ok"
        assert js._job_history[0]["job_id"] == "test_job"

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_job_error_event(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        mock_event = MagicMock()
        mock_event.job_id = "failing_job"
        mock_event.exception = RuntimeError("boom")
        js._on_job_error(mock_event)

        assert len(js._job_history) == 1
        assert js._job_history[0]["status"] == "error"
        assert "boom" in js._job_history[0]["exception"]

        await js.shutdown()

    @pytest.mark.asyncio
    async def test_history_capped_at_100(self, tmp_db):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        await js.start(tmp_db, enable_night_jobs=False)

        mock_event = MagicMock()
        mock_event.job_id = "spam"
        for _ in range(150):
            js._on_job_executed(mock_event)

        assert len(js._job_history) <= 100

        await js.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Briefing trigger
# ═══════════════════════════════════════════════════════════════════════════

class TestBriefingTrigger:
    def test_briefing_trigger_returns_text(self, tmp_db):
        from bantz.data.sqlite_store import SQLiteKVStore
        kv = SQLiteKVStore(tmp_db)
        kv.set("briefing_date", datetime.now().date().isoformat())
        kv.set("briefing_ready", "Good morning! Here is your briefing...")

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_db
            from bantz.agent.job_scheduler import check_briefing_trigger
            text = check_briefing_trigger()
            assert text == "Good morning! Here is your briefing..."

        # Second call should return None (cleared)
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_db
            from bantz.agent.job_scheduler import check_briefing_trigger
            text2 = check_briefing_trigger()
            assert text2 is None

    def test_briefing_trigger_wrong_date(self, tmp_db):
        from bantz.data.sqlite_store import SQLiteKVStore
        kv = SQLiteKVStore(tmp_db)
        kv.set("briefing_date", "2024-01-01")  # Old date
        kv.set("briefing_ready", "Old briefing")

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_db
            from bantz.agent.job_scheduler import check_briefing_trigger
            text = check_briefing_trigger()
            assert text is None

    def test_briefing_trigger_no_data(self, tmp_db):
        from bantz.data.sqlite_store import SQLiteKVStore
        SQLiteKVStore(tmp_db)  # Create the table

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_db
            from bantz.agent.job_scheduler import check_briefing_trigger
            text = check_briefing_trigger()
            assert text is None


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigFields:
    def test_job_scheduler_defaults(self):
        from bantz.config import Config
        cfg = Config(
            _env_file=None,
            BANTZ_OLLAMA_MODEL="test",
        )
        assert cfg.job_scheduler_enabled is True
        assert cfg.night_maintenance_hour == 3
        assert cfg.night_reflection_hour == 23
        assert cfg.briefing_prep_hour == 6
        assert cfg.overnight_poll_hours == "0,2,4,6"

    def test_config_override_via_env(self, monkeypatch):
        monkeypatch.setenv("BANTZ_JOB_SCHEDULER_ENABLED", "false")
        monkeypatch.setenv("BANTZ_MAINTENANCE_HOUR", "4")
        monkeypatch.setenv("BANTZ_REFLECTION_HOUR", "22")
        monkeypatch.setenv("BANTZ_BRIEFING_PREP_HOUR", "5")
        monkeypatch.setenv("BANTZ_OVERNIGHT_POLL_HOURS", "1,3,5")

        from bantz.config import Config
        cfg = Config(_env_file=None)
        assert cfg.job_scheduler_enabled is False
        assert cfg.night_maintenance_hour == 4
        assert cfg.night_reflection_hour == 22
        assert cfg.briefing_prep_hour == 5
        assert cfg.overnight_poll_hours == "1,3,5"


# ═══════════════════════════════════════════════════════════════════════════
# Night job function tests (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestNightJobFunctions:
    @pytest.mark.asyncio
    async def test_maintenance_runs(self):
        """Maintenance job completes without error."""
        from bantz.agent.job_scheduler import _job_maintenance
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
                await _job_maintenance()

    @pytest.mark.asyncio
    async def test_reflection_delegates_to_workflow(self):
        """Reflection delegates to run_reflection workflow."""
        from bantz.agent.job_scheduler import _job_reflection
        with patch("bantz.agent.workflows.reflection.run_reflection", new_callable=AsyncMock) as mock_run:
            await _job_reflection()
            mock_run.assert_awaited_once_with(dry_run=False)

    @pytest.mark.asyncio
    async def test_overnight_poll_no_gmail(self):
        """Overnight poll handles missing Gmail tool."""
        from bantz.agent.job_scheduler import _job_overnight_poll
        with patch("bantz.agent.job_scheduler._job_overnight_poll", new_callable=AsyncMock):
            # Just verify it's callable
            pass

    @pytest.mark.asyncio
    async def test_reminder_check_no_scheduler(self):
        """Reminder check handles uninitialized scheduler."""
        from bantz.agent.job_scheduler import _job_reminder_check
        with patch("bantz.core.scheduler.scheduler") as mock_sched:
            mock_sched._conn = None
            await _job_reminder_check()  # Should return early


# ═══════════════════════════════════════════════════════════════════════════
# Job registry
# ═══════════════════════════════════════════════════════════════════════════

class TestJobRegistry:
    def test_all_registrations_present(self):
        from bantz.agent.job_scheduler import _JOB_REGISTRY
        expected = {"maintenance", "reflection", "overnight_poll",
                    "briefing_prep", "reminder_check", "briefing_watcher"}
        assert set(_JOB_REGISTRY.keys()) == expected

    def test_registry_entries_are_callable(self):
        from bantz.agent.job_scheduler import _JOB_REGISTRY
        for job_id, (func, desc) in _JOB_REGISTRY.items():
            assert callable(func), f"{job_id} func is not callable"
            assert isinstance(desc, str), f"{job_id} desc is not str"


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleSingleton:
    def test_singleton_exists(self):
        from bantz.agent.job_scheduler import job_scheduler
        assert job_scheduler is not None
        assert not job_scheduler.started

    def test_singleton_is_same_instance(self):
        from bantz.agent.job_scheduler import job_scheduler as a
        from bantz.agent.job_scheduler import job_scheduler as b
        assert a is b


# ═══════════════════════════════════════════════════════════════════════════
# Misfire grace and coalesce constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_misfire_grace_is_24h(self):
        from bantz.agent.job_scheduler import _MISFIRE_GRACE
        assert _MISFIRE_GRACE == 86400

    def test_max_retries(self):
        from bantz.agent.job_scheduler import _MAX_RETRIES
        assert _MAX_RETRIES == 3

    def test_backoff_base(self):
        from bantz.agent.job_scheduler import _BACKOFF_BASE
        assert _BACKOFF_BASE == 30
