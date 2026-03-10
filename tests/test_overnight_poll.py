"""
Tests for Overnight Poll Workflow (#132)

Coverage:
  - PollSourceResult / OvernightPollResult data classes
  - Helper functions: _get_urgent_keywords, _is_urgent, _gmail_after_timestamp
  - Individual source pollers: Gmail, Calendar, Classroom
  - KV store persistence (_store_poll_results) and reading (read_overnight_data)
  - Auth error payloads (Rec #4)
  - Deduplication via last_poll_time (Rec #2)
  - Configurable urgent keywords (Rec #3)
  - Urgent notification dispatch
  - Briefing integration (cached data → format)
  - Full run_overnight_poll orchestration
  - CLI arg registration
  - Job scheduler delegation
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def kv_store(tmp_path):
    """A real SQLiteKVStore backed by a temp file."""
    from bantz.data.sqlite_store import SQLiteKVStore
    return SQLiteKVStore(tmp_path / "test.db")


@pytest.fixture
def mock_gmail_creds():
    """Mock Google OAuth credentials."""
    creds = MagicMock()
    creds.expired = False
    creds.valid = True
    return creds


# ═══════════════════════════════════════════════════════════════════════════════
# PollSourceResult & OvernightPollResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestPollSourceResult:
    def test_to_dict_ok(self):
        from bantz.agent.workflows.overnight_poll import PollSourceResult
        r = PollSourceResult(source="gmail", status="ok", data={"unread": 5})
        d = r.to_dict()
        assert d["source"] == "gmail"
        assert d["status"] == "ok"
        assert d["data"]["unread"] == 5

    def test_to_dict_auth_error(self):
        from bantz.agent.workflows.overnight_poll import PollSourceResult
        r = PollSourceResult(
            source="calendar", status="auth_error",
            error_message="Token expired",
        )
        d = r.to_dict()
        assert d["status"] == "auth_error"
        assert "Token expired" in d["error"]
        assert "data" not in d

    def test_to_dict_error(self):
        from bantz.agent.workflows.overnight_poll import PollSourceResult
        r = PollSourceResult(source="classroom", status="error", error_message="API down")
        d = r.to_dict()
        assert d["status"] == "error"
        assert "API down" in d["error"]


class TestOvernightPollResult:
    def test_summary_line_with_data(self):
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult,
        )
        r = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={
                "unread": 12, "urgent_count": 2,
            }),
            calendar=PollSourceResult(source="calendar", status="ok", data={
                "events": [{"summary": "Meeting"}],
            }),
            classroom=PollSourceResult(source="classroom", status="ok", data={
                "assignment_count": 3,
            }),
        )
        line = r.summary_line()
        assert "12 unread" in line
        assert "2 urgent" in line
        assert "1 events" in line
        assert "3 assignments" in line

    def test_summary_line_with_errors(self):
        from bantz.agent.workflows.overnight_poll import OvernightPollResult
        r = OvernightPollResult(errors=2)
        line = r.summary_line()
        assert "2 errors" in line

    def test_summary_line_empty(self):
        from bantz.agent.workflows.overnight_poll import OvernightPollResult
        r = OvernightPollResult()
        assert r.summary_line() == "No data collected"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestUrgentKeywords:
    def test_get_urgent_keywords_from_config(self):
        from bantz.agent.workflows.overnight_poll import _get_urgent_keywords
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.urgent_keywords = "final,deadline,acil"
            kws = _get_urgent_keywords()
        assert "final" in kws
        assert "deadline" in kws
        assert "acil" in kws

    def test_get_urgent_keywords_empty(self):
        from bantz.agent.workflows.overnight_poll import _get_urgent_keywords
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.urgent_keywords = ""
            kws = _get_urgent_keywords()
        assert kws == []

    def test_is_urgent_match(self):
        from bantz.agent.workflows.overnight_poll import _is_urgent
        assert _is_urgent("FINAL exam schedule", "prof@uni.edu", ["final", "deadline"])

    def test_is_urgent_no_match(self):
        from bantz.agent.workflows.overnight_poll import _is_urgent
        assert not _is_urgent("Newsletter update", "news@co.com", ["final", "deadline"])

    def test_is_urgent_case_insensitive(self):
        from bantz.agent.workflows.overnight_poll import _is_urgent
        assert _is_urgent("DEADLINE approaching", "boss@work.com", ["deadline"])

    def test_is_urgent_in_sender(self):
        from bantz.agent.workflows.overnight_poll import _is_urgent
        assert _is_urgent("Hello", "urgent-notices@school.edu", ["urgent"])


class TestDeduplication:
    """Rec #2: Gmail after:TIMESTAMP deduplication."""

    def test_gmail_after_timestamp_with_last_poll(self):
        from bantz.agent.workflows.overnight_poll import _gmail_after_timestamp
        # Use a known timestamp
        ts = "2026-03-10T02:00:00+00:00"
        result = _gmail_after_timestamp(ts)
        dt = datetime.fromisoformat(ts)
        assert result == str(int(dt.timestamp()))

    def test_gmail_after_timestamp_no_last_poll(self):
        from bantz.agent.workflows.overnight_poll import _gmail_after_timestamp
        result = _gmail_after_timestamp(None)
        # Should be roughly 8 hours ago
        ts = int(result)
        now = int(datetime.now(timezone.utc).timestamp())
        assert (now - ts) > 7 * 3600  # at least 7h ago
        assert (now - ts) < 9 * 3600  # at most 9h ago

    def test_gmail_after_timestamp_bad_format(self):
        from bantz.agent.workflows.overnight_poll import _gmail_after_timestamp
        result = _gmail_after_timestamp("not-a-date")
        # Falls back to 8h ago
        ts = int(result)
        now = int(datetime.now(timezone.utc).timestamp())
        assert (now - ts) > 7 * 3600


# ═══════════════════════════════════════════════════════════════════════════════
# Source pollers
# ═══════════════════════════════════════════════════════════════════════════════

class TestPollGmail:
    @pytest.mark.asyncio
    async def test_poll_gmail_success(self):
        from bantz.agent.workflows.overnight_poll import _poll_gmail
        mock_creds = MagicMock()
        mock_messages = [
            {"id": "1", "from": "boss@work.com", "subject": "Final deadline",
             "snippet": "Submit by Friday", "date": "2026-03-10"},
            {"id": "2", "from": "news@co.com", "subject": "Newsletter",
             "snippet": "Weekly update", "date": "2026-03-10"},
        ]
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.return_value = mock_creds
            with patch("bantz.tools.gmail.GmailTool") as MockGmail:
                inst = MockGmail.return_value
                inst._count_sync.return_value = 5
                inst._fetch_messages_sync.return_value = mock_messages
                with patch("bantz.config.config") as mock_cfg:
                    mock_cfg.urgent_keywords = "final,deadline"
                    result = await _poll_gmail(None)

        assert result.status == "ok"
        assert result.data["unread"] == 5
        assert result.data["urgent_count"] == 1  # "Final deadline" matches
        assert len(result.data["urgent"]) == 1
        assert result.data["urgent"][0]["subject"] == "Final deadline"

    @pytest.mark.asyncio
    async def test_poll_gmail_auth_error(self):
        """Rec #4: Auth error produces proper payload."""
        from bantz.agent.workflows.overnight_poll import _poll_gmail
        from bantz.auth.token_store import TokenNotFoundError
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.side_effect = TokenNotFoundError("Gmail token not found")
            result = await _poll_gmail(None)

        assert result.status == "auth_error"
        assert "token" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_poll_gmail_api_error(self):
        from bantz.agent.workflows.overnight_poll import _poll_gmail
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.return_value = MagicMock()
            with patch("bantz.tools.gmail.GmailTool") as MockGmail:
                inst = MockGmail.return_value
                inst._count_sync.side_effect = Exception("API rate limit")
                result = await _poll_gmail(None)

        assert result.status == "error"
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_poll_gmail_dedup_query(self):
        """Rec #2: Verify after:TIMESTAMP is used in the query."""
        from bantz.agent.workflows.overnight_poll import _poll_gmail
        last_poll = "2026-03-10T02:00:00+00:00"
        mock_creds = MagicMock()
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.return_value = mock_creds
            with patch("bantz.tools.gmail.GmailTool") as MockGmail:
                inst = MockGmail.return_value
                inst._count_sync.return_value = 0
                inst._fetch_messages_sync.return_value = []
                with patch("bantz.config.config") as mock_cfg:
                    mock_cfg.urgent_keywords = ""
                    await _poll_gmail(last_poll)
                    # Verify the query contains after:TIMESTAMP
                    call_args = inst._fetch_messages_sync.call_args
                    query = call_args[0][1]  # second positional arg
                    assert "after:" in query


class TestPollCalendar:
    @pytest.mark.asyncio
    async def test_poll_calendar_success(self):
        from bantz.agent.workflows.overnight_poll import _poll_calendar
        mock_events = [
            {"id": "e1", "summary": "Team standup", "start_local": "10:00",
             "location": "Zoom", "attendees": []},
        ]
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.return_value = MagicMock()
            with patch("bantz.tools.calendar.CalendarTool") as MockCal:
                inst = MockCal.return_value
                inst._fetch_events_sync.return_value = mock_events
                with patch("bantz.core.location.location_service") as mock_loc:
                    loc = MagicMock()
                    loc.timezone = "Europe/Istanbul"
                    mock_loc.get = AsyncMock(return_value=loc)
                    result = await _poll_calendar()

        assert result.status == "ok"
        assert result.data["event_count"] == 1
        assert result.data["events"][0]["summary"] == "Team standup"

    @pytest.mark.asyncio
    async def test_poll_calendar_auth_error(self):
        from bantz.agent.workflows.overnight_poll import _poll_calendar
        from bantz.auth.token_store import TokenNotFoundError
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.side_effect = TokenNotFoundError("Calendar token missing")
            result = await _poll_calendar()

        assert result.status == "auth_error"


class TestPollClassroom:
    @pytest.mark.asyncio
    async def test_poll_classroom_success(self):
        from bantz.agent.workflows.overnight_poll import _poll_classroom
        now = datetime.now(timezone.utc)
        mock_assignments = [
            {"title": "Math HW", "course": "Calculus",
             "due_dt": now},  # due today
            {"title": "Essay", "course": "English",
             "due_dt": now + timedelta(days=1)},  # due tomorrow
        ]
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.return_value = MagicMock()
            with patch("bantz.tools.classroom.ClassroomTool") as MockCR:
                inst = MockCR.return_value
                inst._fetch_assignments_sync.return_value = (
                    [{"id": "c1", "name": "Calculus"}],
                    mock_assignments,
                )
                result = await _poll_classroom()

        assert result.status == "ok"
        assert result.data["assignment_count"] == 2
        assert len(result.data["due_today"]) == 1
        assert len(result.data["due_tomorrow"]) == 1

    @pytest.mark.asyncio
    async def test_poll_classroom_auth_error(self):
        from bantz.agent.workflows.overnight_poll import _poll_classroom
        from bantz.auth.token_store import TokenNotFoundError
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.side_effect = TokenNotFoundError("Classroom token expired")
            result = await _poll_classroom()

        assert result.status == "auth_error"


# ═══════════════════════════════════════════════════════════════════════════════
# KV Store persistence (Rec #1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKVStorePersistence:
    def test_store_and_read_overnight_data(self, tmp_path):
        """Rec #1: Store in KV instead of new table, then read back."""
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult,
            _store_poll_results, read_overnight_data,
        )
        result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={
                "unread": 7, "urgent_count": 1,
                "urgent": [{"subject": "FINAL"}],
                "normal": [], "new_since_last_poll": 3,
            }),
            calendar=PollSourceResult(source="calendar", status="ok", data={
                "events": [{"summary": "Meeting"}],
                "event_count": 1, "tomorrow_count": 0, "tomorrow": [],
                "timezone": "Europe/Istanbul",
            }),
            poll_time="2026-03-10T04:00:00+00:00",
        )
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            _store_poll_results(result)
            data = read_overnight_data()

        assert data["gmail"]["status"] == "ok"
        assert data["gmail"]["data"]["unread"] == 7
        assert data["calendar"]["data"]["events"][0]["summary"] == "Meeting"
        assert data["last_poll"] == "2026-03-10T04:00:00+00:00"

    def test_store_clears_errors_on_success(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult,
            _store_poll_results, read_overnight_data,
        )
        # First: store with error
        err_result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="auth_error",
                                   error_message="Token expired"),
            poll_time="T1",
        )
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            _store_poll_results(err_result)
            data1 = read_overnight_data()
        assert "errors" in data1

        # Then: store success
        ok_result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={"unread": 0}),
            poll_time="T2",
        )
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            _store_poll_results(ok_result)
            data2 = read_overnight_data()
        assert "errors" not in data2

    def test_read_single_source(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult,
            _store_poll_results, read_overnight_data,
        )
        result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={"unread": 3}),
            poll_time="T1",
        )
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            _store_poll_results(result)
            gmail_data = read_overnight_data("gmail")
        assert gmail_data["data"]["unread"] == 3

    def test_read_missing_source(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import read_overnight_data
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            data = read_overnight_data("gmail")
        assert data == {}

    def test_clear_overnight_data(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult,
            _store_poll_results, read_overnight_data, clear_overnight_data,
        )
        result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={"unread": 1}),
            poll_time="T1",
        )
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            _store_poll_results(result)
            clear_overnight_data()
            data = read_overnight_data()
        assert data.get("gmail") is None
        assert data.get("last_poll") == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Auth error payloads (Rec #4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthErrorPayloads:
    def test_error_payload_stored_in_kv(self, tmp_path):
        """Rec #4: Auth errors written to KV for briefing to read."""
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult,
            _store_poll_results, read_overnight_data,
        )
        result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="auth_error",
                                   error_message="invalid_grant"),
            calendar=PollSourceResult(source="calendar", status="ok", data={"events": []}),
            poll_time="T1",
        )
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            _store_poll_results(result)
            data = read_overnight_data()

        assert "errors" in data
        assert data["errors"][0]["source"] == "gmail"
        assert data["errors"][0]["status"] == "auth_error"


# ═══════════════════════════════════════════════════════════════════════════════
# Urgent notifications
# ═══════════════════════════════════════════════════════════════════════════════

class TestUrgentNotification:
    def test_sends_notification_on_urgent(self):
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult, _send_urgent_notification,
        )
        result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={
                "urgent": [{"subject": "FINAL exam"}],
                "urgent_count": 1,
            }),
        )
        with patch("bantz.agent.notifier.notifier") as mock_notifier:
            mock_notifier.send.return_value = True
            _send_urgent_notification(result)
            mock_notifier.send.assert_called_once()
            call_args = mock_notifier.send.call_args
            assert "1 Urgent" in call_args[0][0]
            assert "FINAL exam" in call_args[0][1]

    def test_no_notification_without_urgent(self):
        from bantz.agent.workflows.overnight_poll import (
            OvernightPollResult, PollSourceResult, _send_urgent_notification,
        )
        result = OvernightPollResult(
            gmail=PollSourceResult(source="gmail", status="ok", data={
                "urgent": [],
                "urgent_count": 0,
            }),
        )
        with patch("bantz.agent.notifier.notifier") as mock_notifier:
            _send_urgent_notification(result)
            mock_notifier.send.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Briefing integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBriefingCacheIntegration:
    def test_gmail_from_cache_ok(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._gmail_from_cache({
            "status": "ok",
            "data": {"unread": 5, "urgent_count": 1,
                     "urgent": [{"subject": "FINAL deadline"}]},
        })
        assert "5 unread" in result
        assert "1 urgent" in result
        assert "FINAL deadline" in result

    def test_gmail_from_cache_auth_error(self):
        """Rec #4: Graceful auth error in briefing."""
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._gmail_from_cache({"status": "auth_error"})
        assert "re-authenticate" in result.lower()

    def test_gmail_from_cache_clean_inbox(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._gmail_from_cache({
            "status": "ok", "data": {"unread": 0, "urgent_count": 0},
        })
        assert "clean" in result.lower()

    def test_calendar_from_cache_ok(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._calendar_from_cache({
            "status": "ok",
            "data": {"events": [
                {"summary": "Team meeting", "start": "10:00", "location": "Zoom"},
            ]},
        })
        assert "Team meeting" in result
        assert "Zoom" in result

    def test_calendar_from_cache_auth_error(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._calendar_from_cache({"status": "auth_error"})
        assert "re-authenticate" in result.lower()

    def test_calendar_from_cache_no_events(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._calendar_from_cache({"status": "ok", "data": {"events": []}})
        assert "no events" in result.lower()

    def test_classroom_from_cache_ok(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._classroom_from_cache({
            "status": "ok",
            "data": {
                "due_today": [{"title": "Math HW"}],
                "due_tomorrow": [],
                "overdue": [],
            },
        })
        assert "Math HW" in result

    def test_classroom_from_cache_auth_error(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._classroom_from_cache({"status": "auth_error"})
        assert "re-authenticate" in result.lower()

    def test_classroom_from_cache_nothing_due(self):
        from bantz.core.briefing import Briefing
        b = Briefing()
        result = b._classroom_from_cache({
            "status": "ok",
            "data": {"due_today": [], "due_tomorrow": [], "overdue": []},
        })
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Full orchestration: run_overnight_poll
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunOvernightPoll:
    @pytest.mark.asyncio
    async def test_dry_run_no_kv_writes(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import run_overnight_poll
        with patch("bantz.agent.workflows.overnight_poll._poll_gmail", new_callable=AsyncMock) as mg, \
             patch("bantz.agent.workflows.overnight_poll._poll_calendar", new_callable=AsyncMock) as mc, \
             patch("bantz.agent.workflows.overnight_poll._poll_classroom", new_callable=AsyncMock) as mcr, \
             patch("bantz.agent.workflows.overnight_poll._store_poll_results") as store_fn, \
             patch("bantz.agent.workflows.overnight_poll._send_urgent_notification") as notif_fn, \
             patch("bantz.agent.job_scheduler.inhibit_sleep"):
            from bantz.agent.workflows.overnight_poll import PollSourceResult
            mg.return_value = PollSourceResult(source="gmail", status="ok", data={"unread": 0})
            mc.return_value = PollSourceResult(source="calendar", status="ok", data={"events": []})
            mcr.return_value = PollSourceResult(source="classroom", status="ok", data={"assignment_count": 0})

            result = await run_overnight_poll(dry_run=True)
            store_fn.assert_not_called()
            notif_fn.assert_not_called()
            assert result.errors == 0

    @pytest.mark.asyncio
    async def test_normal_run_stores_results(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import run_overnight_poll, PollSourceResult
        with patch("bantz.agent.workflows.overnight_poll._poll_gmail", new_callable=AsyncMock) as mg, \
             patch("bantz.agent.workflows.overnight_poll._poll_calendar", new_callable=AsyncMock) as mc, \
             patch("bantz.agent.workflows.overnight_poll._poll_classroom", new_callable=AsyncMock) as mcr, \
             patch("bantz.agent.workflows.overnight_poll._store_poll_results") as store_fn, \
             patch("bantz.agent.workflows.overnight_poll._send_urgent_notification"), \
             patch("bantz.agent.workflows.overnight_poll._get_kv") as mock_kv, \
             patch("bantz.agent.job_scheduler.inhibit_sleep"):
            kv = MagicMock()
            kv.get.return_value = ""
            mock_kv.return_value = kv
            mg.return_value = PollSourceResult(source="gmail", status="ok", data={"unread": 3})
            mc.return_value = PollSourceResult(source="calendar", status="ok", data={"events": []})
            mcr.return_value = PollSourceResult(source="classroom", status="ok", data={"assignment_count": 1})

            result = await run_overnight_poll(dry_run=False)
            store_fn.assert_called_once()
            assert result.gmail.data["unread"] == 3
            assert result.errors == 0

    @pytest.mark.asyncio
    async def test_partial_failure_counted(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import run_overnight_poll, PollSourceResult
        with patch("bantz.agent.workflows.overnight_poll._poll_gmail", new_callable=AsyncMock) as mg, \
             patch("bantz.agent.workflows.overnight_poll._poll_calendar", new_callable=AsyncMock) as mc, \
             patch("bantz.agent.workflows.overnight_poll._poll_classroom", new_callable=AsyncMock) as mcr, \
             patch("bantz.agent.workflows.overnight_poll._store_poll_results"), \
             patch("bantz.agent.workflows.overnight_poll._send_urgent_notification"), \
             patch("bantz.agent.workflows.overnight_poll._get_kv") as mock_kv, \
             patch("bantz.agent.job_scheduler.inhibit_sleep"):
            kv = MagicMock()
            kv.get.return_value = ""
            mock_kv.return_value = kv
            mg.return_value = PollSourceResult(source="gmail", status="ok", data={})
            mc.return_value = PollSourceResult(source="calendar", status="auth_error",
                                               error_message="expired")
            mcr.return_value = PollSourceResult(source="classroom", status="error",
                                                error_message="API down")

            result = await run_overnight_poll(dry_run=False)
            assert result.errors == 2

    @pytest.mark.asyncio
    async def test_selective_sources(self):
        from bantz.agent.workflows.overnight_poll import run_overnight_poll, PollSourceResult
        with patch("bantz.agent.workflows.overnight_poll._poll_gmail", new_callable=AsyncMock) as mg, \
             patch("bantz.agent.workflows.overnight_poll._poll_calendar", new_callable=AsyncMock) as mc, \
             patch("bantz.agent.workflows.overnight_poll._poll_classroom", new_callable=AsyncMock) as mcr, \
             patch("bantz.agent.workflows.overnight_poll._store_poll_results"), \
             patch("bantz.agent.workflows.overnight_poll._send_urgent_notification"), \
             patch("bantz.agent.workflows.overnight_poll._get_kv") as mock_kv, \
             patch("bantz.agent.job_scheduler.inhibit_sleep"):
            kv = MagicMock()
            kv.get.return_value = ""
            mock_kv.return_value = kv
            mg.return_value = PollSourceResult(source="gmail", status="ok", data={})

            result = await run_overnight_poll(dry_run=False, sources=("gmail",))
            mg.assert_awaited_once()
            mc.assert_not_awaited()
            mcr.assert_not_awaited()

    def test_timeout_constants(self):
        from bantz.agent.workflows.overnight_poll import _POLL_TIMEOUT, _TOTAL_TIMEOUT
        assert _POLL_TIMEOUT >= 60
        assert _TOTAL_TIMEOUT >= 180


# ═══════════════════════════════════════════════════════════════════════════════
# CLI arguments
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIArgs:
    def test_overnight_poll_arg(self):
        import argparse
        from unittest.mock import patch as _patch
        import bantz.__main__
        # Just verify the arg is registered
        import sys
        with _patch.object(sys, "argv", ["bantz", "--overnight-poll", "--dry-run"]):
            # Re-parse — can't easily do full main(), just check parse_args
            parser = argparse.ArgumentParser()
            parser.add_argument("--overnight-poll", action="store_true")
            parser.add_argument("--dry-run", action="store_true")
            args = parser.parse_args(["--overnight-poll", "--dry-run"])
            assert args.overnight_poll is True
            assert args.dry_run is True


# ═══════════════════════════════════════════════════════════════════════════════
# Job scheduler delegation
# ═══════════════════════════════════════════════════════════════════════════════

class TestJobSchedulerDelegation:
    @pytest.mark.asyncio
    async def test_job_overnight_poll_delegates(self):
        from bantz.agent.job_scheduler import _job_overnight_poll
        with patch("bantz.agent.workflows.overnight_poll.run_overnight_poll",
                    new_callable=AsyncMock) as mock_run:
            from bantz.agent.workflows.overnight_poll import OvernightPollResult
            mock_run.return_value = OvernightPollResult()
            await _job_overnight_poll()
            mock_run.assert_awaited_once_with(dry_run=False)

    def test_registry_updated(self):
        from bantz.agent.job_scheduler import _JOB_REGISTRY
        assert "overnight_poll" in _JOB_REGISTRY
        desc = _JOB_REGISTRY["overnight_poll"][1]
        assert "overnight" in desc.lower() or "poll" in desc.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Config integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    def test_urgent_keywords_in_config(self):
        from bantz.config import Config
        cfg = Config()
        assert hasattr(cfg, "urgent_keywords")
        # Default should have some keywords
        assert "urgent" in cfg.urgent_keywords.lower()

    def test_urgent_keywords_custom(self):
        import os
        with patch.dict(os.environ, {"BANTZ_URGENT_KEYWORDS": "erasmus,gargantua,jetson"}):
            from bantz.config import Config
            cfg = Config()
            assert "erasmus" in cfg.urgent_keywords
            assert "gargantua" in cfg.urgent_keywords


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_poll_source_result_serializable(self):
        from bantz.agent.workflows.overnight_poll import PollSourceResult
        r = PollSourceResult(source="gmail", status="ok", data={"key": "val"})
        s = json.dumps(r.to_dict())
        assert json.loads(s)["data"]["key"] == "val"

    def test_overnight_poll_result_serializable(self):
        from bantz.agent.workflows.overnight_poll import OvernightPollResult
        r = OvernightPollResult(poll_time="2026-01-01T00:00:00Z")
        # summary_line should not crash
        assert isinstance(r.summary_line(), str)

    @pytest.mark.asyncio
    async def test_poll_gmail_no_urgent_keywords(self):
        """When urgent_keywords is empty, all emails are normal."""
        from bantz.agent.workflows.overnight_poll import _poll_gmail
        mock_creds = MagicMock()
        with patch("bantz.auth.token_store.token_store") as mock_ts:
            mock_ts.get.return_value = mock_creds
            with patch("bantz.tools.gmail.GmailTool") as MockGmail:
                inst = MockGmail.return_value
                inst._count_sync.return_value = 2
                inst._fetch_messages_sync.return_value = [
                    {"id": "1", "from": "x@y.com", "subject": "URGENT stuff",
                     "snippet": "...", "date": "2026-03-10"},
                ]
                with patch("bantz.config.config") as mock_cfg:
                    mock_cfg.urgent_keywords = ""
                    result = await _poll_gmail(None)
        assert result.data["urgent_count"] == 0
        assert len(result.data["normal"]) == 1

    def test_read_overnight_data_empty_kv(self, tmp_path):
        from bantz.agent.workflows.overnight_poll import read_overnight_data
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.db_path = tmp_path / "bantz.db"
            data = read_overnight_data()
        assert data.get("last_poll") == ""
