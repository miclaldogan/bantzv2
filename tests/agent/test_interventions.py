"""Tests for bantz.agent.interventions (#126)."""
from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bantz.agent.interventions import (
    Intervention,
    InterventionLog,
    InterventionQueue,
    InterventionType,
    Outcome,
    Priority,
    intervention_from_observer,
    intervention_from_rl,
    intervention_from_reminder,
    intervention_from_system,
    ACTION_LABELS,
    SOURCE_LABELS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Intervention dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestIntervention:
    def test_create(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test intervention",
            reason="Testing",
            source="test",
        )
        assert iv.outcome == Outcome.PENDING
        assert iv.action is None
        assert iv.id == 0

    def test_expired_false(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="test",
            ttl=60.0,
        )
        assert not iv.expired

    def test_expired_true(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="test",
            ttl=0.001,
            created_at=time.time() - 1,
        )
        assert iv.expired

    def test_age(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="test",
            created_at=time.time() - 5,
        )
        assert iv.age >= 5

    def test_remaining_ttl(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="test",
            ttl=30.0,
        )
        assert iv.remaining_ttl <= 30.0
        assert iv.remaining_ttl > 0

    def test_remaining_ttl_negative(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="test",
            ttl=1.0,
            created_at=time.time() - 10,
        )
        assert iv.remaining_ttl < 0


# ═══════════════════════════════════════════════════════════════════════════
# InterventionLog
# ═══════════════════════════════════════════════════════════════════════════


class TestInterventionLog:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.log = InterventionLog()
        self.log.init(self.tmp.name)

    def teardown_method(self):
        self.log.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_record_and_total(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test reason",
            source="test",
            outcome=Outcome.ACCEPTED,
        )
        self.log.record(iv)
        assert self.log.total() == 1
        assert iv.id > 0

    def test_recent(self):
        for i in range(5):
            iv = Intervention(
                type=InterventionType.ROUTINE,
                priority=Priority.MEDIUM,
                title=f"Test {i}",
                reason="Test",
                source="test",
                outcome=Outcome.ACCEPTED,
            )
            self.log.record(iv)
        recent = self.log.recent(3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["title"] == "Test 4"

    def test_outcome_counts(self):
        for outcome in [Outcome.ACCEPTED, Outcome.ACCEPTED, Outcome.DISMISSED]:
            iv = Intervention(
                type=InterventionType.ROUTINE,
                priority=Priority.MEDIUM,
                title="Test",
                reason="Test",
                source="test",
                outcome=outcome,
            )
            self.log.record(iv)
        counts = self.log.outcome_counts()
        assert counts["accepted"] == 2
        assert counts["dismissed"] == 1

    def test_acceptance_rate(self):
        for outcome in [Outcome.ACCEPTED, Outcome.DISMISSED, Outcome.DISMISSED]:
            iv = Intervention(
                type=InterventionType.ROUTINE,
                priority=Priority.MEDIUM,
                title="Test",
                reason="Test",
                source="test",
                outcome=outcome,
            )
            self.log.record(iv)
        rate = self.log.acceptance_rate()
        assert abs(rate - 0.333) < 0.01

    def test_acceptance_rate_empty(self):
        assert self.log.acceptance_rate() == 0.0

    def test_stats(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="test",
            outcome=Outcome.ACCEPTED,
        )
        self.log.record(iv)
        s = self.log.stats()
        assert s["total"] == 1
        assert "acceptance_rate" in s

    def test_no_connection(self):
        log = InterventionLog()
        assert log.total() == 0
        assert log.recent() == []
        assert log.outcome_counts() == {}
        assert log.acceptance_rate() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# InterventionQueue
# ═══════════════════════════════════════════════════════════════════════════


class TestInterventionQueue:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.q = InterventionQueue()
        self.q.init(self.tmp.name, rate_limit=3, default_ttl=20.0)

    def teardown_method(self):
        self.q.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _make_iv(self, **kw):
        defaults = dict(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test reason",
            source="test",
            ttl=30.0,
        )
        defaults.update(kw)
        return Intervention(**defaults)

    def test_init(self):
        assert self.q.initialized
        assert self.q.pending_count() == 0

    def test_push_pop(self):
        iv = self._make_iv()
        assert self.q.push(iv)
        assert self.q.pending_count() == 1
        result = self.q.pop()
        assert result is not None
        assert result.title == "Test"

    def test_priority_ordering(self):
        self.q.push(self._make_iv(title="Low", priority=Priority.LOW))
        self.q.push(self._make_iv(title="Critical", priority=Priority.CRITICAL))
        self.q.push(self._make_iv(title="High", priority=Priority.HIGH))
        result = self.q.pop()
        assert result.title == "Critical"

    def test_fifo_within_priority(self):
        self.q.push(self._make_iv(title="First", created_at=time.time() - 10))
        self.q.push(self._make_iv(title="Second", created_at=time.time()))
        result = self.q.pop()
        assert result.title == "First"

    def test_rate_limit(self):
        for i in range(4):
            self.q.push(self._make_iv(title=f"Test {i}"))
        # Pop 3 (rate limit)
        for _ in range(3):
            assert self.q.pop() is not None
        # 4th should be rate-limited
        assert self.q.pop() is None

    def test_rate_limit_critical_bypass(self):
        for i in range(4):
            self.q.push(self._make_iv(title=f"Test {i}", priority=Priority.CRITICAL))
        # All 4 should pop (CRITICAL bypasses rate limit)
        for _ in range(4):
            assert self.q.pop() is not None

    def test_rate_remaining(self):
        assert self.q.rate_remaining() == 3
        self.q.push(self._make_iv())
        self.q.pop()
        assert self.q.rate_remaining() == 2

    def test_quiet_mode_drops_non_critical(self):
        self.q.set_quiet(True)
        assert not self.q.push(self._make_iv(priority=Priority.MEDIUM))
        assert self.q.push(self._make_iv(priority=Priority.CRITICAL))
        assert self.q.pending_count() == 1

    def test_quiet_mode_toggle(self):
        self.q.set_quiet(True)
        assert self.q.quiet
        self.q.set_quiet(False)
        assert not self.q.quiet

    def test_focus_mode_drops_below_high(self):
        self.q.set_focus(True)
        assert not self.q.push(self._make_iv(priority=Priority.LOW))
        assert not self.q.push(self._make_iv(priority=Priority.MEDIUM))
        assert self.q.push(self._make_iv(priority=Priority.HIGH))
        assert self.q.push(self._make_iv(priority=Priority.CRITICAL))
        assert self.q.pending_count() == 2

    def test_focus_mode_toggle(self):
        self.q.set_focus(True)
        assert self.q.focus
        self.q.set_focus(False)
        assert not self.q.focus

    def test_respond_accepted(self):
        self.q.push(self._make_iv(action="launch_docker", state_key="s1"))
        self.q.pop()
        assert self.q.has_active
        result = self.q.respond(Outcome.ACCEPTED)
        assert result is not None
        assert result.outcome == Outcome.ACCEPTED
        assert result.responded_at is not None
        assert not self.q.has_active

    def test_respond_dismissed(self):
        self.q.push(self._make_iv())
        self.q.pop()
        result = self.q.respond(Outcome.DISMISSED)
        assert result.outcome == Outcome.DISMISSED

    def test_respond_never(self):
        self.q.push(self._make_iv())
        self.q.pop()
        result = self.q.respond(Outcome.NEVER)
        assert result.outcome == Outcome.NEVER

    def test_respond_no_active(self):
        assert self.q.respond(Outcome.ACCEPTED) is None

    def test_expire_active(self):
        self.q.push(self._make_iv())
        self.q.pop()
        result = self.q.expire_active()
        assert result is not None
        assert result.outcome == Outcome.AUTO_DISMISSED
        assert not self.q.has_active

    def test_expire_active_no_active(self):
        assert self.q.expire_active() is None

    def test_expired_entries_cleaned_on_pop(self):
        """Expired entries in the queue should be auto-dismissed on pop."""
        self.q.push(self._make_iv(title="Expired", ttl=0.001, created_at=time.time() - 1))
        self.q.push(self._make_iv(title="Fresh", ttl=60.0))
        result = self.q.pop()
        assert result.title == "Fresh"

    def test_push_not_initialized(self):
        q = InterventionQueue()
        assert not q.push(self._make_iv())

    def test_pop_not_initialized(self):
        q = InterventionQueue()
        assert q.pop() is None

    def test_default_ttl_applied(self):
        iv = self._make_iv(ttl=0)  # 0 means use default
        self.q.push(iv)
        assert iv.ttl == 20.0  # default from init

    def test_stats(self):
        self.q.push(self._make_iv())
        s = self.q.stats()
        assert s["initialized"]
        assert s["pending"] == 1
        assert s["rate_remaining"] == 3
        assert s["rate_limit"] == 3
        assert not s["quiet_mode"]
        assert not s["focus_mode"]
        assert not s["has_active"]

    def test_status_line(self):
        line = self.q.status_line()
        assert "pending=" in line
        assert "rate=" in line
        assert "ttl=" in line

    def test_close_expires_pending(self):
        self.q.push(self._make_iv())
        self.q.push(self._make_iv())
        self.q.close()
        assert self.q.pending_count() == 0
        assert not self.q.initialized

    def test_close_expires_active(self):
        self.q.push(self._make_iv())
        self.q.pop()
        assert self.q.has_active
        self.q.close()
        assert not self.q.has_active


# ═══════════════════════════════════════════════════════════════════════════
# Builder helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestInterventionBuilders:
    def test_from_rl(self):
        iv = intervention_from_rl(
            action_value="launch_docker",
            state_key="morning|monday|home|shell",
            reason="Morning Monday routine",
            ttl=25.0,
        )
        assert iv.type == InterventionType.ROUTINE
        assert iv.priority == Priority.MEDIUM
        assert "Docker" in iv.title or "docker" in iv.title.lower()
        assert iv.action == "launch_docker"
        assert iv.state_key == "morning|monday|home|shell"
        assert iv.ttl == 25.0

    def test_from_rl_unknown_action(self):
        iv = intervention_from_rl(
            action_value="unknown_action",
            state_key="s1",
        )
        assert "Suggestion" in iv.title
        assert iv.reason == "Learned routine pattern"

    def test_from_observer_critical(self):
        iv = intervention_from_observer(
            raw_text="Traceback (most recent call last)...",
            severity="critical",
            analysis="NullPointerError in main.py",
        )
        assert iv.type == InterventionType.ERROR_ALERT
        assert iv.priority == Priority.CRITICAL
        assert "🚨" in iv.title
        assert "critical" in iv.reason.lower()

    def test_from_observer_warning(self):
        iv = intervention_from_observer(
            raw_text="DeprecationWarning: old API",
            severity="warning",
        )
        assert iv.priority == Priority.HIGH
        assert "⚠️" in iv.title

    def test_from_observer_info(self):
        iv = intervention_from_observer(
            raw_text="Some info message",
            severity="info",
        )
        assert iv.priority == Priority.MEDIUM

    def test_from_reminder(self):
        iv = intervention_from_reminder(
            title="Meeting with team",
            repeat="daily",
            ttl=30.0,
        )
        assert iv.type == InterventionType.REMINDER
        assert iv.priority == Priority.HIGH
        assert "Meeting with team" in iv.title
        assert "daily" in iv.title
        assert iv.source == "scheduler"

    def test_from_reminder_no_repeat(self):
        iv = intervention_from_reminder(title="One-off task")
        assert "repeats" not in iv.title

    def test_from_system(self):
        iv = intervention_from_system(
            title="Disk at 85%",
            reason="Storage cleanup recommended",
            priority=Priority.LOW,
        )
        assert iv.type == InterventionType.MAINTENANCE
        assert iv.priority == Priority.LOW
        assert "🔧" in iv.title
        assert iv.source == "system"


# ═══════════════════════════════════════════════════════════════════════════
# Auto-dismiss with mild RL penalty
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoDismiss:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.q = InterventionQueue()
        self.q.init(self.tmp.name, rate_limit=10, default_ttl=0.05)

    def teardown_method(self):
        self.q.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_auto_dismiss_expired_active(self):
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Test",
            reason="Test",
            source="rl_engine",
            action="launch_docker",
            state_key="morning|monday|home|shell",
            ttl=0.5,  # Short but not already expired
        )
        self.q.push(iv)
        popped = self.q.pop()
        assert popped is not None
        assert self.q.has_active
        # Wait for TTL to expire
        time.sleep(0.6)
        assert popped.expired
        result = self.q.expire_active()
        assert result.outcome == Outcome.AUTO_DISMISSED
        assert result.responded_at is not None

    def test_expired_queued_items_cleaned(self):
        """Items that expire while still in the queue get auto-dismissed."""
        iv = Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Expired in queue",
            reason="Test",
            source="test",
            ttl=0.001,
            created_at=time.time() - 1,
        )
        self.q.push(iv)
        time.sleep(0.01)
        # Pop should return None (the only item expired)
        assert self.q.pop() is None


# ═══════════════════════════════════════════════════════════════════════════
# Explainability (SOURCE_LABELS, ACTION_LABELS)
# ═══════════════════════════════════════════════════════════════════════════


class TestExplainability:
    def test_action_labels_complete(self):
        """All 8 RL actions should have labels."""
        from bantz.agent.rl_engine import Action
        for action in Action:
            assert action.value in ACTION_LABELS, f"Missing label for {action.value}"

    def test_source_labels_complete(self):
        expected = {"rl_engine", "observer", "scheduler", "system"}
        assert set(SOURCE_LABELS.keys()) == expected

    def test_action_label_format(self):
        for key, (emoji, text) in ACTION_LABELS.items():
            assert len(emoji) > 0, f"Empty emoji for {key}"
            assert len(text) > 0, f"Empty text for {key}"
            assert text.endswith("?"), f"Label should be a question: {text}"


# ═══════════════════════════════════════════════════════════════════════════
# Focus mode (dynamic limits)
# ═══════════════════════════════════════════════════════════════════════════


class TestFocusMode:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.q = InterventionQueue()
        self.q.init(self.tmp.name, rate_limit=10, default_ttl=60.0)

    def teardown_method(self):
        self.q.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_focus_blocks_low_medium(self):
        self.q.set_focus(True)
        assert not self.q.push(Intervention(
            type=InterventionType.MAINTENANCE,
            priority=Priority.LOW,
            title="Low",
            reason="test",
            source="system",
        ))
        assert not self.q.push(Intervention(
            type=InterventionType.ROUTINE,
            priority=Priority.MEDIUM,
            title="Medium",
            reason="test",
            source="rl_engine",
        ))

    def test_focus_allows_high_critical(self):
        self.q.set_focus(True)
        assert self.q.push(Intervention(
            type=InterventionType.REMINDER,
            priority=Priority.HIGH,
            title="Meeting",
            reason="test",
            source="scheduler",
        ))
        assert self.q.push(Intervention(
            type=InterventionType.ERROR_ALERT,
            priority=Priority.CRITICAL,
            title="Crash",
            reason="test",
            source="observer",
        ))
        assert self.q.pending_count() == 2

    def test_focus_off_allows_all(self):
        self.q.set_focus(False)
        assert self.q.push(Intervention(
            type=InterventionType.MAINTENANCE,
            priority=Priority.LOW,
            title="Low",
            reason="test",
            source="system",
        ))
        assert self.q.pending_count() == 1


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestInterventionConfig:
    def test_default_rate_limit(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.intervention_rate_limit == 3

    def test_default_toast_ttl(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.intervention_toast_ttl == 20.0

    def test_default_quiet_mode(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.intervention_quiet_mode is False

    def test_default_focus_mode(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.intervention_focus_mode is False

    def test_env_override_rate_limit(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            BANTZ_INTERVENTION_RATE_LIMIT="5",
            _env_file=None,
        )
        assert cfg.intervention_rate_limit == 5

    def test_env_override_toast_ttl(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            BANTZ_INTERVENTION_TOAST_TTL="30",
            _env_file=None,
        )
        assert cfg.intervention_toast_ttl == 30.0


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class TestEnums:
    def test_intervention_types(self):
        assert len(InterventionType) == 5
        assert InterventionType.ERROR_ALERT.value == "error_alert"

    def test_priorities(self):
        assert Priority.LOW < Priority.MEDIUM < Priority.HIGH < Priority.CRITICAL
        assert Priority.CRITICAL.value == 3

    def test_outcomes(self):
        assert len(Outcome) == 5
        assert Outcome.AUTO_DISMISSED.value == "auto_dismissed"
