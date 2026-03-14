"""Tests for #124 / #220 Sprint 3 Part 5 — Event-driven stderr observer.

Covers:
  - Severity enum ordering
  - ErrorEvent: fingerprint, to_dict
  - ErrorClassifier: regex patterns, severity escalation, unrecognised text
  - ErrorBuffer: batching, dedup, flush
  - StderrReader: push/pop
  - Observer: full pipeline, stats, lifecycle, threshold gate, dedup
  - EventBus integration (subscribe/emit)
  - Config: observer fields
  - LLM analysis (mocked, no aiohttp required)
  - Source audit (no polling patterns)
"""
from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.agent.observer import (
    ErrorBuffer,
    ErrorClassifier,
    ErrorEvent,
    Observer,
    Severity,
    StderrReader,
    observer,
)


# ═══════════════════════════════════════════════════════════════════════════
# Severity
# ═══════════════════════════════════════════════════════════════════════════


class TestSeverity:
    def test_values(self):
        assert Severity.IGNORE.value == "ignore"
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"

    def test_from_string(self):
        assert Severity("warning") == Severity.WARNING
        assert Severity("critical") == Severity.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════
# ErrorEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorEvent:
    def test_fingerprint_stable(self):
        e1 = ErrorEvent(severity=Severity.WARNING, raw_text="Error: foo", pattern_matched="Error:")
        e2 = ErrorEvent(severity=Severity.WARNING, raw_text="Error: foo", pattern_matched="Error:")
        assert e1.fingerprint == e2.fingerprint

    def test_fingerprint_differs(self):
        e1 = ErrorEvent(severity=Severity.WARNING, raw_text="Error: foo", pattern_matched="Error:")
        e2 = ErrorEvent(severity=Severity.WARNING, raw_text="Other: bar", pattern_matched="Other:")
        assert e1.fingerprint != e2.fingerprint

    def test_to_dict(self):
        e = ErrorEvent(
            severity=Severity.CRITICAL,
            raw_text="Traceback ...",
            pattern_matched="Traceback",
            analysis="Division by zero",
            timestamp=1000.0,
        )
        d = e.to_dict()
        assert d["severity"] == "critical"
        assert d["raw_text"] == "Traceback ..."
        assert d["analysis"] == "Division by zero"
        assert d["timestamp"] == 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# ErrorClassifier
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorClassifier:
    def setup_method(self):
        self.clf = ErrorClassifier(enable_llm=False)

    def test_empty_text(self):
        assert self.clf.classify("") is None
        assert self.clf.classify("   ") is None
        assert self.clf.classify(None) is None  # type: ignore

    def test_traceback_is_critical(self):
        text = "Traceback (most recent call last):\n  File ...\nZeroDivisionError: ..."
        event = self.clf.classify(text)
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_segfault_is_critical(self):
        event = self.clf.classify("Segmentation fault (core dumped)")
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_oom_is_critical(self):
        event = self.clf.classify("Cannot allocate memory")
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_fatal_is_critical(self):
        event = self.clf.classify("FATAL: could not open file")
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_panic_is_critical(self):
        event = self.clf.classify("panic: runtime error: index out of range")
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_error_colon_is_warning(self):
        event = self.clf.classify("TypeError: cannot read property 'x' of undefined")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_npm_err_is_warning(self):
        event = self.clf.classify("npm ERR! code E404")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_permission_denied_is_warning(self):
        event = self.clf.classify("bash: /etc/shadow: Permission denied")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_command_not_found_is_warning(self):
        event = self.clf.classify("bash: foo: command not found")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_build_failed_is_warning(self):
        event = self.clf.classify("Build FAILED with 3 errors")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_test_failed_is_warning(self):
        event = self.clf.classify("FAILED tests/test_foo.py::TestBar")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_deprecation_is_info(self):
        event = self.clf.classify("DeprecationWarning: use new_method instead")
        assert event is not None
        assert event.severity == Severity.INFO

    def test_warning_keyword_is_info(self):
        event = self.clf.classify("warning: unused variable 'x'")
        assert event is not None
        assert event.severity == Severity.INFO

    def test_unrecognised_returns_none(self):
        assert self.clf.classify("Hello, world!") is None
        assert self.clf.classify("Compilation succeeded") is None
        assert self.clf.classify("200 OK") is None

    def test_critical_overrides_warning(self):
        """Text matching both critical and warning patterns → critical wins."""
        text = "Traceback (most recent call last):\nValueError: bad input"
        event = self.clf.classify(text)
        assert event is not None
        assert event.severity == Severity.CRITICAL

    def test_error_response_from_daemon(self):
        event = self.clf.classify("Error response from daemon: pull access denied")
        assert event is not None
        assert event.severity == Severity.WARNING

    def test_undefined_reference(self):
        event = self.clf.classify("undefined reference to `main'")
        assert event is not None
        assert event.severity == Severity.WARNING


# ═══════════════════════════════════════════════════════════════════════════
# ErrorBuffer
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorBuffer:
    def test_add_and_flush(self):
        buf = ErrorBuffer(batch_seconds=0.0)
        buf.add_line("line 1")
        buf.add_line("line 2")
        text = buf.flush()
        assert text == "line 1\nline 2"

    def test_flush_empty(self):
        buf = ErrorBuffer()
        assert buf.flush() is None

    def test_should_flush_timing(self):
        buf = ErrorBuffer(batch_seconds=0.01)
        buf.add_line("test")
        time.sleep(0.02)
        assert buf.should_flush() is True

    def test_should_flush_no_lines(self):
        buf = ErrorBuffer(batch_seconds=0.0)
        assert buf.should_flush() is False

    def test_dedup_same_fingerprint(self):
        buf = ErrorBuffer(dedup_window=60.0)
        assert buf.is_duplicate("abc") is False
        assert buf.is_duplicate("abc") is True

    def test_dedup_different_fingerprint(self):
        buf = ErrorBuffer(dedup_window=60.0)
        assert buf.is_duplicate("abc") is False
        assert buf.is_duplicate("def") is False

    def test_dedup_expires(self):
        buf = ErrorBuffer(dedup_window=0.01)
        assert buf.is_duplicate("abc") is False
        time.sleep(0.02)
        assert buf.is_duplicate("abc") is False  # expired

    def test_pending_count(self):
        buf = ErrorBuffer()
        assert buf.pending_count == 0
        buf.add_line("x")
        assert buf.pending_count == 1
        buf.flush()
        assert buf.pending_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# StderrReader
# ═══════════════════════════════════════════════════════════════════════════


class TestStderrReader:
    def test_push_and_pop(self):
        r = StderrReader()
        r.push("line 1")
        r.push("line 2")
        lines = r.pop_all()
        assert lines == ["line 1", "line 2"]
        assert r.pop_all() == []

    def test_push_lines(self):
        r = StderrReader()
        r.push_lines("line 1\nline 2\n\nline 3")
        lines = r.pop_all()
        assert lines == ["line 1", "line 2", "line 3"]  # blank lines skipped

    def test_pending(self):
        r = StderrReader()
        assert r.pending == 0
        r.push("x")
        assert r.pending == 1

    def test_thread_safety(self):
        r = StderrReader()

        def _push_many():
            for i in range(100):
                r.push(f"line-{i}")

        t1 = threading.Thread(target=_push_many)
        t2 = threading.Thread(target=_push_many)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert r.pending == 200


# ═══════════════════════════════════════════════════════════════════════════
# Observer — full pipeline (event-driven, no polling)
# ═══════════════════════════════════════════════════════════════════════════


class TestObserver:
    def _make(self, **kwargs):
        return Observer(
            severity_threshold="warning",
            batch_seconds=0.0,  # flush immediately for test determinism
            dedup_window=60.0,
            enable_llm_analysis=False,
            **kwargs,
        )

    def test_start_stop(self):
        obs = self._make()
        obs.start()
        assert obs.running is True
        obs.stop()
        assert obs.running is False

    def test_feed_triggers_callback(self):
        events = []
        obs = self._make(on_error=events.append)
        obs.start()
        try:
            obs.feed("Traceback (most recent call last):\nZeroDivisionError: ...")
        finally:
            obs.stop()
        assert len(events) >= 1
        assert events[0].severity == Severity.CRITICAL

    def test_threshold_gate(self):
        """Events below threshold should not trigger callback."""
        events = []
        obs = Observer(
            on_error=events.append,
            severity_threshold="critical",
            batch_seconds=0.0,
            dedup_window=60.0,
            enable_llm_analysis=False,
        )
        obs.start()
        try:
            obs.feed("warning: unused variable 'x'")
        finally:
            obs.stop()
        assert len(events) == 0

    def test_dedup_within_window(self):
        """Same error pattern within dedup window should fire once."""
        events = []
        obs = self._make(on_error=events.append)
        obs.start()
        try:
            obs.feed("npm ERR! code E404")
            obs.feed("npm ERR! code E404")
        finally:
            obs.stop()
        # Should have exactly 1 event (second is dedup)
        assert len(events) == 1

    def test_unrecognised_text_ignored(self):
        events = []
        obs = self._make(on_error=events.append)
        obs.start()
        try:
            obs.feed("Everything is fine!")
        finally:
            obs.stop()
        assert len(events) == 0

    def test_stats(self):
        obs = self._make()
        obs.start()
        try:
            obs.feed("Segmentation fault (core dumped)")
        finally:
            obs.stop()
        s = obs.stats()
        assert s["total_lines"] >= 1
        assert s["total_events"] >= 1
        assert s["by_severity"]["critical"] >= 1

    def test_stats_empty(self):
        obs = self._make()
        s = obs.stats()
        assert s["total_lines"] == 0
        assert s["total_events"] == 0
        assert s["running"] is False

    def test_double_start(self):
        obs = self._make()
        obs.start()
        obs.start()  # should not raise
        assert obs.running is True
        obs.stop()

    def test_stop_without_start(self):
        obs = self._make()
        obs.stop()  # should not raise

    def test_feed_before_start(self):
        """Lines fed before start should be processed once started."""
        events = []
        obs = self._make(on_error=events.append)
        obs.feed("Permission denied")  # feed before start
        obs.start()
        try:
            pass
        finally:
            obs.stop()
        assert len(events) >= 1

    def test_multiple_errors_in_batch(self):
        """Multiple error lines in one batch should classify as highest severity."""
        events = []
        obs = self._make(on_error=events.append)
        obs.start()
        try:
            obs.feed("Traceback (most recent call last):\nValueError: bad\nError: also bad")
        finally:
            obs.stop()
        assert len(events) >= 1
        assert events[0].severity == Severity.CRITICAL

    def test_no_thread_created(self):
        """Event-driven observer must NOT create a background thread."""
        obs = self._make()
        threads_before = threading.active_count()
        obs.start()
        threads_after = threading.active_count()
        obs.stop()
        assert threads_after == threads_before


# ═══════════════════════════════════════════════════════════════════════════
# EventBus integration (#220 Part 5)
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverEventBus:
    """Verify Observer subscribes to and emits on EventBus."""

    def test_subscribes_on_start(self):
        obs = Observer(batch_seconds=0.0, enable_llm_analysis=False)
        with patch("bantz.agent.observer.bus") as mock_bus:
            obs.start()
            mock_bus.on.assert_called_once_with("stderr_line", obs._on_stderr_event)
            obs.stop()

    def test_unsubscribes_on_stop(self):
        obs = Observer(batch_seconds=0.0, enable_llm_analysis=False)
        with patch("bantz.agent.observer.bus") as mock_bus:
            obs.start()
            obs.stop()
            mock_bus.off.assert_called_once_with("stderr_line", obs._on_stderr_event)

    def test_emits_observer_error_on_detection(self):
        obs = Observer(
            batch_seconds=0.0, dedup_window=60.0, enable_llm_analysis=False,
        )
        with patch("bantz.agent.observer.bus") as mock_bus:
            # Don't actually subscribe — just test processing
            obs._running = True
            obs.feed("Segmentation fault (core dumped)")
            mock_bus.emit_threadsafe.assert_called_once()
            args, kwargs = mock_bus.emit_threadsafe.call_args
            assert args[0] == "observer_error"
            assert kwargs["severity"] == "critical"
            assert "Segmentation fault" in kwargs["raw_text"]

    def test_on_stderr_event_handler(self):
        """EventBus stderr_line event should trigger processing."""
        events = []
        obs = Observer(
            on_error=events.append, batch_seconds=0.0,
            dedup_window=60.0, enable_llm_analysis=False,
        )
        obs._running = True
        # Simulate an EventBus Event object
        mock_event = MagicMock()
        mock_event.data = {"line": "Permission denied"}
        obs._on_stderr_event(mock_event)
        # Force flush since batch_seconds=0.0
        text = obs.buffer.flush()
        if text:
            obs._process_batch(text)
        assert len(events) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverConfig:
    def test_default_observer_disabled(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            _env_file=None,
        )
        assert cfg.observer_enabled is False

    def test_default_severity_threshold(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            _env_file=None,
        )
        assert cfg.observer_severity_threshold == "warning"

    def test_default_analysis_model(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            _env_file=None,
        )
        assert cfg.observer_analysis_model == "qwen2.5:0.5b"

    def test_default_batch_seconds(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            _env_file=None,
        )
        assert cfg.observer_batch_seconds == 5.0

    def test_default_dedup_window(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            _env_file=None,
        )
        assert cfg.observer_dedup_window == 60.0

    def test_env_override(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            BANTZ_OBSERVER_ENABLED="true",
            BANTZ_OBSERVER_SEVERITY_THRESHOLD="critical",
            _env_file=None,
        )
        assert cfg.observer_enabled is True
        assert cfg.observer_severity_threshold == "critical"


# ═══════════════════════════════════════════════════════════════════════════
# LLM Analysis (mocked — aiohttp lazy-imported, never required)
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifierLLMAnalysis:
    @pytest.mark.asyncio
    async def test_analyze_success(self):
        clf = ErrorClassifier(enable_llm=True)
        event = ErrorEvent(severity=Severity.CRITICAL, raw_text="Traceback ...")

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"response": "Division by zero error."})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await clf.analyze(event)
        assert "Division by zero" in result

    @pytest.mark.asyncio
    async def test_analyze_disabled(self):
        clf = ErrorClassifier(enable_llm=False)
        event = ErrorEvent(severity=Severity.CRITICAL, raw_text="Traceback ...")
        result = await clf.analyze(event)
        assert result == ""

    @pytest.mark.asyncio
    async def test_analyze_error_returns_empty(self):
        clf = ErrorClassifier(enable_llm=True)
        event = ErrorEvent(severity=Severity.CRITICAL, raw_text="Traceback ...")

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(side_effect=Exception("conn err"))

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await clf.analyze(event)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# Source audit (#220 Part 5)
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverSourceAudit:
    """Verify no polling patterns remain in the observer module."""

    def test_no_time_sleep_in_observer_class(self):
        import inspect
        src = inspect.getsource(Observer)
        assert "time.sleep(" not in src, "Observer must not use time.sleep"

    def test_no_while_true_in_observer(self):
        import inspect
        src = inspect.getsource(Observer)
        assert "while True" not in src
        assert "while not" not in src

    def test_no_threading_thread_in_observer(self):
        """Event-driven observer should not spawn its own thread."""
        import inspect
        src = inspect.getsource(Observer)
        assert "threading.Thread" not in src

    def test_no_top_level_aiohttp_import(self):
        """aiohttp must be lazy-imported, never at module level."""
        from bantz.agent import observer as mod
        import inspect
        src = inspect.getsource(mod)
        # Check that 'import aiohttp' only appears inside functions
        lines = src.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "import aiohttp" in stripped and not stripped.startswith("#"):
                # Must be indented (inside a function)
                assert line[0] == " ", f"Line {i}: aiohttp import at module level"

    def test_bus_import_exists(self):
        from bantz.agent import observer as mod
        assert hasattr(mod, "bus")

    def test_singleton_is_observer(self):
        assert isinstance(observer, Observer)
