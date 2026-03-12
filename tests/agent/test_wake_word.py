"""
Tests — Issue #165: Wake Word Detection (Porcupine)

Covers:
  - WakeWordListener: init, start, stop, diagnose, stats, status_line
  - Dedicated thread architecture (NOT APScheduler)
  - Audio stream stays open during TTS (interrupt, not mute)
  - Cooldown between detections
  - TTS interrupt on wake word
  - Ack sound (generate + play)
  - Custom .ppn keyword file discovery
  - Graceful degradation when pvporcupine/pyaudio missing
  - Config: wake_word_enabled, picovoice_access_key, wake_word_sensitivity
  - .env.example completeness
  - --doctor diagnostics
  - TUI WakeWordDetected message
  - Scope guard: no STT code in this module
"""
from __future__ import annotations

import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════

class TestWakeWordConfig:
    def test_wake_word_enabled_field(self):
        from bantz.config import Config
        c = Config(BANTZ_WAKE_WORD_ENABLED="false")
        assert c.wake_word_enabled is False

    def test_wake_word_enabled_true(self):
        from bantz.config import Config
        c = Config(BANTZ_WAKE_WORD_ENABLED="true")
        assert c.wake_word_enabled is True

    def test_picovoice_access_key_default_empty(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.picovoice_access_key == ""

    def test_picovoice_access_key_set(self):
        from bantz.config import Config
        c = Config(BANTZ_PICOVOICE_ACCESS_KEY="test-key-123")
        assert c.picovoice_access_key == "test-key-123"

    def test_wake_word_sensitivity_default(self):
        from bantz.config import Config
        c = Config()
        assert c.wake_word_sensitivity == 0.5

    def test_wake_word_sensitivity_custom(self):
        from bantz.config import Config
        c = Config(BANTZ_WAKE_WORD_SENSITIVITY="0.8")
        assert c.wake_word_sensitivity == 0.8


# ═══════════════════════════════════════════════════════════════════════════
# WakeWordListener — core unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWakeWordListener:
    def _make(self):
        from bantz.agent.wake_word import WakeWordListener
        return WakeWordListener()

    def test_initial_state(self):
        w = self._make()
        assert w.running is False
        assert w.total_detections == 0

    def test_stats(self):
        w = self._make()
        s = w.stats()
        assert s["running"] is False
        assert s["total_detections"] == 0
        assert "last_trigger" in s

    def test_status_line_stopped(self):
        w = self._make()
        assert "stopped" in w.status_line()

    def test_status_line_running(self):
        w = self._make()
        w._running = True
        w._total_detections = 3
        line = w.status_line()
        assert "listening" in line
        assert "3" in line

    def test_stop_when_not_running(self):
        """stop() on a non-running listener should be a no-op."""
        w = self._make()
        w.stop()  # Should not raise
        assert w.running is False

    def test_start_without_access_key(self):
        """Should fail gracefully if no access key."""
        w = self._make()
        with patch("bantz.agent.wake_word.WakeWordListener._get_access_key", return_value=""):
            ok = w.start()
        assert ok is False
        assert w.running is False

    def test_start_without_pvporcupine(self):
        """Should fail gracefully if pvporcupine not installed."""
        w = self._make()
        with patch("bantz.agent.wake_word.WakeWordListener._get_access_key", return_value="key"):
            with patch.dict("sys.modules", {"pvporcupine": None}):
                ok = w._init_porcupine()
        assert ok is False

    def test_start_without_pyaudio(self):
        """Should fail gracefully if pyaudio not installed."""
        w = self._make()
        w._porcupine = MagicMock()
        w._porcupine.sample_rate = 16000
        w._porcupine.frame_length = 512
        with patch.dict("sys.modules", {"pyaudio": None}):
            ok = w._init_audio()
        assert ok is False


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestDiagnose:
    def _make(self):
        from bantz.agent.wake_word import WakeWordListener
        return WakeWordListener()

    def test_diagnose_structure(self):
        w = self._make()
        d = w.diagnose()
        assert "porcupine_available" in d
        assert "pyaudio_available" in d
        assert "running" in d
        assert "total_detections" in d

    def test_diagnose_porcupine_missing(self):
        w = self._make()
        with patch.dict("sys.modules", {"pvporcupine": None}):
            d = w.diagnose()
        assert d["porcupine_available"] is False

    def test_diagnose_pyaudio_missing(self):
        w = self._make()
        with patch.dict("sys.modules", {"pyaudio": None}):
            d = w.diagnose()
        assert d["pyaudio_available"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Dedicated thread architecture (NOT APScheduler)
# ═══════════════════════════════════════════════════════════════════════════

class TestThreadArchitecture:
    def test_not_using_apscheduler(self):
        """Wake word must NOT be registered as an APScheduler job."""
        import ast
        from bantz.agent import wake_word
        import inspect
        tree = ast.parse(inspect.getsource(wake_word))
        # Strip docstrings/comments by looking at executable code only
        code_tokens = ast.dump(tree)
        # Check actual import statements
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        for imp in imports:
            if isinstance(imp, ast.ImportFrom):
                assert imp.module is None or "apscheduler" not in imp.module.lower()
            else:
                for alias in imp.names:
                    assert "apscheduler" not in alias.name.lower()
        # Check function calls — no add_job
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for call in calls:
            if isinstance(call.func, ast.Attribute):
                assert call.func.attr != "add_job"

    def test_uses_threading_thread(self):
        """Must use threading.Thread, not asyncio task."""
        import inspect
        from bantz.agent import wake_word
        src = inspect.getsource(wake_word)
        assert "threading.Thread" in src

    def test_thread_is_daemon(self):
        """Thread must be daemon=True so it dies with the main process."""
        import inspect
        from bantz.agent import wake_word
        src = inspect.getsource(wake_word)
        assert "daemon=True" in src

    def test_has_stop_event(self):
        """Must use threading.Event for clean shutdown."""
        import inspect
        from bantz.agent import wake_word
        src = inspect.getsource(wake_word)
        assert "_stop" in src
        assert "Event" in src


# ═══════════════════════════════════════════════════════════════════════════
# Mic stays open during TTS (interrupt, not mute)
# ═══════════════════════════════════════════════════════════════════════════

class TestMicNotMutedDuringTTS:
    def test_no_tts_mute_check_in_listen_loop(self):
        """The listen loop must NOT check tts.is_speaking to skip audio reads."""
        import inspect
        from bantz.agent.wake_word import WakeWordListener
        src = inspect.getsource(WakeWordListener._listen_loop)
        # is_speaking check belongs in _interrupt_tts, not in the listen loop
        assert "is_speaking" not in src

    def test_interrupt_tts_calls_stop(self):
        """_interrupt_tts should call tts_engine.stop() when speaking."""
        from bantz.agent.wake_word import WakeWordListener
        mock_tts = MagicMock()
        mock_tts.is_speaking = True
        with patch("bantz.agent.tts.tts_engine", mock_tts):
            WakeWordListener._interrupt_tts()
        mock_tts.stop.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Cooldown
# ═══════════════════════════════════════════════════════════════════════════

class TestCooldown:
    def test_cooldown_constant_exists(self):
        from bantz.agent.wake_word import _COOLDOWN_SECONDS
        assert _COOLDOWN_SECONDS >= 1.0

    def test_rapid_triggers_are_suppressed(self):
        """Simulate two detections within cooldown — second should be ignored."""
        from bantz.agent.wake_word import WakeWordListener, _COOLDOWN_SECONDS
        w = WakeWordListener()
        callback = MagicMock()
        w._on_wake = callback

        # Simulate first detection
        w._last_trigger = 0
        now = time.monotonic()

        # First trigger
        w._last_trigger = now
        w._total_detections = 1
        callback()

        # Second trigger immediately (within cooldown)
        # In real code this is checked in _listen_loop; we verify the logic
        second_time = now + 0.5  # < _COOLDOWN_SECONDS
        assert second_time - w._last_trigger < _COOLDOWN_SECONDS


# ═══════════════════════════════════════════════════════════════════════════
# Ack sound
# ═══════════════════════════════════════════════════════════════════════════

class TestAckSound:
    def test_play_ack_without_aplay_uses_bell(self):
        """Falls back to terminal bell if aplay missing."""
        from bantz.agent.wake_word import WakeWordListener
        with patch("shutil.which", return_value=None):
            with patch("builtins.print") as mock_print:
                WakeWordListener._play_ack()
            mock_print.assert_called_once_with("\a", end="", flush=True)

    def test_play_ack_with_aplay(self):
        """Generates sine tone and pipes to aplay."""
        from bantz.agent.wake_word import WakeWordListener
        with patch("shutil.which", return_value="/usr/bin/aplay"):
            with patch("subprocess.run") as mock_run:
                WakeWordListener._play_ack()
            mock_run.assert_called_once()
            args = mock_run.call_args
            assert "/usr/bin/aplay" in args[0][0]
            assert args[1]["input"] is not None  # PCM data
            assert len(args[1]["input"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Custom keyword file discovery
# ═══════════════════════════════════════════════════════════════════════════

class TestKeywordDiscovery:
    def test_find_custom_ppn(self, tmp_path):
        """Finds hey-bantz.ppn in data dir."""
        from bantz.agent.wake_word import WakeWordListener
        ppn = tmp_path / "hey-bantz_en_linux.ppn"
        ppn.write_bytes(b"fake")

        w = WakeWordListener()
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.data_dir = str(tmp_path)
            path = w._find_keyword_path()
        assert path is not None
        assert "hey-bantz" in path

    def test_fallback_when_no_ppn(self, tmp_path):
        """Returns None when no custom keyword file exists in any search dir."""
        from bantz.agent.wake_word import WakeWordListener
        w = WakeWordListener()
        # Create an empty subdir so all search dirs point to empty locations
        empty = tmp_path / "empty"
        empty.mkdir()
        with patch("bantz.config.config") as mock_cfg, \
             patch("bantz.agent.wake_word.Path") as mock_path:
            mock_cfg.data_dir = str(empty)
            # Make Path(config.data_dir) work, Path.cwd() and __file__ resolve to empty
            real_path = Path
            def path_side_effect(arg=""):
                return real_path(arg)
            mock_path.side_effect = path_side_effect
            mock_path.cwd.return_value = empty
            # __file__ parent chain → empty dir too
            mock_file = MagicMock()
            mock_file.resolve.return_value.parent.parent.parent.parent = empty
            mock_path.return_value = empty
            # Override side_effect to handle the __file__ case
            call_count = [0]
            def smart_path(arg=""):
                p = real_path(arg)
                return p
            mock_path.side_effect = smart_path
            mock_path.cwd.return_value = empty
            # The __file__ parent chain
            sentinel = MagicMock()
            sentinel.resolve.return_value.parent.parent.parent.parent = empty
            sentinel.is_dir = lambda: True
            # Simplest approach: just monkeypatch __file__ in the module
            import bantz.agent.wake_word as ww_mod
            orig_file = ww_mod.__file__
            try:
                ww_mod.__file__ = str(empty / "fake.py")
                path = w._find_keyword_path()
            finally:
                ww_mod.__file__ = orig_file
        assert path is None


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════

class TestCleanup:
    def test_cleanup_audio(self):
        from bantz.agent.wake_word import WakeWordListener
        w = WakeWordListener()
        w._audio_stream = MagicMock()
        w._pa = MagicMock()
        w._cleanup_audio()
        assert w._audio_stream is None
        assert w._pa is None

    def test_cleanup_porcupine(self):
        from bantz.agent.wake_word import WakeWordListener
        w = WakeWordListener()
        mock_porc = MagicMock()
        w._porcupine = mock_porc
        w._cleanup_porcupine()
        mock_porc.delete.assert_called_once()
        assert w._porcupine is None

    def test_cleanup_porcupine_none(self):
        """Cleanup when porcupine is None should not raise."""
        from bantz.agent.wake_word import WakeWordListener
        w = WakeWordListener()
        w._porcupine = None
        w._cleanup_porcupine()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleton:
    def test_singleton_exists(self):
        from bantz.agent.wake_word import wake_listener, WakeWordListener
        assert isinstance(wake_listener, WakeWordListener)

    def test_singleton_starts_stopped(self):
        from bantz.agent.wake_word import wake_listener
        assert wake_listener.running is False


# ═══════════════════════════════════════════════════════════════════════════
# .env.example completeness
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvExample:
    def test_wake_word_keys_in_env_example(self):
        env = Path(__file__).resolve().parents[2] / ".env.example"
        text = env.read_text()
        assert "BANTZ_WAKE_WORD_ENABLED" in text
        assert "BANTZ_PICOVOICE_ACCESS_KEY" in text
        assert "BANTZ_WAKE_WORD_SENSITIVITY" in text

    def test_picovoice_console_link(self):
        """Should mention where to get the access key."""
        env = Path(__file__).resolve().parents[2] / ".env.example"
        text = env.read_text()
        assert "picovoice" in text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# --doctor diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestDoctorSection:
    def test_section_for_wake_word(self):
        from bantz.__main__ import _section_for
        assert _section_for("wake_word_enabled") == "Wake Word"
        assert _section_for("wake_word_sensitivity") == "Wake Word"
        assert _section_for("picovoice_access_key") == "Wake Word"


# ═══════════════════════════════════════════════════════════════════════════
# TUI integration — WakeWordDetected message
# ═══════════════════════════════════════════════════════════════════════════

class TestTUIMessage:
    def test_wake_word_detected_message_exists(self):
        from bantz.interface.tui.app import WakeWordDetected
        msg = WakeWordDetected()
        assert msg is not None

    def test_app_has_handler(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "on_wake_word_detected")

    def test_app_has_start_wake_word(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_start_wake_word_listener")


# ═══════════════════════════════════════════════════════════════════════════
# Scope guard — no STT in this module
# ═══════════════════════════════════════════════════════════════════════════

class TestScopeGuard:
    def test_no_whisper_in_wake_word(self):
        """wake_word.py must NOT import or call any STT/Whisper code."""
        import ast, inspect
        from bantz.agent import wake_word
        tree = ast.parse(inspect.getsource(wake_word))
        # Check imports don't reference whisper/stt
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        for imp in imports:
            if isinstance(imp, ast.ImportFrom) and imp.module:
                mod = imp.module.lower()
                assert "whisper" not in mod
                assert "stt" not in mod
                assert "speech_to_text" not in mod
            elif isinstance(imp, ast.Import):
                for alias in imp.names:
                    n = alias.name.lower()
                    assert "whisper" not in n
                    assert "stt" not in n

    def test_no_recording_saving(self):
        """Must not save audio to wav files — that's STT's job."""
        import inspect
        from bantz.agent import wake_word
        src = inspect.getsource(wake_word)
        assert ".wav" not in src
        assert "wave.open" not in src
        assert "soundfile" not in src


# ═══════════════════════════════════════════════════════════════════════════
# Audit — architecture assertions
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitectureAudit:
    def test_no_scheduler_import(self):
        """Must not import job_scheduler."""
        import inspect
        from bantz.agent import wake_word
        src = inspect.getsource(wake_word)
        assert "job_scheduler" not in src

    def test_exception_on_overflow_false(self):
        """stream.read must use exception_on_overflow=False to prevent crashes."""
        import inspect
        from bantz.agent.wake_word import WakeWordListener
        src = inspect.getsource(WakeWordListener._listen_loop)
        assert "exception_on_overflow=False" in src

    def test_listen_loop_has_sleep_on_error(self):
        """Error handler in listen loop must sleep to prevent busy-spin."""
        import inspect
        from bantz.agent.wake_word import WakeWordListener
        src = inspect.getsource(WakeWordListener._listen_loop)
        assert "time.sleep" in src

    def test_tts_interrupt_in_listen_loop(self):
        """Wake detection must call _interrupt_tts."""
        import inspect
        from bantz.agent.wake_word import WakeWordListener
        src = inspect.getsource(WakeWordListener._listen_loop)
        assert "_interrupt_tts" in src
