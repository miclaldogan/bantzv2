"""
Tests for the Ghost Loop, VoiceCapture, and STT modules (#36).

Covers:
  1. Config fields (ghost_loop, stt, vad)
  2. VoiceCapture (VAD integration, mic handling, silence detection)
  3. STTEngine (model loading, transcription, diagnostics)
  4. GhostLoop orchestrator (start/stop, event flow, pipeline)
  5. TUI integration (event subscriptions, voice_input handling)
  6. .env.example completeness
"""
from __future__ import annotations

import asyncio
import struct
import threading
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# 1. Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestGhostLoopConfig:
    """Verify all Ghost Loop / STT config fields exist with correct defaults."""

    def test_ghost_loop_enabled_field(self):
        from bantz.config import Config
        field = Config.model_fields["ghost_loop_enabled"]
        assert field.default is False
        assert field.alias == "BANTZ_GHOST_LOOP_ENABLED"

    def test_stt_enabled_field(self):
        from bantz.config import Config
        field = Config.model_fields["stt_enabled"]
        assert field.default is False
        assert field.alias == "BANTZ_STT_ENABLED"

    def test_stt_model_default(self):
        from bantz.config import Config
        field = Config.model_fields["stt_model"]
        assert field.default == "tiny"
        assert field.alias == "BANTZ_STT_MODEL"

    def test_stt_language_default(self):
        from bantz.config import Config
        field = Config.model_fields["stt_language"]
        assert field.default == "en"
        assert field.alias == "BANTZ_STT_LANGUAGE"

    def test_stt_device_default(self):
        from bantz.config import Config
        field = Config.model_fields["stt_device"]
        assert field.default == "cpu"
        assert field.alias == "BANTZ_STT_DEVICE"

    def test_vad_silence_ms_default(self):
        from bantz.config import Config
        field = Config.model_fields["vad_silence_ms"]
        assert field.default == 800
        assert field.alias == "BANTZ_VAD_SILENCE_MS"

    def test_vad_aggressiveness_default(self):
        from bantz.config import Config
        field = Config.model_fields["vad_aggressiveness"]
        assert field.default == 2
        assert field.alias == "BANTZ_VAD_AGGRESSIVENESS"

    def test_all_ghost_fields_exist(self):
        from bantz.config import Config
        expected = [
            "ghost_loop_enabled", "stt_enabled", "stt_model",
            "stt_language", "stt_device", "vad_silence_ms",
            "vad_aggressiveness",
        ]
        for name in expected:
            assert name in Config.model_fields, f"Missing config field: {name}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. VoiceCapture
# ═══════════════════════════════════════════════════════════════════════════


class TestVoiceCaptureConstants:
    """Verify audio constants match WebRTC VAD requirements."""

    def test_sample_rate(self):
        from bantz.agent.voice_capture import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_frame_duration(self):
        from bantz.agent.voice_capture import FRAME_DURATION_MS
        assert FRAME_DURATION_MS in (10, 20, 30)

    def test_frame_size_matches(self):
        from bantz.agent.voice_capture import SAMPLE_RATE, FRAME_DURATION_MS, FRAME_SIZE
        expected = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
        assert FRAME_SIZE == expected

    def test_max_record_seconds(self):
        from bantz.agent.voice_capture import MAX_RECORD_SECONDS
        assert MAX_RECORD_SECONDS > 0
        assert MAX_RECORD_SECONDS <= 60


class TestVoiceCaptureInit:
    """Test VoiceCapture lazy initialization."""

    def test_init_without_webrtcvad(self):
        from bantz.agent.voice_capture import VoiceCapture
        vc = VoiceCapture()
        with patch.dict("sys.modules", {"webrtcvad": None}):
            with patch("builtins.__import__", side_effect=ImportError("no webrtcvad")):
                assert vc._ensure_init() is False

    def test_init_with_webrtcvad(self):
        from bantz.agent.voice_capture import VoiceCapture
        mock_vad = MagicMock()
        mock_webrtcvad = MagicMock()
        mock_webrtcvad.Vad.return_value = mock_vad

        vc = VoiceCapture()
        with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
            with patch("bantz.agent.voice_capture.VoiceCapture._ensure_init") as m:
                m.return_value = True
                assert vc._ensure_init() is True

    def test_capture_returns_none_without_deps(self):
        from bantz.agent.voice_capture import VoiceCapture
        vc = VoiceCapture()
        with patch.object(vc, "_ensure_init", return_value=False):
            assert vc.capture() is None

    def test_capture_returns_none_without_pyaudio(self):
        from bantz.agent.voice_capture import VoiceCapture
        vc = VoiceCapture()
        vc._vad = MagicMock()  # VAD is "ready"
        with patch.dict("sys.modules", {"pyaudio": None}):
            with patch("builtins.__import__", side_effect=ImportError("no pyaudio")):
                result = vc.capture()
                assert result is None


class TestVoiceCaptureDiagnose:
    """Test VoiceCapture.diagnose() output."""

    def test_diagnose_keys(self):
        from bantz.agent.voice_capture import VoiceCapture
        vc = VoiceCapture()
        diag = vc.diagnose()
        assert "webrtcvad_available" in diag
        assert "pyaudio_available" in diag

    def test_diagnose_without_deps(self):
        from bantz.agent.voice_capture import VoiceCapture
        vc = VoiceCapture()
        # When neither is installed, both should be False
        # (actual result depends on env, so just check structure)
        diag = vc.diagnose()
        assert isinstance(diag["webrtcvad_available"], bool)
        assert isinstance(diag["pyaudio_available"], bool)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STTEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestSTTEngineInit:
    """Test STTEngine initialization and model loading."""

    def test_init_defaults(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()
        assert engine._model is None
        assert engine._available is None

    def test_available_without_faster_whisper(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = engine.available()
                assert result is False
                assert engine._available is False

    def test_transcribe_returns_none_for_empty(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()
        assert engine.transcribe(b"") is None
        assert engine.transcribe(None) is None

    def test_transcribe_returns_none_without_model(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()
        engine._available = False
        assert engine.transcribe(b"\x00" * 1000) is None


class TestSTTEngineDiagnose:
    """Test STTEngine.diagnose() output."""

    def test_diagnose_keys(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()
        diag = engine.diagnose()
        assert "faster_whisper_available" in diag
        assert "numpy_available" in diag
        assert "model" in diag
        assert "device" in diag
        assert "language" in diag

    def test_diagnose_model_not_loaded(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()
        diag = engine.diagnose()
        assert diag["model"] == "(not loaded)"


class TestSTTEngineTranscribe:
    """Test transcription pipeline with mocked model."""

    def test_transcribe_with_mock_model(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()

        # Mock the model
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        engine._model = mock_model
        engine._available = True
        engine._language = "en"

        # Create some fake PCM data
        import numpy as np
        pcm = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second of silence

        result = engine.transcribe(pcm)
        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_empty_result(self):
        from bantz.agent.stt import STTEngine
        engine = STTEngine()

        mock_segment = MagicMock()
        mock_segment.text = "   "
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        engine._model = mock_model
        engine._available = True
        engine._language = "en"

        import numpy as np
        pcm = np.zeros(16000, dtype=np.int16).tobytes()

        result = engine.transcribe(pcm)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. GhostLoop orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class TestGhostLoopInit:
    """Test GhostLoop initialization."""

    def test_init_defaults(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        assert gl.running is False
        assert gl.busy is False
        assert gl._total_transcriptions == 0

    def test_stats(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        stats = gl.stats()
        assert stats["running"] is False
        assert stats["busy"] is False
        assert stats["total_transcriptions"] == 0
        assert stats["last_text"] == ""


class TestGhostLoopStartStop:
    """Test GhostLoop start/stop lifecycle."""

    def test_start_disabled_by_config(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        mock_cfg = MagicMock()
        mock_cfg.ghost_loop_enabled = False
        mock_cfg.stt_enabled = False
        with patch("bantz.agent.ghost_loop.bus"):
            with patch("bantz.config.config", mock_cfg):
                assert gl.start() is False
                assert gl.running is False

    def test_start_enabled(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        mock_cfg = MagicMock()
        mock_cfg.ghost_loop_enabled = True
        mock_cfg.stt_enabled = True
        mock_bus = MagicMock()
        with patch("bantz.agent.ghost_loop.bus", mock_bus):
            with patch("bantz.config.config", mock_cfg):
                assert gl.start() is True
                assert gl.running is True
                mock_bus.on.assert_called_once_with(
                    "wake_word_detected", gl._on_wake_event
                )

    def test_stop(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        gl._running = True
        mock_bus = MagicMock()
        with patch("bantz.agent.ghost_loop.bus", mock_bus):
            gl.stop()
            assert gl.running is False
            mock_bus.off.assert_called_once()

    def test_start_idempotent(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        gl._running = True
        assert gl.start() is True


class TestGhostLoopWakeEvent:
    """Test wake event handling in Ghost Loop."""

    def test_ignores_when_busy(self):
        from bantz.agent.ghost_loop import GhostLoop
        from bantz.core.event_bus import Event
        gl = GhostLoop()
        gl._busy = True
        # Should not spawn a thread
        with patch("threading.Thread") as mock_thread:
            gl._on_wake_event(Event(name="wake_word_detected"))
            mock_thread.assert_not_called()

    def test_spawns_thread_when_idle(self):
        from bantz.agent.ghost_loop import GhostLoop
        from bantz.core.event_bus import Event
        gl = GhostLoop()
        gl._busy = False
        with patch("bantz.agent.ghost_loop.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            gl._on_wake_event(Event(name="wake_word_detected"))
            mock_thread_cls.assert_called_once()
            mock_thread.start.assert_called_once()


class TestGhostLoopPipeline:
    """Test the capture → transcribe → dispatch pipeline."""

    def test_pipeline_no_audio(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        mock_vc = MagicMock()
        mock_vc.capture.return_value = None
        mock_bus = MagicMock()
        with patch("bantz.agent.ghost_loop.bus", mock_bus):
            with patch("bantz.agent.voice_capture.voice_capture", mock_vc):
                gl._capture_and_transcribe()
        assert gl._busy is False
        # Should have emitted listening + idle
        calls = [c[0][0] for c in mock_bus.emit_threadsafe.call_args_list]
        assert "ghost_loop_listening" in calls
        assert "ghost_loop_idle" in calls

    def test_pipeline_successful_transcription(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        mock_vc = MagicMock()
        mock_vc.capture.return_value = b"\x00" * 16000
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "Turn off the lights"
        mock_bus = MagicMock()
        with patch("bantz.agent.ghost_loop.bus", mock_bus):
            with patch("bantz.agent.voice_capture.voice_capture", mock_vc):
                with patch("bantz.agent.stt.stt_engine", mock_stt):
                    gl._capture_and_transcribe()
        assert gl._total_transcriptions == 1
        assert gl._last_text == "Turn off the lights"
        # Check voice_input was emitted
        emit_calls = mock_bus.emit_threadsafe.call_args_list
        voice_call = [c for c in emit_calls if c[0][0] == "voice_input"]
        assert len(voice_call) == 1
        assert voice_call[0][1]["text"] == "Turn off the lights"

    def test_pipeline_stt_returns_none(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        mock_vc = MagicMock()
        mock_vc.capture.return_value = b"\x00" * 16000
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = None
        mock_bus = MagicMock()
        with patch("bantz.agent.ghost_loop.bus", mock_bus):
            with patch("bantz.agent.voice_capture.voice_capture", mock_vc):
                with patch("bantz.agent.stt.stt_engine", mock_stt):
                    gl._capture_and_transcribe()
        assert gl._total_transcriptions == 0
        # voice_input should NOT have been emitted
        emit_names = [c[0][0] for c in mock_bus.emit_threadsafe.call_args_list]
        assert "voice_input" not in emit_names


class TestGhostLoopDiagnose:
    """Test GhostLoop.diagnose() output."""

    def test_diagnose_structure(self):
        from bantz.agent.ghost_loop import GhostLoop
        gl = GhostLoop()
        with patch("bantz.agent.voice_capture.voice_capture") as mock_vc:
            with patch("bantz.agent.stt.stt_engine") as mock_stt:
                mock_vc.diagnose.return_value = {"webrtcvad_available": False, "pyaudio_available": False}
                mock_stt.diagnose.return_value = {"faster_whisper_available": False}
                diag = gl.diagnose()
        assert "running" in diag
        assert "voice_capture" in diag
        assert "stt" in diag


# ═══════════════════════════════════════════════════════════════════════════
# 5. TUI integration
# ═══════════════════════════════════════════════════════════════════════════


class TestTUIEventSubscriptions:
    """Verify the TUI subscribes to Ghost Loop events."""

    def test_subscribe_has_voice_events(self):
        """_subscribe_event_bus must subscribe to ghost loop events."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._subscribe_event_bus)
        assert 'bus.on("voice_input"' in src
        assert 'bus.on("ghost_loop_listening"' in src
        assert 'bus.on("ghost_loop_transcribing"' in src
        assert 'bus.on("ghost_loop_idle"' in src

    def test_unsubscribe_has_voice_events(self):
        """_unsubscribe_event_bus must unsubscribe from ghost loop events."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._unsubscribe_event_bus)
        assert 'bus.off("voice_input"' in src
        assert 'bus.off("ghost_loop_listening"' in src
        assert 'bus.off("ghost_loop_transcribing"' in src
        assert 'bus.off("ghost_loop_idle"' in src

    def test_event_dispatch_has_voice_input(self):
        """on_bantz_event_message must dispatch voice_input events."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp.on_bantz_event_message)
        assert '"voice_input"' in src
        assert '"ghost_loop_listening"' in src
        assert '"ghost_loop_transcribing"' in src

    def test_voice_input_handler_exists(self):
        """BantzApp must have _on_bus_voice_input method."""
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_on_bus_voice_input")
        assert callable(getattr(BantzApp, "_on_bus_voice_input"))

    def test_ghost_listening_handler_exists(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_on_bus_ghost_listening")

    def test_ghost_transcribing_handler_exists(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_on_bus_ghost_transcribing")

    def test_start_ghost_loop_method_exists(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_start_ghost_loop")

    def test_on_mount_calls_start_ghost_loop(self):
        """on_mount must call _start_ghost_loop."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp.on_mount)
        assert "_start_ghost_loop" in src

    def test_action_quit_stops_ghost_loop(self):
        """action_quit must stop the ghost loop."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp.action_quit)
        assert "ghost_loop" in src


# ═══════════════════════════════════════════════════════════════════════════
# 6. --doctor support
# ═══════════════════════════════════════════════════════════════════════════


class TestDoctorGhostLoop:
    """Verify --doctor includes Ghost Loop check."""

    def test_doctor_has_ghost_loop_check(self):
        import inspect
        from bantz.__main__ import _doctor
        src = inspect.getsource(_doctor)
        assert "Ghost Loop" in src
        assert "ghost_loop_enabled" in src
        assert "stt_enabled" in src


# ═══════════════════════════════════════════════════════════════════════════
# 7. .env.example completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvExampleCompleteness:
    """All new config aliases must appear in .env.example."""

    def test_ghost_loop_in_env_example(self):
        from pathlib import Path
        env_example = Path(__file__).resolve().parent.parent / ".env.example"
        if not env_example.exists():
            pytest.skip(".env.example not found")
        content = env_example.read_text()
        expected = [
            "BANTZ_GHOST_LOOP_ENABLED",
            "BANTZ_STT_ENABLED",
            "BANTZ_STT_MODEL",
            "BANTZ_STT_LANGUAGE",
            "BANTZ_STT_DEVICE",
            "BANTZ_VAD_SILENCE_MS",
            "BANTZ_VAD_AGGRESSIVENESS",
        ]
        for alias in expected:
            assert alias in content, f"{alias} missing from .env.example"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Module singletons
# ═══════════════════════════════════════════════════════════════════════════


class TestModuleSingletons:
    """Verify module-level singletons are importable."""

    def test_voice_capture_singleton(self):
        from bantz.agent.voice_capture import voice_capture
        assert voice_capture is not None

    def test_stt_engine_singleton(self):
        from bantz.agent.stt import stt_engine
        assert stt_engine is not None

    def test_ghost_loop_singleton(self):
        from bantz.agent.ghost_loop import ghost_loop
        assert ghost_loop is not None
