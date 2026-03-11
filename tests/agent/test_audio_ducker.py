"""
Tests — Issue #171: Audio Ducking

Covers:
  - AudioDucker: init, duck, restore, fade, diagnose, stats, status_line
  - pactl output parsing (_parse_sink_inputs)
  - PULSE_PROP tagging on TTS aplay (BantzTTS label)
  - BantzTTS streams are excluded from ducking
  - Config: audio_duck_enabled, audio_duck_pct
  - .env.example completeness
  - --doctor diagnostics
  - _section_for mapping
  - Architecture: pactl-only (no wpctl), no PID matching, sync (no async fade)
  - Thread safety (lock usage)
  - TTSEngine speak() duck/restore integration
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════

class TestAudioDuckConfig:
    def test_audio_duck_enabled_default(self):
        from bantz.config import Config
        c = Config()
        assert c.audio_duck_enabled is False

    def test_audio_duck_enabled_true(self):
        from bantz.config import Config
        c = Config(BANTZ_AUDIO_DUCK_ENABLED="true")
        assert c.audio_duck_enabled is True

    def test_audio_duck_pct_default(self):
        from bantz.config import Config
        c = Config()
        assert c.audio_duck_pct == 30

    def test_audio_duck_pct_custom(self):
        from bantz.config import Config
        c = Config(BANTZ_AUDIO_DUCK_PCT="20")
        assert c.audio_duck_pct == 20


# ═══════════════════════════════════════════════════════════════════════════
# AudioDucker — core
# ═══════════════════════════════════════════════════════════════════════════

class TestAudioDucker:
    def _make(self):
        from bantz.agent.audio_ducker import AudioDucker
        return AudioDucker()

    def test_initial_state(self):
        d = self._make()
        assert d.is_ducked is False
        assert d._saved == {}

    def test_available_when_pactl_exists(self):
        d = self._make()
        with patch("shutil.which", return_value="/usr/bin/pactl"):
            assert d.available() is True

    def test_available_when_pactl_missing(self):
        d = self._make()
        with patch("shutil.which", return_value=None):
            assert d.available() is False

    def test_duck_idempotent(self):
        """Second duck() call should be no-op."""
        d = self._make()
        d._ducked = True
        assert d.duck() is True  # no-op, already ducked

    def test_restore_idempotent(self):
        """restore() when not ducked is no-op."""
        d = self._make()
        d.restore()  # should not raise
        assert d.is_ducked is False

    def test_duck_no_pactl(self):
        d = self._make()
        d._pactl = ""
        assert d.duck() is False

    def test_restore_no_pactl(self):
        d = self._make()
        d._ducked = True
        d._saved = {1: 100}
        d._pactl = None
        d.restore()
        assert d.is_ducked is False


# ═══════════════════════════════════════════════════════════════════════════
# pactl output parsing
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_PACTL_OUTPUT = textwrap.dedent("""\
    Sink Input #42
        Driver: protocol-native.c
        Owner Module: 9
        Client: 82
        Sink: 0
        Sample Specification: s16le 2ch 44100Hz
        Channel Map: front-left,front-right
        Format: pcm, format.sample_format = "\"s16le\""
        Corked: no
        Mute: no
        Volume: front-left: 48000 /  73% / -8.09 dB,   front-right: 48000 /  73% / -8.09 dB
                balance 0.00
        Buffer Latency: 0 usec
        Sink Latency: 40000 usec
        Resample method: copy
        Properties:
            media.name = "Playback"
            application.name = "Spotify"
            native-protocol.peer = "UNIX socket client"
            native-protocol.version = "35"
    Sink Input #55
        Driver: protocol-native.c
        Owner Module: 9
        Client: 90
        Sink: 0
        Sample Specification: s16le 1ch 22050Hz
        Channel Map: mono
        Volume: front-left: 65536 / 100% / 0.00 dB
        Properties:
            media.name = "ALSA Playback"
            application.name = "BantzTTS"
""")

_SAMPLE_PACTL_SINGLE = textwrap.dedent("""\
    Sink Input #10
        Volume: front-left: 32768 /  50% / -18.06 dB
        Properties:
            application.name = "Firefox"
""")


class TestParseSinkInputs:
    def test_parse_two_inputs(self):
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs(_SAMPLE_PACTL_OUTPUT)
        assert len(result) == 2

    def test_parse_spotify_entry(self):
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs(_SAMPLE_PACTL_OUTPUT)
        spotify = next(r for r in result if r["app_name"] == "Spotify")
        assert spotify["id"] == 42
        assert spotify["volume"] == 73

    def test_parse_bantz_entry(self):
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs(_SAMPLE_PACTL_OUTPUT)
        bantz = next(r for r in result if r["app_name"] == "BantzTTS")
        assert bantz["id"] == 55
        assert bantz["volume"] == 100

    def test_parse_single_input(self):
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs(_SAMPLE_PACTL_SINGLE)
        assert len(result) == 1
        assert result[0]["id"] == 10
        assert result[0]["app_name"] == "Firefox"
        assert result[0]["volume"] == 50

    def test_parse_empty_output(self):
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs("")
        assert result == []

    def test_parse_no_volume_defaults_100(self):
        raw = "Sink Input #99\n    Properties:\n        application.name = \"VLC\"\n"
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs(raw)
        assert result[0]["volume"] == 100

    def test_parse_no_app_name_defaults_empty(self):
        raw = "Sink Input #99\n    Volume: front-left: 65536 / 100% / 0.00 dB\n"
        from bantz.agent.audio_ducker import AudioDucker
        result = AudioDucker._parse_sink_inputs(raw)
        assert result[0]["app_name"] == ""


# ═══════════════════════════════════════════════════════════════════════════
# BantzTTS exclusion
# ═══════════════════════════════════════════════════════════════════════════

class TestBantzExclusion:
    def test_bantz_stream_not_ducked(self):
        """BantzTTS stream must NOT be volume-ducked."""
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        d._duck_pct = 30

        with patch.object(d, "_get_sink_inputs", return_value=[
            {"id": 42, "app_name": "Spotify", "volume": 100},
            {"id": 55, "app_name": "BantzTTS", "volume": 100},
        ]):
            with patch.object(d, "_set_volume") as mock_vol:
                d.duck()

        # Only Spotify should be ducked, not BantzTTS
        mock_vol.assert_called_once_with(42, 30)
        assert d._saved == {42: 100}

    def test_only_bantz_stream_present(self):
        """If only BantzTTS is playing, duck() returns False (nothing to duck)."""
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"

        with patch.object(d, "_get_sink_inputs", return_value=[
            {"id": 55, "app_name": "BantzTTS", "volume": 100},
        ]):
            result = d.duck()

        assert result is False
        assert not d.is_ducked


# ═══════════════════════════════════════════════════════════════════════════
# Duck + Restore lifecycle
# ═══════════════════════════════════════════════════════════════════════════

class TestDuckRestore:
    def test_duck_and_restore_cycle(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        d._duck_pct = 30

        with patch.object(d, "_get_sink_inputs", return_value=[
            {"id": 1, "app_name": "Firefox", "volume": 80},
            {"id": 2, "app_name": "Spotify", "volume": 65},
        ]):
            with patch.object(d, "_set_volume") as mock_vol:
                d.duck()

        assert d.is_ducked
        assert d._saved == {1: 80, 2: 65}
        assert mock_vol.call_count == 2
        mock_vol.assert_any_call(1, 30)
        mock_vol.assert_any_call(2, 30)

        # Restore
        with patch.object(d, "_set_volume") as mock_vol:
            d.restore()

        assert not d.is_ducked
        assert d._saved == {}
        mock_vol.assert_any_call(1, 80)
        mock_vol.assert_any_call(2, 65)

    def test_duck_no_inputs(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        with patch.object(d, "_get_sink_inputs", return_value=[]):
            assert d.duck() is False


# ═══════════════════════════════════════════════════════════════════════════
# Fade
# ═══════════════════════════════════════════════════════════════════════════

class TestFade:
    def test_duck_with_fade_sets_multiple_volumes(self):
        from bantz.agent.audio_ducker import AudioDucker, _FADE_STEPS
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        d._duck_pct = 30

        with patch.object(d, "_get_sink_inputs", return_value=[
            {"id": 1, "app_name": "Firefox", "volume": 90},
        ]):
            with patch.object(d, "_set_volume") as mock_vol:
                with patch("time.sleep"):
                    d.duck_with_fade()

        assert d.is_ducked
        # Should have called set_volume _FADE_STEPS times for 1 stream
        assert mock_vol.call_count == _FADE_STEPS

    def test_restore_with_fade(self):
        from bantz.agent.audio_ducker import AudioDucker, _FADE_STEPS
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        d._duck_pct = 30
        d._ducked = True
        d._saved = {1: 90}

        with patch.object(d, "_set_volume") as mock_vol:
            with patch("time.sleep"):
                d.restore_with_fade()

        assert not d.is_ducked
        assert mock_vol.call_count == _FADE_STEPS


# ═══════════════════════════════════════════════════════════════════════════
# set_volume clamping
# ═══════════════════════════════════════════════════════════════════════════

class TestSetVolume:
    def test_volume_clamped_to_150(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        with patch("subprocess.run") as mock_run:
            d._set_volume(1, 200)
        # Should clamp to 150%
        args = mock_run.call_args[0][0]
        assert "150%" in args

    def test_volume_clamped_to_0(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        with patch("subprocess.run") as mock_run:
            d._set_volume(1, -10)
        args = mock_run.call_args[0][0]
        assert "0%" in args

    def test_set_volume_no_pactl(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = ""
        assert d._set_volume(1, 50) is False


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestDiagnose:
    def test_diagnose_structure(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        diag = d.diagnose()
        assert "pactl_available" in diag
        assert "ducked" in diag
        assert "saved_streams" in diag
        assert "duck_pct" in diag

    def test_stats(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        s = d.stats()
        assert s["ducked"] is False
        assert s["duck_pct"] == 30

    def test_status_line_unavailable(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = ""
        assert "unavailable" in d.status_line()

    def test_status_line_idle(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        assert "idle" in d.status_line()

    def test_status_line_active(self):
        from bantz.agent.audio_ducker import AudioDucker
        d = AudioDucker()
        d._pactl = "/usr/bin/pactl"
        d._ducked = True
        assert "active" in d.status_line()


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleton:
    def test_singleton_exists(self):
        from bantz.agent.audio_ducker import audio_ducker, AudioDucker
        assert isinstance(audio_ducker, AudioDucker)

    def test_singleton_starts_idle(self):
        from bantz.agent.audio_ducker import audio_ducker
        assert audio_ducker.is_ducked is False


# ═══════════════════════════════════════════════════════════════════════════
# PULSE_PROP tagging in TTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPulsePropTag:
    def test_play_method_sets_pulse_prop(self):
        """_play() must set PULSE_PROP='application.name=BantzTTS' in env."""
        import inspect
        from bantz.agent.tts import TTSEngine
        src = inspect.getsource(TTSEngine._play)
        assert "PULSE_PROP" in src
        assert "BantzTTS" in src

    def test_play_passes_env_to_subprocess(self):
        """_play() must pass env= to create_subprocess_exec."""
        import inspect
        from bantz.agent.tts import TTSEngine
        src = inspect.getsource(TTSEngine._play)
        assert "env=env" in src

    def test_no_pid_matching(self):
        """Must NOT use PID matching to filter Bantz's audio."""
        import inspect
        from bantz.agent import audio_ducker
        src = inspect.getsource(audio_ducker)
        assert "getpid" not in src
        assert "os.getpid" not in src


# ═══════════════════════════════════════════════════════════════════════════
# TTS speak() integration
# ═══════════════════════════════════════════════════════════════════════════

class TestTTSSpeakDucking:
    def test_speak_ducks_and_restores(self):
        """speak() should call duck() before and restore() after playing."""
        import inspect
        from bantz.agent.tts import TTSEngine
        src = inspect.getsource(TTSEngine.speak)
        assert "audio_ducker" in src
        assert ".duck()" in src
        assert ".restore()" in src

    def test_restore_in_finally(self):
        """restore() must be in a finally block for crash safety."""
        import inspect
        from bantz.agent.tts import TTSEngine
        src = inspect.getsource(TTSEngine.speak)
        # The restore call must appear after a 'finally:' line
        lines = src.splitlines()
        in_finally = False
        restore_in_finally = False
        for line in lines:
            if "finally:" in line:
                in_finally = True
            if in_finally and ".restore()" in line:
                restore_in_finally = True
                break
        assert restore_in_finally, "audio_ducker.restore() must be in finally block"


# ═══════════════════════════════════════════════════════════════════════════
# .env.example
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvExample:
    def test_ducking_keys_in_env(self):
        env = Path(__file__).resolve().parents[2] / ".env.example"
        text = env.read_text()
        assert "BANTZ_AUDIO_DUCK_ENABLED" in text
        assert "BANTZ_AUDIO_DUCK_PCT" in text

    def test_pactl_mentioned_in_env(self):
        env = Path(__file__).resolve().parents[2] / ".env.example"
        text = env.read_text()
        assert "pactl" in text.lower()

    def test_pulse_prop_mentioned(self):
        env = Path(__file__).resolve().parents[2] / ".env.example"
        text = env.read_text()
        assert "PULSE_PROP" in text


# ═══════════════════════════════════════════════════════════════════════════
# --doctor section
# ═══════════════════════════════════════════════════════════════════════════

class TestDoctorSection:
    def test_section_for_audio_duck(self):
        from bantz.__main__ import _section_for
        assert _section_for("audio_duck_enabled") == "Audio Ducking"
        assert _section_for("audio_duck_pct") == "Audio Ducking"


# ═══════════════════════════════════════════════════════════════════════════
# Architecture audit
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitectureAudit:
    def test_no_wpctl(self):
        """Must NOT use wpctl (PipeWire-native CLI) — pactl only."""
        import inspect
        from bantz.agent import audio_ducker
        src = inspect.getsource(audio_ducker)
        # wpctl may appear in docstring as explanation; check actual code
        import ast
        tree = ast.parse(inspect.getsource(audio_ducker))
        # Check no subprocess call uses 'wpctl'
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "wpctl" in node.value and node.value != "wpctl":
                    continue  # It's in a larger string (docstring)
                if node.value == "wpctl":
                    pytest.fail("Found wpctl usage in audio_ducker code")

    def test_pactl_only(self):
        """Must use pactl for all PulseAudio operations."""
        import inspect
        from bantz.agent.audio_ducker import AudioDucker
        src = inspect.getsource(AudioDucker._get_sink_inputs)
        assert "pactl" in src

    def test_sync_not_async(self):
        """Duck/restore must be synchronous (no async def)."""
        import inspect
        from bantz.agent.audio_ducker import AudioDucker
        assert not inspect.iscoroutinefunction(AudioDucker.duck)
        assert not inspect.iscoroutinefunction(AudioDucker.restore)

    def test_thread_lock(self):
        """Must use threading.Lock for thread safety."""
        import inspect
        from bantz.agent.audio_ducker import AudioDucker
        src = inspect.getsource(AudioDucker)
        assert "threading.Lock" in src or "_lock" in src

    def test_bantz_app_name_constant(self):
        from bantz.agent.audio_ducker import _BANTZ_APP_NAME
        assert _BANTZ_APP_NAME == "BantzTTS"

    def test_fade_constants(self):
        from bantz.agent.audio_ducker import _FADE_STEPS, _FADE_STEP_MS
        assert _FADE_STEPS >= 2
        assert _FADE_STEP_MS >= 5
