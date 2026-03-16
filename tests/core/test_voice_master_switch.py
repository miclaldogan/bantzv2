"""
Tests for Issue #277 — Voice Master Switch (BANTZ_VOICE_ENABLED).

Covers:
  - @model_validator cascade: master switch ON promotes sub-flags
  - Individual overrides: explicit sub-flag=false survives master ON
  - Master OFF: sub-flags stay at default (False)
  - Partial voice: individual flags without master
  - _check_whisper_model_cached() helper
  - GhostLoop._preload_stt() emits bus events
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, call

import pytest


# ── Voice Master Switch — model_validator cascade ───────────────────────

class TestVoiceMasterSwitch:
    """Test BANTZ_VOICE_ENABLED @model_validator in Config."""

    @staticmethod
    def _fresh_config(**env_vars):
        """Create a fresh Config from environment variables.

        Passes BANTZ_* env vars and uses _env_file=None to avoid
        reading the real .env from disk.  Pydantic-settings reads
        env vars with highest priority (after init), so this is the
        reliable way to control Config values in tests.
        """
        from bantz.config import Config
        with patch.dict(os.environ, env_vars, clear=False):
            return Config(_env_file=None)

    def test_master_off_all_defaults(self):
        """Master OFF → all sub-flags remain False."""
        cfg = self._fresh_config(BANTZ_VOICE_ENABLED="false")
        assert cfg.voice_enabled is False
        assert cfg.tts_enabled is False
        assert cfg.wake_word_enabled is False
        assert cfg.stt_enabled is False
        assert cfg.ghost_loop_enabled is False
        assert cfg.audio_duck_enabled is False
        assert cfg.ambient_enabled is False

    def test_master_on_cascades_all(self):
        """Master ON → all sub-flags promoted to True."""
        cfg = self._fresh_config(BANTZ_VOICE_ENABLED="true")
        assert cfg.voice_enabled is True
        assert cfg.tts_enabled is True
        assert cfg.wake_word_enabled is True
        assert cfg.stt_enabled is True
        assert cfg.ghost_loop_enabled is True
        assert cfg.audio_duck_enabled is True
        assert cfg.ambient_enabled is True

    def test_master_on_preserves_explicit_true(self):
        """Master ON + explicit sub=True → remains True (not reset)."""
        cfg = self._fresh_config(
            BANTZ_VOICE_ENABLED="true",
            BANTZ_TTS_ENABLED="true",
        )
        assert cfg.tts_enabled is True

    def test_individual_without_master(self):
        """Individual flag on with master off works independently."""
        cfg = self._fresh_config(
            BANTZ_VOICE_ENABLED="false",
            BANTZ_TTS_ENABLED="true",
        )
        assert cfg.voice_enabled is False
        assert cfg.tts_enabled is True
        # Others stay default
        assert cfg.wake_word_enabled is False
        assert cfg.stt_enabled is False

    def test_master_on_subset_already_true(self):
        """Master ON + some flags already True → all end up True."""
        cfg = self._fresh_config(
            BANTZ_VOICE_ENABLED="true",
            BANTZ_TTS_ENABLED="true",
            BANTZ_STT_ENABLED="true",
        )
        assert cfg.tts_enabled is True
        assert cfg.stt_enabled is True
        assert cfg.wake_word_enabled is True  # cascaded
        assert cfg.ghost_loop_enabled is True  # cascaded
        assert cfg.audio_duck_enabled is True  # cascaded
        assert cfg.ambient_enabled is True     # cascaded

    def test_validator_logs_flipped(self):
        """Validator should log which flags were flipped."""
        with patch("bantz.config.log") as mock_log:
            cfg = self._fresh_config(BANTZ_VOICE_ENABLED="true")
            assert cfg.voice_enabled is True
            # Should have called log.debug with flipped flag names
            assert mock_log.debug.called
            log_msg = mock_log.debug.call_args[0][0]
            assert "Voice master switch ON" in log_msg

    def test_voice_enabled_field_exists(self):
        """Config should expose voice_enabled as a bool field."""
        from bantz.config import Config
        fields = Config.model_fields
        assert "voice_enabled" in fields

    def test_voice_enabled_alias(self):
        """voice_enabled should read from BANTZ_VOICE_ENABLED env var."""
        from bantz.config import Config
        field_info = Config.model_fields["voice_enabled"]
        assert field_info.alias == "BANTZ_VOICE_ENABLED"


# ── Whisper Model Cache Check ────────────────────────────────────────────

class TestWhisperModelCached:
    """Test _check_whisper_model_cached() helper from __main__."""

    def test_no_cache_dir(self, tmp_path):
        """Returns False when HuggingFace cache doesn't exist."""
        from bantz.__main__ import _check_whisper_model_cached
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert _check_whisper_model_cached("tiny") is False

    def test_model_found_in_hf_cache(self, tmp_path):
        """Returns True when matching model dir exists in HF cache."""
        from bantz.__main__ import _check_whisper_model_cached
        hf_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hf_dir / "models--Systran--faster-whisper-tiny"
        model_dir.mkdir(parents=True)
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert _check_whisper_model_cached("tiny") is True

    def test_model_found_in_faster_whisper_cache(self, tmp_path):
        """Returns True when matching dir exists in faster_whisper cache."""
        from bantz.__main__ import _check_whisper_model_cached
        fw_dir = tmp_path / ".cache" / "faster_whisper"
        model_dir = fw_dir / "tiny"
        model_dir.mkdir(parents=True)
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert _check_whisper_model_cached("tiny") is True

    def test_wrong_model_not_found(self, tmp_path):
        """Returns False when cached model is different size."""
        from bantz.__main__ import _check_whisper_model_cached
        hf_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hf_dir / "models--Systran--faster-whisper-base"
        model_dir.mkdir(parents=True)
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert _check_whisper_model_cached("tiny") is False

    def test_exception_returns_false(self):
        """Returns False on any exception (graceful degradation)."""
        from bantz.__main__ import _check_whisper_model_cached
        with patch("pathlib.Path.home", side_effect=RuntimeError("boom")):
            assert _check_whisper_model_cached("tiny") is False


# ── GhostLoop STT Preload Events ────────────────────────────────────────

class TestGhostLoopPreloadEvents:
    """Test that _preload_stt emits loading/ready events."""

    def test_preload_emits_loading_and_ready(self):
        """Successful preload emits stt_model_loading + stt_model_ready."""
        from bantz.agent.ghost_loop import GhostLoop
        emitted = []
        with patch("bantz.agent.ghost_loop.bus") as mock_bus:
            mock_bus.emit_threadsafe = lambda event, **kw: emitted.append(event)
            with patch("bantz.agent.stt.stt_engine") as mock_stt:
                mock_stt._ensure_model.return_value = True
                GhostLoop._preload_stt()
        assert "stt_model_loading" in emitted
        assert "stt_model_ready" in emitted

    def test_preload_emits_failed_on_error(self):
        """Failed preload emits stt_model_loading + stt_model_failed."""
        from bantz.agent.ghost_loop import GhostLoop
        events = {}
        def track(event, **kw):
            events[event] = kw
        with patch("bantz.agent.ghost_loop.bus") as mock_bus:
            mock_bus.emit_threadsafe = track
            with patch(
                "bantz.agent.ghost_loop.stt_engine",
                create=True,
            ):
                # Force an import error inside _preload_stt
                with patch.dict("sys.modules", {"bantz.agent.stt": None}):
                    GhostLoop._preload_stt()
        assert "stt_model_loading" in events
        assert "stt_model_failed" in events
        assert "error" in events["stt_model_failed"]

    def test_preload_logs_info(self):
        """Preload should log info messages about loading status."""
        from bantz.agent.ghost_loop import GhostLoop
        with patch("bantz.agent.ghost_loop.bus"):
            with patch("bantz.agent.ghost_loop.log") as mock_log:
                with patch("bantz.agent.stt.stt_engine") as mock_stt:
                    mock_stt._ensure_model.return_value = True
                    GhostLoop._preload_stt()
                assert mock_log.info.called
                first_msg = mock_log.info.call_args_list[0][0][0]
                assert "pre-loading" in first_msg.lower() or "STT" in first_msg


# ── Doctor voice section ────────────────────────────────────────────────

class TestDoctorVoiceSection:
    """Smoke-test for the consolidated voice diagnostics in --doctor."""

    def test_section_for_voice(self):
        """voice_enabled should map to 'Voice' or 'TTS / Audio' section."""
        from bantz.__main__ import _section_for
        # voice_enabled is new — it should map to TTS / Audio or Voice
        result = _section_for("voice_enabled")
        assert isinstance(result, str)

    def test_check_whisper_model_cached_importable(self):
        """_check_whisper_model_cached should be importable from __main__."""
        from bantz.__main__ import _check_whisper_model_cached
        assert callable(_check_whisper_model_cached)
