"""Tests for Issue #434 — Guided Voice Setup Wizard.

Covers:
  - _voice_test_mic() pass / fail / skip paths
  - _voice_test_stt() pass / fail paths
  - _voice_test_tts() pass / fail paths
  - _setup_voice() .env writing logic (keys written / removed correctly)
  - _setup_voice() picovoice key only written when provided
  - _setup_voice() BANTZ_WAKE_WORD_ENABLED only set when key is supplied
  - _handle_setup(["voice"]) dispatches to _setup_voice
  - _VOICE_PACKAGES constant contains all expected packages
  - _VOICE_ENV_PREFIXES constant contains required keys
  - No os.system() calls in _setup_voice (use subprocess for security)
"""
from __future__ import annotations

import inspect
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch



# ── Module-level constant sanity checks ───────────────────────────────────────

class TestVoicePackagesConstant:
    def test_faster_whisper_present(self):
        from bantz.cli.setup import _VOICE_PACKAGES
        assert "faster-whisper" in _VOICE_PACKAGES

    def test_pyaudio_present(self):
        from bantz.cli.setup import _VOICE_PACKAGES
        assert "pyaudio" in _VOICE_PACKAGES

    def test_webrtcvad_present(self):
        from bantz.cli.setup import _VOICE_PACKAGES
        assert "webrtcvad" in _VOICE_PACKAGES

    def test_pvporcupine_present(self):
        from bantz.cli.setup import _VOICE_PACKAGES
        assert "pvporcupine" in _VOICE_PACKAGES

    def test_import_names_are_strings(self):
        from bantz.cli.setup import _VOICE_PACKAGES
        for k, v in _VOICE_PACKAGES.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


class TestVoiceEnvPrefixesConstant:
    def test_voice_enabled_present(self):
        from bantz.cli.setup import _VOICE_ENV_PREFIXES
        assert "BANTZ_VOICE_ENABLED=" in _VOICE_ENV_PREFIXES

    def test_stt_enabled_present(self):
        from bantz.cli.setup import _VOICE_ENV_PREFIXES
        assert "BANTZ_STT_ENABLED=" in _VOICE_ENV_PREFIXES

    def test_ghost_loop_present(self):
        from bantz.cli.setup import _VOICE_ENV_PREFIXES
        assert "BANTZ_GHOST_LOOP_ENABLED=" in _VOICE_ENV_PREFIXES

    def test_picovoice_key_present(self):
        from bantz.cli.setup import _VOICE_ENV_PREFIXES
        assert "BANTZ_PICOVOICE_ACCESS_KEY=" in _VOICE_ENV_PREFIXES

    def test_wake_word_present(self):
        from bantz.cli.setup import _VOICE_ENV_PREFIXES
        assert "BANTZ_WAKE_WORD_ENABLED=" in _VOICE_ENV_PREFIXES


# ── _voice_test_mic ────────────────────────────────────────────────────────────

class TestVoiceTestMic:
    def test_pass_when_pyaudio_returns_data(self):
        from bantz.cli.setup import _voice_test_mic
        mock_stream = MagicMock()
        mock_stream.read.return_value = b"\x00" * 960  # 480 frames * 2 bytes
        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio_mod = MagicMock()
        mock_pyaudio_mod.PyAudio.return_value = mock_pa
        mock_pyaudio_mod.paInt16 = 8

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio_mod}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_mic()
        assert result is True
        assert "PASS" in buf.getvalue()

    def test_fail_when_no_audio_data(self):
        from bantz.cli.setup import _voice_test_mic
        mock_stream = MagicMock()
        mock_stream.read.return_value = b""  # empty
        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio_mod = MagicMock()
        mock_pyaudio_mod.PyAudio.return_value = mock_pa
        mock_pyaudio_mod.paInt16 = 8

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio_mod}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_mic()
        assert result is False
        assert "FAIL" in buf.getvalue()

    def test_skip_when_pyaudio_missing(self):
        from bantz.cli.setup import _voice_test_mic
        with patch.dict("sys.modules", {"pyaudio": None}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_mic()
        assert result is False
        assert "SKIP" in buf.getvalue() or "FAIL" in buf.getvalue()

    def test_fail_on_exception(self):
        from bantz.cli.setup import _voice_test_mic
        mock_pyaudio_mod = MagicMock()
        mock_pyaudio_mod.PyAudio.side_effect = OSError("no mic")
        mock_pyaudio_mod.paInt16 = 8

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio_mod}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_mic()
        assert result is False
        assert "FAIL" in buf.getvalue()


# ── _voice_test_stt ────────────────────────────────────────────────────────────

class TestVoiceTestSTT:
    def test_pass_when_faster_whisper_importable(self):
        from bantz.cli.setup import _voice_test_stt
        mock_fw = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_stt()
        assert result is True
        assert "PASS" in buf.getvalue()

    def test_fail_when_faster_whisper_missing(self):
        from bantz.cli.setup import _voice_test_stt
        with patch.dict("sys.modules", {"faster_whisper": None}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_stt()
        assert result is False
        assert "FAIL" in buf.getvalue()


# ── _voice_test_tts ────────────────────────────────────────────────────────────

class TestVoiceTestTTS:
    def test_pass_when_piper_binary_found(self, tmp_path):
        from bantz.cli.setup import _voice_test_tts
        # Create a fake piper binary
        piper_bin = tmp_path / "piper"
        piper_bin.write_text("#!/bin/sh\n")
        piper_bin.chmod(0o755)

        with patch("shutil.which", side_effect=lambda x: str(piper_bin) if x == "piper" else None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_tts()
        assert result is True
        assert "PASS" in buf.getvalue()

    def test_fail_when_no_piper_binary(self, tmp_path):
        from bantz.cli.setup import _voice_test_tts
        # Redirect home so no ~/.local/bin/piper or miniforge3/bin/piper found
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "home", return_value=tmp_path):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _voice_test_tts()
        assert result is False
        assert "FAIL" in buf.getvalue()


# ── _setup_voice .env writing ─────────────────────────────────────────────────

class TestSetupVoiceEnvWriting:
    """Test that _setup_voice() writes correct .env entries."""

    def _run_setup_voice(
        self,
        tmp_path: Path,
        *,
        existing_env: str = "",
        picovoice_key: str = "",
        install_answer: str = "n",
    ) -> str:
        """Helper: run _setup_voice() with mocked inputs; return resulting .env text."""
        from bantz.cli.setup import _setup_voice

        env_path = tmp_path / ".env"
        if existing_env:
            env_path.write_text(existing_env, encoding="utf-8")

        inputs = iter([
            install_answer,   # "Install now?" prompt
            picovoice_key,    # Picovoice Access Key prompt
        ])

        with patch("builtins.input", side_effect=lambda _: next(inputs)), \
             patch("bantz.cli.setup._voice_test_mic", return_value=True), \
             patch("bantz.cli.setup._voice_test_stt", return_value=True), \
             patch("bantz.cli.setup._voice_test_tts", return_value=True), \
             patch("subprocess.run"), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            _setup_voice()

        return env_path.read_text(encoding="utf-8")

    def test_voice_enabled_written(self, tmp_path):
        text = self._run_setup_voice(tmp_path)
        assert "BANTZ_VOICE_ENABLED=true" in text

    def test_tts_enabled_written(self, tmp_path):
        text = self._run_setup_voice(tmp_path)
        assert "BANTZ_TTS_ENABLED=true" in text

    def test_stt_enabled_written(self, tmp_path):
        text = self._run_setup_voice(tmp_path)
        assert "BANTZ_STT_ENABLED=true" in text

    def test_ghost_loop_enabled_written(self, tmp_path):
        text = self._run_setup_voice(tmp_path)
        assert "BANTZ_GHOST_LOOP_ENABLED=true" in text

    def test_picovoice_key_written_when_provided(self, tmp_path):
        text = self._run_setup_voice(tmp_path, picovoice_key="test-key-abc")
        assert "BANTZ_PICOVOICE_ACCESS_KEY=test-key-abc" in text

    def test_wake_word_enabled_when_key_provided(self, tmp_path):
        text = self._run_setup_voice(tmp_path, picovoice_key="test-key-abc")
        assert "BANTZ_WAKE_WORD_ENABLED=true" in text

    def test_picovoice_key_not_written_when_blank(self, tmp_path):
        text = self._run_setup_voice(tmp_path, picovoice_key="")
        assert "BANTZ_PICOVOICE_ACCESS_KEY=" not in text

    def test_wake_word_not_enabled_when_no_key(self, tmp_path):
        text = self._run_setup_voice(tmp_path, picovoice_key="")
        assert "BANTZ_WAKE_WORD_ENABLED=" not in text

    def test_existing_voice_entries_replaced(self, tmp_path):
        existing = (
            "BANTZ_VOICE_ENABLED=false\n"
            "BANTZ_STT_ENABLED=false\n"
            "SOME_OTHER_KEY=keep\n"
        )
        text = self._run_setup_voice(tmp_path, existing_env=existing)
        assert "BANTZ_VOICE_ENABLED=true" in text
        assert "BANTZ_VOICE_ENABLED=false" not in text
        assert "BANTZ_STT_ENABLED=true" in text
        assert "BANTZ_STT_ENABLED=false" not in text
        # Unrelated keys should be preserved
        assert "SOME_OTHER_KEY=keep" in text

    def test_env_file_permissions_are_0600(self, tmp_path):
        self._run_setup_voice(tmp_path)
        env_path = tmp_path / ".env"
        stat = env_path.stat()
        perms = oct(stat.st_mode & 0o777)
        assert perms == "0o600"

    def test_all_packages_present_skips_install_prompt(self, tmp_path):
        """When all packages are installed, no install prompt is shown."""
        from bantz.cli.setup import _setup_voice

        # Patch _VOICE_PACKAGES to empty dict → no missing packages → no install prompt
        # Only one input needed: picovoice key (blank)
        inputs = iter([""])  # picovoice key blank

        with patch("bantz.cli.setup._VOICE_PACKAGES", {}), \
             patch("builtins.input", side_effect=lambda _: next(inputs)), \
             patch("bantz.cli.setup._voice_test_mic", return_value=True), \
             patch("bantz.cli.setup._voice_test_stt", return_value=True), \
             patch("bantz.cli.setup._voice_test_tts", return_value=True), \
             patch("subprocess.run"), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            _setup_voice()

        assert (tmp_path / ".env").exists()


# ── _handle_setup dispatch ────────────────────────────────────────────────────

class TestHandleSetupVoiceDispatch:
    def test_voice_dispatches_to_setup_voice(self):
        from bantz.cli.setup import _handle_setup
        with patch("bantz.cli.setup._setup_voice") as mock_setup:
            _handle_setup(["voice"])
        mock_setup.assert_called_once_with()

    def test_voice_uppercase_dispatches(self):
        from bantz.cli.setup import _handle_setup
        with patch("bantz.cli.setup._setup_voice") as mock_setup:
            _handle_setup(["VOICE"])
        mock_setup.assert_called_once_with()

    def test_help_text_includes_voice(self):
        from bantz.cli.setup import _handle_setup
        buf = io.StringIO()
        with redirect_stdout(buf):
            _handle_setup(["unknown-target"])
        assert "voice" in buf.getvalue()


# ── Security: no os.system() in _setup_voice ──────────────────────────────────

class TestSetupVoiceSecurity:
    def test_no_os_system_calls(self):
        """_setup_voice must use subprocess, not os.system() (shell injection risk)."""
        from bantz.cli import setup as setup_mod
        src = inspect.getsource(setup_mod._setup_voice)
        assert "os.system(" not in src

    def test_pip_invoked_via_sys_executable(self):
        """pip is invoked as [sys.executable, '-m', 'pip', ...] not bare 'pip'."""
        from bantz.cli import setup as setup_mod
        src = inspect.getsource(setup_mod._setup_voice)
        assert "sys.executable" in src
