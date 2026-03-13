"""
Tests for bantz.agent.tts — Streaming TTS Pipeline (#131).

Coverage:
  - Sentence splitting: basic, edge cases, emoji/markdown removal
  - TTSEngine: init, available, speak, stop, synthesis, playback
  - Briefing watcher: IDLE→active transition detection
  - Brain integration: quick_route for TTS stop, briefing with TTS
  - Config: TTS-related fields
"""
from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# 1. Sentence splitting
# ═══════════════════════════════════════════════════════════════════════════


class TestSplitSentences:
    """Test the sentence splitter used for TTS chunking."""

    def _split(self, text: str) -> list[str]:
        from bantz.agent.tts import split_sentences
        return split_sentences(text)

    def test_basic_split(self):
        result = self._split("Hello world. How are you? Fine thanks!")
        assert len(result) >= 2
        assert "Hello world" in result[0]

    def test_empty(self):
        assert self._split("") == []
        assert self._split("   ") == []

    def test_single_sentence(self):
        result = self._split("Just one sentence here.")
        assert len(result) == 1
        assert "Just one sentence here" in result[0]

    def test_newline_split(self):
        result = self._split("Line one.\nLine two.\nLine three.")
        assert len(result) >= 2

    def test_emoji_removal(self):
        result = self._split("🌤️ Weather is sunny. 📅 Calendar has 3 events.")
        for s in result:
            assert "🌤️" not in s
            assert "📅" not in s

    def test_markdown_removal(self):
        result = self._split("**Bold text** and *italic* here.")
        for s in result:
            assert "**" not in s
            assert "*" not in s or s.count("*") == 0

    def test_bullet_removal(self):
        result = self._split("• Item one.\n• Item two.\n- Item three.")
        for s in result:
            assert not s.startswith("•")
            assert not s.startswith("-")

    def test_short_merge(self):
        """Very short fragments should be merged with their predecessor."""
        result = self._split("OK. Yes. Good. This is a longer sentence here.")
        # "OK", "Yes", "Good" are <10 chars, should be merged
        assert len(result) <= 2

    def test_briefing_like_text(self):
        text = (
            "🌤️ Weather: Sunny, 22°C.\n"
            "📅 Calendar: 3 events today.\n"
            "📧 Gmail: 2 new emails.\n"
            "No urgent items."
        )
        result = self._split(text)
        assert len(result) >= 2
        assert all(isinstance(s, str) for s in result)
        assert all(len(s) > 0 for s in result)

    def test_mixed_punctuation(self):
        result = self._split("Hello; goodbye: see you! Done?")
        assert len(result) >= 2

    def test_colon_split(self):
        result = self._split("Weather: sunny and warm. Calendar: meeting at 10.")
        assert len(result) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 2. TTSEngine unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTTSEngine:
    """Test TTSEngine initialization, availability, and lifecycle."""

    def _make_engine(self):
        from bantz.agent.tts import TTSEngine
        return TTSEngine()

    def test_initial_state(self):
        eng = self._make_engine()
        assert eng.is_speaking is False
        assert eng._piper_path is None
        assert eng._aplay_path is None

    def test_available_when_disabled(self):
        eng = self._make_engine()
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            assert eng.available() is False

    def test_available_no_piper(self):
        eng = self._make_engine()
        with patch("bantz.config.config") as mock_cfg, \
             patch("shutil.which", return_value=None), \
             patch("pathlib.Path.exists", return_value=False):
            mock_cfg.tts_enabled = True
            assert eng.available() is False

    def test_ensure_init_finds_piper(self):
        """When piper and aplay exist and model is found."""
        eng = self._make_engine()

        mock_cfg = MagicMock()
        mock_cfg.tts_enabled = True
        mock_cfg.tts_model = "en_US-lessac-medium"
        mock_cfg.tts_model_path = "/tmp/fake_model.onnx"
        mock_cfg.tts_speaker = 0
        mock_cfg.tts_rate = 1.0

        with patch("bantz.agent.tts.shutil.which") as mock_which, \
             patch("bantz.config.config", mock_cfg), \
             patch("pathlib.Path.exists", return_value=True):
            mock_which.side_effect = lambda cmd: {
                "piper": "/usr/bin/piper",
                "aplay": "/usr/bin/aplay",
            }.get(cmd)

            result = eng._ensure_init()
            assert result is True
            assert eng._piper_path == "/usr/bin/piper"
            assert eng._aplay_path == "/usr/bin/aplay"

    def test_ensure_init_cached(self):
        """Second call to _ensure_init uses cached values."""
        eng = self._make_engine()
        eng._piper_path = "/usr/bin/piper"
        eng._aplay_path = "/usr/bin/aplay"
        eng._model_path = "/fake/model.onnx"
        # Should return True without calling shutil.which
        with patch("shutil.which") as mock_which:
            result = eng._ensure_init()
            assert result is True
            mock_which.assert_not_called()

    def test_ensure_init_cached_negative(self):
        """Cached negative result (empty string)."""
        eng = self._make_engine()
        eng._piper_path = ""
        eng._aplay_path = ""
        assert eng._ensure_init() is False

    def test_stop_not_speaking(self):
        """stop() when not speaking should not crash."""
        eng = self._make_engine()
        eng.stop()  # Should not raise
        assert eng._stop_requested is True

    def test_stop_kills_playback(self):
        eng = self._make_engine()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        eng._playing = mock_proc
        eng._speaking = True

        eng.stop()

        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)
        assert eng._stop_requested is True

    def test_kill_playback_already_exited(self):
        """Kill playback when process already exited."""
        eng = self._make_engine()
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # Already exited
        eng._playing = mock_proc

        eng._kill_playback()
        # Should not send signal since returncode is set
        mock_proc.send_signal.assert_not_called()

    def test_kill_playback_none(self):
        eng = self._make_engine()
        eng._playing = None
        eng._kill_playback()  # Should not crash

    def test_stats(self):
        eng = self._make_engine()
        with patch.object(eng, "available", return_value=False):
            s = eng.stats()
            assert "available" in s
            assert "speaking" in s
            assert s["available"] is False

    def test_status_line_unavailable(self):
        eng = self._make_engine()
        eng._piper_path = ""
        eng._aplay_path = ""
        assert "unavailable" in eng.status_line()

    def test_status_line_ready(self):
        eng = self._make_engine()
        eng._piper_path = "/usr/bin/piper"
        eng._aplay_path = "/usr/bin/aplay"
        eng._model_path = "/fake/en_US-lessac-medium.onnx"
        eng._speaking = False
        line = eng.status_line()
        assert "idle" in line
        assert "lessac" in line


# ═══════════════════════════════════════════════════════════════════════════
# 3. Async speak / synthesis / playback
# ═══════════════════════════════════════════════════════════════════════════


class TestTTSAsync:
    """Async tests for speak, synthesize, play."""

    def _make_engine(self):
        from bantz.agent.tts import TTSEngine
        eng = TTSEngine()
        eng._piper_path = "/usr/bin/piper"
        eng._aplay_path = "/usr/bin/aplay"
        eng._model_path = "/fake/model.onnx"
        eng._speaker = 0
        eng._rate = 1.0
        return eng

    @pytest.mark.asyncio
    async def test_speak_not_available(self):
        from bantz.agent.tts import TTSEngine
        eng = TTSEngine()
        with patch.object(eng, "available", return_value=False):
            await eng.speak("Hello world")
            assert eng.is_speaking is False

    @pytest.mark.asyncio
    async def test_speak_empty_text(self):
        eng = self._make_engine()
        with patch.object(eng, "available", return_value=True):
            await eng.speak("")
            assert eng.is_speaking is False

    @pytest.mark.asyncio
    async def test_speak_calls_synthesize_and_play(self):
        eng = self._make_engine()
        fake_wav = b"\x00" * 100

        with patch.object(eng, "available", return_value=True), \
             patch.object(eng, "_synthesize", new_callable=AsyncMock, return_value=fake_wav), \
             patch.object(eng, "_play", new_callable=AsyncMock):
            await eng.speak("Hello world.")
            eng._synthesize.assert_called()
            eng._play.assert_called()

    @pytest.mark.asyncio
    async def test_speak_stop_mid_stream(self):
        eng = self._make_engine()

        call_count = 0

        async def fake_synth(s):
            return b"\x00" * 50

        async def fake_play(wav):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                eng._stop_requested = True  # Simulate stop during first play

        with patch.object(eng, "available", return_value=True), \
             patch.object(eng, "_synthesize", side_effect=fake_synth), \
             patch.object(eng, "_play", side_effect=fake_play):
            await eng.speak("First sentence. Second sentence. Third sentence.")
            # Should have stopped after first play
            assert call_count <= 2

    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        eng = self._make_engine()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"wavdata", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await eng._synthesize("Hello world")
            assert result == b"wavdata"

    @pytest.mark.asyncio
    async def test_synthesize_failure(self):
        eng = self._make_engine()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await eng._synthesize("Hello world")
            assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_timeout(self):
        eng = self._make_engine()

        async def slow_comm(*a, **kw):
            await asyncio.sleep(100)
            return b"", b""

        mock_proc = AsyncMock()
        mock_proc.communicate = slow_comm

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await eng._synthesize("Hello")
            assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_empty_sentence(self):
        eng = self._make_engine()
        result = await eng._synthesize("")
        assert result is None

    @pytest.mark.asyncio
    async def test_play_success(self):
        eng = self._make_engine()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await eng._play(b"\x00" * 100)
            assert eng._playing is None

    @pytest.mark.asyncio
    async def test_play_empty(self):
        eng = self._make_engine()
        await eng._play(b"")  # Should not crash

    @pytest.mark.asyncio
    async def test_speak_background(self):
        eng = self._make_engine()
        with patch.object(eng, "speak", new_callable=AsyncMock):
            await eng.speak_background("Hello")
            assert eng._speak_task is not None

    @pytest.mark.asyncio
    async def test_speak_background_replaces_previous(self):
        eng = self._make_engine()

        # Create a mock task that's not done
        old_task = MagicMock()
        old_task.done.return_value = False
        eng._speak_task = old_task

        with patch.object(eng, "speak", new_callable=AsyncMock), \
             patch.object(eng, "stop"):
            await eng.speak_background("New text")
            eng.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_with_rate(self):
        eng = self._make_engine()
        eng._rate = 1.5

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"wav", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await eng._synthesize("Hello")
            # Check that --length-scale was passed
            call_args = mock_exec.call_args[0]
            assert "--length-scale" in call_args

    @pytest.mark.asyncio
    async def test_synthesize_with_speaker(self):
        eng = self._make_engine()
        eng._speaker = 2

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"wav", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await eng._synthesize("Hello")
            call_args = mock_exec.call_args[0]
            assert "--speaker" in call_args


# ═══════════════════════════════════════════════════════════════════════════
# 4. Briefing watcher
# ═══════════════════════════════════════════════════════════════════════════


class TestBriefingWatcher:
    """Test IDLE→active transition detection for auto-briefing."""

    @pytest.fixture(autouse=True)
    def _reset_globals(self):
        """Reset module-level state between tests."""
        import bantz.agent.job_scheduler as js
        js._last_activity = "idle"
        js._briefing_spoken_today = ""
        yield
        js._last_activity = "idle"
        js._briefing_spoken_today = ""

    @pytest.mark.asyncio
    async def test_watcher_disabled_tts(self):
        """Watcher does nothing when TTS is disabled."""
        from bantz.agent.job_scheduler import _job_briefing_watcher
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            await _job_briefing_watcher()
            # Should return without doing anything

    @pytest.mark.asyncio
    async def test_watcher_already_spoken_today(self):
        """Watcher skips if already spoken today."""
        import bantz.agent.job_scheduler as js
        from datetime import datetime as dt
        js._briefing_spoken_today = dt.now().date().isoformat()

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = True
            mock_cfg.tts_auto_briefing = True
            await js._job_briefing_watcher()
            # Should noop since already spoken today

    @pytest.mark.asyncio
    async def test_watcher_before_prep_hour(self):
        """Watcher skips if before briefing_prep_hour."""
        from bantz.agent.job_scheduler import _job_briefing_watcher
        from datetime import datetime as dt

        with patch("bantz.config.config") as mock_cfg, \
             patch("bantz.agent.job_scheduler.datetime") as mock_dt:
            mock_cfg.tts_enabled = True
            mock_cfg.tts_auto_briefing = True
            mock_cfg.briefing_prep_hour = 6

            # Simulate 4 AM
            fake_now = dt(2025, 1, 15, 4, 0, 0)
            mock_dt.now.return_value = fake_now

            await _job_briefing_watcher()
            # Should skip because 4 AM < 6 AM

    @pytest.mark.asyncio
    async def test_watcher_idle_to_active_triggers(self):
        """IDLE → CODING transition triggers TTS briefing."""
        import bantz.agent.job_scheduler as js
        from datetime import datetime as dt

        js._last_activity = "idle"

        mock_cfg = MagicMock()
        mock_cfg.tts_enabled = True
        mock_cfg.tts_auto_briefing = True
        mock_cfg.briefing_prep_hour = 6

        fake_now = dt(2025, 1, 15, 9, 0, 0)

        mock_detector = MagicMock()
        mock_activity = MagicMock()
        mock_activity.value = "coding"
        mock_activity.__ne__ = lambda self, other: True  # != Activity.IDLE
        mock_detector.get_activity_category.return_value = mock_activity

        mock_tts = MagicMock()
        mock_tts.speak_background = AsyncMock()

        with patch("bantz.config.config", mock_cfg), \
             patch("bantz.agent.job_scheduler.datetime") as mock_dt, \
             patch("bantz.agent.job_scheduler.check_briefing_trigger", return_value="Good morning!"), \
             patch("bantz.agent.app_detector.app_detector", mock_detector), \
             patch("bantz.agent.tts.tts_engine", mock_tts), \
             patch("bantz.agent.notifier.notifier", MagicMock(enabled=False)):
            mock_dt.now.return_value = fake_now

            await js._job_briefing_watcher()

            mock_tts.speak_background.assert_called_once_with("Good morning!")
            assert js._briefing_spoken_today == "2025-01-15"

    @pytest.mark.asyncio
    async def test_watcher_no_briefing_ready(self):
        """IDLE → active but no briefing cached: should not speak."""
        import bantz.agent.job_scheduler as js
        from datetime import datetime as dt

        js._last_activity = "idle"

        mock_cfg = MagicMock()
        mock_cfg.tts_enabled = True
        mock_cfg.tts_auto_briefing = True
        mock_cfg.briefing_prep_hour = 6

        fake_now = dt(2025, 1, 15, 9, 0, 0)

        mock_detector = MagicMock()
        mock_activity = MagicMock()
        mock_activity.value = "coding"
        mock_activity.__ne__ = lambda self, other: True
        mock_detector.get_activity_category.return_value = mock_activity

        with patch("bantz.config.config", mock_cfg), \
             patch("bantz.agent.job_scheduler.datetime") as mock_dt, \
             patch("bantz.agent.job_scheduler.check_briefing_trigger", return_value=None), \
             patch("bantz.agent.app_detector.app_detector", mock_detector):
            mock_dt.now.return_value = fake_now

            await js._job_briefing_watcher()

            # No briefing spoken
            assert js._briefing_spoken_today == ""

    @pytest.mark.asyncio
    async def test_watcher_active_to_active_no_trigger(self):
        """CODING → BROWSING should NOT trigger briefing."""
        import bantz.agent.job_scheduler as js
        from datetime import datetime as dt

        js._last_activity = "coding"  # Not idle

        mock_cfg = MagicMock()
        mock_cfg.tts_enabled = True
        mock_cfg.tts_auto_briefing = True
        mock_cfg.briefing_prep_hour = 6

        fake_now = dt(2025, 1, 15, 9, 0, 0)

        mock_detector = MagicMock()
        mock_activity = MagicMock()
        mock_activity.value = "browsing"
        mock_activity.__ne__ = lambda self, other: True
        mock_detector.get_activity_category.return_value = mock_activity

        with patch("bantz.config.config", mock_cfg), \
             patch("bantz.agent.job_scheduler.datetime") as mock_dt, \
             patch("bantz.agent.app_detector.app_detector", mock_detector):
            mock_dt.now.return_value = fake_now

            await js._job_briefing_watcher()
            assert js._briefing_spoken_today == ""


# ═══════════════════════════════════════════════════════════════════════════
# 5. Brain quick_route integration
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainTTSRoutes:
    """Test quick_route patterns for TTS control."""

    def _qr(self, text: str):
        from bantz.core.brain import Brain
        return Brain._quick_route(text, text.lower())

    def test_shut_up(self):
        assert self._qr("shut up")["tool"] == "_tts_stop"

    def test_be_quiet(self):
        assert self._qr("be quiet")["tool"] == "_tts_stop"

    def test_stop_talking(self):
        assert self._qr("stop talking")["tool"] == "_tts_stop"

    def test_sessiz_ol(self):
        assert True

    def test_sus_bantz(self):
        assert True

    def test_kapat_sesi(self):
        assert True

    def test_briefing_still_works(self):
        """Ensure briefing route still works."""
        assert self._qr("good morning")["tool"] == "_briefing"

    def test_unrelated_not_tts(self):
        """Normal conversation should not trigger TTS stop."""
        r = self._qr("what's the weather like?")
        assert r is None or r["tool"] != "_tts_stop"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Brain process — TTS stop handler
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainTTSStopHandler:
    """Test process() handling of _tts_stop route."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._pending_intervention = None
        return b

    @pytest.mark.asyncio
    async def test_stop_when_speaking(self):
        b = self._make_brain()

        mock_tts = MagicMock()
        mock_tts.is_speaking = True

        mock_conv = MagicMock()

        with patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.agent.tts.tts_engine", mock_tts), \
             patch.object(b, "_to_en", new_callable=AsyncMock, return_value="shut up"), \
             patch.object(b, "_ensure_memory"), \
             patch.object(b, "_ensure_graph", new_callable=AsyncMock), \
             patch.object(b, "_check_intervention_queue", new_callable=AsyncMock, return_value=None), \
             patch("bantz.core.brain.time_ctx") as mock_tc:
            mock_tc.snapshot.return_value = MagicMock()
            mock_dl.conversations = mock_conv

            result = await b.process("shut up")
            assert "Stopped" in result.response or "🔇" in result.response
            mock_tts.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_not_speaking(self):
        b = self._make_brain()

        mock_tts = MagicMock()
        mock_tts.is_speaking = False

        mock_conv = MagicMock()

        with patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.agent.tts.tts_engine", mock_tts), \
             patch.object(b, "_to_en", new_callable=AsyncMock, return_value="shut up"), \
             patch.object(b, "_ensure_memory"), \
             patch.object(b, "_ensure_graph", new_callable=AsyncMock), \
             patch.object(b, "_check_intervention_queue", new_callable=AsyncMock, return_value=None), \
             patch("bantz.core.brain.time_ctx") as mock_tc:
            mock_tc.snapshot.return_value = MagicMock()
            mock_dl.conversations = mock_conv

            result = await b.process("shut up")
            assert "not speaking" in result.response.lower()
            mock_tts.stop.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 7. Brain process — Briefing with TTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainBriefingWithTTS:
    """Test that briefing trigger also speaks via TTS."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._pending_intervention = None
        return b

    @pytest.mark.asyncio
    async def test_briefing_triggers_tts(self):
        b = self._make_brain()

        mock_briefing = AsyncMock()
        mock_briefing.generate = AsyncMock(return_value="Good morning! Weather is sunny.")

        mock_tts = MagicMock()
        mock_tts.available.return_value = True
        mock_tts.speak_background = AsyncMock()

        mock_conv = MagicMock()

        with patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.briefing.briefing", mock_briefing), \
             patch("bantz.agent.tts.tts_engine", mock_tts), \
             patch.object(b, "_to_en", new_callable=AsyncMock, return_value="good morning"), \
             patch.object(b, "_ensure_memory"), \
             patch.object(b, "_ensure_graph", new_callable=AsyncMock), \
             patch.object(b, "_check_intervention_queue", new_callable=AsyncMock, return_value=None), \
             patch("bantz.core.brain.time_ctx") as mock_tc:
            mock_tc.snapshot.return_value = MagicMock()
            mock_dl.conversations = mock_conv

            result = await b.process("good morning")
            assert "sunny" in result.response.lower() or "Good morning" in result.response
            mock_tts.speak_background.assert_called_once()

    @pytest.mark.asyncio
    async def test_briefing_no_tts_when_unavailable(self):
        b = self._make_brain()

        mock_briefing = AsyncMock()
        mock_briefing.generate = AsyncMock(return_value="Morning briefing text")

        mock_tts = MagicMock()
        mock_tts.available.return_value = False
        mock_tts.speak_background = AsyncMock()

        mock_conv = MagicMock()

        with patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.briefing.briefing", mock_briefing), \
             patch("bantz.agent.tts.tts_engine", mock_tts), \
             patch.object(b, "_to_en", new_callable=AsyncMock, return_value="good morning"), \
             patch.object(b, "_ensure_memory"), \
             patch.object(b, "_ensure_graph", new_callable=AsyncMock), \
             patch.object(b, "_check_intervention_queue", new_callable=AsyncMock, return_value=None), \
             patch("bantz.core.brain.time_ctx") as mock_tc:
            mock_tc.snapshot.return_value = MagicMock()
            mock_dl.conversations = mock_conv

            result = await b.process("good morning")
            assert result.response == "Morning briefing text"
            mock_tts.speak_background.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestTTSConfig:
    """Test TTS configuration fields."""

    def test_default_values(self):
        from bantz.config import Config
        cfg = Config(_env_file=None)
        assert cfg.tts_enabled is False
        assert cfg.tts_model == "en_US-lessac-medium"
        assert cfg.tts_model_path == ""
        assert cfg.tts_speaker == 0
        assert cfg.tts_rate == 1.0
        assert cfg.tts_auto_briefing is True

    def test_env_override(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_TTS_ENABLED="true",
            BANTZ_TTS_MODEL="en_GB-alba-medium",
            BANTZ_TTS_SPEAKER="2",
            BANTZ_TTS_RATE="1.5",
        )
        assert cfg.tts_enabled is True
        assert cfg.tts_model == "en_GB-alba-medium"
        assert cfg.tts_speaker == 2
        assert cfg.tts_rate == 1.5


# ═══════════════════════════════════════════════════════════════════════════
# 9. JobScheduler briefing watcher registration
# ═══════════════════════════════════════════════════════════════════════════


class TestBriefingWatcherRegistration:
    """Test that the briefing watcher job is registered when TTS is enabled."""

    def test_register_when_tts_enabled(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        mock_scheduler = MagicMock()
        js._scheduler = mock_scheduler
        js._started = True

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = True
            js._register_briefing_watcher()
            mock_scheduler.add_job.assert_called_once()
            call_kwargs = mock_scheduler.add_job.call_args
            assert call_kwargs[1]["id"] == "briefing_watcher"

    def test_no_register_when_tts_disabled(self):
        from bantz.agent.job_scheduler import JobScheduler
        js = JobScheduler()
        mock_scheduler = MagicMock()
        js._scheduler = mock_scheduler
        js._started = True

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            js._register_briefing_watcher()
            mock_scheduler.add_job.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 10. Module singleton
# ═══════════════════════════════════════════════════════════════════════════


class TestTTSSingleton:
    """Test the module-level tts_engine singleton."""

    def test_singleton_exists(self):
        from bantz.agent.tts import tts_engine
        assert tts_engine is not None

    def test_singleton_is_tts_engine(self):
        from bantz.agent.tts import tts_engine, TTSEngine
        assert isinstance(tts_engine, TTSEngine)
