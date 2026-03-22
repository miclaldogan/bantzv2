"""Tests for Issue #166 integrations — ambient ↔ wake_word, RL engine, reflection."""
from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch




# ═══════════════════════════════════════════════════════════════════════════
# Wake Word → Ambient piggybacking
# ═══════════════════════════════════════════════════════════════════════════

class TestWakeWordAmbientPiggybacking:
    """Verify that _listen_loop feeds frames to ambient_analyzer."""

    def _make_listener(self):
        from bantz.agent.wake_word import WakeWordListener
        listener = WakeWordListener()
        return listener

    def test_listen_loop_calls_feed_frames_when_ambient_enabled(self):
        """When ambient_enabled=True, the listen loop feeds frames."""
        listener = self._make_listener()
        listener._running = True

        # Create mock porcupine and stream
        mock_porc = MagicMock()
        mock_porc.frame_length = 512
        mock_porc.process.return_value = -1  # no wake word
        listener._porcupine = mock_porc

        # Build PCM bytes for one frame
        pcm_bytes = struct.pack(f"{512}h", *([0] * 512))

        # Stream returns one frame then triggers stop
        call_count = [0]
        def mock_read(n, exception_on_overflow=False):
            call_count[0] += 1
            if call_count[0] > 1:
                listener._stop.set()
            return pcm_bytes
        mock_stream = MagicMock()
        mock_stream.read = mock_read
        listener._audio_stream = mock_stream

        mock_analyzer = MagicMock()

        mock_config = SimpleNamespace(ambient_enabled=True)
        with patch("bantz.agent.wake_word.log"), \
             patch.dict("sys.modules", {
                 "bantz.agent.ambient": MagicMock(ambient_analyzer=mock_analyzer),
                 "bantz.config": MagicMock(config=mock_config),
             }):
            listener._stop.clear()
            listener._listen_loop()

        # ambient_analyzer.feed_frames should have been called
        assert mock_analyzer.feed_frames.called

    def test_listen_loop_skips_ambient_when_disabled(self):
        """When ambient_enabled=False, no feed_frames calls."""
        listener = self._make_listener()
        listener._running = True

        mock_porc = MagicMock()
        mock_porc.frame_length = 512
        mock_porc.process.return_value = -1
        listener._porcupine = mock_porc

        pcm_bytes = struct.pack(f"{512}h", *([0] * 512))
        call_count = [0]
        def mock_read(n, exception_on_overflow=False):
            call_count[0] += 1
            if call_count[0] > 1:
                listener._stop.set()
            return pcm_bytes
        mock_stream = MagicMock()
        mock_stream.read = mock_read
        listener._audio_stream = mock_stream

        mock_analyzer = MagicMock()
        mock_config = SimpleNamespace(ambient_enabled=False)
        with patch("bantz.agent.wake_word.log"), \
             patch.dict("sys.modules", {
                 "bantz.agent.ambient": MagicMock(ambient_analyzer=mock_analyzer),
                 "bantz.config": MagicMock(config=mock_config),
             }):
            listener._stop.clear()
            listener._listen_loop()

        # ambient_analyzer should NOT have been imported/used
        assert not mock_analyzer.feed_frames.called

    def test_ambient_exception_does_not_crash_listen_loop(self):
        """If feed_frames raises, the loop continues without crashing."""
        listener = self._make_listener()
        listener._running = True

        mock_porc = MagicMock()
        mock_porc.frame_length = 512
        mock_porc.process.return_value = -1
        listener._porcupine = mock_porc

        pcm_bytes = struct.pack(f"{512}h", *([0] * 512))
        call_count = [0]
        def mock_read(n, exception_on_overflow=False):
            call_count[0] += 1
            if call_count[0] > 2:
                listener._stop.set()
            return pcm_bytes
        mock_stream = MagicMock()
        mock_stream.read = mock_read
        listener._audio_stream = mock_stream

        mock_analyzer = MagicMock()
        mock_analyzer.feed_frames.side_effect = RuntimeError("boom")
        mock_config = SimpleNamespace(ambient_enabled=True)
        with patch("bantz.agent.wake_word.log"), \
             patch.dict("sys.modules", {
                 "bantz.agent.ambient": MagicMock(ambient_analyzer=mock_analyzer),
                 "bantz.config": MagicMock(config=mock_config),
             }):
            listener._stop.clear()
            # Should NOT raise
            listener._listen_loop()

        # Loop ran and called feed_frames (even though it raised)
        assert mock_analyzer.feed_frames.call_count >= 1


# ═══════════════════════════════════════════════════════════════════════════
# RL Engine — State extension with ambient
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Affinity Engine — no state encoding needed (#221)
# ═══════════════════════════════════════════════════════════════════════════

class TestAffinityEngineAmbient:
    """Verify affinity engine initialises cleanly (ambient context N/A)."""

    def test_affinity_engine_importable(self):
        from bantz.agent.affinity_engine import AffinityEngine
        ae = AffinityEngine()
        assert ae.initialized is False
        assert ae.get_score() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Reflection — ambient_summary integration
# ═══════════════════════════════════════════════════════════════════════════

class TestReflectionAmbient:
    """Verify nightly reflection includes ambient data."""

    def test_reflection_result_has_ambient_summary_field(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult()
        assert hasattr(r, "ambient_summary")
        assert r.ambient_summary == ""

    def test_reflection_result_to_dict_includes_ambient_when_set(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(ambient_summary="Ambient today (5 samples): silence: 60%, speech: 40%")
        d = r.to_dict()
        assert "ambient_summary" in d
        assert "60%" in d["ambient_summary"]

    def test_reflection_result_to_dict_excludes_ambient_when_empty(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult()
        d = r.to_dict()
        assert "ambient_summary" not in d

    def test_summary_line_includes_ambient(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(
            date="2026-03-12",
            sessions=3,
            total_messages=15,
            ambient_summary="Ambient today: silence 80%, speech 20%",
        )
        line = r.summary_line()
        assert "🎤" in line
        assert "Ambient today" in line

    def test_reflection_prompt_has_ambient_section_placeholder(self):
        """_REFLECTION_USER template includes {ambient_section}."""
        from bantz.agent.workflows.reflection import _REFLECTION_USER
        assert "{ambient_section}" in _REFLECTION_USER


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientConfig:
    def test_config_has_ambient_fields(self):
        from bantz.config import Config
        fields = {f for f in Config.model_fields}
        assert "ambient_enabled" in fields
        assert "ambient_interval" in fields
        assert "ambient_window" in fields

    def test_default_values(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.ambient_enabled is False
        assert c.ambient_interval == 600
        assert c.ambient_window == 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Safety: ambient does NOT touch TTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientDoesNotTouchTTS:
    """Issue #166 design constraint: ambient data must NOT affect TTS decisions."""

    def test_ambient_module_has_no_tts_import(self):
        """ambient.py must not import from bantz.agent.tts."""
        import inspect
        import bantz.agent.ambient as mod
        source = inspect.getsource(mod)
        assert "from bantz.agent.tts" not in source
        assert "import tts" not in source

    def test_ambient_module_has_no_audio_ducker_import(self):
        """ambient.py must not import audio_ducker — that's TTS's job."""
        import inspect
        import bantz.agent.ambient as mod
        source = inspect.getsource(mod)
        assert "audio_ducker" not in source


# ═══════════════════════════════════════════════════════════════════════════
# Safety: ambient never opens its own mic stream
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientNoOwnMic:
    """Issue #166 Trap #1: ambient must NEVER open its own pyaudio stream."""

    def test_no_pyaudio_in_ambient(self):
        import ast
        import bantz.agent.ambient as mod
        source = open(mod.__file__).read()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any("pyaudio" in i for i in imports)

    def test_no_stream_open_in_ambient(self):
        import inspect
        import bantz.agent.ambient as mod
        source = inspect.getsource(mod)
        assert "pa.open" not in source
        assert ".open(" not in source

    def test_no_numpy_dependency(self):
        """MVP: pure stdlib math, no numpy."""
        import inspect
        import bantz.agent.ambient as mod
        source = inspect.getsource(mod)
        assert "import numpy" not in source
        assert "from numpy" not in source

    def test_no_fft_spectral(self):
        """MVP: no FFT / spectral centroid."""
        import ast
        import bantz.agent.ambient as mod
        source = open(mod.__file__).read()
        tree = ast.parse(source)
        # Check function/method names and actual code — not comments/docstrings
        func_names = [node.name.lower() for node in ast.walk(tree)
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert not any("fft" in n or "spectral" in n for n in func_names)
