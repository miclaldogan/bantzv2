"""Tests for #441 — StandaloneAmbientSampler: decouple AmbientEngine from Picovoice.

Covers:
  - StandaloneAmbientSampler.start() returns False when pyaudio missing
  - start() opens a PyAudio stream and launches a daemon thread
  - stop() closes the stream and terminates the thread
  - The sampling thread feeds PCM frames into ambient_analyzer
  - maybe_start_standalone() fires only when ambient_enabled=True, wake_word_enabled=False
  - maybe_start_standalone() is a no-op when wake_word is enabled
  - maybe_start_standalone() is a no-op when ambient is disabled
"""
from __future__ import annotations

import struct
import sys
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_pyaudio_stub(frame_size: int = 512):
    """Return (pyaudio_module_stub, pa_instance_mock, stream_mock)."""
    silent_frame = struct.pack(f"{frame_size}h", *([0] * frame_size))
    stream_mock = MagicMock()
    stream_mock.read.return_value = silent_frame
    pa_instance = MagicMock()
    pa_instance.open.return_value = stream_mock
    pyaudio_mod = MagicMock()
    pyaudio_mod.PyAudio.return_value = pa_instance
    pyaudio_mod.paInt16 = 8  # realistic constant
    return pyaudio_mod, pa_instance, stream_mock


@contextmanager
def _stub_pyaudio(frame_size: int = 512):
    """Context manager that injects a pyaudio stub into sys.modules."""
    pyaudio_mod, pa_instance, stream_mock = _make_pyaudio_stub(frame_size)
    with patch.dict(sys.modules, {"pyaudio": pyaudio_mod}):
        yield pyaudio_mod, pa_instance, stream_mock


# ─── StandaloneAmbientSampler tests ───────────────────────────────────────────

class TestStandaloneAmbientSamplerNoAudio:
    def test_start_returns_false_when_pyaudio_missing(self):
        from bantz.agent.ambient import StandaloneAmbientSampler
        sampler = StandaloneAmbientSampler()
        with patch.dict(sys.modules, {"pyaudio": None}):
            ok = sampler.start()
        assert ok is False
        assert sampler.running is False

    def test_start_returns_false_when_mic_open_fails(self):
        from bantz.agent.ambient import StandaloneAmbientSampler
        sampler = StandaloneAmbientSampler()

        pyaudio_mod, pa_instance, _ = _make_pyaudio_stub()
        pa_instance.open.side_effect = OSError("no mic")

        with (
            patch.dict(sys.modules, {"pyaudio": pyaudio_mod}),
            patch("bantz.agent.voice_capture.suppress_alsa_stderr"),
        ):
            ok = sampler.start()

        assert ok is False
        assert sampler.running is False


class TestStandaloneAmbientSamplerLifecycle:
    def test_start_launches_daemon_thread(self):
        from bantz.agent.ambient import StandaloneAmbientSampler, _STANDALONE_FRAME_SIZE
        sampler = StandaloneAmbientSampler()

        read_event = threading.Event()
        silent = struct.pack(f"{_STANDALONE_FRAME_SIZE}h", *([0] * _STANDALONE_FRAME_SIZE))
        pyaudio_mod, pa_instance, stream_mock = _make_pyaudio_stub(_STANDALONE_FRAME_SIZE)

        def slow_read(*a, **kw):
            read_event.wait(timeout=0.05)
            return silent

        stream_mock.read.side_effect = slow_read

        with (
            patch.dict(sys.modules, {"pyaudio": pyaudio_mod}),
            patch("bantz.agent.voice_capture.suppress_alsa_stderr"),
        ):
            ok = sampler.start()

        assert ok is True
        assert sampler.running is True
        assert sampler._thread is not None
        assert sampler._thread.daemon is True
        assert sampler._thread.is_alive()

        read_event.set()
        sampler.stop()
        assert sampler.running is False

    def test_stop_releases_resources(self):
        from bantz.agent.ambient import StandaloneAmbientSampler, _STANDALONE_FRAME_SIZE
        sampler = StandaloneAmbientSampler()
        pyaudio_mod, pa_instance, stream_mock = _make_pyaudio_stub(_STANDALONE_FRAME_SIZE)

        with (
            patch.dict(sys.modules, {"pyaudio": pyaudio_mod}),
            patch("bantz.agent.voice_capture.suppress_alsa_stderr"),
        ):
            sampler.start()
            sampler.stop()

        stream_mock.stop_stream.assert_called()
        stream_mock.close.assert_called()
        pa_instance.terminate.assert_called()

    def test_start_is_idempotent(self):
        from bantz.agent.ambient import StandaloneAmbientSampler, _STANDALONE_FRAME_SIZE
        sampler = StandaloneAmbientSampler()
        pyaudio_mod, pa_instance, _ = _make_pyaudio_stub(_STANDALONE_FRAME_SIZE)

        with (
            patch.dict(sys.modules, {"pyaudio": pyaudio_mod}),
            patch("bantz.agent.voice_capture.suppress_alsa_stderr"),
        ):
            ok1 = sampler.start()
            ok2 = sampler.start()

        assert ok1 is True
        assert ok2 is True
        assert pa_instance.open.call_count == 1
        sampler.stop()

    def test_sample_loop_feeds_ambient_analyzer(self):
        """The sampling thread must call ambient_analyzer.feed_frames() with PCM."""
        from bantz.agent.ambient import StandaloneAmbientSampler, _STANDALONE_FRAME_SIZE
        sampler = StandaloneAmbientSampler()
        pyaudio_mod, _, _ = _make_pyaudio_stub(_STANDALONE_FRAME_SIZE)

        fed_frames: list = []
        feed_event = threading.Event()

        def capture_feed(pcm):
            fed_frames.append(pcm)
            if len(fed_frames) >= 2:
                feed_event.set()

        with (
            patch.dict(sys.modules, {"pyaudio": pyaudio_mod}),
            patch("bantz.agent.voice_capture.suppress_alsa_stderr"),
            patch("bantz.agent.ambient.ambient_analyzer") as mock_aa,
        ):
            mock_aa.feed_frames.side_effect = capture_feed
            sampler.start()
            feed_event.wait(timeout=2.0)
            sampler.stop()

        assert len(fed_frames) >= 2
        assert len(fed_frames[0]) == _STANDALONE_FRAME_SIZE


# ─── maybe_start_standalone() guard tests ─────────────────────────────────────

class TestMaybeStartStandalone:
    def test_starts_when_ambient_enabled_and_wake_word_disabled(self):
        from bantz.agent.ambient import maybe_start_standalone

        mock_sampler = MagicMock()
        mock_sampler.start.return_value = True
        mock_config = MagicMock()
        mock_config.ambient_enabled = True
        mock_config.wake_word_enabled = False

        with (
            patch("bantz.agent.ambient.standalone_ambient_sampler", mock_sampler),
            patch("bantz.config.config", mock_config),
        ):
            result = maybe_start_standalone()

        mock_sampler.start.assert_called_once()
        assert result is True

    def test_noop_when_wake_word_enabled(self):
        from bantz.agent.ambient import maybe_start_standalone

        mock_sampler = MagicMock()
        mock_config = MagicMock()
        mock_config.ambient_enabled = True
        mock_config.wake_word_enabled = True

        with (
            patch("bantz.agent.ambient.standalone_ambient_sampler", mock_sampler),
            patch("bantz.config.config", mock_config),
        ):
            result = maybe_start_standalone()

        mock_sampler.start.assert_not_called()
        assert result is False

    def test_noop_when_ambient_disabled(self):
        from bantz.agent.ambient import maybe_start_standalone

        mock_sampler = MagicMock()
        mock_config = MagicMock()
        mock_config.ambient_enabled = False
        mock_config.wake_word_enabled = False

        with (
            patch("bantz.agent.ambient.standalone_ambient_sampler", mock_sampler),
            patch("bantz.config.config", mock_config),
        ):
            result = maybe_start_standalone()

        mock_sampler.start.assert_not_called()
        assert result is False

    def test_survives_exception(self):
        """maybe_start_standalone() must not propagate exceptions from start()."""
        from bantz.agent.ambient import maybe_start_standalone

        mock_sampler = MagicMock()
        mock_sampler.start.side_effect = RuntimeError("boom")
        mock_config = MagicMock()
        mock_config.ambient_enabled = True
        mock_config.wake_word_enabled = False

        with (
            patch("bantz.agent.ambient.standalone_ambient_sampler", mock_sampler),
            patch("bantz.config.config", mock_config),
        ):
            result = maybe_start_standalone()

        assert result is False

