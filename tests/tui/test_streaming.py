"""
Tests — Issue #175: TUI Streaming Delta Buffer (Option C)

Covers:
  - stream_start / stream_token / stream_end lifecycle
  - Delta buffering: only the current partial line is in _streaming_buffer
  - Completed lines are written permanently (not re-written each token)
  - _full_text accumulates correctly for stream_end() return
  - Single-line streaming (no newlines)
  - Multi-line streaming with newlines
  - Empty tokens, consecutive newlines, trailing newline
  - State reset between consecutive streaming sessions
  - No old rendered_count / state leakage between sessions
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip('textual')


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_chatlog():
    """Create a ChatLog with a real ``lines`` list and a recording ``write``.

    RichLog.write() defers rendering when the widget is unmounted (no app),
    so we patch ``lines`` to a plain list and ``write`` to append + return self.
    ``scroll_end`` is also stubbed out.
    """
    from bantz.interface.tui.panels.chat import ChatLog

    cl = ChatLog(max_lines=1000)
    cl.lines = []  # override — Textual's Strip-based list needs a running app

    _orig_write = cl.write

    def _test_write(content, **kw):
        cl.lines.append(str(content))
        return cl

    cl.write = _test_write
    cl.scroll_end = lambda **kw: None
    return cl


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle basics
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamLifecycle:
    def test_stream_start_resets_state(self):
        cl = _make_chatlog()
        cl._streaming_buffer = "leftover"
        cl._full_text = "old"
        cl.stream_start()
        assert cl._streaming_buffer == ""
        assert cl._full_text == ""
        assert cl._stream_started is True

    def test_stream_end_returns_full_text(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("Hello ")
        cl.stream_token("world")
        result = cl.stream_end()
        assert result == "Hello world"
        assert cl._stream_started is False
        assert cl._streaming_buffer == ""
        assert cl._full_text == ""

    def test_stream_end_with_newlines(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("line1\nline2\nline3")
        result = cl.stream_end()
        assert result == "line1\nline2\nline3"


# ═══════════════════════════════════════════════════════════════════════════
# Delta buffer — _streaming_buffer only holds current partial
# ═══════════════════════════════════════════════════════════════════════════

class TestDeltaBuffer:
    def test_buffer_holds_only_partial(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("Hello\nWorld")
        # "Hello\n" was flushed permanently; only "World" remains in buffer
        assert cl._streaming_buffer == "World"

    def test_buffer_empty_after_trailing_newline(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("Done\n")
        assert cl._streaming_buffer == ""

    def test_buffer_accumulates_within_line(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("Mer")
        assert cl._streaming_buffer == "Mer"
        cl.stream_token("haba")
        assert cl._streaming_buffer == "Merhaba"

    def test_buffer_resets_on_newline_boundary(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("Merhaba")
        cl.stream_token("\n")
        assert cl._streaming_buffer == ""
        cl.stream_token("Ben ")
        assert cl._streaming_buffer == "Ben "

    def test_multiple_newlines_in_single_token(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("a\nb\nc")
        # "a" and "b" flushed, "c" remains
        assert cl._streaming_buffer == "c"
        assert cl._full_text == "a\nb\nc"


# ═══════════════════════════════════════════════════════════════════════════
# Line count — no duplicates
# ═══════════════════════════════════════════════════════════════════════════

class TestNoDuplicateLines:
    """The core bug: old code re-wrote all completed lines on every token."""

    def test_single_line_no_growth(self):
        """Streaming 5 tokens on one line should keep exactly 1 content line."""
        cl = _make_chatlog()
        cl.stream_start()
        initial = len(cl.lines)  # includes stream_start placeholder

        for tok in ["H", "e", "l", "l", "o"]:
            cl.stream_token(tok)

        # stream_start wrote 1 placeholder → popped → replaced.
        # After 5 tokens there should be exactly initial lines
        # (placeholder popped, one partial written each time, net = initial).
        assert len(cl.lines) == initial

    def test_multiline_exact_count(self):
        """'Hello\\nWorld\\n' should produce exactly 2 permanent + 1 partial."""
        cl = _make_chatlog()
        cl.stream_start()
        initial = len(cl.lines)  # placeholder

        cl.stream_token("Hello\nWorld\n")
        # placeholder popped → "Hello" permanent, "World" permanent, "" partial
        # net = initial - 1 (popped) + 3 (Hello, World, "") = initial + 2
        assert len(cl.lines) == initial + 2

    def test_no_exponential_growth(self):
        """Simulate 100 tokens across 10 lines — line count must be O(n_lines)."""
        cl = _make_chatlog()
        cl.stream_start()

        for i in range(100):
            tok = f"w{i}"
            if i % 10 == 9:
                tok += "\n"
            cl.stream_token(tok)

        # 10 completed lines + 1 partial + stream_start(1) - popped
        # Exact: stream_start writes 1, first pop removes it = 0 base
        # Then 10 permanent lines + 1 partial = 11 lines above base
        # With old buggy code this would be hundreds of lines.
        assert len(cl.lines) < 20  # generous upper bound


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamEdgeCases:
    def test_empty_token(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("")
        assert cl._streaming_buffer == ""
        assert cl._full_text == ""

    def test_consecutive_newlines(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("\n\n\n")
        # Three completed empty lines, buffer = ""
        assert cl._streaming_buffer == ""
        assert cl._full_text == "\n\n\n"

    def test_only_newline_token(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("hello")
        cl.stream_token("\n")
        assert cl._streaming_buffer == ""
        assert cl._full_text == "hello\n"

    def test_full_text_matches_all_tokens(self):
        cl = _make_chatlog()
        cl.stream_start()
        tokens = ["Merhaba", "\n", "Ben ", "Bantz", "\n", "Nasılsın?"]
        for t in tokens:
            cl.stream_token(t)
        assert cl.stream_end() == "Merhaba\nBen Bantz\nNasılsın?"


# ═══════════════════════════════════════════════════════════════════════════
# Session isolation — no state leakage
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionIsolation:
    def test_second_session_works_correctly(self):
        """After stream_end, a new stream_start must work from clean state."""
        cl = _make_chatlog()

        # Session 1: 20-line response
        cl.stream_start()
        for i in range(20):
            cl.stream_token(f"line{i}\n")
        text1 = cl.stream_end()
        assert "line0" in text1
        assert "line19" in text1

        count_after_s1 = len(cl.lines)

        # Session 2: short response
        cl.stream_start()
        cl.stream_token("Kısa cevap")
        text2 = cl.stream_end()

        assert text2 == "Kısa cevap"
        # Session 2 should only add stream_start placeholder + 1 partial
        # (placeholder popped, partial written = net +1 from session 1 end + 1 new)
        # Point: it must NOT be silent (old bug: rendered_count=20 would skip first 20 lines)
        assert len(cl.lines) > count_after_s1

    def test_buffer_clean_between_sessions(self):
        cl = _make_chatlog()
        cl.stream_start()
        cl.stream_token("data")
        cl.stream_end()

        assert cl._streaming_buffer == ""
        assert cl._full_text == ""
        assert cl._stream_started is False


# ═══════════════════════════════════════════════════════════════════════════
# add_bantz and add_user — sanity (non-streaming)
# ═══════════════════════════════════════════════════════════════════════════

class TestNonStreamingMethods:
    def test_add_user(self):
        cl = _make_chatlog()
        cl.add_user("merhaba")
        assert len(cl.lines) == 1

    def test_add_bantz_single_line(self):
        cl = _make_chatlog()
        cl.add_bantz("hello")
        assert len(cl.lines) == 1

    def test_add_bantz_multiline(self):
        cl = _make_chatlog()
        cl.add_bantz("line1\nline2\nline3")
        assert len(cl.lines) == 3

    def test_add_system(self):
        cl = _make_chatlog()
        cl.add_system("info")
        assert len(cl.lines) == 1

    def test_add_error(self):
        cl = _make_chatlog()
        cl.add_error("oops")
        assert len(cl.lines) == 1

    def test_add_tool(self):
        cl = _make_chatlog()
        cl.add_tool("web_search")
        assert len(cl.lines) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Audit: no old accumulation pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamingAudit:
    def test_no_full_buffer_split_on_every_token(self):
        """_streaming_buffer must never grow beyond the current line fragment."""
        cl = _make_chatlog()
        cl.stream_start()
        long_text = "word " * 50 + "\n"
        # Feed one char at a time
        for ch in long_text:
            cl.stream_token(ch)
            # Buffer should never exceed one line's worth of chars
            assert "\n" not in cl._streaming_buffer

    def test_full_text_accumulates_everything(self):
        cl = _make_chatlog()
        cl.stream_start()
        expected = ""
        for tok in ["abc", "\n", "def", "ghi", "\n", "jkl"]:
            cl.stream_token(tok)
            expected += tok
        assert cl._full_text == expected
