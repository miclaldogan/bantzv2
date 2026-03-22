"""Tests for bantz.core.translation_layer (#226).

Covers:
  - get_bridge() caching & fallback
  - to_en() delegation & timeout
  - resolve_message_ref() ordinals, keyword matching, fallback
  - detect_feedback() positive/negative/neutral + boundary safety
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.core.translation_layer import (
    to_en,
    resolve_message_ref,
    detect_feedback,
    POSITIVE_FEEDBACK_KWS,
    NEGATIVE_FEEDBACK_KWS,
)


# ═══════════════════════════════════════════════════════════════════════════
# get_bridge
# ═══════════════════════════════════════════════════════════════════════════


class TestGetBridge:
    """Bridge singleton cache."""

    def setup_method(self):
        """Reset the module-level cache before each test."""
        import bantz.core.translation_layer as mod
        mod._bridge_cache = None

    def test_returns_bridge_when_available(self):
        mock_bridge = MagicMock()
        import bantz.core.translation_layer as mod
        mod._bridge_cache = None
        fake_module = MagicMock(bridge=mock_bridge)
        with patch.dict("sys.modules", {"bantz.i18n.bridge": fake_module}):
            result = mod.get_bridge()
        assert result is mock_bridge

    def test_returns_none_when_import_fails(self):
        import bantz.core.translation_layer as mod
        mod._bridge_cache = None
        with patch.dict("sys.modules", {"bantz.i18n.bridge": None}):
            mod._bridge_cache = None
            result = mod.get_bridge()
        assert result is None

    def test_caches_result(self):
        import bantz.core.translation_layer as mod
        sentinel = MagicMock()
        mod._bridge_cache = sentinel
        result = mod.get_bridge()
        assert result is sentinel


# ═══════════════════════════════════════════════════════════════════════════
# to_en
# ═══════════════════════════════════════════════════════════════════════════


class TestToEn:
    """Async translation to English."""

    def setup_method(self):
        import bantz.core.translation_layer as mod
        mod._bridge_cache = None

    @pytest.mark.asyncio
    async def test_passthrough_when_bridge_disabled(self):
        import bantz.core.translation_layer as mod
        mod._bridge_cache = False  # bridge unavailable
        result = await to_en("merhaba dünya")
        assert result == "merhaba dünya"

    @pytest.mark.asyncio
    async def test_translates_when_bridge_enabled(self):
        mock_bridge = MagicMock()
        mock_bridge.is_enabled.return_value = True
        mock_bridge.to_english = AsyncMock(return_value="hello world")

        import bantz.core.translation_layer as mod
        mod._bridge_cache = mock_bridge
        result = await to_en("merhaba dünya")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self):
        mock_bridge = MagicMock()
        mock_bridge.is_enabled.return_value = True

        async def slow_translate(text):
            await asyncio.sleep(100)  # Will timeout
            return "translated"

        mock_bridge.to_english = slow_translate

        import bantz.core.translation_layer as mod
        mod._bridge_cache = mock_bridge

        # Patch asyncio.wait_for timeout to be very short
        result = await to_en("test")
        # The 10s timeout won't actually fire in test, but the function
        # should gracefully handle it. Since we can't easily test real timeout
        # in a unit test, test the error path directly:
        mock_bridge.to_english = AsyncMock(side_effect=asyncio.TimeoutError)
        result = await to_en("test")
        assert result == "test"  # falls back to original


# ═══════════════════════════════════════════════════════════════════════════
# resolve_message_ref
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveMessageRef:
    """Email/message reference resolution."""

    MESSAGES = [
        {"id": "msg_1", "from": "ali@test.com", "subject": "LinkedIn Update"},
        {"id": "msg_2", "from": "google-cloud@google.com", "subject": "Billing Alert"},
        {"id": "msg_3", "from": "boss@work.com", "subject": "Quarterly Review"},
    ]

    def test_empty_messages_returns_none(self):
        assert resolve_message_ref("read the first one", []) is None

    def test_ordinal_first(self):
        assert resolve_message_ref("read the first one", self.MESSAGES) == "msg_1"

    def test_ordinal_second(self):
        assert resolve_message_ref("open the second one", self.MESSAGES) == "msg_2"

    def test_ordinal_last(self):
        assert resolve_message_ref("show me the last email", self.MESSAGES) == "msg_3"

    def test_ordinal_out_of_range(self):
        assert resolve_message_ref("read the fifth one", self.MESSAGES) is None

    def test_keyword_match_sender(self):
        assert resolve_message_ref("read the google one", self.MESSAGES) == "msg_2"

    def test_keyword_match_subject(self):
        assert resolve_message_ref("open the linkedin email", self.MESSAGES) == "msg_1"

    def test_fallback_to_first(self):
        # No ordinal or keyword match → first message
        assert resolve_message_ref("read that email", self.MESSAGES) == "msg_1"

    def test_skip_words_not_matched(self):
        # "please read the email" — all are skip words → fallback
        assert resolve_message_ref("please read the email", self.MESSAGES) == "msg_1"


# ═══════════════════════════════════════════════════════════════════════════
# detect_feedback
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectFeedback:
    """Feedback detection with word-boundary safety."""

    def test_positive_english(self):
        assert detect_feedback("good job on that") == "positive"

    def test_positive_turkish(self):
        assert detect_feedback("aferin sana") == "positive"

    def test_negative_english(self):
        assert detect_feedback("you are wrong about this") == "negative"

    def test_negative_turkish(self):
        assert detect_feedback("yanlış cevap verdin") == "negative"

    def test_neutral_returns_none(self):
        assert detect_feedback("what is the weather today") is None

    def test_negative_priority(self):
        # Negative checked first — if both match, negative wins
        assert detect_feedback("bravo, but you are wrong") == "negative"

    def test_case_insensitive(self):
        assert detect_feedback("GOOD JOB") == "positive"

    def test_word_boundary_bus_stop(self):
        """'bus stop' should NOT trigger feedback."""
        assert detect_feedback("Can you tell me about the bus stop?") is None

    def test_keyword_tuples_nonempty(self):
        assert len(POSITIVE_FEEDBACK_KWS) > 0
        assert len(NEGATIVE_FEEDBACK_KWS) > 0
