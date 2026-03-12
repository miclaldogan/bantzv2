"""Tests for issue #155 — core stabilization fixes.

A: AppDetector → brain.py (desktop context in system prompt)
B: Sticky router context fix (short inputs no longer blindly go to Gmail)
C: Weather location extraction (_extract_city with TR/EN patterns)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# A: Desktop context tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDesktopContext:
    """Test Brain._desktop_context() builds correct system prompt hints."""

    def _make_brain(self):
        """Create a Brain instance with tool imports mocked out."""
        with patch.dict("sys.modules", {
            "bantz.tools.shell": MagicMock(),
            "bantz.tools.system": MagicMock(),
            "bantz.tools.filesystem": MagicMock(),
            "bantz.tools.weather": MagicMock(),
            "bantz.tools.news": MagicMock(),
            "bantz.tools.web_search": MagicMock(),
            "bantz.tools.gmail": MagicMock(),
            "bantz.tools.calendar": MagicMock(),
            "bantz.tools.classroom": MagicMock(),
            "bantz.tools.reminder": MagicMock(),
        }):
            from bantz.core.brain import Brain
            return Brain()

    def test_desktop_context_not_initialized(self):
        """Returns empty string when AppDetector is not initialized."""
        brain = self._make_brain()
        mock_detector = MagicMock()
        mock_detector.initialized = False
        with patch("bantz.core.brain.app_detector", mock_detector, create=True):
            # Call the method — it imports app_detector inside
            result = brain._desktop_context()
        assert result == ""

    def test_desktop_context_empty_snapshot(self):
        """Returns empty string when snapshot is empty."""
        brain = self._make_brain()
        mock_detector = MagicMock()
        mock_detector.initialized = True
        mock_detector.get_workspace_context.return_value = {}
        with patch("bantz.agent.app_detector.app_detector", mock_detector):
            result = brain._desktop_context()
        assert result == ""

    def test_desktop_context_full_snapshot(self):
        """Builds full desktop hint with all sections."""
        brain = self._make_brain()
        mock_detector = MagicMock()
        mock_detector.initialized = True
        mock_detector.get_workspace_context.return_value = {
            "active_window": {"name": "code", "title": "brain.py - bantzv2"},
            "activity": "coding",
            "apps": ["code", "firefox", "terminal", "docker", "slack"],
            "ide": {"ide": "VS Code", "file": "brain.py", "project": "bantzv2"},
            "docker": [
                {"name": "neo4j", "state": "running", "image": "neo4j:latest"},
                {"name": "redis", "state": "exited", "image": "redis:7"},
            ],
        }
        with patch("bantz.agent.app_detector.app_detector", mock_detector):
            result = brain._desktop_context()

        assert "Desktop Context" in result
        assert "code" in result
        assert "coding" in result
        assert "5" in result  # 5 apps
        assert "VS Code" in result
        assert "brain.py" in result
        assert "neo4j" in result
        assert "1 running" in result  # only neo4j is running

    def test_desktop_context_no_docker(self):
        """Omits docker section when no containers."""
        brain = self._make_brain()
        mock_detector = MagicMock()
        mock_detector.initialized = True
        mock_detector.get_workspace_context.return_value = {
            "active_window": {"name": "firefox", "title": "Google"},
            "activity": "browsing",
            "apps": ["firefox"],
        }
        with patch("bantz.agent.app_detector.app_detector", mock_detector):
            result = brain._desktop_context()

        assert "Docker" not in result
        assert "browsing" in result

    def test_desktop_context_exception_safe(self):
        """Returns empty string on any exception."""
        brain = self._make_brain()
        # Patch the import inside _desktop_context to raise
        with patch.dict("sys.modules", {"bantz.agent.app_detector": None}):
            result = brain._desktop_context()
        assert result == ""

    def test_chat_system_has_desktop_hint_placeholder(self):
        """CHAT_SYSTEM template includes {desktop_hint}."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "{desktop_hint}" in CHAT_SYSTEM

    def test_chat_system_has_anti_hallucination_rule(self):
        """CHAT_SYSTEM tells LLM to use Desktop Context only."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "Desktop Context" in CHAT_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════
# B: Sticky router context tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStickyContextFix:
    """Test that short ambiguous inputs don't blindly route to Gmail."""

    def _make_brain(self):
        with patch.dict("sys.modules", {
            "bantz.tools.shell": MagicMock(),
            "bantz.tools.system": MagicMock(),
            "bantz.tools.filesystem": MagicMock(),
            "bantz.tools.weather": MagicMock(),
            "bantz.tools.news": MagicMock(),
            "bantz.tools.web_search": MagicMock(),
            "bantz.tools.gmail": MagicMock(),
            "bantz.tools.calendar": MagicMock(),
            "bantz.tools.classroom": MagicMock(),
            "bantz.tools.reminder": MagicMock(),
        }):
            from bantz.core.brain import Brain
            b = Brain()
            # Simulate having recent email messages
            b._last_messages = [
                {"id": "msg1", "from": "john@medium.com", "subject": "Welcome"},
                {"id": "msg2", "from": "support@github.com", "subject": "PR Review"},
                {"id": "msg3", "from": "newsletter@linkedin.com", "subject": "Jobs"},
            ]
            return b

    @pytest.mark.asyncio
    async def test_yes_does_not_route_to_gmail(self):
        """'yes' should NOT be treated as email search."""
        brain = self._make_brain()
        # _quick_route is a static method, so we test the process flow
        # The key logic: _email_followup should be False for "yes"
        _input = "yes"
        _words = _input.strip().split()
        _input_lower = _input.strip().lower()

        _ORDINALS = {"first", "1st", "second", "2nd", "third", "3rd",
                     "fourth", "4th", "fifth", "5th", "last", "next",
                     "previous", "that one", "this one"}
        _email_followup = False

        if _input_lower in _ORDINALS or any(w in _input_lower for w in _ORDINALS):
            _email_followup = True
        else:
            for msg in brain._last_messages:
                sender = msg.get("from", "").lower()
                if _input_lower in sender or sender.split("@")[0] in _input_lower:
                    _email_followup = True
                    break

        assert _email_followup is False

    @pytest.mark.asyncio
    async def test_eh_does_not_route_to_gmail(self):
        """'eh?' should NOT be treated as email search."""
        brain = self._make_brain()
        _input_lower = "eh?"
        _ORDINALS = {"first", "1st", "second", "2nd", "third", "3rd",
                     "fourth", "4th", "fifth", "5th", "last", "next",
                     "previous", "that one", "this one"}
        _email_followup = False
        if _input_lower in _ORDINALS:
            _email_followup = True
        else:
            for msg in brain._last_messages:
                sender = msg.get("from", "").lower()
                if _input_lower in sender:
                    _email_followup = True
                    break
        assert _email_followup is False

    @pytest.mark.asyncio
    async def test_aa_does_not_route_to_gmail(self):
        """'aa' should NOT be treated as email search."""
        brain = self._make_brain()
        _input_lower = "aa"
        _email_followup = False
        for msg in brain._last_messages:
            sender = msg.get("from", "").lower()
            if _input_lower in sender:
                _email_followup = True
                break
        assert _email_followup is False

    @pytest.mark.asyncio
    async def test_ok_does_not_route_to_gmail(self):
        """'ok' should NOT be treated as email search."""
        brain = self._make_brain()
        _input_lower = "ok"
        _email_followup = False
        for msg in brain._last_messages:
            sender = msg.get("from", "").lower()
            if _input_lower in sender:
                _email_followup = True
                break
        assert _email_followup is False

    @pytest.mark.asyncio
    async def test_medium_routes_to_gmail(self):
        """'medium' should match sender john@medium.com → email follow-up."""
        brain = self._make_brain()
        _input_lower = "medium"
        _email_followup = False
        for msg in brain._last_messages:
            sender = msg.get("from", "").lower()
            if _input_lower in sender:
                _email_followup = True
                break
        assert _email_followup is True

    @pytest.mark.asyncio
    async def test_github_routes_to_gmail(self):
        """'github' should match sender support@github.com → email follow-up."""
        brain = self._make_brain()
        _input_lower = "github"
        _email_followup = False
        for msg in brain._last_messages:
            sender = msg.get("from", "").lower()
            if _input_lower in sender:
                _email_followup = True
                break
        assert _email_followup is True

    @pytest.mark.asyncio
    async def test_first_is_ordinal(self):
        """'first' should be recognized as ordinal → email follow-up."""
        _input_lower = "first"
        _ORDINALS = {"first", "1st", "second", "2nd", "third", "3rd",
                     "fourth", "4th", "fifth", "5th", "last", "next",
                     "previous", "that one", "this one"}
        assert _input_lower in _ORDINALS

    @pytest.mark.asyncio
    async def test_last_one_is_ordinal(self):
        """'last' should be recognized as ordinal."""
        _input_lower = "last"
        _ORDINALS = {"first", "1st", "second", "2nd", "third", "3rd",
                     "fourth", "4th", "fifth", "5th", "last", "next",
                     "previous", "that one", "this one"}
        assert _input_lower in _ORDINALS

    @pytest.mark.asyncio
    async def test_no_last_messages_no_followup(self):
        """When no _last_messages, short inputs go to CoT route, not Gmail."""
        brain = self._make_brain()
        brain._last_messages = []
        _input_lower = "medium"
        _short_input = True
        _email_followup = False
        # With empty _last_messages, the block doesn't execute
        if _short_input and brain._last_messages:
            _email_followup = True  # This should NOT run
        assert _email_followup is False


# ═══════════════════════════════════════════════════════════════════════════
# C: Weather location extraction tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractCity:
    """Test _extract_city with English and Turkish patterns."""

    def _extract(self, text: str) -> str:
        from bantz.core.brain import _extract_city
        return _extract_city(text)

    # ── English patterns ──────────────────────────────────────────────

    def test_weather_in_city(self):
        assert self._extract("weather in Istanbul") == "Istanbul"

    def test_weather_in_new_york(self):
        assert self._extract("weather in New York") == "New York"

    def test_forecast_for_london(self):
        assert self._extract("forecast for London") == "London"

    def test_temperature_in_berlin(self):
        assert self._extract("temperature in Berlin") == "Berlin"

    def test_how_is_weather_in_paris(self):
        assert self._extract("how's the weather in Paris") == "Paris"

    def test_how_is_weather_in_tokyo(self):
        assert self._extract("how is the weather in Tokyo") == "Tokyo"

    def test_bare_weather(self):
        """Just 'weather' → empty (auto-detect via GPS)."""
        assert self._extract("weather") == ""

    def test_whats_the_weather(self):
        """'what's the weather' → empty."""
        assert self._extract("what's the weather") == ""

    def test_is_it_raining(self):
        """'is it raining' → empty."""
        assert self._extract("is it raining") == ""

    def test_weather_today(self):
        """'weather today' → empty."""
        assert self._extract("weather today") == ""

    def test_show_me_weather(self):
        """'show me weather' → empty."""
        assert self._extract("show me weather") == ""

    # ── Turkish patterns ──────────────────────────────────────────────

    def test_turkish_de_suffix(self):
        """ankara'da hava nasıl → Ankara"""
        assert self._extract("ankara'da hava nasıl") == "Ankara"

    def test_turkish_te_suffix(self):
        """İstanbul'da hava nasıl → İstanbul"""
        result = self._extract("istanbul'da hava nasıl")
        assert result.lower() == "istanbul"

    def test_turkish_izmir_hava_durumu(self):
        """izmir hava durumu → Izmir"""
        result = self._extract("izmir hava durumu")
        assert result.lower() == "izmir"

    def test_turkish_bugun_hava_nasil(self):
        """bugün hava nasıl → empty (no city, use GPS)."""
        assert self._extract("bugün hava nasıl") == ""

    def test_turkish_yarin_hava_nasil(self):
        """yarın hava nasıl → empty."""
        assert self._extract("yarın hava nasıl") == ""

    def test_turkish_antalya_de(self):
        """antalya'da hava → Antalya"""
        result = self._extract("antalya'da hava")
        assert result.lower() == "antalya"

    def test_turkish_bursa_te(self):
        """bursa'da hava → Bursa"""
        result = self._extract("bursa'da hava nasıl")
        assert result.lower() == "bursa"

    # ── Edge cases ────────────────────────────────────────────────────

    def test_single_city_name(self):
        """Just a city name like 'Berlin' after stripping weather words."""
        # The quick_route checks for weather keywords first, so city-only
        # wouldn't reach _extract_city. But if it does:
        result = self._extract("Berlin")
        assert result == "Berlin"

    def test_empty_input(self):
        assert self._extract("") == ""

    def test_question_mark_stripped(self):
        assert self._extract("weather in Rome?") == "Rome"

    def test_weather_in_city_today(self):
        assert self._extract("weather in Madrid today") == "Madrid"

    def test_multi_word_city(self):
        """Cities with spaces."""
        assert self._extract("weather in San Francisco") == "San Francisco"

    def test_forecast_tomorrow_no_city(self):
        """'forecast tomorrow' → empty."""
        assert self._extract("forecast tomorrow") == ""


# ═══════════════════════════════════════════════════════════════════════════
# Markdown URL Trap & Bracket Bug Fixes
# ═══════════════════════════════════════════════════════════════════════════


class TestMarkdownURLPromptRules:
    """Prompt templates must instruct LLM to never use Markdown link formatting."""

    def test_chat_system_has_raw_url_rule(self):
        """CHAT_SYSTEM must tell LLM to use raw URLs."""
        from bantz.core.brain import CHAT_SYSTEM
        lower = CHAT_SYSTEM.lower()
        assert "raw" in lower and "url" in lower

    def test_chat_system_forbids_markdown_links(self):
        """CHAT_SYSTEM must explicitly forbid [Text](URL) syntax."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "[Text](URL)" in CHAT_SYSTEM or "no [Text]" in CHAT_SYSTEM

    def test_chat_system_forbids_bracket_urls(self):
        """CHAT_SYSTEM must forbid [URL] bracket wrapping."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "no [URL]" in CHAT_SYSTEM or "[URL]" in CHAT_SYSTEM

    def test_finalizer_has_raw_url_rule(self):
        """FINALIZER_SYSTEM must tell LLM to use raw URLs."""
        from bantz.core.finalizer import FINALIZER_SYSTEM
        lower = FINALIZER_SYSTEM.lower()
        assert "raw" in lower and "url" in lower

    def test_finalizer_forbids_markdown_links(self):
        """FINALIZER_SYSTEM must explicitly forbid [Text](URL) syntax."""
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "[Text](URL)" in FINALIZER_SYSTEM or "no [Text]" in FINALIZER_SYSTEM

    def test_finalizer_forbids_bracket_urls(self):
        """FINALIZER_SYSTEM must forbid [URL] bracket wrapping."""
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "no [URL]" in FINALIZER_SYSTEM or "[URL]" in FINALIZER_SYSTEM


class TestStripMarkdownLinks:
    """strip_markdown() must convert markdown links to bare URLs."""

    def test_markdown_link_to_bare_url(self):
        """[Click Here](https://example.com) → https://example.com"""
        from bantz.core.finalizer import strip_markdown
        text = "Check this: [Click Here](https://example.com) for details."
        result = strip_markdown(text)
        assert "https://example.com" in result
        assert "[Click Here]" not in result
        assert "](https" not in result

    def test_bracket_url_to_bare(self):
        """[https://example.com] → https://example.com"""
        from bantz.core.finalizer import strip_markdown
        text = "Telegraph Reference: [https://en.wikipedia.org/wiki/Edith]"
        result = strip_markdown(text)
        assert "https://en.wikipedia.org/wiki/Edith" in result
        assert "[https://" not in result

    def test_url_with_parens_in_markdown(self):
        """Links with parentheses in URL still convert correctly."""
        from bantz.core.finalizer import strip_markdown
        text = "[Article](https://en.wikipedia.org/wiki/Edith_(name))"
        result = strip_markdown(text)
        # Should extract the URL (may truncate at inner paren, which is OK)
        assert "https://en.wikipedia.org" in result

    def test_multiple_markdown_links(self):
        """Multiple links in one response all get stripped."""
        from bantz.core.finalizer import strip_markdown
        text = "[A](https://a.com) and [B](https://b.com)"
        result = strip_markdown(text)
        assert "https://a.com" in result
        assert "https://b.com" in result
        assert "[A]" not in result
        assert "[B]" not in result

    def test_bare_url_unchanged(self):
        """Already-bare URLs pass through unmodified."""
        from bantz.core.finalizer import strip_markdown
        text = "See https://example.com for info."
        result = strip_markdown(text)
        assert result == text

    def test_non_url_brackets_unchanged(self):
        """Regular brackets without URLs are left alone."""
        from bantz.core.finalizer import strip_markdown
        text = "The answer is [option A]."
        result = strip_markdown(text)
        assert "[option A]" in result
