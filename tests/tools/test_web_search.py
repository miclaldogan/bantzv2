"""Tests for the web tools (web_search / web_research / web_news).

These cover the current bantz-web-backed implementation: tool registration,
metadata, empty-input handling, delegation to the underlying functions, and the
research cancel flag. The heavy lifting lives in vendor/bantz-web, so the
network-touching functions are mocked here — these stay fast and hermetic.

(The previous suite tested an older in-module DuckDuckGo implementation
— `_format_results`, `_ddg_search`, `SearchOutcome`, `_classify_http_error`,
`_last_results_store` — that was replaced by the bantz-web integration.)
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

from bantz.tools import registry
from bantz.tools.web_search import (
    WebNewsTool,
    WebResearchTool,
    WebSearchTool,
)


def _run(coro):
    return asyncio.run(coro)


class TestRegistration:
    def test_all_three_tools_registered(self):
        for name in ("web_search", "web_research", "web_news"):
            assert registry.get(name) is not None, f"{name} not registered"

    def test_tool_names_and_risk(self):
        assert WebSearchTool.name == "web_search"
        assert WebResearchTool.name == "web_research"
        assert WebNewsTool.name == "web_news"
        # All three are read-only / safe.
        assert WebSearchTool.risk_level == "safe"
        assert WebResearchTool.risk_level == "safe"
        assert WebNewsTool.risk_level == "safe"


class TestDescriptions:
    def test_web_search_has_specificity_rule(self):
        desc = WebSearchTool.description.lower()
        assert "specific" in desc or "vague" in desc

    def test_web_search_distinguishes_from_research(self):
        desc = WebSearchTool.description.lower()
        assert "quick" in desc and "report" in desc

    def test_web_research_mentions_deep_report(self):
        desc = WebResearchTool.description.lower()
        assert "report" in desc and ("deep" in desc or "research" in desc)


class TestWebSearchTool:
    def test_empty_query_returns_error(self):
        result = _run(WebSearchTool().execute(query=""))
        assert result.success is False
        assert "query" in (result.error or "").lower()

    def test_delegates_to_execute_web_search(self):
        with patch(
            "bantz.tools.web_search.execute_web_search", return_value="RESULTS"
        ) as m:
            result = _run(WebSearchTool().execute(query="python asyncio"))
        assert result.success is True
        assert result.output == "RESULTS"
        assert result.data["query"] == "python asyncio"
        m.assert_called_once()

    def test_text_kwarg_fallback(self):
        with patch(
            "bantz.tools.web_search.execute_web_search", return_value="ok"
        ):
            result = _run(WebSearchTool().execute(text="weather"))
        assert result.success is True

    def test_underlying_exception_is_caught(self):
        with patch(
            "bantz.tools.web_search.execute_web_search",
            side_effect=RuntimeError("boom"),
        ):
            result = _run(WebSearchTool().execute(query="x"))
        assert result.success is False
        assert "boom" in (result.error or "")


class TestWebNewsTool:
    def test_defaults_topic_to_general(self):
        with patch(
            "bantz.tools.web_search.execute_web_news", return_value="news"
        ) as m:
            result = _run(WebNewsTool().execute(topic=""))
        assert result.success is True
        m.assert_called_once_with("general")

    def test_passes_topic_through(self):
        with patch(
            "bantz.tools.web_search.execute_web_news", return_value="n"
        ) as m:
            _run(WebNewsTool().execute(topic="elections"))
        m.assert_called_once_with("elections")


class TestWebResearchTool:
    def test_empty_topic_returns_error(self):
        result = _run(WebResearchTool().execute(topic=""))
        assert result.success is False
        assert "topic" in (result.error or "").lower()

    def test_has_cancel_event(self):
        tool = WebResearchTool()
        assert isinstance(tool._research_cancelled, threading.Event)
        # Cancel flag toggles cleanly.
        tool._research_cancelled.set()
        assert tool._research_cancelled.is_set()
        tool._research_cancelled.clear()
        assert not tool._research_cancelled.is_set()

    def test_successful_research_returns_report(self):
        with patch(
            "bantz.tools.web_search.execute_web_research", return_value="REPORT"
        ):
            result = _run(WebResearchTool().execute(topic="quantum computing"))
        assert result.success is True
        assert result.output == "REPORT"
        assert result.data["topic"] == "quantum computing"

    def test_research_emits_structured_progress_not_chat_tokens(self):
        """web_research streams structured research_progress events (#490),
        never raw chat_token text, so the UI can render a progress widget."""
        events: list[tuple[str, dict]] = []
        with patch(
            "bantz.tools.web_search.execute_web_research", return_value="REPORT"
        ), patch(
            "bantz.tools.web_search.bus.emit_threadsafe",
            side_effect=lambda name, **kw: events.append((name, kw)),
        ):
            _run(WebResearchTool().execute(topic="tea"))

        names = [n for n, _ in events]
        # A fast (mocked) run emits exactly a start + a terminal event…
        assert names == ["research_progress", "research_progress"]
        # …and never the old noisy chat_token path.
        assert "chat_token" not in names

        start = events[0][1]
        assert start["stage"] == "searching"
        assert start["state"] == "running"
        assert "tea" in start["detail"]

        done = events[-1][1]
        assert done["stage"] == "done"
        assert done["state"] == "done"
        assert done["elapsed"] >= 0

    def test_emit_research_progress_payload_shape(self):
        from bantz.tools.web_search import _emit_research_progress

        events: list[tuple[str, dict]] = []
        with patch(
            "bantz.tools.web_search.bus.emit_threadsafe",
            side_effect=lambda name, **kw: events.append((name, kw)),
        ):
            _emit_research_progress(
                "cancelled", "Research cancelled.", elapsed=12, state="cancelled"
            )
        assert events == [(
            "research_progress",
            {"stage": "cancelled", "detail": "Research cancelled.",
             "elapsed": 12, "state": "cancelled"},
        )]
