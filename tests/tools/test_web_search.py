"""Tests for web_search URL deduplication and formatting."""
from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# _format_results deduplication
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatResultsDedup:
    """_format_results must deduplicate URLs at format time."""

    def _format(self, results, query="test", cached=False):
        from bantz.tools.web_search import _format_results
        return _format_results(results, query, cached)

    def test_unique_urls_all_shown(self):
        results = [
            {"title": "A", "url": "https://a.com", "snippet": "a"},
            {"title": "B", "url": "https://b.com", "snippet": "b"},
        ]
        out = self._format(results)
        assert "https://a.com" in out
        assert "https://b.com" in out
        assert "1. A" in out
        assert "2. B" in out

    def test_duplicate_urls_removed(self):
        results = [
            {"title": "Wiki 1", "url": "https://en.wikipedia.org/wiki/Edith", "snippet": "a"},
            {"title": "Wiki 2", "url": "https://en.wikipedia.org/wiki/Edith", "snippet": "b"},
            {"title": "Other", "url": "https://other.com", "snippet": "c"},
        ]
        out = self._format(results)
        # Only 2 numbered results (1 and 2), not 3
        assert "1. Wiki 1" in out
        assert "2. Other" in out
        # The duplicate is skipped
        assert "3." not in out

    def test_triple_duplicate_reduced_to_one(self):
        results = [
            {"title": f"R{i}", "url": "https://same.com/page", "snippet": f"s{i}"}
            for i in range(3)
        ]
        out = self._format(results)
        assert out.count("https://same.com/page") == 1

    def test_empty_url_not_deduped(self):
        """Results without URLs should all be included."""
        results = [
            {"title": "A", "url": "", "snippet": "a"},
            {"title": "B", "url": "", "snippet": "b"},
        ]
        out = self._format(results)
        assert "1. A" in out
        assert "2. B" in out

    def test_no_results(self):
        out = self._format([])
        assert "No results found" in out

    def test_cached_tag(self):
        results = [{"title": "X", "url": "https://x.com", "snippet": "x"}]
        out = self._format(results, cached=True)
        assert "(cached)" in out

    def test_format_uses_bare_urls_not_markdown(self):
        """_format_results must output bare URLs (no markdown links)."""
        results = [
            {"title": "Test", "url": "https://example.com/path", "snippet": "s"},
        ]
        out = self._format(results)
        assert "URL: https://example.com/path" in out
        assert "[" not in out  # no brackets
        assert "](" not in out  # no markdown link syntax


# ═══════════════════════════════════════════════════════════════════════════
# _ddg_search deduplication (unit check)
# ═══════════════════════════════════════════════════════════════════════════


class TestDDGSearchDedup:
    """_ddg_search must have URL dedup via seen_urls set."""

    def test_ddg_search_uses_seen_urls(self):
        """Source code must use a seen_urls set for deduplication."""
        import inspect
        from bantz.tools.web_search import _ddg_search
        src = inspect.getsource(_ddg_search)
        assert "seen_urls" in src
        assert "set()" in src

    def test_format_results_uses_seen_urls(self):
        """_format_results source must also use seen_urls set."""
        import inspect
        from bantz.tools.web_search import _format_results
        src = inspect.getsource(_format_results)
        assert "seen_urls" in src
