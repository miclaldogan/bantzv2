"""Tests for web_search URL deduplication, formatting, error handling (#256)."""
from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
        # extract http(s) URLs and assert presence
        urls = re.findall(r"https?://[^\s\)\]]+", out)
        assert any(u.startswith("https://a.com") for u in urls)
        assert any(u.startswith("https://b.com") for u in urls)
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
        urls = re.findall(r"https?://[^\s\)\]]+", out)
        # deduplication: only one occurrence of the canonical URL
        assert sum(1 for u in urls if u.startswith("https://same.com/page")) == 1

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
        # extract http(s) URLs and assert the expected URL is present
        urls = re.findall(r"https?://[^\s\)\]]+", out)
        assert any(u.startswith("https://example.com/path") for u in urls)
        # ensure there are no markdown link patterns like [text](http...)
        assert not re.search(r"\[.*\]\(https?://", out)


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


# ═══════════════════════════════════════════════════════════════════════════
# SearchOutcome dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchOutcome:
    """Verify the SearchOutcome dataclass structure."""

    def test_defaults(self):
        from bantz.tools.web_search import SearchOutcome
        o = SearchOutcome()
        assert o.results == []
        assert o.source == "none"
        assert o.infrastructure_error is None

    def test_with_error(self):
        from bantz.tools.web_search import SearchOutcome
        o = SearchOutcome(results=[], source="none", infrastructure_error="API: timeout | HTML: HTTP 403 (bot-blocked)")
        assert o.infrastructure_error is not None
        assert "403" in o.infrastructure_error

    def test_successful_search(self):
        from bantz.tools.web_search import SearchOutcome
        o = SearchOutcome(results=[{"title": "a"}], source="api")
        assert o.results
        assert o.infrastructure_error is None


# ═══════════════════════════════════════════════════════════════════════════
# _classify_http_error
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifyHTTPError:
    """Verify HTTP error code classification."""

    def _classify(self, code):
        from bantz.tools.web_search import _classify_http_error
        return _classify_http_error(code)

    def test_403_bot_blocked(self):
        assert self._classify(403) == "bot-blocked"

    def test_429_rate_limited(self):
        assert self._classify(429) == "rate-limited"

    def test_503_unavailable(self):
        assert self._classify(503) == "service-unavailable"

    def test_500_server_error(self):
        assert self._classify(500) == "server-error"

    def test_400_client_error(self):
        assert self._classify(400) == "client-error"

    def test_unknown(self):
        assert self._classify(301) == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# _LastResultsStore (session-scoped LRU)
# ═══════════════════════════════════════════════════════════════════════════


class TestLastResultsStore:
    """Verify the bounded session-scoped results store."""

    def _store(self, maxsize=3):
        from bantz.tools.web_search import _LastResultsStore
        return _LastResultsStore(maxsize=maxsize)

    def test_put_get(self):
        s = self._store()
        s.put("sess1", [{"title": "a"}])
        assert len(s.get("sess1")) == 1

    def test_get_empty(self):
        s = self._store()
        assert s.get("nonexistent") == []

    def test_lru_eviction(self):
        s = self._store(maxsize=2)
        s.put("s1", [{"title": "a"}])
        s.put("s2", [{"title": "b"}])
        s.put("s3", [{"title": "c"}])  # evicts s1
        assert s.get("s1") == []
        assert len(s.get("s2")) == 1
        assert len(s.get("s3")) == 1

    def test_access_refreshes_lru(self):
        s = self._store(maxsize=2)
        s.put("s1", [{"title": "a"}])
        s.put("s2", [{"title": "b"}])
        s.get("s1")  # refresh s1, making s2 oldest
        s.put("s3", [{"title": "c"}])  # evicts s2
        assert len(s.get("s1")) == 1
        assert s.get("s2") == []


# ═══════════════════════════════════════════════════════════════════════════
# _ddg_search error classification
# ═══════════════════════════════════════════════════════════════════════════


class TestDDGSearchErrorHandling:
    """Verify _ddg_search returns proper SearchOutcome on failures."""

    @pytest.mark.asyncio
    async def test_both_methods_timeout(self):
        """When both API and HTML time out, returns infrastructure_error."""
        from bantz.tools.web_search import _ddg_search

        async def timeout_get(*args, **kwargs):
            raise httpx.ReadTimeout("read timed out")

        mock_client = AsyncMock()
        mock_client.get = timeout_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            outcome = await _ddg_search("test query")

        assert outcome.results == []
        assert outcome.infrastructure_error is not None
        assert "timeout" in outcome.infrastructure_error.lower()

    @pytest.mark.asyncio
    async def test_api_403_html_403(self):
        """When both methods return 403, error includes bot-blocked."""
        from bantz.tools.web_search import _ddg_search

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {}

        async def forbidden_get(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "403 Forbidden", request=MagicMock(), response=mock_response
            )

        mock_client = AsyncMock()
        mock_client.get = forbidden_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            outcome = await _ddg_search("blocked query")

        assert outcome.results == []
        assert outcome.infrastructure_error is not None
        assert "403" in outcome.infrastructure_error
        assert "bot-blocked" in outcome.infrastructure_error

    @pytest.mark.asyncio
    async def test_api_429_rate_limited(self):
        """429 rate-limit is classified correctly."""
        from bantz.tools.web_search import _ddg_search

        mock_response = MagicMock()
        mock_response.status_code = 429

        async def limited_get(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "429 Too Many Requests", request=MagicMock(), response=mock_response
            )

        mock_client = AsyncMock()
        mock_client.get = limited_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            outcome = await _ddg_search("rate limited")

        assert "rate-limited" in (outcome.infrastructure_error or "")


# ═══════════════════════════════════════════════════════════════════════════
# WebSearchTool.execute() — Circuit Breaker integration
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteErrorHandling:
    """Verify execute() returns success=False on infra failures."""

    @pytest.mark.asyncio
    async def test_infra_failure_returns_success_false(self):
        """Infra error → success=False (Circuit Breaker triggers)."""
        from bantz.tools.web_search import WebSearchTool, SearchOutcome

        tool = WebSearchTool()
        outcome = SearchOutcome(
            results=[], source="none",
            infrastructure_error="API: timeout | HTML: HTTP 403 (bot-blocked)",
        )

        with patch("bantz.tools.web_search._ddg_search", return_value=outcome), \
             patch("bantz.tools.web_search._cache") as mock_cache:
            mock_cache.get.return_value = None
            result = await tool.execute(query="test")

        assert result.success is False
        assert "unreachable or blocked" in result.error
        assert "403" in result.error

    @pytest.mark.asyncio
    async def test_genuine_zero_hits_returns_success_true(self):
        """Zero results but no infra error → success=True."""
        from bantz.tools.web_search import WebSearchTool, SearchOutcome

        tool = WebSearchTool()
        outcome = SearchOutcome(results=[], source="both", infrastructure_error=None)

        with patch("bantz.tools.web_search._ddg_search", return_value=outcome), \
             patch("bantz.tools.web_search._cache") as mock_cache:
            mock_cache.get.return_value = None
            result = await tool.execute(query="xyzzy nonexistent gibberish")

        assert result.success is True
        assert "No results found" in result.output
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_successful_search_has_source(self):
        """Successful search includes source in data."""
        from bantz.tools.web_search import WebSearchTool, SearchOutcome

        tool = WebSearchTool()
        outcome = SearchOutcome(
            results=[{"title": "Py", "url": "https://python.org", "snippet": "s", "source": "DDG"}],
            source="api",
        )

        with patch("bantz.tools.web_search._ddg_search", return_value=outcome), \
             patch("bantz.tools.web_search._cache") as mock_cache:
            mock_cache.get.return_value = None
            mock_cache.recent_results.return_value = []
            result = await tool.execute(query="python")

        assert result.success is True
        assert result.data["source"] == "api"
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_session_scoped_follow_up(self):
        """Follow-up index uses session-scoped store, not global."""
        from bantz.tools.web_search import WebSearchTool, _last_results_store

        tool = WebSearchTool()
        _last_results_store.put("sess_A", [
            {"title": "R1", "url": "https://r1.com", "snippet": "s1", "source": "DDG"},
        ])
        _last_results_store.put("sess_B", [
            {"title": "R2", "url": "https://r2.com", "snippet": "s2", "source": "DDG"},
        ])

        result_a = await tool.execute(index=1, session_id="sess_A")
        assert result_a.success is True
        assert "R1" in result_a.output

        result_b = await tool.execute(index=1, session_id="sess_B")
        assert result_b.success is True
        assert "R2" in result_b.output

    @pytest.mark.asyncio
    async def test_no_previous_results_error(self):
        """Follow-up with no previous results returns helpful error."""
        from bantz.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        result = await tool.execute(index=1, session_id="fresh_session_xyz")
        assert result.success is False
        assert "No previous search results" in result.error

    @pytest.mark.asyncio
    async def test_no_global_last_results(self):
        """Module no longer has a global _last_results list."""
        import bantz.tools.web_search as mod
        assert not hasattr(mod, "_last_results") or not isinstance(getattr(mod, "_last_results", None), list)
