"""Bantz v2 - Web Search Tool (#256)

DuckDuckGo-based internet search with TTL caching, source URL tracking,
result deduplication, and proper error classification.

Implements #79 (web_search tool), #65 (caching + source tracking),
and #256 (silent failure fix + global mutable state removal).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import httpx

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.web_search")


# ── SearchOutcome ───────────────────────────────────────────────────────────────────

@dataclass
class SearchOutcome:
    """Result of a search attempt with error classification.

    Distinguishes genuine “zero hits” (infrastructure worked, no matches)
    from infrastructure failures (network error, 403/429 block, timeout).
    This feeds into the Circuit Breaker (PR #260) via success=False.
    """
    results: list[dict] = field(default_factory=list)
    source: str = "none"                 # "api", "html", "both", "none"
    infrastructure_error: str | None = None  # None = infra OK

# ── Cache ──────────────────────────────────────────────────────────────────────

DEFAULT_CACHE_TTL = 900  # 15 minutes


@dataclass
class _CacheEntry:
    query: str
    results: list[dict]
    timestamp: float


class _SearchCache:
    """Simple in-memory TTL cache for search results."""

    def __init__(self, ttl: int = DEFAULT_CACHE_TTL) -> None:
        self._ttl = ttl
        self._store: dict[str, _CacheEntry] = {}

    def get(self, query: str) -> list[dict] | None:
        key = self._key(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry.timestamp > self._ttl:
            del self._store[key]
            return None
        log.debug("Cache hit for: %s", query)
        return entry.results

    def put(self, query: str, results: list[dict]) -> None:
        key = self._key(query)
        self._store[key] = _CacheEntry(
            query=query, results=results, timestamp=time.time()
        )
        # Evict old entries (keep max 50)
        if len(self._store) > 50:
            oldest_key = min(self._store, key=lambda k: self._store[k].timestamp)
            del self._store[oldest_key]

    def recent_results(self, limit: int = 20) -> list[dict]:
        """Get all recent results across queries for deduplication."""
        all_results: list[dict] = []
        now = time.time()
        for entry in sorted(self._store.values(), key=lambda e: e.timestamp, reverse=True):
            if now - entry.timestamp <= self._ttl:
                all_results.extend(entry.results)
            if len(all_results) >= limit:
                break
        return all_results[:limit]

    @staticmethod
    def _key(query: str) -> str:
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()


# Global cache instance
_cache = _SearchCache()

# ── Session-scoped last results (bounded LRU) ───────────────────────────────────

_MAX_SESSIONS = 20


class _LastResultsStore:
    """Bounded dict keyed by session_id to avoid memory leaks.

    When the tool is a singleton, naively storing results per session
    without cleanup leads to unbounded growth.  This class uses an
    OrderedDict capped at ``_MAX_SESSIONS`` entries (LRU eviction).
    """

    def __init__(self, maxsize: int = _MAX_SESSIONS) -> None:
        self._store: OrderedDict[str, list[dict]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, session_id: str) -> list[dict]:
        results = self._store.get(session_id, [])
        if results:
            self._store.move_to_end(session_id)
        return results

    def put(self, session_id: str, results: list[dict]) -> None:
        self._store[session_id] = results
        self._store.move_to_end(session_id)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)


_last_results_store = _LastResultsStore()


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

TIMEOUT = 10.0


async def _ddg_search(query: str, max_results: int = 8) -> SearchOutcome:
    """
    Search DuckDuckGo via the HTML lite endpoint + instant answer API.
    Returns a SearchOutcome with results and error classification.
    """
    results: list[dict] = []
    seen_urls: set[str] = set()
    api_error: str | None = None
    html_error: str | None = None
    source = "none"

    # Method 1: DuckDuckGo Instant Answer API (fast, structured)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                headers={"User-Agent": "Bantz/2.0"},
            )
            resp.raise_for_status()
            data = resp.json()

            # Abstract / instant answer
            if data.get("AbstractText") and data.get("AbstractURL"):
                results.append({
                    "title": data.get("Heading", query),
                    "url": data["AbstractURL"],
                    "snippet": data["AbstractText"][:300],
                    "source": data.get("AbstractSource", ""),
                })
                seen_urls.add(data["AbstractURL"])

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("FirstURL"):
                    url = topic["FirstURL"]
                    if url not in seen_urls:
                        text = topic.get("Text", "")
                        results.append({
                            "title": text[:80] if text else "",
                            "url": url,
                            "snippet": text[:300],
                            "source": "DuckDuckGo",
                        })
                        seen_urls.add(url)

            # Results from API sub-topics
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and "Topics" in topic:
                    for sub in topic["Topics"][:3]:
                        if sub.get("FirstURL") and sub["FirstURL"] not in seen_urls:
                            results.append({
                                "title": sub.get("Text", "")[:80],
                                "url": sub["FirstURL"],
                                "snippet": sub.get("Text", "")[:300],
                                "source": "DuckDuckGo",
                            })
                            seen_urls.add(sub["FirstURL"])

            source = "api"

    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        api_error = f"HTTP {code} ({_classify_http_error(code)})"
        log.warning("DDG API failed: %s", api_error)
    except httpx.TimeoutException:
        api_error = "timeout"
        log.warning("DDG API timed out")
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        log.warning("DDG API failed: %s", api_error)

    # Method 2: DuckDuckGo HTML lite (fallback for web results)
    if len(results) < 3:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Bantz/2.0"},
                )
                resp.raise_for_status()
                html = resp.text

                # Parse results from HTML lite page
                links = re.findall(
                    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.+?)</a>',
                    html,
                )
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.+?)</a>',
                    html,
                    re.DOTALL,
                )

                for i, (url, title) in enumerate(links[:max_results]):
                    # Clean up URL (DDG wraps in redirect)
                    url_match = re.search(r"uddg=([^&]+)", url)
                    if url_match:
                        from urllib.parse import unquote
                        url = unquote(url_match.group(1))

                    if url in seen_urls:
                        continue

                    clean_title = re.sub(r"<[^>]+>", "", title).strip()
                    clean_snippet = ""
                    if i < len(snippets):
                        clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()[:300]

                    results.append({
                        "title": clean_title[:80],
                        "url": url,
                        "snippet": clean_snippet,
                        "source": "DuckDuckGo",
                    })
                    seen_urls.add(url)

                source = "both" if source == "api" else "html"

        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            html_error = f"HTTP {code} ({_classify_http_error(code)})"
            log.warning("DDG HTML failed: %s", html_error)
        except httpx.TimeoutException:
            html_error = "timeout"
            log.warning("DDG HTML timed out")
        except Exception as exc:
            html_error = f"{type(exc).__name__}: {exc}"
            log.warning("DDG HTML failed: %s", html_error)

    # Build infrastructure error string if BOTH methods failed
    infra_error: str | None = None
    if not results and (api_error or html_error):
        parts = []
        if api_error:
            parts.append(f"API: {api_error}")
        if html_error:
            parts.append(f"HTML: {html_error}")
        infra_error = " | ".join(parts)

    return SearchOutcome(
        results=results[:max_results],
        source=source,
        infrastructure_error=infra_error,
    )


def _classify_http_error(code: int) -> str:
    """Human-readable classification of HTTP error codes."""
    if code == 403:
        return "bot-blocked"
    if code == 429:
        return "rate-limited"
    if code == 503:
        return "service-unavailable"
    if 400 <= code < 500:
        return "client-error"
    if 500 <= code < 600:
        return "server-error"
    return "unknown"


def _deduplicate(new_results: list[dict], recent: list[dict]) -> list[dict]:
    """Remove results already seen in recent searches."""
    recent_urls = {r["url"] for r in recent if r.get("url")}
    deduped = []
    for r in new_results:
        if r.get("url") not in recent_urls:
            deduped.append(r)
        else:
            log.debug("Dedup: skipping %s", r.get("url"))
    # If everything was deduped, return originals (user explicitly searched again)
    return deduped if deduped else new_results


def _format_results(results: list[dict], query: str, cached: bool) -> str:
    """Format search results with source URLs for the user.

    Deduplicates URLs at format time as a safety net — even if
    ``_ddg_search`` already filters, callers may inject duplicates.
    """
    if not results:
        return f"No results found for: {query}"

    lines: list[str] = []
    tag = " (cached)" if cached else ""
    lines.append(f"Search results for \"{query}\"{tag}:")
    lines.append("")

    seen_urls: set[str] = set()
    idx = 0
    for r in results:
        url = r.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        idx += 1

        title = r.get("title", "Untitled")
        snippet = r.get("snippet", "")

        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet[:200]}")
        lines.append("")

    lines.append("Want me to tell you more about any of these?")
    return "\n".join(lines)


# ── Tool class ─────────────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the internet via DuckDuckGo. "
        "Use for: search, look up, find online, google, research, "
        "anything about, is there any information about."
    )
    risk_level = "safe"

    async def execute(self, query: str = "", index: int = 0, **kwargs: Any) -> ToolResult:
        """
        Search the web.
        - query: search query string
        - index: if > 0, return details about the Nth result from last search
        """
        # Session key for scoped last-results storage
        session_id = kwargs.get("session_id", "_default")

        if not query and not index:
            return ToolResult(success=False, output="", error="No search query provided")

        # Follow-up: "tell me more about the first/second result"
        if index:
            last_results = _last_results_store.get(session_id)
            if last_results:
                idx = index - 1
                if 0 <= idx < len(last_results):
                    r = last_results[idx]
                    detail = (
                        f"Result #{index}: {r.get('title', '')}\n"
                        f"URL: {r.get('url', '')}\n"
                        f"Source: {r.get('source', '')}\n\n"
                        f"{r.get('snippet', 'No additional details available.')}"
                    )
                    return ToolResult(
                        success=True,
                        output=detail,
                        data={"result": r, "index": index},
                    )
                return ToolResult(
                    success=False, output="",
                    error=f"No result at index {index}. Last search had {len(last_results)} results.",
                )
            return ToolResult(
                success=False, output="",
                error="No previous search results in this session.",
            )

        # Check cache
        cached_results = _cache.get(query)
        if cached_results is not None:
            _last_results_store.put(session_id, cached_results)
            output = _format_results(cached_results, query, cached=True)
            return ToolResult(
                success=True,
                output=output,
                data={"results": cached_results, "cached": True, "count": len(cached_results)},
            )

        # Fresh search
        outcome = await _ddg_search(query)

        # Infrastructure failure → success=False (feeds Circuit Breaker)
        if outcome.infrastructure_error and not outcome.results:
            return ToolResult(
                success=False,
                output="",
                error=f"Search engine unreachable or blocked: {outcome.infrastructure_error}",
            )

        # Genuine zero hits (infra worked, no matches)
        if not outcome.results:
            return ToolResult(
                success=True,
                output=f"No results found for: {query}",
                data={"results": [], "count": 0, "source": outcome.source},
            )

        # Deduplicate against recent results
        recent = _cache.recent_results()
        results = _deduplicate(outcome.results, recent)

        # Store in cache + session-scoped state
        _cache.put(query, results)
        _last_results_store.put(session_id, results)

        output = _format_results(results, query, cached=False)
        return ToolResult(
            success=True,
            output=output,
            data={
                "results": results,
                "cached": False,
                "count": len(results),
                "source": outcome.source,
            },
        )


# Register
registry.register(WebSearchTool())
