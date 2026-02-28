"""
Bantz v2 — Web Search Tool
DuckDuckGo-based internet search with TTL caching, source URL tracking,
and result deduplication.

Implements #79 (web_search tool) and #65 (caching + source tracking).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.web_search")

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

# ── Session state for follow-ups ───────────────────────────────────────────────

_last_results: list[dict] = []


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

TIMEOUT = 10.0


async def _ddg_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Search DuckDuckGo via the HTML lite endpoint + instant answer API.
    Returns list of {"title": ..., "url": ..., "snippet": ...}
    """
    results: list[dict] = []
    seen_urls: set[str] = set()

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

    except Exception as exc:
        log.debug("DDG instant answer failed: %s", exc)

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
                # Each result block: <a rel="nofollow" class="result__a" href="...">title</a>
                # Snippet: <a class="result__snippet" ...>text</a>
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

                    # Clean HTML tags from title and snippet
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

        except Exception as exc:
            log.debug("DDG HTML lite failed: %s", exc)

    return results[:max_results]


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
    """Format search results with source URLs for the user."""
    if not results:
        return f"No results found for: {query}"

    lines = []
    tag = " (cached)" if cached else ""
    lines.append(f"Search results for \"{query}\"{tag}:")
    lines.append("")

    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")

        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            # Truncate snippet to 2 lines max
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
        global _last_results

        if not query and not index:
            return ToolResult(success=False, output="", error="No search query provided")

        # Follow-up: "tell me more about the first/second result"
        if index and _last_results:
            idx = index - 1
            if 0 <= idx < len(_last_results):
                r = _last_results[idx]
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
                error=f"No result at index {index}. Last search had {len(_last_results)} results.",
            )

        # Check cache
        cached_results = _cache.get(query)
        if cached_results is not None:
            _last_results = cached_results
            output = _format_results(cached_results, query, cached=True)
            return ToolResult(
                success=True,
                output=output,
                data={"results": cached_results, "cached": True, "count": len(cached_results)},
            )

        # Fresh search
        try:
            results = await _ddg_search(query)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Search failed: {exc}")

        if not results:
            return ToolResult(
                success=True,
                output=f"No results found for: {query}",
                data={"results": [], "count": 0},
            )

        # Deduplicate against recent results
        recent = _cache.recent_results()
        results = _deduplicate(results, recent)

        # Store in cache + session state
        _cache.put(query, results)
        _last_results = results

        output = _format_results(results, query, cached=False)
        return ToolResult(
            success=True,
            output=output,
            data={"results": results, "cached": False, "count": len(results)},
        )


# Register
registry.register(WebSearchTool())
