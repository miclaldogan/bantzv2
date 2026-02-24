"""
Bantz v3 — Web Search Tool (DuckDuckGo)
Privacy-first internet search. No API key needed.
Uses DuckDuckGo HTML scraping via httpx.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from bantz.tools import BaseTool, ToolResult, registry

TIMEOUT = 10.0
DDG_URL = "https://html.duckduckgo.com/html/"
MAX_RESULTS = 5


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Searches the internet via DuckDuckGo. "
        "Use for: search online, look up, find information, google, "
        "current events, who is, what is, how to."
    )
    risk_level = "safe"

    async def execute(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query:
            return ToolResult(success=False, output="", error="Query required.")

        try:
            results = await self._search(query)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Search failed: {exc}")

        if not results:
            return ToolResult(
                success=True,
                output=f"No results found for: {query}",
                data={"query": query, "count": 0},
            )

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append(f"   {r['url']}")

        return ToolResult(
            success=True,
            output=f"Search: {query}\n\n" + "\n".join(lines),
            data={"query": query, "count": len(results), "results": results},
        )

    async def _search(self, query: str) -> list[dict]:
        """Fetch DuckDuckGo HTML results and parse."""
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = await client.post(
                DDG_URL,
                data={"q": query, "kl": "us-en"},
                headers=headers,
            )
            resp.raise_for_status()
            return self._parse(resp.text)

    def _parse(self, html: str) -> list[dict]:
        """Extract results from DuckDuckGo HTML."""
        results = []

        # Find result blocks: <div class="result__body">
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        for url, title_html, snippet_html in blocks[:MAX_RESULTS]:
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
            # DuckDuckGo wraps URLs — extract real URL
            if "uddg=" in url:
                url_match = re.search(r"uddg=([^&]+)", url)
                if url_match:
                    from urllib.parse import unquote
                    url = unquote(url_match.group(1))
            if title:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })

        # Fallback: simpler extraction
        if not results:
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            urls = re.findall(r'class="result__url"[^>]*>(.*?)</span>', html, re.DOTALL)
            for title_html, url_html in zip(titles[:MAX_RESULTS], urls[:MAX_RESULTS]):
                title = re.sub(r"<[^>]+>", "", title_html).strip()
                url = re.sub(r"<[^>]+>", "", url_html).strip()
                if title:
                    results.append({"title": title, "url": url, "snippet": ""})

        return results


registry.register(WebSearchTool())
