"""
Bantz v3 — Deep Web Reader Tool (#182 / Plan A)

"The Butler Reads the Full Article"

Fetches a webpage by URL, strips HTML/scripts/styles, and returns
clean text content.  This complements web_search (which only returns
2-3 line snippets) by letting the planner read the *full* page before
summarizing or citing.

Usage (via registry):
    tool = registry.get("read_url")
    result = await tool.execute(url="https://example.com/article")
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.web_reader")

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TEXT_LENGTH = 10_000   # chars — keep context window manageable
FETCH_TIMEOUT = 15.0       # seconds
MAX_HTML_SIZE = 2_000_000  # ~2 MB — refuse huge pages

_INVISIBLE_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})


# ── HTML → plain-text stripper (stdlib only) ─────────────────────────────────

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter that drops invisible elements."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth: int = 0   # inside an invisible tag

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INVISIBLE_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _INVISIBLE_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._pieces)
        # Collapse whitespace
        return re.sub(r"\s+", " ", raw).strip()


def strip_html(html: str) -> str:
    """Convert HTML to clean plain text."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ── Tool class ─────────────────────────────────────────────────────────────────

class WebReaderTool(BaseTool):
    name = "read_url"
    description = (
        "Fetch and read the full text content of a specific URL / webpage. "
        "Use when you need the complete article text, not just a search snippet. "
        "Returns clean plain text (HTML stripped)."
    )
    risk_level = "safe"

    async def execute(self, url: str = "", **kwargs: Any) -> ToolResult:
        """Fetch *url*, strip HTML, return truncated plain text."""
        if not url:
            return ToolResult(
                success=False, output="",
                error="No URL provided. Pass a 'url' parameter.",
            )

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False, output="",
                error=f"Invalid URL (must start with http:// or https://): {url}",
            )

        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Bantz/3.0 (Web Reader)"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

                # Guard against absurdly large pages
                raw_html = resp.text
                if len(raw_html) > MAX_HTML_SIZE:
                    raw_html = raw_html[:MAX_HTML_SIZE]
                    log.warning("Truncated HTML from %s (exceeded %d bytes)",
                                url, MAX_HTML_SIZE)

        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False, output="",
                error=f"HTTP {exc.response.status_code} fetching {url}",
            )
        except Exception as exc:
            return ToolResult(
                success=False, output="",
                error=f"Failed to fetch URL: {exc}",
            )

        # Strip HTML → plain text
        text = strip_html(raw_html)

        if not text:
            return ToolResult(
                success=True,
                output=f"(Page at {url} returned no readable text content.)",
                data={"url": url, "length": 0},
            )

        # Truncate to keep context window happy
        truncated = len(text) > MAX_TEXT_LENGTH
        text = text[:MAX_TEXT_LENGTH]

        # Append the source URL at the bottom — #182 citation
        footer = f"\n\nTelegraph Reference: {url}"
        output = text + footer

        log.info("read_url: fetched %d chars from %s%s",
                 len(text), url, " (truncated)" if truncated else "")

        return ToolResult(
            success=True,
            output=output,
            data={
                "url": url,
                "length": len(text),
                "truncated": truncated,
            },
        )


# ── Register ─────────────────────────────────────────────────────────────────

registry.register(WebReaderTool())
