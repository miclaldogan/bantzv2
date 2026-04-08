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
import random
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

# Modern browser User-Agents for rotation — avoids naive bot detection (#257)
_BROWSER_UAS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
)

_MAX_RETRIES = 2
_MIN_READABLE_LENGTH = 20  # chars — anything shorter is likely a blocked/empty page

# Shared client to leverage HTTP connection pooling, reducing TCP/TLS overhead
# and speeding up repeated tool executions.
_shared_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient()
    return _shared_client


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
        "Read the full text content of a specific URL. "
        "Params: url (str) = the webpage URL to read. "
        "Returns clean plain text with HTML stripped. "
        "Use when you have a specific URL and need its full content. "
        "For searching the web without a URL, use web_search instead. "
        "NEVER fabricate or guess URLs."
    )
    risk_level = "safe"

    async def execute(self, url: str = "", **kwargs: Any) -> ToolResult:
        """Fetch *url*, strip HTML, return truncated plain text.

        Uses UA rotation and a single retry on 401/403 to bypass
        naive bot-detection walls (#257).
        """
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

        # ── Fetch with UA rotation + retry on 401/403 (#257) ──────────────
        used_uas: list[str] = []
        resp: httpx.Response | None = None
        last_status: int | None = None

        for attempt in range(_MAX_RETRIES):
            # Pick a UA we haven't tried yet
            remaining = [ua for ua in _BROWSER_UAS if ua not in used_uas]
            ua = random.choice(remaining) if remaining else random.choice(_BROWSER_UAS)
            used_uas.append(ua)

            try:
                client = _get_client()
                resp = await client.get(
                    url,
                    timeout=FETCH_TIMEOUT,
                    follow_redirects=True,
                    headers={"User-Agent": ua},
                )
                last_status = resp.status_code

                if last_status in (401, 403) and attempt < _MAX_RETRIES - 1:
                    log.warning(
                        "read_url: HTTP %d from %s (attempt %d), retrying with different UA",
                        last_status, url, attempt + 1,
                    )
                    continue  # retry with different UA

                # Any other status — break out (success or final failure)
                break

            except Exception as exc:
                # Network-level failure — no point retrying with a different UA
                return ToolResult(
                    success=False, output="",
                    error=f"Failed to fetch URL: {exc}",
                )

        # ── Handle final HTTP errors ──────────────────────────────────────
        assert resp is not None  # loop always runs at least once
        if resp.status_code >= 400:
            return ToolResult(
                success=False,
                output=(
                    f"Error: Could not read URL. HTTP Status: {resp.status_code}. "
                    "The site might be blocking automated access."
                ),
                error=f"HTTP {resp.status_code} fetching {url}",
            )

        # Guard against absurdly large pages
        raw_html = resp.text
        if len(raw_html) > MAX_HTML_SIZE:
            raw_html = raw_html[:MAX_HTML_SIZE]
            log.warning("Truncated HTML from %s (exceeded %d bytes)",
                        url, MAX_HTML_SIZE)

        # ── Strip HTML → plain text ───────────────────────────────────────
        text = strip_html(raw_html)

        # Empty / dangerously short content — likely JS challenge or captcha (#257)
        if len(text) < _MIN_READABLE_LENGTH:
            return ToolResult(
                success=False,
                output=(
                    "Error: The page returned empty content or requires "
                    "JavaScript/Captcha to view."
                ),
                data={"url": url, "length": len(text)},
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
