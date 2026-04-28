"""
Bantz v2 — FeedTool: RSS/Atom feed parser (#289)

Fetches and parses RSS/Atom feeds using ``curl`` + ``defusedxml.ElementTree``
— replacing any news API dependency with direct feed consumption.

Supports:
  - RSS 2.0 (``<item>`` nodes)
  - Atom 1.0 (``<entry>`` nodes)
  - Image extraction from ``<media:content>``, ``<enclosure>``, ``<media:thumbnail>``
  - Feed URL registry in ``config/feeds.yaml``

Usage:
    from bantz.tools.feed_tool import feed_tool
    result = await feed_tool.execute(action="fetch", category="tech")
"""
from __future__ import annotations

import logging
import subprocess
import defusedxml.ElementTree as ET
from defusedxml.common import DefusedXmlException
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import yaml

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tool.feed")

# ── Feed registry paths ──────────────────────────────────────────────────────

# Look for feeds.yaml in these locations (first match wins):
#   1. <project_root>/config/feeds.yaml  (dev / repo checkout)
#   2. ~/.config/bantz/feeds.yaml        (user override)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/bantz/tools → project root
_REGISTRY_PATHS = [
    _PROJECT_ROOT / "config" / "feeds.yaml",
    Path.home() / ".config" / "bantz" / "feeds.yaml",
]

# ── Default feeds (used when no feeds.yaml exists) ────────────────────────────

_DEFAULT_FEEDS: dict[str, list[dict[str, str]]] = {
    "tech": [
        {"name": "Hacker News", "url": "https://hnrss.org/frontpage"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
    ],
    "world": [
        {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
        {"name": "Reuters Top News", "url": "https://www.reutersagency.com/feed/"},
    ],
    "tr_news": [
        {"name": "NTV", "url": "https://www.ntv.com.tr/son-dakika.rss"},
        {"name": "Habertürk", "url": "https://www.haberturk.com/rss"},
    ],
}


# ── Data classes ──────────────────────────────────────────────────────────────

class FeedToolError(Exception):
    """Raised when feed fetching or parsing fails."""


@dataclass
class FeedItem:
    """A single item from an RSS/Atom feed."""
    title: str
    link: str
    summary: str
    image_url: str | None = None
    published_at: datetime | None = None
    source_name: str = ""


# ── Feed registry ─────────────────────────────────────────────────────────────

def _load_feed_registry() -> dict[str, list[dict[str, str]]]:
    """Load feed registry from YAML, falling back to defaults."""
    for path in _REGISTRY_PATHS:
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "feeds" in data:
                    log.debug("Loaded feed registry from %s", path)
                    return data["feeds"]
            except Exception as exc:
                log.warning("Failed to load %s: %s", path, exc)

    log.debug("No feeds.yaml found, using defaults")
    return _DEFAULT_FEEDS


# ── XML parsing ───────────────────────────────────────────────────────────────

# Namespace map for media:content / media:thumbnail
_NS = {
    "media": "http://search.yahoo.com/mrss/",
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse RFC 2822 (RSS) or ISO 8601 (Atom) dates.

    Always returns a naive (UTC) datetime to avoid sorting issues
    between timezone-aware and naive datetimes.
    """
    if not date_str:
        return None
    date_str = date_str.strip()

    def _to_naive(dt: datetime) -> datetime:
        """Convert to naive UTC datetime."""
        if dt.tzinfo is not None:
            from datetime import timezone
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    # RFC 2822 (RSS pubDate): "Mon, 01 Jan 2024 12:00:00 GMT"
    try:
        return _to_naive(parsedate_to_datetime(date_str))
    except Exception:
        pass

    # ISO 8601 (Atom updated/published): "2024-01-01T12:00:00Z"
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return _to_naive(datetime.strptime(date_str, fmt))
        except ValueError:
            continue

    return None


def _extract_image(node: ET.Element, ns: dict[str, str]) -> str | None:
    """Extract image URL from media:content, media:thumbnail, or enclosure."""
    # media:content
    media = node.find("media:content", ns)
    if media is not None:
        url = media.get("url")
        if url:
            return url

    # media:thumbnail
    thumb = node.find("media:thumbnail", ns)
    if thumb is not None:
        url = thumb.get("url")
        if url:
            return url

    # enclosure (often used for podcast art / news images)
    enclosure = node.find("enclosure")
    if enclosure is not None:
        mime = enclosure.get("type", "")
        url = enclosure.get("url")
        if url and ("image" in mime or not mime):
            return url

    return None


def _strip_html(text: str) -> str:
    """Rough HTML tag stripping for feed descriptions."""
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#\d+;", "", text)
    return text.strip()


def _parse_rss(root: ET.Element, source_name: str = "") -> list[FeedItem]:
    """Parse RSS 2.0 feed (//channel/item nodes)."""
    items: list[FeedItem] = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        summary = item.findtext("description") or ""
        pub_date = item.findtext("pubDate") or item.findtext("dc:date", namespaces=_NS)
        image = _extract_image(item, _NS)

        items.append(FeedItem(
            title=title.strip(),
            link=link.strip(),
            summary=_strip_html(summary)[:300],
            image_url=image,
            published_at=_parse_date(pub_date),
            source_name=source_name,
        ))
    return items


def _parse_atom(root: ET.Element, source_name: str = "") -> list[FeedItem]:
    """Parse Atom 1.0 feed (//entry nodes)."""
    items: list[FeedItem] = []
    # Atom may have default namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    for entry in root.findall(f".//{ns}entry"):
        title = entry.findtext(f"{ns}title") or ""

        # Atom link is an attribute, not text
        link = ""
        link_elem = entry.find(f"{ns}link[@rel='alternate']")
        if link_elem is None:
            link_elem = entry.find(f"{ns}link")
        if link_elem is not None:
            link = link_elem.get("href", "")

        summary = entry.findtext(f"{ns}summary") or entry.findtext(f"{ns}content") or ""
        published = entry.findtext(f"{ns}published") or entry.findtext(f"{ns}updated") or ""
        image = _extract_image(entry, _NS)

        items.append(FeedItem(
            title=title.strip(),
            link=link.strip(),
            summary=_strip_html(summary)[:300],
            image_url=image,
            published_at=_parse_date(published),
            source_name=source_name,
        ))
    return items


def parse_feed(xml_text: str, source_name: str = "") -> list[FeedItem]:
    """Parse an XML feed string (RSS 2.0 or Atom 1.0).

    Raises ``FeedToolError`` if the XML cannot be parsed.
    """
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise FeedToolError(
            f"Failed to parse feed XML: {exc}. "
            "Expected valid RSS/Atom — got invalid data (captive portal or dead domain?)."
        ) from exc

    # Detect format: RSS has <channel>/<item>, Atom has <entry>
    if root.findall(".//item"):
        items = _parse_rss(root, source_name)
    elif root.tag.endswith("feed") or root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        items = _parse_atom(root, source_name)
    else:
        # Try RSS anyway (some feeds have unusual structure)
        items = _parse_rss(root, source_name)

    return sorted(
        items,
        key=lambda x: x.published_at or datetime.min,
        reverse=True,
    )


# ── Feed fetcher ──────────────────────────────────────────────────────────────

def _fetch_raw(url: str, timeout: int = 12) -> str:
    """Fetch raw XML from a feed URL using curl."""
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
            timeout=timeout + 3,
        )
        if result.returncode != 0:
            raise FeedToolError(f"curl failed for {url}: exit {result.returncode}")
        if not result.stdout.strip():
            raise FeedToolError(f"Empty response from {url}")
        return result.stdout
    except subprocess.TimeoutExpired:
        raise FeedToolError(f"Timeout fetching {url}")


# ── Tool class ────────────────────────────────────────────────────────────────

class FeedTool(BaseTool):
    """Fetch and parse RSS/Atom feeds.

    Actions:
      - ``fetch``: Fetch a specific feed URL
      - ``category``: Fetch all feeds in a category (from feeds.yaml)
      - ``list``: List available categories
    """

    name = "feed"
    description = (
        "Fetch and parse RSS/Atom feeds for news, blogs, and content. "
        "Params: action (fetch|category|list), "
        "url (str) = direct feed URL for action=fetch, "
        "category (str) = feed category from registry for action=category, "
        "limit (int) = max items to return (default 10). "
        "Use for: 'fetch RSS feed', 'show tech news from feeds', 'list feed categories'."
    )
    risk_level = "safe"

    def __init__(self) -> None:
        self._registry: dict[str, list[dict[str, str]]] | None = None

    @property
    def feed_registry(self) -> dict[str, list[dict[str, str]]]:
        if self._registry is None:
            self._registry = _load_feed_registry()
        return self._registry

    def reload_registry(self) -> None:
        """Force reload of feed registry from disk."""
        self._registry = None

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "fetch")
        limit = int(kwargs.get("limit", 10))

        if action == "list":
            return self._list_categories()
        elif action == "category":
            category = kwargs.get("category", "")
            return await self._fetch_category(category, limit)
        else:  # "fetch" or default
            url = kwargs.get("url", "")
            if not url:
                # If no URL, try to treat it as category
                category = kwargs.get("category", "")
                if category:
                    return await self._fetch_category(category, limit)
                return ToolResult(
                    success=False, output="",
                    error="Please provide a feed URL or category name.",
                )
            return await self._fetch_url(url, limit)

    def _list_categories(self) -> ToolResult:
        """List available feed categories."""
        reg = self.feed_registry
        if not reg:
            return ToolResult(success=True, output="No feed categories configured.")

        lines = ["📡 Available feed categories:"]
        for cat, feeds in reg.items():
            names = ", ".join(f["name"] for f in feeds)
            lines.append(f"  • {cat}: {names}")
        return ToolResult(success=True, output="\n".join(lines))

    async def _fetch_url(self, url: str, limit: int) -> ToolResult:
        """Fetch and parse a single feed URL."""
        try:
            raw = _fetch_raw(url)
            items = parse_feed(raw)
            return self._format_items(items[:limit])
        except FeedToolError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:
            log.warning("FeedTool error: %s", exc)
            return ToolResult(
                success=False, output="",
                error=f"Feed fetch failed: {exc}",
            )

    async def _fetch_category(self, category: str, limit: int) -> ToolResult:
        """Fetch all feeds in a category."""
        reg = self.feed_registry
        cat_key = category.strip().lower().replace(" ", "_")

        if cat_key not in reg:
            available = ", ".join(reg.keys())
            return ToolResult(
                success=False, output="",
                error=f"Unknown category '{category}'. Available: {available}",
            )

        all_items: list[FeedItem] = []
        errors: list[str] = []

        for feed_def in reg[cat_key]:
            name = feed_def.get("name", "")
            url = feed_def.get("url", "")
            if not url:
                continue
            try:
                raw = _fetch_raw(url)
                items = parse_feed(raw, source_name=name)
                all_items.extend(items)
            except FeedToolError as exc:
                errors.append(f"{name}: {exc}")
                log.debug("Feed error for %s: %s", name, exc)

        if not all_items and errors:
            return ToolResult(
                success=False, output="",
                error=f"All feeds failed in '{category}': {'; '.join(errors)}",
            )

        # Re-sort merged items by date
        all_items.sort(
            key=lambda x: x.published_at or datetime.min,
            reverse=True,
        )

        result = self._format_items(all_items[:limit])
        if errors:
            result.output += f"\n\n⚠ Some feeds failed: {'; '.join(errors)}"
        return result

    @staticmethod
    def _format_items(items: list[FeedItem]) -> ToolResult:
        """Format feed items into readable text output."""
        if not items:
            return ToolResult(success=True, output="No items found in feed.")

        lines: list[str] = []
        for i, item in enumerate(items, 1):
            source = f" [{item.source_name}]" if item.source_name else ""
            date = ""
            if item.published_at:
                date = f" ({item.published_at.strftime('%d %b %H:%M')})"

            line = f"{i}. {item.title}{source}{date}"
            if item.summary:
                # Truncate summary for readability
                summary = item.summary[:150]
                if len(item.summary) > 150:
                    summary += "…"
                line += f"\n   {summary}"
            if item.link:
                line += f"\n   {item.link}"
            lines.append(line)

        header = f"📰 {len(items)} feed items:"
        output = header + "\n\n" + "\n\n".join(lines)

        # Include image URLs in data for downstream consumers
        data: dict[str, Any] = {
            "items": [
                {
                    "title": it.title,
                    "link": it.link,
                    "image_url": it.image_url,
                    "published_at": it.published_at.isoformat() if it.published_at else None,
                    "source": it.source_name,
                }
                for it in items
            ],
        }

        return ToolResult(success=True, output=output, data=data)


# ── Auto-register ─────────────────────────────────────────────────────────────
feed_tool = FeedTool()
registry.register(feed_tool)
