"""
Bantz v2 — News Tool
Fetches headlines from Hacker News API and Google News RSS.
LLM summarizes results into a natural paragraph.
15-minute TTL in-memory cache.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from bantz.core.location import location_service
from bantz.tools import BaseTool, ToolResult, registry

TIMEOUT = 8.0
CACHE_TTL = 900  # 15 minutes


class _Cache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        if key in self._data:
            ts, val = self._data[key]
            if time.time() - ts < CACHE_TTL:
                return val
            del self._data[key]
        return None

    def set(self, key: str, val: Any) -> None:
        self._data[key] = (time.time(), val)


_cache = _Cache()

# News summarizer prompt — called from brain._finalize via tool output flag
NEWS_SUMMARY_PROMPT = """\
You are Bantz. Below are today's top headlines.
Write a 3-4 sentence natural summary: what are the main themes today?
Be conversational, no bullet points, no markdown.
Finish with one sentence about what stands out most.\
"""


class NewsTool(BaseTool):
    name = "news"
    description = (
        "Fetches and summarizes latest news headlines. "
        "Use for: news, headlines, what's happening, hacker news, tech news, breaking news."
    )
    risk_level = "safe"

    async def execute(
        self,
        source: str = "all",   # "hn" | "google" | "all"
        limit: int = 5,
        summarize: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        raw_items: list[str] = []
        sections: list[str] = []

        if source in ("hn", "all"):
            hn = await self._fetch_hn(limit)
            if hn:
                sections.append("🟠 Hacker News\n" + "\n".join(f"  • {h}" for h in hn))
                raw_items.extend(hn)

        if source in ("google", "all"):
            loc = await location_service.get()
            gl = "tr" if loc.is_turkey else "us"
            hl = "tr" if loc.is_turkey else "en"
            google = await self._fetch_google(limit, gl=gl, hl=hl)
            if google:
                sections.append("🔵 Google News\n" + "\n".join(f"  • {h}" for h in google))
                raw_items.extend(google)

        if not sections:
            return ToolResult(success=False, output="", error="Could not fetch news.")

        raw_output = "\n\n".join(sections)

        # Get LLM summary if requested
        summary = ""
        if summarize and raw_items:
            summary = await self._summarize(raw_items)

        final_output = summary if summary else raw_output

        return ToolResult(
            success=True,
            output=final_output,
            data={
                "source": source,
                "raw": raw_output,       # always available for debugging
                "summarized": bool(summary),
                "item_count": len(raw_items),
            },
        )

    async def _summarize(self, items: list[str]) -> str:
        """Ask Ollama to summarize the headlines into a natural paragraph."""
        try:
            from bantz.llm.ollama import ollama
            headlines = "\n".join(f"- {item}" for item in items)
            raw = await ollama.chat([
                {"role": "system", "content": NEWS_SUMMARY_PROMPT},
                {"role": "user", "content": f"Headlines:\n{headlines}"},
            ])
            # Strip markdown artifacts
            import re
            raw = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
            raw = re.sub(r"^\d+\.\s+", "", raw, flags=re.MULTILINE)
            return raw.strip()
        except Exception:
            return ""  # fallback: raw output shown instead

    # ── Hacker News ───────────────────────────────────────────────────────

    async def _fetch_hn(self, limit: int) -> list[str]:
        cache_key = f"hn_{limit}"
        if cached := _cache.get(cache_key):
            return cached
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                ids_resp = await client.get(
                    "https://hacker-news.firebaseio.com/v0/topstories.json"
                )
                ids_resp.raise_for_status()
                ids = ids_resp.json()[:limit]

                titles = []
                for story_id in ids:
                    item_resp = await client.get(
                        f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    )
                    item_resp.raise_for_status()
                    item = item_resp.json()
                    if title := item.get("title"):
                        score = item.get("score", 0)
                        titles.append(f"{title}  ({score} pts)")

            _cache.set(cache_key, titles)
            return titles
        except Exception:
            return []

    # ── Google News RSS ───────────────────────────────────────────────────

    async def _fetch_google(self, limit: int, gl: str = "us", hl: str = "en") -> list[str]:
        cache_key = f"google_{gl}_{limit}"
        if cached := _cache.get(cache_key):
            return cached
        url = f"https://news.google.com/rss?hl={hl}&gl={gl}&ceid={gl.upper()}:{hl}"
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()

            root = ET.fromstring(resp.text)
            items = root.findall(".//item")[:limit]
            titles = []
            for item in items:
                title_el = item.find("title")
                source_el = item.find("source")
                if title_el is not None and title_el.text:
                    source = f"  [{source_el.text}]" if source_el is not None else ""
                    titles.append(f"{title_el.text.strip()}{source}")

            _cache.set(cache_key, titles)
            return titles
        except Exception:
            return []


registry.register(NewsTool())