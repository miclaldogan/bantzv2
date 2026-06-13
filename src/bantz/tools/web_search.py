"""Web tools backed by the bundled bantz-web pipeline (vendor/ submodule).

Wraps bantz-web's search / deep-research / news functions so bantzv2 calls
them in-process — no HTTP, no subprocess.

Import-isolation note: the vendor dir is *appended* to sys.path (not
inserted at 0). bantz-web ships flat modules with generic names — notably
``telegram.py`` — and prepending would shadow bantzv2's installed
``python-telegram-bot`` (`import telegram`). Appending lets installed
packages win while bantz-web's own modules (``searcher``, ``config`` …)
still resolve, since bantzv2 has no top-level modules by those names.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any

from bantz.core.event_bus import bus
from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.web")

# vendor/bantz-web lives at <repo>/vendor/bantz-web; this file is src/bantz/tools/
_VENDOR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vendor/bantz-web"))
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.append(_VENDOR)

# bantz-web auto-commits pipeline artifacts via git_ops; as a submodule that
# would create commits inside vendor/bantz-web. Neutralize it once.
_GIT_SILENCED = False


def _silence_bantz_web_git() -> None:
    global _GIT_SILENCED
    if _GIT_SILENCED:
        return
    try:
        import main as _bw_main  # noqa: PLC0415
        _bw_main.git_commit = lambda *a, **k: None  # type: ignore[attr-defined]
        _GIT_SILENCED = True
    except Exception as exc:  # pragma: no cover
        log.debug("could not silence bantz-web git_commit: %s", exc)


# bantz-web's CLI default is 50 sources (≈hours). Capped for interactive tool use.
DEFAULT_RESEARCH_RESULTS = 15


# ── wrapped functions ───────────────────────────────────────────────────────

def execute_web_search(query: str, max_results: int = 5) -> str:
    """One-shot web search via bantz-web's tiered searcher. Returns a report string."""
    from searcher import search  # noqa: PLC0415
    results = search(query, max_results)
    if not results:
        return f"No results found for: {query}"
    lines = [f'Search results for "{query}":', ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title') or 'Untitled'}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
        if r.get("content"):
            lines.append(f"   {r['content'][:200]}")
        lines.append("")
    return "\n".join(lines).strip()


def execute_web_news(topic: str = "general") -> str:
    """Latest news for a topic via bantz-web's news searcher. Returns a list string."""
    from news_searcher import search_news  # noqa: PLC0415
    articles = search_news(topic)
    if not articles:
        return f"No news found for: {topic}"
    lines = [f"Latest news — {topic}:", ""]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a.get('title') or 'Untitled'}")
        meta = " · ".join(x for x in (a.get("date", ""), a.get("url", "")) if x)
        if meta:
            lines.append(f"   {meta}")
        if a.get("content"):
            lines.append(f"   {a['content'][:200]}")
        lines.append("")
    return "\n".join(lines).strip()


def execute_web_research(topic: str, max_results: int = DEFAULT_RESEARCH_RESULTS) -> str:
    """Deep multi-source research via bantz-web. Blocking/heavy — call in a thread.

    run_research takes a session, not a topic, so we replicate the CLI's
    orchestration (new_session → session_path → save → run_research) and read
    the report back off the session.
    """
    import config as bw_config  # noqa: PLC0415
    import main as bw_main  # noqa: PLC0415
    import storage  # noqa: PLC0415

    _silence_bantz_web_git()
    session = storage.new_session(topic, "research")
    sess_path = storage.session_path(topic)
    storage.save(sess_path, session)
    bw_main.run_research(session, sess_path, max_results, bw_config.REPORT_DIR)
    return session.get("final_report") or "Research completed but produced no report."


# ── tools ───────────────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the internet for information — answers factual questions, looks up "
        "people, places, concepts, current events. Params: query (str). "
        "'search X', 'look it up', 'find info about', 'ara', 'google Z' → web_search. "
        "For a quick lookup, not an in-depth report. NOT for opening apps or clicking."
    )
    risk_level = "safe"

    async def execute(self, query: str = "", **kwargs: Any) -> ToolResult:
        query = (query or kwargs.get("text", "")).strip()
        if not query:
            return ToolResult(success=False, output="", error="No search query provided")
        try:
            max_results = int(kwargs.get("max_results", 5))
        except (TypeError, ValueError):
            max_results = 5
        try:
            output = await asyncio.to_thread(execute_web_search, query, max_results)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"web search failed: {exc}")
        return ToolResult(success=True, output=output, data={"query": query})


class WebResearchTool(BaseTool):
    name = "web_research"
    description = (
        "Deep, multi-source research producing a structured report. Slow (minutes). "
        "Params: topic (str). 'research X', 'deep dive into Y', 'araştır', 'investigate', "
        "'write me a report on Z' → web_research. Use web_search instead for quick lookups."
    )
    risk_level = "safe"

    async def execute(self, topic: str = "", query: str = "", **kwargs: Any) -> ToolResult:
        topic = (topic or query or kwargs.get("text", "")).strip()
        if not topic:
            return ToolResult(success=False, output="", error="No research topic provided")

        fut = asyncio.ensure_future(asyncio.to_thread(execute_web_research, topic))
        start = time.time()
        bus.emit_threadsafe("log", level="info", msg=f"[web_research] starting: {topic}")
        log.info("[web_research] starting: %s", topic)
        try:
            while True:
                try:
                    report = await asyncio.wait_for(asyncio.shield(fut), timeout=30)
                    break
                except asyncio.TimeoutError:
                    elapsed = int(time.time() - start)
                    bus.emit_threadsafe(
                        "log", level="info",
                        msg=f"[web_research] '{topic}' still running — {elapsed}s elapsed…")
                    log.info("[web_research] '%s' still running — %ds elapsed", topic, elapsed)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"web research failed: {exc}")
        return ToolResult(success=True, output=report, data={"topic": topic})


class WebNewsTool(BaseTool):
    name = "web_news"
    description = (
        "Fetch the latest news / current headlines for a topic. Params: topic (str, "
        "optional — defaults to general). 'news about X', 'haberler', 'gündem', "
        "'what happened with Y', 'latest on Z' → web_news."
    )
    risk_level = "safe"

    async def execute(self, topic: str = "", query: str = "", **kwargs: Any) -> ToolResult:
        topic = (topic or query or kwargs.get("text", "") or "general").strip() or "general"
        try:
            output = await asyncio.to_thread(execute_web_news, topic)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"web news failed: {exc}")
        return ToolResult(success=True, output=output, data={"topic": topic})


registry.register(WebSearchTool())
registry.register(WebResearchTool())
registry.register(WebNewsTool())
