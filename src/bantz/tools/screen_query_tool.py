"""
Bantz v2 — screen_query tool

Thin registry wrapper around :class:`bantz.screen_query.ScreenQueryHandler`.
One screenshot + one vision call: describe the screen, read text from it,
or click an element the user described ("şu ikona tıkla" = click that icon).
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tool.screen_query")

_handler = None  # lazy singleton — ScreenControl probes backends on init


def _get_handler():
    global _handler
    if _handler is None:
        from bantz.screen_query import ScreenQueryHandler
        _handler = ScreenQueryHandler()
    return _handler


class ScreenQueryTool(BaseTool):
    name = "screen_query"
    description = (
        "Look at the current screen and answer about it or click on it. "
        "query=<the user's request>. Modes auto-detected: describe "
        "('what do you see on screen', Turkish 'ekranda ne var' = what's on "
        "the screen), read ('what does this say', Turkish 'ne yazıyor' = "
        "what does it say), click ('click the blue button', Turkish "
        "'şu ikona tıkla' = click that icon)."
    )
    risk_level = "moderate"  # may click

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = str(
            kwargs.get("query") or kwargs.get("utterance") or kwargs.get("text") or ""
        ).strip()
        if not query:
            return ToolResult(False, "", error="screen_query needs a query")
        try:
            ok, text = await _get_handler().handle(query)
        except Exception as exc:
            log.error("screen_query failed: %s", exc)
            return ToolResult(False, "", error=f"screen_query error: {exc}")
        return ToolResult(ok, text, error="" if ok else text)


registry.register(ScreenQueryTool())
