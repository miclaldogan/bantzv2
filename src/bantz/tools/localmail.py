"""
LocalMailTool (#552) — read/search mail from the local notmuch index.

Mail arrives via mbsync (systemd user timer, see deploy/README-mail.md)
into a local maildir; notmuch indexes it. This tool answers count /
recent / search / read / unread_summary entirely from local files — it
works offline and never touches the Gmail API. Sending and label
management stay with the gmail tool.

Backend: the ``notmuch`` CLI with ``--format=json`` (present wherever
notmuch is installed; the Python bindings are distro-packaged and not
always importable, so the CLI is the portable path).
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.localmail")

_DEFAULT_LIMIT = 10


def _run_notmuch(args: list[str], timeout: float = 10.0) -> tuple[bool, str]:
    """Run a notmuch CLI command. Returns (ok, stdout)."""
    binary = shutil.which("notmuch")
    if not binary:
        return False, "notmuch is not installed (pacman -S notmuch)"
    try:
        r = subprocess.run(
            [binary, *args], capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "notmuch timed out"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip()[:300]
    return True, r.stdout


def _build_query(kwargs: dict[str, Any]) -> str:
    """Map tool args to a notmuch query string."""
    raw = str(kwargs.get("query") or kwargs.get("full_text") or "").strip()
    parts: list[str] = []
    if raw:
        parts.append(raw)
    if kwargs.get("from_sender"):
        parts.append(f'from:{kwargs["from_sender"]}')
    if kwargs.get("subject_filter"):
        parts.append(f'subject:"{kwargs["subject_filter"]}"')
    if kwargs.get("unread", False):
        parts.append("tag:unread")
    try:
        days = int(kwargs.get("days_ago") or 0)
    except (TypeError, ValueError):
        days = 0
    if days > 0:
        parts.append(f"date:{days}d..")
    return " and ".join(parts) if parts else "tag:unread"


def _format_threads(threads: list[dict], header: str) -> str:
    if not threads:
        return f"{header}: nothing found."
    lines = [header, ""]
    for i, t in enumerate(threads, 1):
        authors = t.get("authors", "?")
        subject = t.get("subject", "(no subject)")
        date = t.get("date_relative", "")
        tags = ",".join(t.get("tags", []))
        lines.append(f"{i}. {subject}")
        lines.append(f"   {authors} · {date} · [{tags}] · id:{t.get('thread', '?')}")
    return "\n".join(lines)


class LocalMailTool(BaseTool):
    name = "localmail"
    description = (
        "Read/search email from the LOCAL mail index (offline, instant — "
        "preferred over gmail for reading when enabled). Params: action "
        "(count|recent|search|read|unread_summary), query (notmuch syntax "
        "or free text), from_sender, subject_filter, days_ago, limit, "
        "thread_id (for read). 'how many unread emails', 'any mail from X', "
        "'search my mail for Y' → localmail. Sending stays with gmail."
    )
    risk_level = "safe"

    async def execute(self, action: str = "unread_summary", **kwargs: Any) -> ToolResult:
        from bantz.config import config
        if not config.localmail_enabled:
            return ToolResult(
                success=False, output="",
                error="Local mail is disabled (BANTZ_LOCALMAIL_ENABLED=false).",
            )
        handler = {
            "count": self._count,
            "recent": self._recent,
            "search": self._search,
            "read": self._read,
            "unread_summary": self._unread_summary,
        }.get(action, self._unread_summary)
        return await handler(kwargs)

    # ── actions ───────────────────────────────────────────────────────────

    async def _count(self, kwargs: dict) -> ToolResult:
        ok, out = await asyncio.to_thread(
            _run_notmuch, ["count", _build_query({**kwargs, "unread": True})],
        )
        if not ok:
            return ToolResult(success=False, output="", error=out)
        n = out.strip() or "0"
        return ToolResult(
            success=True,
            output=f"You have {n} unread email{'s' if n != '1' else ''} (local index).",
            data={"count": int(n)},
        )

    async def _recent(self, kwargs: dict) -> ToolResult:
        limit = int(kwargs.get("limit") or _DEFAULT_LIMIT)
        ok, out = await asyncio.to_thread(
            _run_notmuch,
            ["search", "--format=json", f"--limit={limit}",
             "--sort=newest-first", _build_query(kwargs)],
        )
        if not ok:
            return ToolResult(success=False, output="", error=out)
        threads = json.loads(out or "[]")
        return ToolResult(
            success=True,
            output=_format_threads(threads, "Recent mail (local)"),
            data={"threads": threads},
        )

    async def _search(self, kwargs: dict) -> ToolResult:
        limit = int(kwargs.get("limit") or _DEFAULT_LIMIT)
        query = _build_query(kwargs)
        ok, out = await asyncio.to_thread(
            _run_notmuch,
            ["search", "--format=json", f"--limit={limit}",
             "--sort=newest-first", query],
        )
        if not ok:
            return ToolResult(success=False, output="", error=out)
        threads = json.loads(out or "[]")
        return ToolResult(
            success=True,
            output=_format_threads(threads, f'Mail matching "{query}"'),
            data={"threads": threads, "query": query},
        )

    async def _read(self, kwargs: dict) -> ToolResult:
        tid = str(kwargs.get("thread_id") or kwargs.get("message_id")
                  or kwargs.get("id") or "").strip()
        if not tid:
            return ToolResult(success=False, output="", error="No message/thread id given.")
        if not tid.startswith(("id:", "thread:")):
            tid = f"thread:{tid}"
        ok, out = await asyncio.to_thread(
            _run_notmuch, ["show", "--format=json", "--body=true", tid],
        )
        if not ok:
            return ToolResult(success=False, output="", error=out)
        try:
            text = _extract_bodies(json.loads(out))
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"parse error: {exc}")
        return ToolResult(success=True, output=text[:6000] or "(empty message)")

    async def _unread_summary(self, kwargs: dict) -> ToolResult:
        limit = int(kwargs.get("limit") or 15)
        ok, out = await asyncio.to_thread(
            _run_notmuch,
            ["search", "--format=json", f"--limit={limit}",
             "--sort=newest-first", "tag:unread"],
        )
        if not ok:
            return ToolResult(success=False, output="", error=out)
        threads = json.loads(out or "[]")
        if not threads:
            return ToolResult(success=True, output="Inbox is clean (local index).",
                              data={"count": 0})
        # Group with the same categorizer the briefing uses.
        try:
            from bantz.tools.gmail import categorize, _BRIEFING_CATEGORIES
            keep, noise = [], 0
            for t in threads:
                cat = categorize(t.get("authors", ""), t.get("subject", ""))
                if cat in _BRIEFING_CATEGORIES:
                    keep.append(t)
                else:
                    noise += 1
            lines = [_format_threads(keep, f"Unread that matters ({len(keep)})")]
            if noise:
                lines.append(f"\n(+{noise} notification/service emails skipped)")
            return ToolResult(success=True, output="\n".join(lines),
                              data={"threads": keep, "skipped": noise})
        except Exception:
            return ToolResult(
                success=True,
                output=_format_threads(threads, "Unread mail (local)"),
                data={"threads": threads},
            )


def _extract_bodies(nodes: Any, out: list[str] | None = None) -> str:
    """Walk notmuch show's nested JSON and collect text/plain parts."""
    if out is None:
        out = []
    if isinstance(nodes, list):
        for n in nodes:
            _extract_bodies(n, out)
    elif isinstance(nodes, dict):
        headers = nodes.get("headers")
        if headers:
            out.append(
                f"From: {headers.get('From', '?')}\n"
                f"Subject: {headers.get('Subject', '?')}\n"
                f"Date: {headers.get('Date', '?')}\n"
            )
        for part in nodes.get("body", []) or []:
            _extract_bodies(part, out)
        if nodes.get("content-type") == "text/plain" and isinstance(nodes.get("content"), str):
            out.append(nodes["content"])
        elif isinstance(nodes.get("content"), list):
            _extract_bodies(nodes["content"], out)
    return "\n".join(out).strip()


registry.register(LocalMailTool())
