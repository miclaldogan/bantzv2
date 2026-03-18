"""
Bantz v3 — Graph Context Builder

Queries the Neo4j graph and builds an LLM-ready context string from
relevant entities (people, tasks, decisions, events, keyword matches).

Usage:
    from bantz.memory.context_builder import build_context
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Awaitable

log = logging.getLogger("bantz.context_builder")

# Token budget for build_context() output (#219)
MAX_CONTEXT_TOKENS: int = 2000
_MAX_CONTEXT_CHARS: int = MAX_CONTEXT_TOKENS * 4  # ~1 token ≈ 4 chars

# Stop-words for keyword extraction
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "do", "does",
    "did", "will", "can", "could", "would", "should", "my", "me",
    "i", "you", "we", "it", "he", "she", "they", "this", "that",
    "what", "how", "when", "where", "who", "which", "to", "in",
    "on", "at", "for", "of", "and", "or", "but", "not", "with",
    "from", "by", "about", "any", "all", "some", "up", "out",
    "just", "now", "today", "tomorrow", "yesterday", "please",
    "check", "show", "tell", "get", "want", "need", "have", "has",
    "hey", "hi", "hello", "thanks", "ok", "okay", "yeah", "yes", "no",
})


def extract_keywords(text: str) -> list[str]:
    """Pull significant words from user input for graph search."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in _STOP_WORDS][:5]


async def keyword_search(
    keywords: list[str],
    query_fn: Callable[..., Awaitable[list[dict]]],
) -> list[str]:
    """Search graph nodes for keyword matches."""
    results: list[str] = []
    for kw in keywords:
        pattern = f"(?i).*{re.escape(kw)}.*"
        rows = await query_fn(
            "MATCH (n) WHERE any(k IN keys(n) WHERE "
            "toString(n[k]) =~ $pattern) "
            "RETURN labels(n)[0] AS label, "
            "coalesce(n.name, n.title, n.description, n.what, n.path) AS val "
            "LIMIT 3",
            pattern=pattern,
        )
        for r in rows:
            entry = f"[{r['label']}] {r['val']}"
            if entry not in results:
                results.append(entry)
    return results[:8]


async def build_context(
    user_msg: str,
    query_fn: Callable[..., Awaitable[list[dict]]],
) -> str:
    """
    Query the graph for entities relevant to the user's message
    and return a compact multi-line string the LLM can use as context.

    Parameters:
        user_msg:  The user's input text
        query_fn:  Async function(cypher, **params) -> list[dict]
    """
    try:
        parts: list[str] = []

        # 1. Recent people
        people = await query_fn(
            "MATCH (p:Person) RETURN p.name AS name, p.last_seen AS seen "
            "ORDER BY p.last_seen DESC LIMIT 8"
        )
        if people:
            names = [r["name"] for r in people]
            parts.append(f"Known people: {', '.join(names)}")

        # 2. Open tasks
        tasks = await query_fn(
            "MATCH (t:Task {status: 'open'}) "
            "RETURN t.description AS desc, t.priority AS prio "
            "ORDER BY t.created_at DESC LIMIT 5"
        )
        if tasks:
            lines = [f"  - [{t['prio']}] {t['desc']}" for t in tasks]
            parts.append("Open tasks:\n" + "\n".join(lines))

        # 3. Recent decisions
        decisions = await query_fn(
            "MATCH (d:Decision) RETURN d.what AS what, d.date AS date "
            "ORDER BY d.date DESC LIMIT 3"
        )
        if decisions:
            lines = [f"  - {d['what']}" for d in decisions]
            parts.append("Recent decisions:\n" + "\n".join(lines))

        # 4. Recent events
        events = await query_fn(
            "MATCH (e:Event) RETURN e.title AS title, e.date AS date "
            "ORDER BY e.updated_at DESC LIMIT 5"
        )
        if events:
            lines = [f"  - {e['title']} ({e['date'] or '?'})" for e in events]
            parts.append("Recent events:\n" + "\n".join(lines))

        # 5. Active commitments
        commitments = await query_fn(
            "MATCH (c:Commitment {status: 'active'}) "
            "RETURN c.what AS what, c.date AS date "
            "ORDER BY c.date DESC LIMIT 5"
        )
        if commitments:
            lines = [f"  - {c['what']}" for c in commitments]
            parts.append("Active commitments:\n" + "\n".join(lines))

        # 6. Active reminders
        reminders = await query_fn(
            "MATCH (r:Reminder {status: 'active'}) "
            "RETURN r.title AS title, r.trigger_type AS trigger, "
            "r.fire_at AS fire_at, r.trigger_place AS place "
            "ORDER BY r.created_at DESC LIMIT 5"
        )
        if reminders:
            lines = []
            for r in reminders:
                if r.get("place"):
                    lines.append(f"  - {r['title']} (at {r['place']})")
                elif r.get("fire_at"):
                    lines.append(f"  - {r['title']} ({r['fire_at']})")
                else:
                    lines.append(f"  - {r['title']}")
            parts.append("Active reminders:\n" + "\n".join(lines))

        # 7. Keyword-matching
        keywords = extract_keywords(user_msg)
        if keywords:
            relevant = await keyword_search(keywords, query_fn)
            if relevant:
                parts.append(
                    "Related context:\n" + "\n".join(f"  - {r}" for r in relevant)
                )

        if not parts:
            return ""

        result = "=== Graph Memory ===\n" + "\n".join(parts) + "\n=== End Graph ==="

        # Enforce token budget (#219)
        if len(result) > _MAX_CONTEXT_CHARS:
            result = result[:_MAX_CONTEXT_CHARS] + "\n…"

        return result

    except Exception as exc:
        log.debug("Graph context error: %s", exc)
        return ""
