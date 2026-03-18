"""
Bantz — MemoryManager: Neo4j-backed conversation memory API (#293)

Public surface:
    MemoryManager.store(turn)               — extract entities and write to graph
    MemoryManager.query(text, top_k)        — full-text search, returns memories
    MemoryManager.summarize_context(topic)  — narrative summary for a topic node

Usage:
    from bantz.memory.memory_manager import memory_manager

    await memory_manager.store({"user": "msg", "assistant": "reply"})
    memories = await memory_manager.query("what did we decide about auth?")
    ctx = await memory_manager.summarize_context("authentication")
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("bantz.memory_manager")

# Full-text index name created by the migration
_FT_INDEX = "bantz_fulltext"


class MemoryManager:
    """High-level API for Neo4j conversation memory.

    Wraps ``GraphMemory`` (graph.py) and exposes the three methods required
    by the #293 acceptance criteria.

    The class is intentionally thin — heavy lifting is delegated to
    GraphMemory and context_builder so existing behaviour is unchanged.
    """

    # ── store ─────────────────────────────────────────────────────────────

    async def store(self, turn: dict[str, Any]) -> None:
        """Extract entities from a conversation turn and write them to Neo4j.

        Args:
            turn: dict with keys:
                  - ``user``      (str)  — user message
                  - ``assistant`` (str)  — assistant reply
                  - ``tool``      (str | None)  — tool name if used
                  - ``tool_data`` (dict | None) — tool result data
        """
        from bantz.memory.graph import graph_memory

        if not graph_memory.enabled:
            return

        user_msg = turn.get("user", "") or ""
        assistant_msg = turn.get("assistant", "") or ""
        tool_used = turn.get("tool")
        tool_data = turn.get("tool_data")

        try:
            await graph_memory.extract_and_store(
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                tool_used=tool_used,
                tool_result_data=tool_data,
            )
        except Exception as exc:
            log.debug("MemoryManager.store error: %s", exc)

    # ── query ─────────────────────────────────────────────────────────────

    async def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Full-text search across graph nodes.

        Uses Neo4j's procedure-based full-text index when available;
        falls back to regex-based keyword search.

        Returns:
            List of dicts: {"label": str, "value": str, "score": float}
        """
        from bantz.memory.graph import graph_memory

        if not graph_memory.enabled:
            return []

        # Try full-text index first
        try:
            rows = await graph_memory._query(
                f"CALL db.index.fulltext.queryNodes('{_FT_INDEX}', $q) "
                "YIELD node, score "
                "RETURN labels(node)[0] AS label, "
                "coalesce(node.name, node.title, node.description, "
                "         node.what, node.text, node.path) AS value, "
                "score "
                f"LIMIT {top_k}",
                q=text,
            )
            if rows:
                return rows
        except Exception:
            pass  # index not yet created — fall through to keyword search

        # Fallback: keyword regex search
        from bantz.memory.context_builder import extract_keywords, keyword_search

        keywords = extract_keywords(text)
        if not keywords:
            return []

        try:
            hits = await keyword_search(keywords, graph_memory._query)
            results = []
            for h in hits[:top_k]:
                # h is "[Label] value"
                if h.startswith("["):
                    label_end = h.index("]")
                    label = h[1:label_end]
                    value = h[label_end + 2:]
                else:
                    label, value = "Node", h
                results.append({"label": label, "value": value, "score": 1.0})
            return results
        except Exception as exc:
            log.debug("MemoryManager.query fallback error: %s", exc)
            return []

    # ── summarize_context ─────────────────────────────────────────────────

    async def summarize_context(self, topic: str) -> str:
        """Return a narrative summary of everything the graph knows about a topic.

        Queries decisions, tasks, events, people, and facts related to the
        topic node, then formats them into a human-readable paragraph.

        Args:
            topic: Topic name to summarise (matched case-insensitively).

        Returns:
            A multi-sentence summary string, or empty string if nothing found.
        """
        from bantz.memory.graph import graph_memory

        if not graph_memory.enabled:
            return ""

        try:
            pattern = f"(?i).*{topic}.*"

            decisions = await graph_memory._query(
                "MATCH (d:Decision) WHERE d.what =~ $pat OR d.context =~ $pat "
                "RETURN d.what AS text ORDER BY d.date DESC LIMIT 3",
                pat=pattern,
            )
            tasks = await graph_memory._query(
                "MATCH (t:Task) WHERE t.description =~ $pat "
                "RETURN t.description AS text, t.status AS status LIMIT 3",
                pat=pattern,
            )
            events = await graph_memory._query(
                "MATCH (e:Event) WHERE e.title =~ $pat "
                "RETURN e.title AS text, e.date AS date LIMIT 3",
                pat=pattern,
            )
            people = await graph_memory._query(
                "MATCH (p:Person)-[*1..2]-(n) WHERE any(k IN keys(n) "
                "WHERE toString(n[k]) =~ $pat) "
                "RETURN DISTINCT p.name AS text LIMIT 5",
                pat=pattern,
            )
            facts = await graph_memory._query(
                "MATCH (f:Fact) WHERE f.text =~ $pat "
                "RETURN f.text AS text LIMIT 3",
                pat=pattern,
            )

            parts: list[str] = []

            if decisions:
                d_texts = "; ".join(r["text"] for r in decisions)
                parts.append(f"Decisions about {topic!r}: {d_texts}.")

            if tasks:
                t_texts = ", ".join(
                    f"{r['text']} ({r.get('status', '?')})" for r in tasks
                )
                parts.append(f"Related tasks: {t_texts}.")

            if events:
                e_texts = ", ".join(
                    f"{r['text']}" + (f" on {r['date']}" if r.get("date") else "")
                    for r in events
                )
                parts.append(f"Related events: {e_texts}.")

            if people:
                names = ", ".join(r["text"] for r in people)
                parts.append(f"People involved: {names}.")

            if facts:
                f_texts = " ".join(r["text"] for r in facts)
                parts.append(f"Facts: {f_texts}")

            if not parts:
                return ""

            return f"Context for {topic!r}: " + " ".join(parts)

        except Exception as exc:
            log.debug("MemoryManager.summarize_context error: %s", exc)
            return ""


# ── Module singleton ──────────────────────────────────────────────────────────

memory_manager = MemoryManager()
