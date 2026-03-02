"""
Bantz v2 — Neo4j Graph Memory

Extracts entities (Person, Topic, Decision, Task, Event, Location) from
conversations and stores them as a knowledge graph.  The graph gives Bantz
long-term contextual memory that goes beyond raw chat history.

Usage:
    from bantz.memory.graph import graph_memory

    # On app startup
    await graph_memory.init()

    # After each assistant reply
    await graph_memory.extract_and_store(user_msg, assistant_msg, tool_used)

    # Before generating a response — inject relevant context
    ctx = await graph_memory.context_for(user_msg)
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from bantz.memory.nodes import NODE_LABELS, REL_TYPES, extract_entities
from bantz.memory.context_builder import build_context

log = logging.getLogger("bantz.graph")


class GraphMemory:
    """Neo4j-backed knowledge graph for Bantz."""

    def __init__(self) -> None:
        self._driver = None
        self._enabled = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def init(self) -> bool:
        """Connect to Neo4j.  Returns True if connection succeeded."""
        from bantz.config import config

        if not config.neo4j_enabled:
            log.info("Neo4j disabled in config")
            return False

        try:
            import neo4j
        except ImportError:
            log.warning("neo4j driver not installed — graph memory off")
            return False

        try:
            self._driver = neo4j.AsyncGraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            self._enabled = True
            log.info("Neo4j connected: %s", config.neo4j_uri)
            await self._ensure_indexes()
            return True
        except Exception as exc:
            log.warning("Neo4j connection failed: %s — graph memory off", exc)
            self._driver = None
            self._enabled = False
            return False

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _ensure_indexes(self) -> None:
        """Create indexes for fast lookups."""
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Person)   ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic)    ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Decision) ON (d.what)",
            "CREATE INDEX IF NOT EXISTS FOR (tk:Task)    ON (tk.description)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event)    ON (e.title)",
            "CREATE INDEX IF NOT EXISTS FOR (l:Location) ON (l.name)",
            "CREATE INDEX IF NOT EXISTS FOR (dc:Document) ON (dc.path)",
        ]
        async with self._driver.session() as s:
            for q in queries:
                try:
                    await s.run(q)
                except Exception:
                    pass  # index may already exist

    # ── Writing — extract entities from conversation ───────────────────────

    async def extract_and_store(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: str | None = None,
        tool_result_data: dict | None = None,
    ) -> None:
        """
        Parse a user↔assistant exchange and upsert entities into the graph.
        This runs after every exchange — keep it fast and non-blocking.
        """
        if not self._enabled:
            return

        now = datetime.now().isoformat(timespec="seconds")

        try:
            # Extract entities from the conversation text
            entities = self._extract_entities(user_msg, assistant_msg, tool_used, tool_result_data)

            if entities:
                await self._upsert_entities(entities, now)
        except Exception as exc:
            log.debug("Graph extract error: %s", exc)

    def _extract_entities(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: str | None,
        tool_data: dict | None,
    ) -> list[dict]:
        """Delegate to bantz.memory.nodes module."""
        return extract_entities(user_msg, assistant_msg, tool_used, tool_data)

    async def _upsert_entities(self, entities: list[dict], now: str) -> None:
        """MERGE entities into Neo4j — create if new, update if existing."""
        async with self._driver.session() as session:
            for ent in entities:
                label = ent["label"]
                key = ent["key"]
                props = ent["props"]
                key_val = props[key]

                # MERGE on the key property, SET remaining props
                set_clause = ", ".join(
                    f"n.{k} = ${k}" for k in props if k != key
                )
                query = (
                    f"MERGE (n:{label} {{{key}: ${key}}})\n"
                    f"ON CREATE SET {set_clause}\n"
                    f"ON MATCH SET {set_clause}"
                )
                try:
                    await session.run(query, **props)
                except Exception as exc:
                    log.debug("Upsert %s failed: %s", label, exc)

    # ── Reading — build context for LLM ────────────────────────────────────

    async def context_for(self, user_msg: str) -> str:
        """Delegate to bantz.memory.context_builder module."""
        if not self._enabled:
            return ""
        return await build_context(user_msg, self._query)

    async def _query(self, cypher: str, **params) -> list[dict]:
        """Run a read query, return list of record dicts."""
        async with self._driver.session() as s:
            result = await s.run(cypher, **params)
            records = await result.data()
            return records

    # ── Stats ──────────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Quick stats for --doctor."""
        if not self._enabled:
            return {"enabled": False}

        try:
            counts = await self._query(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt "
                "ORDER BY cnt DESC"
            )
            rels = await self._query(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt "
                "ORDER BY cnt DESC"
            )
            total_nodes = sum(c["cnt"] for c in counts)
            total_rels = sum(r["cnt"] for r in rels)
            return {
                "enabled": True,
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "nodes_by_label": {c["label"]: c["cnt"] for c in counts},
                "rels_by_type": {r["rel"]: r["cnt"] for r in rels},
            }
        except Exception as exc:
            return {"enabled": True, "error": str(exc)}

    async def status_line(self) -> str:
        """One-line summary for --doctor."""
        s = await self.stats()
        if not s.get("enabled"):
            return "Graph memory: off"
        if "error" in s:
            return f"Graph memory: error — {s['error']}"
        return (
            f"Graph memory: {s['total_nodes']} nodes, "
            f"{s['total_relationships']} rels"
        )


# Singleton
graph_memory = GraphMemory()
