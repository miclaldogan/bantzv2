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

import logging
from datetime import datetime

from bantz.memory.nodes import extract_entities
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
            "CREATE INDEX IF NOT EXISTS FOR (p:Person)     ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic)      ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Decision)   ON (d.what)",
            "CREATE INDEX IF NOT EXISTS FOR (tk:Task)      ON (tk.description)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Event)      ON (e.title)",
            "CREATE INDEX IF NOT EXISTS FOR (l:Location)   ON (l.name)",
            "CREATE INDEX IF NOT EXISTS FOR (dc:Document)  ON (dc.path)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Reminder)   ON (r.title)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Commitment) ON (c.what)",
            "CREATE INDEX IF NOT EXISTS FOR (pr:Project)   ON (pr.name)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Fact)       ON (f.text)",
        ]
        async with self._driver.session() as s:
            for q in queries:
                try:
                    await s.run(q)
                except Exception:
                    pass  # index may already exist
            # Full-text index (migration 002)
            try:
                await s.run(
                    "CALL db.index.fulltext.createNodeIndex("
                    "  'bantz_fulltext',"
                    "  ['Person','Topic','Decision','Task','Event',"
                    "   'Location','Document','Reminder','Commitment','Project','Fact'],"
                    "  ['name','title','description','what','text','path','context']"
                    ")"
                )
            except Exception:
                pass  # already exists

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
        """MERGE entities into Neo4j — create if new, update if existing.
        Also create relationships between entities."""
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

                # Create relationships
                for rel in ent.get("rels", []):
                    await self._upsert_relationship(
                        session, label, key, key_val, rel
                    )

    async def _upsert_relationship(
        self,
        session,
        src_label: str,
        src_key: str,
        src_val: str,
        rel: dict,
    ) -> None:
        """MERGE a relationship between two nodes."""
        rel_type = rel["type"]
        tgt_label = rel["target_label"]
        tgt_key = rel["target_key"]
        tgt_val = rel["target_val"]

        query = (
            f"MATCH (a:{src_label} {{{src_key}: $src_val}})\n"
            f"MATCH (b:{tgt_label} {{{tgt_key}: $tgt_val}})\n"
            f"MERGE (a)-[r:{rel_type}]->(b)\n"
            f"ON CREATE SET r.created_at = $now\n"
            f"ON MATCH SET r.updated_at = $now"
        )
        try:
            await session.run(
                query,
                src_val=src_val,
                tgt_val=tgt_val,
                now=datetime.now().isoformat(),
            )
        except Exception as exc:
            log.debug("Rel %s failed: %s", rel_type, exc)

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

    # ── Growth tracking ────────────────────────────────────────────────────

    async def growth_since(self, since_iso: str) -> dict:
        """Count nodes and relationships created since a given ISO timestamp.

        Returns {"new_nodes": N, "new_rels": N, "by_label": {...}}.
        """
        if not self._enabled:
            return {"new_nodes": 0, "new_rels": 0, "by_label": {}}

        try:
            nodes = await self._query(
                "MATCH (n) WHERE n.created_at >= $since OR n.updated_at >= $since "
                "OR n.last_seen >= $since OR n.accessed_at >= $since "
                "RETURN labels(n)[0] AS label, count(n) AS cnt",
                since=since_iso,
            )
            rels = await self._query(
                "MATCH ()-[r]->() WHERE r.created_at >= $since "
                "RETURN count(r) AS cnt",
                since=since_iso,
            )
            by_label = {r["label"]: r["cnt"] for r in nodes}
            return {
                "new_nodes": sum(r["cnt"] for r in nodes),
                "new_rels": rels[0]["cnt"] if rels else 0,
                "by_label": by_label,
            }
        except Exception as exc:
            log.debug("Growth query failed: %s", exc)
            return {"new_nodes": 0, "new_rels": 0, "by_label": {}}

    # ── Delete operations ──────────────────────────────────────────────────

    async def delete_node(
        self,
        label: str,
        key: str,
        value: str,
    ) -> bool:
        """Delete a specific node and its relationships.

        Returns True if a node was deleted.
        """
        if not self._enabled:
            return False

        try:
            result = await self._query(
                f"MATCH (n:{label} {{{key}: $val}}) "
                f"DETACH DELETE n RETURN count(n) AS deleted",
                val=value,
            )
            return result[0]["deleted"] > 0 if result else False
        except Exception as exc:
            log.debug("Delete %s failed: %s", label, exc)
            return False

    async def delete_by_label(
        self,
        label: str,
        limit: int = 0,
    ) -> int:
        """Delete all nodes of a given label (with optional limit).

        Returns count of deleted nodes.
        """
        if not self._enabled:
            return 0

        try:
            if limit > 0:
                result = await self._query(
                    f"MATCH (n:{label}) WITH n LIMIT $limit "
                    f"DETACH DELETE n RETURN count(n) AS deleted",
                    limit=limit,
                )
            else:
                result = await self._query(
                    f"MATCH (n:{label}) DETACH DELETE n "
                    f"RETURN count(n) AS deleted"
                )
            return result[0]["deleted"] if result else 0
        except Exception as exc:
            log.debug("Delete all %s failed: %s", label, exc)
            return 0

    async def delete_relationship(
        self,
        src_label: str,
        src_key: str,
        src_val: str,
        rel_type: str,
        tgt_label: str,
        tgt_key: str,
        tgt_val: str,
    ) -> bool:
        """Delete a specific relationship between two nodes."""
        if not self._enabled:
            return False

        try:
            result = await self._query(
                f"MATCH (a:{src_label} {{{src_key}: $src_val}})"
                f"-[r:{rel_type}]->"
                f"(b:{tgt_label} {{{tgt_key}: $tgt_val}}) "
                f"DELETE r RETURN count(r) AS deleted",
                src_val=src_val,
                tgt_val=tgt_val,
            )
            return result[0]["deleted"] > 0 if result else False
        except Exception as exc:
            log.debug("Delete rel %s failed: %s", rel_type, exc)
            return False


# Singleton
graph_memory = GraphMemory()
