"""
Bantz v3 — Neo4j Graph Memory

Connects to Neo4j via bolt:// protocol.
Falls back gracefully if Neo4j is not available.

Node types: Person, Event, Task, Topic, Decision, Location, Document, Reminder, Commitment
Relation types: KNOWS, ASSIGNED_TO, RELATED_TO, DECIDED_IN, WORKS_ON, LOCATED_AT, REFERENCES, COMMITTED_TO, FOLLOWS_UP
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GraphMemory:
    """Neo4j graph memory with graceful fallback."""

    def __init__(self) -> None:
        self._driver = None
        self._available = False

    def connect(self, uri: str, user: str, password: str) -> bool:
        """Attempt connection. Returns True if successful."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            self._available = True
            self._ensure_schema()
            logger.info("Neo4j connected: %s", uri)
            return True
        except Exception as e:
            logger.warning("Neo4j unavailable: %s — using SQLite fallback", e)
            self._available = False
            return False

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    @property
    def is_available(self) -> bool:
        return self._available

    def _run(self, query: str, **params: Any) -> list[dict]:
        if not self._available or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def _ensure_schema(self) -> None:
        """Create constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date)",
            "CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status)",
        ]
        for c in constraints:
            try:
                self._run(c)
            except Exception:
                pass

    # ── Person nodes ──────────────────────────────────────────────────────

    def upsert_person(
        self,
        name: str,
        relationship_type: str = "contact",
        importance: int = 5,
    ) -> None:
        self._run(
            """
            MERGE (p:Person {name: $name})
            SET p.relationship_type = $rel,
                p.importance = $importance,
                p.last_seen = $now
            """,
            name=name, rel=relationship_type,
            importance=importance, now=datetime.now().isoformat(),
        )

    def get_person(self, name: str) -> Optional[dict]:
        rows = self._run(
            "MATCH (p:Person {name: $name}) RETURN p",
            name=name,
        )
        return rows[0]["p"] if rows else None

    # ── Event nodes ───────────────────────────────────────────────────────

    def add_event(
        self,
        title: str,
        date: str,
        attendees: list[str] | None = None,
    ) -> str:
        """Create an Event node and link attendees."""
        rows = self._run(
            """
            CREATE (e:Event {title: $title, date: $date, created_at: $now})
            RETURN elementId(e) as id
            """,
            title=title, date=date, now=datetime.now().isoformat(),
        )
        event_id = rows[0]["id"] if rows else ""
        if event_id and attendees:
            for person_name in attendees:
                self.upsert_person(person_name)
                self._run(
                    """
                    MATCH (e:Event) WHERE elementId(e) = $eid
                    MATCH (p:Person {name: $name})
                    MERGE (p)-[:LOCATED_AT]->(e)
                    """,
                    eid=event_id, name=person_name,
                )
        return event_id

    # ── Decision nodes ────────────────────────────────────────────────────

    def add_decision(
        self,
        what: str,
        context: str = "",
        topic: str = "",
    ) -> None:
        self._run(
            """
            CREATE (d:Decision {what: $what, context: $context, date: $date})
            """,
            what=what, context=context, date=datetime.now().isoformat(),
        )
        if topic:
            self._run(
                """
                MATCH (d:Decision {what: $what})
                MERGE (t:Topic {name: $topic})
                MERGE (d)-[:RELATED_TO]->(t)
                """,
                what=what, topic=topic,
            )

    # ── Task nodes ────────────────────────────────────────────────────────

    def add_task(
        self,
        description: str,
        priority: str = "medium",
        deadline: str = "",
    ) -> None:
        self._run(
            """
            CREATE (t:Task {
                description: $desc,
                priority: $priority,
                deadline: $deadline,
                status: 'open',
                created_at: $now
            })
            """,
            desc=description, priority=priority,
            deadline=deadline, now=datetime.now().isoformat(),
        )

    # ── Context building ──────────────────────────────────────────────────

    def build_context(self, topic: str = "", person: str = "", n: int = 10) -> str:
        """
        Query graph and return LLM-ready context string.
        Used by Gemini finalizer for meeting briefings, etc.
        """
        if not self._available:
            return ""

        parts: list[str] = []

        if person:
            rows = self._run(
                """
                MATCH (p:Person {name: $name})
                OPTIONAL MATCH (p)-[:COMMITTED_TO]->(t:Task)
                OPTIONAL MATCH (p)-[:LOCATED_AT]->(e:Event)
                RETURN p, collect(t.description)[..5] as tasks,
                       collect(e.title)[..5] as events
                LIMIT 1
                """,
                name=person,
            )
            if rows:
                r = rows[0]
                parts.append(f"Person: {person}")
                if r.get("tasks"):
                    parts.append(f"  Open tasks: {', '.join(r['tasks'])}")
                if r.get("events"):
                    parts.append(f"  Recent events: {', '.join(r['events'])}")

        if topic:
            rows = self._run(
                """
                MATCH (t:Topic {name: $topic})
                OPTIONAL MATCH (d:Decision)-[:RELATED_TO]->(t)
                OPTIONAL MATCH (task:Task)-[:RELATED_TO]->(t)
                RETURN collect(d.what)[..5] as decisions,
                       collect(task.description)[..5] as tasks
                LIMIT 1
                """,
                topic=topic,
            )
            if rows:
                r = rows[0]
                parts.append(f"Topic: {topic}")
                if r.get("decisions"):
                    parts.append(f"  Decisions: {'; '.join(r['decisions'])}")
                if r.get("tasks"):
                    parts.append(f"  Related tasks: {', '.join(r['tasks'])}")

        # Recent decisions (always include)
        rows = self._run(
            """
            MATCH (d:Decision)
            RETURN d.what as what, d.date as date
            ORDER BY d.date DESC LIMIT $n
            """,
            n=n,
        )
        if rows:
            parts.append("Recent decisions:")
            for r in rows:
                parts.append(f"  [{r['date'][:10]}] {r['what']}")

        return "\n".join(parts) if parts else ""

    def stats(self) -> dict:
        """Node and relation counts for the status panel."""
        if not self._available:
            return {"nodes": 0, "relations": 0, "available": False}
        try:
            node_count = self._run("MATCH (n) RETURN count(n) as c")[0]["c"]
            rel_count = self._run("MATCH ()-[r]->() RETURN count(r) as c")[0]["c"]
            return {"nodes": node_count, "relations": rel_count, "available": True}
        except Exception:
            return {"nodes": 0, "relations": 0, "available": False}


graph_memory = GraphMemory()
