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

log = logging.getLogger("bantz.graph")

# ── Node / Relationship constants ─────────────────────────────────────────

NODE_LABELS = ("Person", "Topic", "Decision", "Task", "Event", "Location", "Document")

REL_TYPES = {
    "KNOWS":        ("Person", "Person"),
    "ASSIGNED_TO":  ("Task", "Person"),
    "RELATED_TO":   (None, None),          # flexible
    "DECIDED_IN":   ("Decision", "Event"),
    "WORKS_ON":     ("Person", "Topic"),
    "LOCATED_AT":   (None, "Location"),
    "REFERENCES":   (None, "Document"),
    "COMMITTED_TO": ("Person", "Task"),
    "FOLLOWS_UP":   ("Task", "Decision"),
}


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
        """
        Rule-based entity extraction from conversation.
        Returns list of {"label": ..., "props": {...}, "rels": [...]}
        """
        entities: list[dict] = []
        combined = f"{user_msg} {assistant_msg}".lower()
        tool_data = tool_data or {}

        # ── People mentioned ──
        # Look for name patterns: "Ali said", "meeting with Ahmet", "email from John"
        people_patterns = [
            r"(?:from|with|to|by|about|ask|tell|call|email|meet)\s+([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)",
        ]
        found_people: set[str] = set()
        for pat in people_patterns:
            for m in re.finditer(pat, f"{user_msg} {assistant_msg}"):
                name = m.group(1).strip()
                # Filter out common non-name words
                skip = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                        "Saturday", "Sunday", "January", "February", "March",
                        "April", "May", "June", "July", "August", "September",
                        "October", "November", "December", "Today", "Tomorrow",
                        "Error", "Done", "Event", "Calendar", "Gmail", "News",
                        "Desktop", "Downloads", "Documents", "Home", "Bantz",
                        "Linux", "Python", "English", "Turkish"}
                if name not in skip and len(name) > 2:
                    found_people.add(name)

        for name in found_people:
            entities.append({
                "label": "Person",
                "key": "name",
                "props": {"name": name, "last_seen": datetime.now().isoformat()},
                "rels": [],
            })

        # ── Topics from tool usage ──
        tool_topic_map = {
            "calendar": "calendar",
            "gmail": "email",
            "news": "news",
            "weather": "weather",
            "classroom": "university",
            "web_search": "research",
            "document": "documents",
            "shell": "system",
        }
        if tool_used and tool_used in tool_topic_map:
            topic_name = tool_topic_map[tool_used]
            entities.append({
                "label": "Topic",
                "key": "name",
                "props": {"name": topic_name, "last_accessed": datetime.now().isoformat()},
                "rels": [],
            })

        # ── Calendar events → Event nodes ──
        if tool_used == "calendar" and tool_data:
            events = tool_data.get("events", [])
            if isinstance(events, list):
                for ev in events[:5]:
                    title = ev.get("summary") or ev.get("title", "")
                    if title:
                        start = ev.get("start", "")
                        entities.append({
                            "label": "Event",
                            "key": "title",
                            "props": {"title": title, "date": start,
                                      "updated_at": datetime.now().isoformat()},
                            "rels": [],
                        })
            # Created event from message
            if "added" in assistant_msg.lower() or "created" in assistant_msg.lower():
                title_m = re.search(r"[\"'](.+?)[\"']|Event:\s*(.+?)(?:\n|$)", assistant_msg)
                if title_m:
                    t = (title_m.group(1) or title_m.group(2) or "").strip()
                    if t:
                        entities.append({
                            "label": "Event",
                            "key": "title",
                            "props": {"title": t, "updated_at": datetime.now().isoformat()},
                            "rels": [],
                        })

        # ── Gmail → potential person nodes ──
        if tool_used == "gmail" and tool_data:
            messages = tool_data.get("messages", [])
            for msg in messages[:5]:
                sender = msg.get("from", "")
                if sender:
                    # Extract name from "Name <email>" format
                    name_match = re.match(r"([^<]+)", sender)
                    if name_match:
                        name = name_match.group(1).strip().strip('"')
                        if name and len(name) > 2 and "@" not in name:
                            entities.append({
                                "label": "Person",
                                "key": "name",
                                "props": {"name": name, "email": sender.strip(),
                                          "last_seen": datetime.now().isoformat()},
                                "rels": [],
                            })

        # ── Decisions — "let's use X", "we'll go with X", "decided to X" ──
        decision_patterns = [
            r"(?:let'?s|we(?:'ll)?\s+(?:go\s+with|use|pick|choose|decided?\s+to))\s+(.+?)(?:\.|$|!)",
            r"(?:i'?ll|going to|plan to)\s+(.+?)(?:\.|$|!)",
        ]
        for pat in decision_patterns:
            m = re.search(pat, combined)
            if m:
                what = m.group(1).strip()[:100]
                if len(what) > 5:
                    entities.append({
                        "label": "Decision",
                        "key": "what",
                        "props": {"what": what,
                                  "date": datetime.now().isoformat(),
                                  "context": user_msg[:200]},
                        "rels": [],
                    })
                break  # one decision per exchange

        # ── Tasks — "remind me to", "i need to", "add task" ──
        task_patterns = [
            r"(?:remind\s+me\s+to|i\s+need\s+to|todo|add\s+task|task:)\s+(.+?)(?:\.|$|!)",
            r"(?:don'?t\s+forget\s+to)\s+(.+?)(?:\.|$|!)",
        ]
        for pat in task_patterns:
            m = re.search(pat, combined)
            if m:
                desc = m.group(1).strip()[:150]
                if len(desc) > 3:
                    entities.append({
                        "label": "Task",
                        "key": "description",
                        "props": {"description": desc, "status": "open",
                                  "priority": "medium",
                                  "created_at": datetime.now().isoformat()},
                        "rels": [],
                    })
                break

        # ── Documents ──
        if tool_used == "document":
            path = tool_data.get("path", "")
            if path:
                entities.append({
                    "label": "Document",
                    "key": "path",
                    "props": {"path": path,
                              "accessed_at": datetime.now().isoformat()},
                    "rels": [],
                })

        # ── Locations ──
        loc_patterns = [
            r"(?:in|at|from|going to|travel to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            r"(?:\s|,|\.|\?|!|$)",
        ]
        skip_locs = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday", "January", "February", "March",
                     "April", "May", "June", "July", "August", "September",
                     "October", "November", "December", "Today", "Tomorrow",
                     "Error", "Done", "English", "Turkish", "Bantz", "Ollama",
                     "Gmail", "Google", "Linux", "Python", "Neo"}
        for pat in loc_patterns:
            for m in re.finditer(pat, f"{user_msg} {assistant_msg}"):
                loc = m.group(1).strip()
                if loc not in skip_locs and len(loc) > 2:
                    entities.append({
                        "label": "Location",
                        "key": "name",
                        "props": {"name": loc,
                                  "last_mentioned": datetime.now().isoformat()},
                        "rels": [],
                    })

        return entities

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
        """
        Query the graph for entities relevant to the user's message
        and return a compact multi-line string the LLM can use as context.
        """
        if not self._enabled:
            return ""

        try:
            parts: list[str] = []

            # 1. Recent people
            people = await self._query(
                "MATCH (p:Person) RETURN p.name AS name, p.last_seen AS seen "
                "ORDER BY p.last_seen DESC LIMIT 8"
            )
            if people:
                names = [r["name"] for r in people]
                parts.append(f"Known people: {', '.join(names)}")

            # 2. Open tasks
            tasks = await self._query(
                "MATCH (t:Task {status: 'open'}) "
                "RETURN t.description AS desc, t.priority AS prio "
                "ORDER BY t.created_at DESC LIMIT 5"
            )
            if tasks:
                lines = [f"  - [{t['prio']}] {t['desc']}" for t in tasks]
                parts.append("Open tasks:\n" + "\n".join(lines))

            # 3. Recent decisions
            decisions = await self._query(
                "MATCH (d:Decision) RETURN d.what AS what, d.date AS date "
                "ORDER BY d.date DESC LIMIT 3"
            )
            if decisions:
                lines = [f"  - {d['what']}" for d in decisions]
                parts.append("Recent decisions:\n" + "\n".join(lines))

            # 4. Recent events
            events = await self._query(
                "MATCH (e:Event) RETURN e.title AS title, e.date AS date "
                "ORDER BY e.updated_at DESC LIMIT 5"
            )
            if events:
                lines = [f"  - {e['title']} ({e['date'] or '?'})" for e in events]
                parts.append("Recent events:\n" + "\n".join(lines))

            # 5. Keyword-matching — find nodes whose names/descriptions match user query
            keywords = self._extract_keywords(user_msg)
            if keywords:
                relevant = await self._keyword_search(keywords)
                if relevant:
                    parts.append("Related context:\n" + "\n".join(f"  - {r}" for r in relevant))

            if not parts:
                return ""

            return "=== Graph Memory ===\n" + "\n".join(parts) + "\n=== End Graph ==="

        except Exception as exc:
            log.debug("Graph context error: %s", exc)
            return ""

    async def _query(self, cypher: str, **params) -> list[dict]:
        """Run a read query, return list of record dicts."""
        async with self._driver.session() as s:
            result = await s.run(cypher, **params)
            records = await result.data()
            return records

    def _extract_keywords(self, text: str) -> list[str]:
        """Pull significant words from user input for graph search."""
        stop = {"the", "a", "an", "is", "are", "was", "were", "do", "does",
                "did", "will", "can", "could", "would", "should", "my", "me",
                "i", "you", "we", "it", "he", "she", "they", "this", "that",
                "what", "how", "when", "where", "who", "which", "to", "in",
                "on", "at", "for", "of", "and", "or", "but", "not", "with",
                "from", "by", "about", "any", "all", "some", "up", "out",
                "just", "now", "today", "tomorrow", "yesterday", "please",
                "check", "show", "tell", "get", "want", "need", "have", "has",
                "hey", "hi", "hello", "thanks", "ok", "okay", "yeah", "yes", "no"}
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return [w for w in words if w not in stop][:5]

    async def _keyword_search(self, keywords: list[str]) -> list[str]:
        """Search graph nodes for keyword matches."""
        results: list[str] = []
        for kw in keywords:
            pattern = f"(?i).*{re.escape(kw)}.*"
            rows = await self._query(
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
