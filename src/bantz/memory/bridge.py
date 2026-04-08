"""
Bantz — MemPalace Bridge Adapter

Drop-in adapter that replaces the old memory subsystem (Neo4j graph + SQLite
vectors + Ollama embeddings + deep_probe + distiller) with MemPalace's unified
architecture (ChromaDB + SQLite knowledge graph + 4-layer stack).

Design goals:
  1. Same public API surface as the modules it replaces — callers don't change
     (yet). The bridge exposes the same method signatures so `omni_memory.py`,
     `memory_injector.py`, `brain.py`, etc. can swap imports without rewriting
     logic.
  2. All heavy lifting delegated to MemPalace — no duplicate embedding pipelines
     or vector stores. ChromaDB handles everything.
  3. Session conversation log stays in SQLite (core/memory.py) — MemPalace
     doesn't store raw messages, so the existing `Memory` class is preserved
     for `context(n)`, `add()`, `last_n()` etc.

Replaces:
  - memory/graph.py          → KnowledgeGraph (SQLite, temporal)
  - memory/nodes.py          → entity_detector + entity_registry
  - memory/context_builder.py → MemoryStack layers (L0-L3)
  - memory/vector_store.py    → ChromaDB built-in vectors
  - memory/embeddings.py      → ChromaDB built-in embeddings
  - memory/deep_probe.py      → spontaneous L3 search with rate limiting
  - memory/memory_manager.py  → thin wrapper, replaced by bridge methods
  - memory/distiller.py       → convo mining into palace + KG triples

Usage:
    from bantz.memory.bridge import palace_bridge
    await palace_bridge.init()
    result = await palace_bridge.recall("what was that decision about auth?")
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("bantz.memory.bridge")


# ── Lazy imports to avoid circular dependencies ──────────────────────────

def _get_config():
    from bantz.config import config
    return config


# ── Spontaneous Memory Probe (replaces deep_probe.py) ───────────────────

class SpontaneousProbe:
    """Rate-limited spontaneous memory surfacing via MemPalace L3.

    Replaces deep_probe.py. Instead of brute-force SQLite cosine scan,
    uses ChromaDB semantic search with rate limiting and dedup.
    """

    _INJECT_PREAMBLE = (
        "Spontaneous Human Memory — the following recollections surfaced "
        "naturally in your mind as a human butler. Weave them in subtly, "
        "as if you personally recall overhearing or being told these things. "
        "Do NOT say 'according to my records' or anything robotic.\n"
    )

    def __init__(
        self,
        *,
        rate_every_n: int = 3,
        max_results: int = 3,
        min_similarity: float = 0.45,
        decay_half_life_days: float = 30.0,
        dedup_ttl_seconds: float = 600.0,
    ) -> None:
        self._rate_every_n = rate_every_n
        self._max_results = max_results
        self._min_similarity = min_similarity
        self._decay_half_life = decay_half_life_days
        self._dedup_ttl = dedup_ttl_seconds
        self._call_counter: int = 0
        self._recently_used: dict[str, float] = {}  # drawer_id → mono time

    def probe(self, user_message: str, l3) -> str:
        """Return a spontaneous memory hint or empty string.

        Rate-limited: only fires every N calls. Uses L3 deep search,
        filters by similarity, deduplicates, and returns butler-lore
        formatted text.
        """
        cfg = _get_config()
        if not getattr(cfg, "deep_memory_enabled", True):
            return ""

        self._call_counter += 1
        if self._call_counter % self._rate_every_n != 0:
            return ""

        max_results = getattr(cfg, "deep_memory_max_results", self._max_results)

        try:
            raw_hits = l3.search_raw(user_message, n_results=max_results + 5)
        except Exception as exc:
            log.debug("Spontaneous probe search failed: %s", exc)
            return ""

        if not raw_hits:
            return ""

        # Filter by similarity threshold
        min_sim = getattr(cfg, "deep_memory_threshold", self._min_similarity)
        candidates = [h for h in raw_hits if h.get("similarity", 0) >= min_sim]
        if not candidates:
            return ""

        # Dedup: skip recently surfaced memories
        self._evict_stale()
        now = time.monotonic()
        filtered = []
        for h in candidates:
            did = self._drawer_id(h)
            if did not in self._recently_used:
                filtered.append(h)
        if not filtered:
            return ""

        # Take top-N and mark as used
        top = filtered[:max_results]
        for h in top:
            self._recently_used[self._drawer_id(h)] = now

        return self._format_hint(top)

    def reset(self) -> None:
        self._call_counter = 0
        self._recently_used.clear()

    def _drawer_id(self, hit: dict) -> str:
        text = hit.get("text", "")[:100]
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale = [k for k, t in self._recently_used.items() if now - t > self._dedup_ttl]
        for k in stale:
            del self._recently_used[k]

    def _format_hint(self, memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = [self._INJECT_PREAMBLE]
        for m in memories:
            text = m.get("text", "").strip()
            if len(text) > 200:
                text = text[:197] + "..."
            wing = m.get("wing", "?")
            room = m.get("room", "?")
            lines.append(f"  • [{wing}/{room}] {text}")
        return "\n".join(lines)


# ── MemPalace Bridge ────────────────────────────────────────────────────

class MemPalaceBridge:
    """Adapter that provides the old memory API surface on top of MemPalace.

    Singletons this replaces:
      - graph_memory  (GraphMemory)
      - memory_manager (MemoryManager)
      - deep_probe    (DeepMemoryProbe)
      - embedder      (Embedder)
      - distill_session / store_distillation  (distiller functions)

    Singletons this does NOT replace:
      - memory        (core/memory.py — SQLite conversation log)
      - session_store (Redis — session state, queues, pubsub)
      - omni_memory   (will be rewritten to call bridge instead)
    """

    def __init__(self) -> None:
        self._stack = None          # MemoryStack
        self._kg = None             # KnowledgeGraph
        self._registry = None       # EntityRegistry
        self._probe = SpontaneousProbe()
        self._initialized = False
        self._wing: str = "bantz"

    # ── Lifecycle ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when MemPalace is initialized and ready."""
        return self._initialized and self._stack is not None

    @property
    def kg(self):
        """Direct access to the KnowledgeGraph for callers that need it."""
        return self._kg

    @property
    def stack(self):
        """Direct access to MemoryStack."""
        return self._stack

    @property
    def registry(self):
        """Direct access to EntityRegistry."""
        return self._registry

    async def init(self) -> None:
        """Initialize MemPalace components. Replaces graph_memory.init()."""
        cfg = _get_config()

        if not getattr(cfg, "mempalace_enabled", True):
            log.info("MemPalace disabled via config")
            return

        try:
            from mempalace.layers import MemoryStack
            from mempalace.knowledge_graph import KnowledgeGraph
            from mempalace.entity_registry import EntityRegistry

            palace_path = cfg.resolved_palace_path
            kg_path = cfg.resolved_kg_path
            identity_path = cfg.resolved_identity_path
            self._wing = getattr(cfg, "mempalace_wing", "bantz")

            # Ensure dirs exist
            Path(palace_path).mkdir(parents=True, exist_ok=True)
            Path(kg_path).parent.mkdir(parents=True, exist_ok=True)

            self._stack = MemoryStack(
                palace_path=palace_path,
                identity_path=identity_path,
            )
            self._kg = KnowledgeGraph(db_path=kg_path)
            self._registry = EntityRegistry.load()
            self._initialized = True

            # ── First-run onboarding ─────────────────────────────────
            from bantz.memory.onboarding import is_onboarding_done
            palace_parent = str(Path(palace_path).parent)
            if not is_onboarding_done(palace_parent):
                try:
                    import sys
                    if sys.stdin.isatty():
                        from bantz.memory.onboarding import run_onboarding
                        run_onboarding(
                            identity_path=identity_path,
                            kg=self._kg,
                            registry=self._registry,
                            palace_parent=palace_parent,
                        )
                        # Reload L0 after identity file was written
                        self._stack = MemoryStack(
                            palace_path=palace_path,
                            identity_path=identity_path,
                        )
                except Exception as onb_exc:
                    log.debug("Onboarding skipped: %s", onb_exc)

            log.info(
                "MemPalace bridge initialized "
                "(palace=%s, kg=%s, wing=%s)",
                palace_path, kg_path, self._wing,
            )
        except Exception as exc:
            log.warning("MemPalace bridge init failed: %s", exc)
            self._initialized = False

    def close(self) -> None:
        """Clean shutdown. Replaces graph_memory.close()."""
        self._initialized = False
        self._stack = None
        self._kg = None
        self._registry = None
        log.debug("MemPalace bridge closed")

    # ── Context retrieval (replaces context_builder.build_context) ────

    def wake_up_context(self) -> str:
        """Return L0 + L1 wake-up text (~600-900 tokens).

        Replaces the always-on portion of graph context.
        """
        if not self.enabled:
            return ""
        try:
            return self._stack.wake_up(wing=self._wing)
        except Exception as exc:
            log.debug("wake_up_context failed: %s", exc)
            return ""

    def graph_context(self, user_msg: str) -> str:
        """Return knowledge graph context for a user message.

        Replaces graph_memory.context_for().
        Queries KG entities mentioned in the message.
        """
        if not self.enabled or not self._kg:
            return ""
        try:
            return self._kg_context_for(user_msg)
        except Exception as exc:
            log.debug("graph_context failed: %s", exc)
            return ""

    def vector_context(self, user_msg: str, limit: int = 5) -> str:
        """Semantic search via MemPalace L3. Replaces vector_store + embedder.

        ChromaDB handles embedding internally — no Ollama needed.
        """
        if not self.enabled:
            return ""
        try:
            return self._stack.search(
                user_msg,
                wing=self._wing,
                n_results=limit,
            )
        except Exception as exc:
            log.debug("vector_context failed: %s", exc)
            return ""

    def deep_memory(self, user_msg: str) -> str:
        """Spontaneous memory surfacing. Replaces deep_probe.probe()."""
        if not self.enabled:
            return ""
        try:
            return self._probe.probe(user_msg, self._stack.l3)
        except Exception as exc:
            log.debug("deep_memory failed: %s", exc)
            return ""

    # ── Storage (replaces graph_memory.extract_and_store) ────────────

    async def store_exchange(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: str | None = None,
        tool_data: dict | None = None,
    ) -> None:
        """Store an exchange in the palace. Replaces graph_memory.extract_and_store().

        1. Adds the exchange as a drawer in ChromaDB (via miner.add_drawer)
        2. Extracts entities → upserts into knowledge graph
        """
        if not self.enabled:
            return

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._store_exchange_sync,
                user_msg, assistant_msg, tool_used, tool_data,
            )
        except Exception as exc:
            log.debug("store_exchange failed: %s", exc)

    def _store_exchange_sync(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: str | None,
        tool_data: dict | None,
    ) -> None:
        """Synchronous store — runs in executor."""
        from mempalace.miner import get_collection, add_drawer

        cfg = _get_config()
        palace_path = cfg.resolved_palace_path
        col = get_collection(palace_path)

        # Build exchange text
        exchange = f"User: {user_msg}\nAssistant: {assistant_msg}"
        if tool_used:
            exchange += f"\n[Tool: {tool_used}]"

        # Detect room from content
        room = self._detect_room_from_exchange(user_msg, assistant_msg, tool_used)

        # Add as drawer
        ts = datetime.now().isoformat()
        add_drawer(
            col,
            wing=self._wing,
            room=room,
            content=exchange,
            source_file=f"bantz_session_{ts[:10]}",
            chunk_index=int(time.time() * 1000) % 999999,
            agent="bantz",
        )

        # Extract and store KG triples
        self._extract_kg_triples(user_msg, assistant_msg, tool_used, tool_data)

    def _detect_room_from_exchange(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: str | None,
    ) -> str:
        """Map an exchange to a MemPalace room name."""
        # Tool-based routing (most reliable signal)
        tool_room_map = {
            "calendar": "events",
            "gmail": "email",
            "email": "email",
            "weather": "daily",
            "news": "daily",
            "reminder": "tasks",
            "filesystem": "technical",
            "shell": "technical",
            "web_search": "research",
            "document": "documents",
            "classroom": "education",
        }
        if tool_used:
            for key, room in tool_room_map.items():
                if key in tool_used.lower():
                    return room

        # Keyword-based fallback
        combined = (user_msg + " " + assistant_msg).lower()
        keyword_rooms = {
            "decisions": ["decided", "let's go with", "we'll use", "decision"],
            "tasks": ["remind me", "todo", "need to", "task", "deadline"],
            "preferences": ["i prefer", "i like", "always use", "favorite"],
            "problems": ["bug", "error", "crash", "broken", "issue"],
            "events": ["meeting", "appointment", "schedule", "event"],
            "people": ["told me", "said that", "according to", "with "],
            "technical": ["code", "deploy", "server", "database", "api"],
        }
        best_room = "general"
        best_score = 0
        for room, keywords in keyword_rooms.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_room = room

        return best_room

    def _extract_kg_triples(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: str | None,
        tool_data: dict | None,
    ) -> None:
        """Extract entities and relationships → store in KG.

        Multi-lingual (EN + TR) regex extraction — stores into
        MemPalace's temporal KnowledgeGraph (SQLite).
        """
        if not self._kg:
            log.debug("_extract_kg_triples: no KG instance — skipping")
            return

        import re

        combined = user_msg + "\n" + assistant_msg
        today = datetime.now().strftime("%Y-%m-%d")
        src = f"bantz_session_{today}"

        # ── 1. Self-introduction patterns (EN + TR) ──────────────────
        intro_patterns = [
            # English — word-boundary guarded to avoid mid-word matches
            r"\b(?:my name is|call me)\s+(\S+)",
            r"\bi[' ]?m\s+(\S+)",
            # Turkish
            r"(?:benim\s+adım|adım)\s+(\S+)",
        ]
        # Names that are clearly not person names
        _INTRO_SKIP = {
            "a", "an", "the", "here", "there", "not", "just",
            "very", "so", "really", "also", "doing", "going",
            "fine", "good", "well", "ok", "okay", "sure",
            "ready", "happy", "sorry", "glad", "tired", "busy",
            "looking", "trying", "working", "using", "thinking",
        }
        for pat in intro_patterns:
            for m in re.finditer(pat, combined, re.IGNORECASE):
                name = m.group(1).strip().rstrip(".,!?;:")
                if len(name) < 2 or name.lower() in _INTRO_SKIP:
                    continue
                # Must start with a letter (skip numbers, symbols)
                if not name[0].isalpha():
                    continue
                try:
                    self._kg.add_entity(name, entity_type="person")
                    self._kg.add_triple(
                        "user", "name_is", name,
                        valid_from=today, source_file=src,
                    )
                except Exception as exc:
                    log.warning("KG intro triple failed for %r: %s", name, exc)

        # ── 2. Profession / role patterns ────────────────────────────
        profession_patterns = [
            # English
            r"i(?:'m| am) (?:a |an )?(\w[\w\s]{2,30}?)(?:\.|,|$)",
            r"i work (?:as|in) (?:a |an )?(.+?)(?:\.|,|$)",
            r"my (?:job|profession|occupation) is (.+?)(?:\.|,|$)",
            # Turkish — suffix-based (yazılımcıyım, mühendisim, etc.)
            r"(\w+(?:cıyım|ciyim|çıyım|çiyim|cuyum|cüyüm|çuyum|çüyüm))\b",
            r"(\w+(?:ım|im|um|üm))\b(?=\s*[.,!?]|\s*$)",
            # Turkish — explicit
            r"(?:mesleğim|işim)\s+(.+?)(?:\.|,|$)",
        ]
        for pat in profession_patterns:
            for m in re.finditer(pat, combined, re.IGNORECASE):
                role = m.group(1).strip().rstrip(".,!?;:")
                if len(role) < 3 or len(role) > 40:
                    continue
                # Skip common false positives
                if role.lower() in {
                    "here", "there", "fine", "good", "well", "ok",
                    "sure", "ready", "happy", "sorry", "glad",
                }:
                    continue
                try:
                    self._kg.add_triple(
                        "user", "works_as", role,
                        valid_from=today, source_file=src,
                    )
                except Exception as exc:
                    log.warning("KG profession triple failed for %r: %s", role, exc)

        # ── 3. Person names (Unicode-aware, case-sensitive) ───────────
        _STOP_WORDS = {
            "the", "this", "that", "here", "there", "what", "when",
            "where", "how", "which", "then", "also", "just", "very",
            "much", "some", "will", "have", "been", "about", "from",
            "with", "user", "assistant", "merhaba", "selam", "hello",
            "nice", "meet", "good", "great", "sure", "well",
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "january", "february", "march",
            "april", "may", "june", "july", "august", "september",
            "october", "november", "december",
        }
        # Title-case name after common prepositions — NO IGNORECASE
        # First char must genuinely be uppercase for this set of patterns.
        _UC = r"[\u00C0-\u00D6\u00D8-\u00DEA-Z\u0100\u0102\u0104\u0106\u0108\u010A\u010C\u010E\u0110\u0112\u0114\u0116\u0118\u011A\u011C\u011E\u0120\u0122\u0124\u0126\u0128\u012A\u012C\u012E\u0130\u0132\u0134\u0136\u0139\u013B\u013D\u013F\u0141\u0143\u0145\u0147\u014A\u014C\u014E\u0150\u0152\u0154\u0156\u0158\u015A\u015C\u015E\u0160\u0162\u0164\u0166\u0168\u016A\u016C\u016E\u0170\u0172\u0174\u0176\u0178\u0179\u017B\u017D]"
        _LC = r"[\u00E0-\u00F6\u00F8-\u00FFa-z\u0101\u0103\u0105\u0107\u0109\u010B\u010D\u010F\u0111\u0113\u0115\u0117\u0119\u011B\u011D\u011F\u0121\u0123\u0125\u0127\u0129\u012B\u012D\u012F\u0131\u0133\u0135\u0137\u013A\u013C\u013E\u0140\u0142\u0144\u0146\u0148\u014B\u014D\u014F\u0151\u0153\u0155\u0157\u0159\u015B\u015D\u015F\u0161\u0163\u0165\u0167\u0169\u016B\u016D\u016F\u0171\u0173\u0175\u0177\u017A\u017C\u017E]"
        _NAME = f"{_UC}{_LC}+"
        person_patterns = [
            # English prepositions → single Title-Case word only
            rf"\b(?:with|from|for|about|named)\s+({_NAME})\b",
            # Turkish prepositions
            rf"\b(?:ile|için|hakkında)\s+({_NAME})\b",
            # Greeting + name (case-sensitive — name must be Title Case)
            rf"\b(?:Merhaba|Selam|Hey|Hi|Hello)\s+({_NAME})\b",
        ]
        people: set[str] = set()
        for pat in person_patterns:
            for m in re.finditer(pat, combined):  # case-sensitive!
                name = m.group(1).strip().rstrip(".,!?;:")
                if len(name) > 1 and name.lower() not in _STOP_WORDS:
                    people.add(name)

        for person in people:
            try:
                self._kg.add_entity(person, entity_type="person")
            except Exception as exc:
                log.warning("KG add_entity failed for %r: %s", person, exc)

        # ── 4. Tool-based triples ────────────────────────────────────
        if tool_used:
            try:
                topic = tool_used.split("_")[0] if "_" in tool_used else tool_used
                self._kg.add_triple(
                    "user", f"used_{tool_used}", topic,
                    valid_from=today, source_file=src,
                )
            except Exception as exc:
                log.warning("KG tool triple failed: %s", exc)

        # ── 5. Decision triples ──────────────────────────────────────
        decision_patterns = [
            r"(?:let's|lets|we'll|we will)\s+(?:use|go with|switch to)\s+(.+?)(?:[.,;!\n]|$)",
            r"decided\s+(?:to|on)\s+(.+?)(?:[.,;!\n]|$)",
            # Turkish
            r"(?:karar verdik|kullanacağız|geçelim)\s+(.+?)(?:[.,;!\n]|$)",
        ]
        for pat in decision_patterns:
            for m in re.finditer(pat, combined, re.IGNORECASE):
                decision = m.group(1).strip()[:80]
                if decision:
                    try:
                        self._kg.add_triple(
                            "user", "decided", decision,
                            valid_from=today, source_file=src,
                        )
                    except Exception as exc:
                        log.warning("KG decision triple failed: %s", exc)

        # ── 6. Preference triples ────────────────────────────────────
        pref_patterns = [
            r"i (?:prefer|like|love|always use|enjoy)\s+(.+?)(?:[.,;!\n]|$)",
            r"my favo(?:u)?rite (?:is |)\s*(.+?)(?:[.,;!\n]|$)",
            # Turkish
            r"(?:tercih ederim|seviyorum|kullanıyorum)\s+(.+?)(?:[.,;!\n]|$)",
        ]
        for pat in pref_patterns:
            for m in re.finditer(pat, combined, re.IGNORECASE):
                pref = m.group(1).strip()[:60]
                if pref and len(pref) > 2:
                    try:
                        self._kg.add_triple(
                            "user", "prefers", pref,
                            valid_from=today, source_file=src,
                        )
                    except Exception as exc:
                        log.warning("KG preference triple failed: %s", exc)

    def _kg_context_for(self, user_msg: str) -> str:
        """Build context string from KnowledgeGraph. Replaces context_builder."""
        if not self._kg:
            return ""

        lines = []

        # Extract potential entity names from the query
        words = [w for w in user_msg.split() if len(w) > 2 and w[0].isupper()]

        # Also try the entity registry for better detection
        if self._registry:
            try:
                detected = self._registry.extract_people_from_query(user_msg)
                words.extend(detected)
            except Exception:
                pass

        # Query each potential entity
        seen = set()
        for word in words:
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            try:
                facts = self._kg.query_entity(word)
                if facts:
                    lines.append(f"[{word}]")
                    for fact in facts[:5]:  # limit per entity
                        subj = fact.get("subject", "?")
                        pred = fact.get("predicate", "?")
                        obj = fact.get("object", "?")
                        lines.append(f"  {subj} → {pred} → {obj}")
            except Exception:
                pass

        # Recent triples (last 24h) as general context
        try:
            stats = self._kg.stats()
            if stats.get("total_triples", 0) > 0:
                # Get recent activity
                recent = self._kg_recent_triples(limit=5)
                if recent:
                    lines.append("[Recent]")
                    for t in recent:
                        lines.append(f"  {t['subject']} → {t['predicate']} → {t['object']}")
        except Exception:
            pass

        if not lines:
            return ""

        return "=== Knowledge Graph ===\n" + "\n".join(lines)

    def _kg_recent_triples(self, limit: int = 5) -> list[dict]:
        """Get most recent KG triples."""
        if not self._kg:
            return []
        try:
            import sqlite3
            conn = sqlite3.connect(self._kg.db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT subject, predicate, object, valid_from
                   FROM triples
                   ORDER BY extracted_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ── Distillation (replaces distiller.py) ─────────────────────────

    async def distill_session(self, session_id: int) -> dict:
        """Mine a completed session into the palace.

        Replaces distiller.distill_session(). Instead of LLM summarization
        + embedding, we chunk the session transcript and add drawers
        directly to ChromaDB. The knowledge graph also gets updated.

        Returns a dict with summary info.
        """
        if not self.enabled:
            return {"summary": "", "drawers_added": 0}

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._distill_session_sync, session_id,
            )
            return result
        except Exception as exc:
            log.debug("distill_session failed: %s", exc)
            return {"summary": "", "drawers_added": 0}

    def _distill_session_sync(self, session_id: int) -> dict:
        """Synchronous distillation — runs in executor."""
        from bantz.data.connection_pool import get_pool
        from mempalace.miner import get_collection, add_drawer

        cfg = _get_config()

        # Fetch messages
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT role, content, tool_used, created_at
                   FROM messages
                   WHERE conversation_id = ?
                   ORDER BY created_at ASC""",
                (session_id,),
            ).fetchall()

        if not rows or len(rows) < 4:
            return {"summary": "", "drawers_added": 0}

        messages = [dict(r) for r in rows]

        # Build exchange pairs and add as drawers
        col = get_collection(cfg.resolved_palace_path)
        drawers_added = 0

        # Group into exchange pairs (user + assistant)
        exchanges = []
        current_exchange = []
        for msg in messages:
            current_exchange.append(msg)
            if msg["role"] == "assistant" and len(current_exchange) >= 2:
                exchanges.append(current_exchange)
                current_exchange = []

        for i, exchange in enumerate(exchanges):
            parts = []
            for msg in exchange:
                role = msg["role"].upper()
                content = msg["content"]
                parts.append(f"{role}: {content}")
                if msg.get("tool_used"):
                    parts.append(f"[Tool: {msg['tool_used']}]")

            text = "\n".join(parts)
            if len(text) < 20:
                continue

            # Detect room from exchange content
            user_parts = [m["content"] for m in exchange if m["role"] == "user"]
            asst_parts = [m["content"] for m in exchange if m["role"] == "assistant"]
            tool_parts = [m.get("tool_used", "") for m in exchange if m.get("tool_used")]

            room = self._detect_room_from_exchange(
                " ".join(user_parts),
                " ".join(asst_parts),
                tool_parts[0] if tool_parts else None,
            )

            success = add_drawer(
                col,
                wing=self._wing,
                room=room,
                content=text,
                source_file=f"bantz_session_{session_id}",
                chunk_index=i,
                agent="bantz_distiller",
            )
            if success:
                drawers_added += 1

            # Also extract KG triples from this exchange
            self._extract_kg_triples(
                " ".join(user_parts),
                " ".join(asst_parts),
                tool_parts[0] if tool_parts else None,
                None,
            )

        summary = (
            f"Session {session_id}: {len(exchanges)} exchanges → "
            f"{drawers_added} drawers mined into palace"
        )
        log.info(summary)
        return {"summary": summary, "drawers_added": drawers_added}

    # ── Stats (replaces graph_memory.stats / growth_since) ────────────

    def stats(self) -> dict:
        """Return combined stats for palace + KG."""
        result = {
            "palace_enabled": self.enabled,
            "total_drawers": 0,
            "kg_entities": 0,
            "kg_triples": 0,
        }
        if not self.enabled:
            return result

        try:
            status = self._stack.status()
            result["total_drawers"] = status.get("total_drawers", 0)
        except Exception:
            pass

        try:
            kg_stats = self._kg.stats()
            result["kg_entities"] = kg_stats.get("entities", 0)
            result["kg_triples"] = kg_stats.get("triples", 0)
        except Exception:
            pass

        return result

    def status_line(self) -> str:
        """One-line status string."""
        s = self.stats()
        return (
            f"MemPalace: {s['total_drawers']} drawers, "
            f"{s['kg_entities']} entities, "
            f"{s['kg_triples']} triples"
        )

    def growth_since(self, since_iso: str) -> dict:
        """Count new KG items since a timestamp."""
        result = {"entities": 0, "triples": 0}
        if not self._kg:
            return result
        try:
            import sqlite3
            conn = sqlite3.connect(self._kg.db_path, timeout=5)
            row = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE created_at >= ?",
                (since_iso,),
            ).fetchone()
            result["entities"] = row[0] if row else 0
            row = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE extracted_at >= ?",
                (since_iso,),
            ).fetchone()
            result["triples"] = row[0] if row else 0
            conn.close()
        except Exception:
            pass
        return result


# ── Module singleton ─────────────────────────────────────────────────────
palace_bridge = MemPalaceBridge()
