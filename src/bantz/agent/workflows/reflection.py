"""
Bantz — Nightly Memory Reflection Workflow (#130)

Runs at 11 PM daily via APScheduler.  Compresses the day's conversations
into a coherent daily reflection with entity extraction and graph updates.

Architecture (user's 4 strategic fixes applied):

  1. **Hierarchical Summarisation** — reads today's *session distillation*
     summaries (Issue #118) instead of raw messages.  Saves tokens, avoids
     blowing the context window on 50+ message days.

  2. **Generous Timeout** — 10-min total cap (instead of 2 min).  This is
     a night job running while the user sleeps; no rush.

  3. **Vector Orphan Cleanup** — when pruning raw messages older than 30
     days, also deletes their corresponding embeddings from vector_store
     so no "ghost vectors" remain in semantic search.

  4. **Entity Resolution** — before asking the LLM to extract entities,
     fetches existing Person/Topic/Decision nodes from Neo4j and injects
     them into the prompt so the model merges "Ali (study group)" with
     "Ali" instead of creating duplicates.

Pipeline:
    1. Collect today's session distillation summaries
    2. LLM daily reflection prompt (using summaries, not raw messages)
    3. Parse structured JSON output
    4. Entity extraction with graph context injection
    5. Store reflection in vector DB + KV store
    6. Optional raw-message pruning (> 30 days) with vector cleanup
    7. Report (notification + Telegram)

Usage:
    from bantz.agent.workflows.reflection import run_reflection
    report = await run_reflection(dry_run=False)

    # CLI
    bantz --reflect              # run now
    bantz --reflect --dry-run    # simulate
    bantz --reflections          # view past reflections
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

log = logging.getLogger("bantz.reflection")

# ── Constants ────────────────────────────────────────────────────────────

_TOTAL_TIMEOUT = 600            # 10 min total (generous for local LLM)
_LLM_TIMEOUT = 120             # 2 min per LLM call
_PRUNE_RAW_DAYS = 30           # delete raw messages older than this
_REFLECTION_RETENTION_DAYS = 0  # 0 = keep forever

# ── Prompts ──────────────────────────────────────────────────────────────

_REFLECTION_SYSTEM = """\
You are Bantz's nightly self-reflection engine.  Given today's conversation \
summaries, produce a structured daily reflection in JSON format.

OUTPUT — valid JSON only, no markdown fences:
{
  "summary": "2-3 sentence overview of the day",
  "decisions": ["decision 1", "decision 2"],
  "tasks_created": ["task 1"],
  "tasks_completed": ["task 1"],
  "people_mentioned": ["Name (context)"],
  "reflection": "Pattern observations, meta-insights, suggestions",
  "unresolved": ["item 1"]
}

RULES:
— Be factual: only include what actually appears in the summaries.
— Keep 'summary' to max 3 sentences.
— Use empty lists [] if none found — never omit keys.
— 'reflection' should note patterns, context-switches, or focus suggestions.
— 'people_mentioned' should include context in parentheses if available.
"""

_REFLECTION_USER = """\
Today is {date}.  Here are today's conversation session summaries:

{summaries}
{ambient_section}
Produce the daily reflection JSON.
"""

_ENTITY_SYSTEM = """\
You are an entity extraction assistant.  Given a daily reflection and a \
list of EXISTING entities in the knowledge graph, extract new entities and \
relationships.  MERGE with existing entities when possible (same person, \
same topic) — do NOT create duplicates.

EXISTING ENTITIES:
{existing_entities}

OUTPUT — valid JSON array, no markdown fences:
[
  {{"label": "Person|Topic|Decision|Task|Event|Location", "key_prop": "name|what|description|title", "value": "...", "context": "..."}}
]

RULES:
— Match names case-insensitively against EXISTING ENTITIES.
— If "Ali (study group)" exists and you see "Ali", use the existing entry.
— Only extract entities that are clearly mentioned.
— Return [] if no entities found.
"""

_ENTITY_USER = """\
Daily reflection for {date}:
{reflection_json}

Extract entities and relationships.
"""


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class ReflectionResult:
    """Structured output from nightly reflection."""
    date: str = ""
    sessions: int = 0
    total_messages: int = 0
    summary: str = ""
    decisions: list[str] = field(default_factory=list)
    tasks_created: list[str] = field(default_factory=list)
    tasks_completed: list[str] = field(default_factory=list)
    people_mentioned: list[str] = field(default_factory=list)
    reflection: str = ""
    unresolved: list[str] = field(default_factory=list)
    entities_extracted: int = 0
    raw_pruned: int = 0
    vectors_pruned: int = 0
    ambient_summary: str = ""  # #166: day-level ambient environment digest

    def to_dict(self) -> dict:
        d = {
            "date": self.date,
            "sessions": self.sessions,
            "total_messages": self.total_messages,
            "summary": self.summary,
            "decisions": self.decisions,
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "people_mentioned": self.people_mentioned,
            "reflection": self.reflection,
            "unresolved": self.unresolved,
            "entities_extracted": self.entities_extracted,
            "raw_pruned": self.raw_pruned,
            "vectors_pruned": self.vectors_pruned,
        }
        if self.ambient_summary:
            d["ambient_summary"] = self.ambient_summary
        return d

    def summary_line(self) -> str:
        """One-paragraph human-readable summary."""
        parts = [f"🤔 Reflection ({self.date}): {self.sessions} sessions, {self.total_messages} msgs"]
        if self.ambient_summary:
            parts.append(f"   🎤 {self.ambient_summary}")
        if self.summary:
            parts.append(f"   {self.summary}")
        if self.decisions:
            parts.append(f"   Decisions: {', '.join(self.decisions)}")
        if self.tasks_created:
            parts.append(f"   Tasks created: {', '.join(self.tasks_created)}")
        if self.tasks_completed:
            parts.append(f"   Tasks done: {', '.join(self.tasks_completed)}")
        if self.people_mentioned:
            parts.append(f"   People: {', '.join(self.people_mentioned)}")
        if self.reflection:
            parts.append(f"   💡 {self.reflection}")
        if self.unresolved:
            parts.append(f"   ❓ Unresolved: {', '.join(self.unresolved)}")
        if self.entities_extracted:
            parts.append(f"   🔗 {self.entities_extracted} entities → graph")
        if self.raw_pruned:
            parts.append(f"   🗑 Pruned {self.raw_pruned} old messages + {self.vectors_pruned} vectors")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _data_dir():
    from bantz.config import config
    return config.db_path.parent


def _get_pool():
    """Return the shared SQLite connection pool."""
    from bantz.data.connection_pool import get_pool
    return get_pool()


# ── Step 1: Collect today's session distillations ────────────────────────

def _collect_today_distillations(
    date_str: str,
) -> list[dict]:
    """
    Hierarchical summarisation (Recommendation #1):
    Read today's session distillation summaries from #118
    instead of raw messages — massive token savings.
    """
    pool = _get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT sd.conversation_id, sd.summary, sd.topics,
                      sd.decisions, sd.people, sd.tools_used,
                      sd.exchange_count, sd.created_at
               FROM session_distillations sd
               WHERE sd.created_at LIKE ?
               ORDER BY sd.created_at ASC""",
            (f"{date_str}%",),
        ).fetchall()
        return [dict(r) for r in rows]


def _collect_today_sessions_meta(
    date_str: str,
) -> tuple[int, int]:
    """Count today's sessions and total messages."""
    pool = _get_pool()
    with pool.connection() as conn:
        sessions = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE started_at LIKE ?",
            (f"{date_str}%",),
        ).fetchone()[0]
        messages = conn.execute(
            """SELECT COUNT(*) FROM messages m
               JOIN conversations c ON c.id = m.conversation_id
               WHERE c.started_at LIKE ?""",
            (f"{date_str}%",),
        ).fetchone()[0]
        return sessions, messages


def _collect_undistilled_sessions(
    date_str: str,
) -> list[dict]:
    """Fallback: if no distillations exist, fetch raw session summaries.

    Returns lightweight summaries — first/last user messages only.
    """
    pool = _get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT c.id as conversation_id,
                      (SELECT GROUP_CONCAT(content, ' | ')
                       FROM (SELECT content FROM messages
                             WHERE conversation_id = c.id AND role = 'user'
                             ORDER BY created_at LIMIT 3)) as user_preview,
                      COUNT(m.id) as msg_count
               FROM conversations c
               LEFT JOIN messages m ON m.conversation_id = c.id
               LEFT JOIN session_distillations sd ON sd.conversation_id = c.id
               WHERE c.started_at LIKE ? AND sd.id IS NULL
               GROUP BY c.id
               HAVING msg_count > 0
               ORDER BY c.started_at ASC""",
            (f"{date_str}%",),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Step 2: LLM daily reflection ────────────────────────────────────────

def _build_summaries_text(
    distillations: list[dict],
    undistilled: list[dict],
) -> str:
    """Build the summaries block for the reflection prompt."""
    parts = []
    for i, d in enumerate(distillations, 1):
        topics = d.get("topics", "")
        decisions = d.get("decisions", "")
        people = d.get("people", "")
        summary = d.get("summary", "")
        parts.append(
            f"Session {i} ({d.get('exchange_count', '?')} exchanges):\n"
            f"  Summary: {summary}\n"
            f"  Topics: {topics or 'n/a'}\n"
            f"  Decisions: {decisions or 'none'}\n"
            f"  People: {people or 'none'}"
        )

    for i, u in enumerate(undistilled, len(distillations) + 1):
        parts.append(
            f"Session {i} ({u.get('msg_count', '?')} messages, no distillation):\n"
            f"  Preview: {(u.get('user_preview') or 'empty')[:300]}"
        )

    return "\n\n".join(parts) if parts else "(No conversations today)"


async def _llm_reflect(date_str: str, summaries_text: str, ambient_text: str = "") -> str:
    """Call LLM to produce the daily reflection JSON."""
    # Build the optional ambient section (#166)
    ambient_section = ""
    if ambient_text:
        ambient_section = (
            "\n\nAmbient environment observations today:\n"
            f"{ambient_text}\n"
            "Consider the user's physical environment when writing the reflection.\n"
        )

    messages = [
        {"role": "system", "content": _REFLECTION_SYSTEM},
        {"role": "user", "content": _REFLECTION_USER.format(
            date=date_str, summaries=summaries_text,
            ambient_section=ambient_section,
        )},
    ]

    # Try Gemini first (fast)
    try:
        from bantz.llm.gemini import gemini
        if gemini.is_enabled():
            return await asyncio.wait_for(
                gemini.chat(messages, temperature=0.2),
                timeout=_LLM_TIMEOUT,
            )
    except asyncio.TimeoutError:
        log.warning("Gemini reflection timed out after %ds", _LLM_TIMEOUT)
    except Exception:
        pass

    # Ollama fallback
    from bantz.llm.ollama import ollama
    return await asyncio.wait_for(
        ollama.chat(messages),
        timeout=_LLM_TIMEOUT,
    )


def _parse_reflection_json(raw: str) -> dict:
    """Parse the LLM's JSON output, tolerant of markdown fences."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to find JSON object within the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: extract what we can
    log.warning("Could not parse reflection JSON, using raw text as summary")
    return {
        "summary": text[:500],
        "decisions": [],
        "tasks_created": [],
        "tasks_completed": [],
        "people_mentioned": [],
        "reflection": "",
        "unresolved": [],
    }


# ── Step 3: Entity extraction with graph context (Rec #4) ───────────────

async def _fetch_existing_entities_from_graph() -> str:
    """
    Entity Resolution (Recommendation #4):
    Fetch existing Person/Topic/Decision nodes from Neo4j
    so the LLM can merge instead of creating duplicates.
    """
    try:
        from bantz.memory.graph import graph_memory
        if not graph_memory.enabled:
            return "(Graph not available)"

        entities = []
        for label in ("Person", "Topic", "Decision", "Task"):
            try:
                rows = await graph_memory._query(
                    f"MATCH (n:{label}) RETURN n.name AS name "
                    f"ORDER BY n.last_seen DESC LIMIT 50"
                )
                for r in rows:
                    name = r.get("name")
                    if name:
                        entities.append(f"{label}: {name}")
            except Exception:
                # Some labels use different key props
                try:
                    key = {"Decision": "what", "Task": "description"}.get(label, "name")
                    rows = await graph_memory._query(
                        f"MATCH (n:{label}) RETURN n.{key} AS name "
                        f"ORDER BY n.created_at DESC LIMIT 30"
                    )
                    for r in rows:
                        name = r.get("name")
                        if name:
                            entities.append(f"{label}: {name}")
                except Exception:
                    pass

        return "\n".join(entities) if entities else "(Empty graph — this is the first run)"
    except (ImportError, Exception) as exc:
        log.debug("Could not fetch graph entities: %s", exc)
        return "(Graph not available)"


async def _llm_extract_entities(
    date_str: str,
    reflection_json: dict,
    existing_entities: str,
) -> list[dict]:
    """
    LLM-based entity extraction with graph context injection.
    Falls back to rule-based extraction if LLM fails.
    """
    messages = [
        {"role": "system", "content": _ENTITY_SYSTEM.format(
            existing_entities=existing_entities,
        )},
        {"role": "user", "content": _ENTITY_USER.format(
            date=date_str,
            reflection_json=json.dumps(reflection_json, ensure_ascii=False, indent=2),
        )},
    ]

    raw = None
    try:
        from bantz.llm.gemini import gemini
        if gemini.is_enabled():
            raw = await asyncio.wait_for(
                gemini.chat(messages, temperature=0.1),
                timeout=_LLM_TIMEOUT,
            )
    except Exception:
        pass

    if not raw:
        try:
            from bantz.llm.ollama import ollama
            raw = await asyncio.wait_for(
                ollama.chat(messages),
                timeout=_LLM_TIMEOUT,
            )
        except Exception as exc:
            log.debug("LLM entity extraction failed: %s", exc)

    if raw:
        entities = _parse_entity_json(raw)
        if entities:
            return entities

    # Fallback: rule-based extraction from the reflection summary
    return _rule_based_entity_extraction(reflection_json)


def _parse_entity_json(raw: str) -> list[dict]:
    """Parse entity extraction JSON from LLM."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def _rule_based_entity_extraction(reflection: dict) -> list[dict]:
    """Fallback rule-based extraction from reflection dict."""
    entities = []

    for person in reflection.get("people_mentioned", []):
        name = person.split("(")[0].strip()
        if name and len(name) > 1:
            entities.append({
                "label": "Person",
                "key_prop": "name",
                "value": name,
                "context": person,
            })

    for decision in reflection.get("decisions", []):
        if decision:
            entities.append({
                "label": "Decision",
                "key_prop": "what",
                "value": decision[:100],
                "context": "",
            })

    for task in reflection.get("tasks_created", []):
        if task:
            entities.append({
                "label": "Task",
                "key_prop": "description",
                "value": task[:100],
                "context": "created",
            })

    for task in reflection.get("tasks_completed", []):
        if task:
            entities.append({
                "label": "Task",
                "key_prop": "description",
                "value": task[:100],
                "context": "completed",
            })

    return entities


async def _store_entities_to_graph(entities: list[dict]) -> int:
    """Upsert extracted entities into Neo4j graph."""
    try:
        from bantz.memory.graph import graph_memory
        if not graph_memory.enabled:
            return 0

        stored = 0
        now = datetime.now().isoformat(timespec="seconds")

        for ent in entities:
            label = ent.get("label", "")
            key_prop = ent.get("key_prop", "name")
            value = ent.get("value", "")
            context = ent.get("context", "")

            if not label or not value:
                continue

            # Build entity dict compatible with graph_memory._upsert_entities
            graph_ent = {
                "label": label,
                "key": key_prop,
                "props": {
                    key_prop: value,
                    "last_seen": now,
                    "source": "reflection",
                },
                "rels": [],
            }
            if context:
                graph_ent["props"]["context"] = context

            try:
                await graph_memory._upsert_entities([graph_ent], now)
                stored += 1
            except Exception as exc:
                log.debug("Graph upsert failed for %s: %s", value, exc)

        return stored
    except (ImportError, Exception) as exc:
        log.debug("Graph entity storage failed: %s", exc)
        return 0


# ── Step 4: Store reflection ────────────────────────────────────────────

async def _store_reflection(
    result: ReflectionResult,
) -> None:
    """Store reflection in KV store, vector DB, and memory."""
    # KV store for --reflections and morning briefing
    try:
        from bantz.data.sqlite_store import SQLiteKVStore
        kv = SQLiteKVStore(_data_dir() / "bantz.db")
        kv.set(f"reflection_{result.date}", json.dumps(result.to_dict(), ensure_ascii=False))
        kv.set("reflection_latest", json.dumps(result.to_dict(), ensure_ascii=False))
        kv.set("reflection_latest_date", result.date)
    except Exception as exc:
        log.warning("Failed to store reflection in KV: %s", exc)

    # Embed reflection summary into vector space
    try:
        from bantz.config import config
        if config.embedding_enabled:
            from bantz.memory.embeddings import embedder
            embed_text = f"Daily reflection {result.date}: {result.summary} {result.reflection}"
            vec = await embedder.embed(embed_text)
            if vec:
                from bantz.memory.distiller import store_distillation, DistillationResult
                # Store as a special "reflection" distillation (conversation_id = -date_hash)
                # Use negative ID to distinguish from session distillations
                date_hash = abs(hash(result.date)) % 10_000_000
                dr = DistillationResult(
                    session_id=-date_hash,
                    summary=f"[Daily Reflection] {result.summary}",
                    topics=[],
                    decisions=result.decisions,
                    people=result.people_mentioned,
                    tools_used=[],
                    exchange_count=result.total_messages,
                )
                try:
                    store_distillation(
                        -date_hash, dr,
                        embedding=vec,
                        embed_model=embedder.model,
                    )
                except Exception:
                    # Might conflict on session_id — update instead
                    pass
    except Exception as exc:
        log.debug("Reflection embedding failed: %s", exc)

    # Log to memory
    try:
        from bantz.core.memory import memory
        if memory._initialized:
            memory.add("assistant", result.summary_line(), tool_used="reflection")
    except Exception:
        pass


# ── Step 5: Prune old raw messages (Rec #3: vector orphan cleanup) ───────

def _prune_old_messages(
    keep_days: int = _PRUNE_RAW_DAYS,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Delete raw messages older than keep_days, KEEPING distilled summaries.
    Also cleans up orphaned vector embeddings (Recommendation #3).

    Returns (messages_deleted, vectors_deleted).
    """
    cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
    pool = _get_pool()

    # Find message IDs that will be deleted
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT m.id FROM messages m
               JOIN conversations c ON c.id = m.conversation_id
               WHERE c.last_active < ?
                 AND c.id IN (
                    SELECT conversation_id FROM session_distillations
                 )""",
            (cutoff,),
        ).fetchall()
    msg_ids = [r[0] if isinstance(r, (tuple, list)) else r["id"] for r in rows]

    if dry_run:
        return len(msg_ids), len(msg_ids)  # estimate vectors = messages

    if not msg_ids:
        return 0, 0

    # Delete vector embeddings FIRST (Rec #3: no ghost vectors)
    vectors_deleted = 0
    try:
        placeholders = ",".join("?" * len(msg_ids))
        with pool.connection(write=True) as conn:
            cur = conn.execute(
                f"DELETE FROM message_vectors WHERE message_id IN ({placeholders})",
                msg_ids,
            )
            vectors_deleted = cur.rowcount
    except Exception as exc:
        log.debug("Vector pruning error: %s", exc)

    # Then delete the messages
    messages_deleted = 0
    try:
        placeholders = ",".join("?" * len(msg_ids))
        with pool.connection(write=True) as conn:
            cur = conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                msg_ids,
            )
            messages_deleted = cur.rowcount
    except Exception as exc:
        log.debug("Message pruning error: %s", exc)

    # Clean up empty conversations
    try:
        with pool.connection(write=True) as conn:
            conn.execute(
                """DELETE FROM conversations
                   WHERE last_active < ?
                     AND id NOT IN (SELECT DISTINCT conversation_id FROM messages)""",
                (cutoff,),
            )
    except Exception:
        pass

    log.info("Pruned %d messages + %d vectors (older than %d days)",
             messages_deleted, vectors_deleted, keep_days)
    return messages_deleted, vectors_deleted


# ── Step 6: Report ──────────────────────────────────────────────────────

async def _send_report(result: ReflectionResult, dry_run: bool) -> None:
    """Desktop notification + Telegram summary."""
    tag = " (dry-run)" if dry_run else ""
    summary = result.summary_line()

    # Desktop notification
    try:
        from bantz.agent.notifier import notifier
        if notifier.enabled:
            one_line = (
                f"🤔 Reflection{tag}: "
                f"{result.sessions} sessions, "
                f"{result.total_messages} msgs"
            )
            notifier.send(one_line, urgency="normal", expire_ms=10_000)
    except Exception:
        pass

    # Telegram
    try:
        from bantz.config import config
        if config.telegram_bot_token and config.telegram_allowed_users:
            import httpx
            users = [u.strip() for u in config.telegram_allowed_users.split(",") if u.strip()]
            for uid in users:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(
                            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                            json={"chat_id": uid, "text": summary, "parse_mode": ""},
                        )
                except Exception:
                    pass
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# View past reflections
# ═══════════════════════════════════════════════════════════════════════════

def list_reflections(limit: int = 10) -> list[dict]:
    """List recent daily reflections from KV store."""
    try:
        from bantz.data.sqlite_store import SQLiteKVStore
        kv = SQLiteKVStore(_data_dir() / "bantz.db")
        all_keys = kv.all()
        reflection_keys = sorted(
            [k for k in all_keys if k.startswith("reflection_2")],
            reverse=True,
        )[:limit]
        results = []
        for key in reflection_keys:
            try:
                data = json.loads(kv.get(key, "{}"))
                results.append(data)
            except Exception:
                pass
        return results
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Main entrypoint
# ═══════════════════════════════════════════════════════════════════════════

async def run_reflection(
    *,
    dry_run: bool = False,
    date_override: str = "",
) -> ReflectionResult:
    """Execute the nightly reflection workflow.

    Args:
        dry_run: If True, print what would be done without LLM calls/writes.
        date_override: ISO date string (YYYY-MM-DD) to reflect on. Defaults to today.

    Returns:
        ReflectionResult with the day's reflection and metadata.
    """
    from bantz.agent.job_scheduler import inhibit_sleep

    date_str = date_override or datetime.now().strftime("%Y-%m-%d")
    tag = " (DRY-RUN)" if dry_run else ""
    log.info("🤔 Reflection starting for %s%s...", date_str, tag)

    result = ReflectionResult(date=date_str)

    with inhibit_sleep("Bantz nightly reflection"):
        deadline = time.monotonic() + _TOTAL_TIMEOUT

        # ── 1. Collect today's data ──────────────────────────────────────
        sessions, messages = _collect_today_sessions_meta(date_str)
        result.sessions = sessions
        result.total_messages = messages

        if sessions == 0:
            result.summary = "No conversations today."
            result.reflection = "Quiet day — no interactions recorded."
            log.info("🤔 Reflection: 0 sessions for %s, skipping LLM", date_str)
            # Still store the empty reflection
            if not dry_run:
                await _store_reflection(result)
            await _send_report(result, dry_run)
            return result

        # ── 1b. Ambient environment digest (#166) ────────────────────
        ambient_text = ""
        try:
            from bantz.agent.ambient import ambient_analyzer
            amb_summary = ambient_analyzer.day_summary()
            if amb_summary and "No ambient" not in amb_summary:
                ambient_text = amb_summary
                result.ambient_summary = amb_summary
                log.info("Reflection: %s", amb_summary)
        except Exception:
            pass

        # ── 2. Hierarchical summarisation (Rec #1) ──────────────────────
        distillations = _collect_today_distillations(date_str)
        undistilled = _collect_undistilled_sessions(date_str)
        summaries_text = _build_summaries_text(distillations, undistilled)

        log.info("Reflection: %d distillations + %d undistilled for %s",
                 len(distillations), len(undistilled), date_str)

        if dry_run:
            result.summary = f"[DRY-RUN] Would reflect on {sessions} sessions ({messages} messages)"
            result.reflection = f"Summaries text:\n{summaries_text[:500]}"
            await _send_report(result, dry_run)
            return result

        # ── 3. LLM reflection ───────────────────────────────────────────
        if time.monotonic() > deadline:
            result.summary = "Timed out before LLM call"
            return result

        try:
            raw = await _llm_reflect(date_str, summaries_text, ambient_text)
            parsed = _parse_reflection_json(raw)

            result.summary = parsed.get("summary", "")
            result.decisions = parsed.get("decisions", [])
            result.tasks_created = parsed.get("tasks_created", [])
            result.tasks_completed = parsed.get("tasks_completed", [])
            result.people_mentioned = parsed.get("people_mentioned", [])
            result.reflection = parsed.get("reflection", "")
            result.unresolved = parsed.get("unresolved", [])
        except asyncio.TimeoutError:
            log.warning("Reflection LLM timed out")
            result.summary = "LLM timed out during reflection"
        except Exception as exc:
            log.warning("Reflection LLM failed: %s", exc)
            result.summary = f"LLM error: {exc}"

        # ── 4. Entity extraction with graph context (Rec #4) ─────────────
        if time.monotonic() < deadline:
            try:
                existing = await _fetch_existing_entities_from_graph()
                entities = await _llm_extract_entities(
                    date_str, result.to_dict(), existing,
                )
                if entities:
                    stored = await _store_entities_to_graph(entities)
                    result.entities_extracted = stored
            except Exception as exc:
                log.debug("Entity extraction failed: %s", exc)

                # Fallback to rule-based
                fallback = _rule_based_entity_extraction(result.to_dict())
                if fallback:
                    stored = await _store_entities_to_graph(fallback)
                    result.entities_extracted = stored

        # ── 5. Store reflection ──────────────────────────────────────────
        if time.monotonic() < deadline:
            await _store_reflection(result)

        # ── 6. Prune old raw messages + vectors (Rec #3) ────────────────
        if time.monotonic() < deadline:
            try:
                result.raw_pruned, result.vectors_pruned = _prune_old_messages(
                    keep_days=_PRUNE_RAW_DAYS,
                )
            except Exception as exc:
                log.debug("Pruning failed: %s", exc)

        # ── 7. Report ───────────────────────────────────────────────────
        await _send_report(result, dry_run)

    log.info("🤔 Reflection complete for %s: %s", date_str,
             result.summary_line().split("\n")[0])
    return result
