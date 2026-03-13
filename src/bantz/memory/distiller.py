"""
Bantz v3 — Session Distiller (#118)

Automatic session distillation into long-term semantic memory.
When a new session starts, the previous session is summarized via LLM
and stored as a vector-searchable distillation record.

Pipeline:
  session ends → fetch messages → LLM summarise → embed summary →
  store in session_distillations → optionally extract graph entities

Usage:
    from bantz.memory.distiller import distill_session
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("bantz.distiller")


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class DistillationResult:
    """Result of distilling a session."""
    session_id: int
    summary: str
    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    exchange_count: int = 0
    entities_extracted: int = 0


# ── Prompt ─────────────────────────────────────────────────────────────────

DISTILL_SYSTEM = """\
You are a precise note-taking assistant. Given a conversation transcript, \
produce a structured summary.

OUTPUT FORMAT (use exactly these headers):
SUMMARY: 2-3 sentence overview of what was discussed and accomplished.
TOPICS: comma-separated list of main topics (e.g. email, calendar, weather)
DECISIONS: comma-separated list of decisions made (or "none")
PEOPLE: comma-separated list of people mentioned by name (or "none")
TOOLS: comma-separated list of tools/services used (or "none")

RULES:
- Be factual — only include what actually appears in the conversation.
- Keep the summary concise — max 3 sentences.
- If no decisions were made, write "none".
- If no people were mentioned, write "none".
"""

DISTILL_USER = """\
Summarise this conversation session:\n\n{transcript}\
"""


# ── Schema ─────────────────────────────────────────────────────────────────

def migrate_distillation_table() -> None:
    """Create the session_distillations table if it doesn't exist."""
    sql = """\
CREATE TABLE IF NOT EXISTS session_distillations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL UNIQUE,
    summary         TEXT    NOT NULL,
    topics          TEXT    NOT NULL DEFAULT '',
    decisions       TEXT    NOT NULL DEFAULT '',
    people          TEXT    NOT NULL DEFAULT '',
    tools_used      TEXT    NOT NULL DEFAULT '',
    exchange_count  INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,
    embed_dim       INTEGER,
    embed_model     TEXT,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
)"""
    from bantz.data.connection_pool import get_pool
    with get_pool().connection(write=True) as conn:
        conn.execute(sql)


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_transcript(messages: list[dict], max_chars: int = 6000) -> str:
    """Build a readable transcript from message dicts."""
    lines: list[str] = []
    total = 0
    for m in messages:
        role = m.get("role", "?").upper()
        content = m.get("content", "")
        tool = m.get("tool_used")
        line = f"{role}: {content}"
        if tool:
            line += f" [tool: {tool}]"
        if total + len(line) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _count_exchanges(messages: list[dict]) -> int:
    """Count user↔assistant exchange pairs."""
    user_count = sum(1 for m in messages if m.get("role") == "user")
    return user_count


def _parse_llm_output(raw: str) -> dict:
    """Parse the structured LLM output into a dict."""
    result = {
        "summary": "",
        "topics": [],
        "decisions": [],
        "people": [],
        "tools": [],
    }

    for line in raw.strip().split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
        elif upper.startswith("TOPICS:"):
            val = line.split(":", 1)[1].strip()
            result["topics"] = _parse_csv(val)
        elif upper.startswith("DECISIONS:"):
            val = line.split(":", 1)[1].strip()
            result["decisions"] = _parse_csv(val)
        elif upper.startswith("PEOPLE:"):
            val = line.split(":", 1)[1].strip()
            result["people"] = _parse_csv(val)
        elif upper.startswith("TOOLS:"):
            val = line.split(":", 1)[1].strip()
            result["tools"] = _parse_csv(val)

    # If no SUMMARY header found, use the whole text as summary
    if not result["summary"] and raw.strip():
        result["summary"] = raw.strip()[:500]

    return result


def _parse_csv(val: str) -> list[str]:
    """Parse a comma-separated string, filtering 'none'."""
    if not val or val.lower().strip() in ("none", "n/a", "-"):
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


# ── Core distillation ─────────────────────────────────────────────────────

def fetch_session_messages(
    session_id: int,
) -> list[dict]:
    """Fetch all messages for a session, in chronological order."""
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        rows = conn.execute(
            """SELECT role, content, tool_used, created_at
               FROM messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


async def _llm_summarise(transcript: str) -> str:
    """Call LLM to summarise a transcript. Tries Gemini then Ollama."""
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM},
        {"role": "user", "content": DISTILL_USER.format(transcript=transcript)},
    ]

    # Try Gemini first (fast, cheap)
    try:
        from bantz.llm.gemini import gemini
        if gemini.is_enabled():
            return await gemini.chat(messages, temperature=0.2)
    except Exception:
        pass

    # Ollama fallback
    from bantz.llm.ollama import ollama
    return await ollama.chat(messages)


def store_distillation(
    session_id: int,
    result: DistillationResult,
    embedding: Optional[list[float]] = None,
    embed_model: str = "",
) -> int:
    """Store a distillation record. Returns the rowid."""
    import struct

    now = datetime.now().isoformat(timespec="seconds")
    blob = None
    dim = None
    if embedding:
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        dim = len(embedding)

    from bantz.data.connection_pool import get_pool
    with get_pool().connection(write=True) as conn:
        cur = conn.execute(
            """INSERT OR REPLACE INTO session_distillations
               (conversation_id, summary, topics, decisions, people,
                tools_used, exchange_count, embedding, embed_dim,
                embed_model, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                result.summary,
                ",".join(result.topics),
                ",".join(result.decisions),
                ",".join(result.people),
                ",".join(result.tools_used),
                result.exchange_count,
                blob,
                dim,
                embed_model or None,
                now,
            ),
        )
        return cur.lastrowid


async def distill_session(
    session_id: int,
    *,
    min_exchanges: int = 5,
    embed: bool = True,
    extract_graph: bool = True,
) -> Optional[DistillationResult]:
    """
    Distill a completed session into long-term semantic memory.

    1. Fetch session messages
    2. Check min_exchanges threshold
    3. LLM summarisation
    4. Embed the summary into vector space
    5. Store in session_distillations table
    6. Optionally extract entities for graph memory

    Returns DistillationResult or None if session is too short.
    """
    # 1. Fetch messages
    messages = fetch_session_messages(session_id)
    exchanges = _count_exchanges(messages)

    if exchanges < min_exchanges:
        log.debug(
            "Session %d has %d exchanges (< %d), skipping distillation",
            session_id, exchanges, min_exchanges,
        )
        return None

    # 2. Build transcript & summarise
    transcript = _build_transcript(messages)
    try:
        raw = await _llm_summarise(transcript)
    except Exception as exc:
        log.warning("Distillation LLM failed for session %d: %s", session_id, exc)
        return None

    parsed = _parse_llm_output(raw)

    # Collect tools from actual messages
    actual_tools = list({
        m["tool_used"] for m in messages
        if m.get("tool_used")
    })

    result = DistillationResult(
        session_id=session_id,
        summary=parsed["summary"],
        topics=parsed["topics"],
        decisions=parsed["decisions"],
        people=parsed["people"],
        tools_used=actual_tools or parsed["tools"],
        exchange_count=exchanges,
    )

    # 3. Embed the summary
    embedding = None
    embed_model = ""
    if embed:
        try:
            from bantz.config import config
            if config.embedding_enabled:
                from bantz.memory.embeddings import embedder
                embedding = await embedder.embed(result.summary)
                embed_model = embedder.model
        except Exception as exc:
            log.debug("Distillation embedding failed: %s", exc)

    # 4. Store
    try:
        store_distillation(
            session_id, result,
            embedding=embedding, embed_model=embed_model,
        )
        log.info(
            "Distilled session %d → %d chars, %d topics, %d decisions",
            session_id, len(result.summary),
            len(result.topics), len(result.decisions),
        )
    except Exception as exc:
        log.warning("Failed to store distillation for session %d: %s", session_id, exc)
        return None

    # 5. Graph entity extraction (fire-and-forget)
    if extract_graph:
        try:
            from bantz.memory.nodes import extract_entities
            # Summarise the conversation into a single user/assistant pair
            user_parts = " ".join(
                m["content"] for m in messages if m.get("role") == "user"
            )[:500]
            entities = extract_entities(
                user_msg=user_parts,
                assistant_msg=result.summary,
                tool_used=actual_tools[0] if actual_tools else None,
                tool_data=None,
            )
            result.entities_extracted = len(entities)

            # Store in graph if available
            try:
                from bantz.memory.graph import graph_memory
                if graph_memory and graph_memory.enabled:
                    for ent in entities:
                        await graph_memory.merge_entity(ent)
            except (ImportError, Exception):
                pass  # graph is optional, entities still counted

        except Exception as exc:
            log.debug("Distillation entity extraction failed: %s", exc)

    return result


# ── Vector search on distillations ─────────────────────────────────────────

def search_distillations(
    query_vec: list[float],
    limit: int = 5,
    min_score: float = 0.3,
) -> list[dict]:
    """Search distillation summaries by vector similarity."""
    import struct

    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        rows = conn.execute(
            """SELECT id, conversation_id, summary, topics, decisions,
                      people, tools_used, exchange_count, embedding,
                      embed_dim, created_at
               FROM session_distillations
               WHERE embedding IS NOT NULL"""
        ).fetchall()

    from bantz.memory.vector_store import _cosine_similarity, _blob_to_vec

    scored: list[tuple[float, dict]] = []
    for row in rows:
        vec = _blob_to_vec(row["embedding"], row["embed_dim"])
        score = _cosine_similarity(query_vec, vec)
        if score >= min_score:
            scored.append((score, {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "summary": row["summary"],
                "topics": row["topics"],
                "decisions": row["decisions"],
                "people": row["people"],
                "tools_used": row["tools_used"],
                "exchange_count": row["exchange_count"],
                "score": round(score, 4),
                "created_at": row["created_at"],
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def get_distillation(
    session_id: int,
) -> Optional[dict]:
    """Get the distillation record for a specific session."""
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        row = conn.execute(
            """SELECT id, conversation_id, summary, topics, decisions,
                      people, tools_used, exchange_count, created_at
               FROM session_distillations
               WHERE conversation_id = ?""",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


def distillation_stats() -> dict:
    """Get distillation statistics."""
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM session_distillations"
        ).fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM session_distillations WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        total_sessions = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        return {
            "total_distillations": total,
            "embedded_distillations": embedded,
            "total_sessions": total_sessions,
            "coverage_pct": round(total / max(total_sessions, 1) * 100, 1),
        }
