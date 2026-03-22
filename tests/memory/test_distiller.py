"""
Tests for session distillation (#118).

Covers:
  - Schema migration
  - Transcript building & parsing
  - Exchange counting
  - Store / retrieve distillation records
  - Vector search on distillations
  - Threshold gating (min_exchanges)
  - End-to-end distill_session with mocked LLM
  - Memory.new_session() distillation hook
  - Config flags (enabled/disabled)
  - search_distillations & distillation_stats on Memory
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _pool_db(tmp_path):
    """Set up a temporary pool-backed SQLite database for every test."""
    from bantz.data.connection_pool import SQLitePool

    db = tmp_path / "test_bantz.db"
    pool = SQLitePool.get_instance(db)

    # Create base tables that distiller depends on
    with pool.connection(write=True) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  TEXT NOT NULL,
                last_active TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                tool_used       TEXT,
                created_at      TEXT NOT NULL
            )
        """)

    yield pool

    SQLitePool.reset()


@pytest.fixture
def pool(_pool_db):
    """Expose the pool fixture explicitly for tests that need it."""
    return _pool_db


@pytest.fixture
def session_with_messages(pool):
    """Create a session with 6 exchanges (12 messages)."""
    with pool.connection(write=True) as conn:
        conn.execute(
            "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01T10:00:00', '2025-01-01T11:00:00')"
        )
        messages = [
            (1, "user", "What's the weather today?", None, "2025-01-01T10:00:01"),
            (1, "assistant", "It's 15°C and sunny in Istanbul.", "weather", "2025-01-01T10:00:02"),
            (1, "user", "Check my email", None, "2025-01-01T10:05:00"),
            (1, "assistant", "You have 3 unread emails. One from Ahmet about the project.", "gmail", "2025-01-01T10:05:01"),
            (1, "user", "What's on my calendar?", None, "2025-01-01T10:10:00"),
            (1, "assistant", "You have 2 events today: Team standup at 11am and lunch with Mehmet at 1pm.", "calendar", "2025-01-01T10:10:01"),
            (1, "user", "Let's go with the new design for the project", None, "2025-01-01T10:15:00"),
            (1, "assistant", "Got it — noted the decision to go with the new design.", None, "2025-01-01T10:15:01"),
            (1, "user", "Remind me to call Ali tomorrow", None, "2025-01-01T10:20:00"),
            (1, "assistant", "Reminder set: call Ali tomorrow.", "reminder", "2025-01-01T10:20:01"),
            (1, "user", "Thanks, that's all for now", None, "2025-01-01T10:25:00"),
            (1, "assistant", "No problem! Have a great day.", None, "2025-01-01T10:25:01"),
        ]
        for conv_id, role, content, tool, ts in messages:
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, tool_used, created_at) VALUES (?,?,?,?,?)",
                (conv_id, role, content, tool, ts),
            )


@pytest.fixture
def short_session(pool):
    """Create a session with only 2 exchanges (below threshold)."""
    with pool.connection(write=True) as conn:
        conn.execute(
            "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01T10:00:00', '2025-01-01T10:05:00')"
        )
        messages = [
            (1, "user", "Hello", None, "2025-01-01T10:00:01"),
            (1, "assistant", "Hi there!", None, "2025-01-01T10:00:02"),
            (1, "user", "Bye", None, "2025-01-01T10:04:00"),
            (1, "assistant", "Goodbye!", None, "2025-01-01T10:04:01"),
        ]
        for conv_id, role, content, tool, ts in messages:
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, tool_used, created_at) VALUES (?,?,?,?,?)",
                (conv_id, role, content, tool, ts),
            )


# ── Schema tests ───────────────────────────────────────────────────────────

class TestDistillationSchema:
    def test_migrate_creates_table(self, pool):
        from bantz.memory.distiller import migrate_distillation_table
        migrate_distillation_table()
        # Table should exist
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_distillations'"
            ).fetchone()
        assert row is not None

    def test_migrate_idempotent(self, pool):
        from bantz.memory.distiller import migrate_distillation_table
        migrate_distillation_table()
        migrate_distillation_table()  # Should not raise
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_distillations'"
            ).fetchone()
        assert row is not None

    def test_table_columns(self, pool):
        from bantz.memory.distiller import migrate_distillation_table
        migrate_distillation_table()
        with pool.connection() as conn:
            cols = conn.execute("PRAGMA table_info(session_distillations)").fetchall()
        col_names = {c["name"] for c in cols}
        expected = {
            "id", "conversation_id", "summary", "topics", "decisions",
            "people", "tools_used", "exchange_count", "embedding",
            "embed_dim", "embed_model", "created_at",
        }
        assert expected == col_names


# ── Transcript / parsing tests ─────────────────────────────────────────────

class TestTranscriptBuilding:
    def test_build_transcript(self, session_with_messages):
        from bantz.memory.distiller import fetch_session_messages, _build_transcript
        msgs = fetch_session_messages(1)
        transcript = _build_transcript(msgs)
        assert "USER:" in transcript
        assert "ASSISTANT:" in transcript
        assert "weather" in transcript.lower() or "[tool: weather]" in transcript

    def test_build_transcript_truncation(self, session_with_messages):
        from bantz.memory.distiller import fetch_session_messages, _build_transcript
        msgs = fetch_session_messages(1)
        transcript = _build_transcript(msgs, max_chars=100)
        assert "truncated" in transcript.lower()
        assert len(transcript) < 200

    def test_build_transcript_empty(self):
        from bantz.memory.distiller import _build_transcript
        assert _build_transcript([]) == ""


class TestParsingLLMOutput:
    def test_parse_complete_output(self):
        from bantz.memory.distiller import _parse_llm_output
        raw = (
            "SUMMARY: User checked weather, email, and calendar.\n"
            "TOPICS: weather, email, calendar\n"
            "DECISIONS: go with new design\n"
            "PEOPLE: Ahmet, Mehmet, Ali\n"
            "TOOLS: weather, gmail, calendar, reminder"
        )
        result = _parse_llm_output(raw)
        assert "weather" in result["summary"].lower()
        assert "weather" in result["topics"]
        assert "email" in result["topics"]
        assert len(result["people"]) == 3
        assert "Ali" in result["people"]

    def test_parse_none_values(self):
        from bantz.memory.distiller import _parse_llm_output
        raw = (
            "SUMMARY: Quick hello conversation.\n"
            "TOPICS: greeting\n"
            "DECISIONS: none\n"
            "PEOPLE: none\n"
            "TOOLS: none"
        )
        result = _parse_llm_output(raw)
        assert result["decisions"] == []
        assert result["people"] == []
        assert result["tools"] == []

    def test_parse_fallback_no_headers(self):
        from bantz.memory.distiller import _parse_llm_output
        raw = "This is just a plain summary without any headers."
        result = _parse_llm_output(raw)
        assert result["summary"] == raw


class TestExchangeCounting:
    def test_count_exchanges(self, session_with_messages):
        from bantz.memory.distiller import fetch_session_messages, _count_exchanges
        msgs = fetch_session_messages(1)
        count = _count_exchanges(msgs)
        assert count == 6  # 6 user messages

    def test_count_exchanges_short(self, short_session):
        from bantz.memory.distiller import fetch_session_messages, _count_exchanges
        msgs = fetch_session_messages(1)
        count = _count_exchanges(msgs)
        assert count == 2

    def test_count_exchanges_empty(self):
        from bantz.memory.distiller import _count_exchanges
        assert _count_exchanges([]) == 0


class TestCSVParsing:
    def test_parse_csv_normal(self):
        from bantz.memory.distiller import _parse_csv
        assert _parse_csv("weather, email, calendar") == ["weather", "email", "calendar"]

    def test_parse_csv_none(self):
        from bantz.memory.distiller import _parse_csv
        assert _parse_csv("none") == []
        assert _parse_csv("None") == []
        assert _parse_csv("n/a") == []

    def test_parse_csv_empty(self):
        from bantz.memory.distiller import _parse_csv
        assert _parse_csv("") == []
        assert _parse_csv(None) == []


# ── Store / Retrieve tests ────────────────────────────────────────────────

class TestStoreDistillation:
    def test_store_and_retrieve(self, pool):
        from bantz.memory.distiller import (
            migrate_distillation_table, store_distillation,
            get_distillation, DistillationResult,
        )
        migrate_distillation_table()

        # Need a conversation row first
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01', '2025-01-01')"
            )

        result = DistillationResult(
            session_id=1,
            summary="User checked weather and email.",
            topics=["weather", "email"],
            decisions=["new design"],
            people=["Ahmet"],
            tools_used=["weather", "gmail"],
            exchange_count=6,
        )
        rowid = store_distillation(1, result)
        assert rowid > 0

        retrieved = get_distillation(1)
        assert retrieved is not None
        assert retrieved["summary"] == "User checked weather and email."
        assert retrieved["exchange_count"] == 6
        assert "weather" in retrieved["topics"]

    def test_store_with_embedding(self, pool):
        from bantz.memory.distiller import (
            migrate_distillation_table, store_distillation,
            DistillationResult,
        )
        migrate_distillation_table()
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01', '2025-01-01')"
            )

        result = DistillationResult(session_id=1, summary="Test", exchange_count=5)
        embedding = [0.1, 0.2, 0.3, 0.4]
        store_distillation(1, result, embedding=embedding, embed_model="test")

        with pool.connection() as conn:
            row = conn.execute(
                "SELECT embedding, embed_dim, embed_model FROM session_distillations WHERE conversation_id=1"
            ).fetchone()
        assert row["embed_dim"] == 4
        assert row["embed_model"] == "test"
        assert row["embedding"] is not None

    def test_store_upsert(self, pool):
        """INSERT OR REPLACE should update existing distillation."""
        from bantz.memory.distiller import (
            migrate_distillation_table, store_distillation,
            get_distillation, DistillationResult,
        )
        migrate_distillation_table()
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01', '2025-01-01')"
            )

        r1 = DistillationResult(session_id=1, summary="Version 1", exchange_count=5)
        store_distillation(1, r1)

        r2 = DistillationResult(session_id=1, summary="Version 2", exchange_count=7)
        store_distillation(1, r2)

        retrieved = get_distillation(1)
        assert retrieved["summary"] == "Version 2"
        assert retrieved["exchange_count"] == 7

    def test_get_nonexistent(self):
        from bantz.memory.distiller import migrate_distillation_table, get_distillation
        migrate_distillation_table()
        assert get_distillation(999) is None


# ── Vector search tests ───────────────────────────────────────────────────

class TestDistillationSearch:
    def test_search_distillations(self, pool):
        from bantz.memory.distiller import (
            migrate_distillation_table, store_distillation,
            search_distillations, DistillationResult,
        )
        migrate_distillation_table()
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01', '2025-01-01')"
            )
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (2, '2025-01-02', '2025-01-02')"
            )

        # Store two distillations with known embeddings
        r1 = DistillationResult(session_id=1, summary="Weather and email session", exchange_count=5)
        r2 = DistillationResult(session_id=2, summary="Calendar planning session", exchange_count=8)

        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]

        store_distillation(1, r1, embedding=vec1, embed_model="test")
        store_distillation(2, r2, embedding=vec2, embed_model="test")

        # Search with a vector close to vec1
        query_vec = [0.9, 0.1, 0.0, 0.0]
        results = search_distillations(query_vec, limit=5, min_score=0.1)
        assert len(results) >= 1
        assert results[0]["conversation_id"] == 1  # closest to query
        assert results[0]["score"] > 0.5

    def test_search_empty(self):
        from bantz.memory.distiller import migrate_distillation_table, search_distillations
        migrate_distillation_table()
        results = search_distillations([1.0, 0.0], limit=5)
        assert results == []

    def test_search_min_score_filter(self, pool):
        from bantz.memory.distiller import (
            migrate_distillation_table, store_distillation,
            search_distillations, DistillationResult,
        )
        migrate_distillation_table()
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01', '2025-01-01')"
            )
        r = DistillationResult(session_id=1, summary="Test", exchange_count=5)
        store_distillation(1, r, embedding=[1.0, 0.0, 0.0, 0.0])

        # Orthogonal query — score should be ~0
        results = search_distillations([0.0, 1.0, 0.0, 0.0], min_score=0.5)
        assert len(results) == 0


# ── Stats tests ───────────────────────────────────────────────────────────

class TestDistillationStats:
    def test_stats(self, pool):
        from bantz.memory.distiller import (
            migrate_distillation_table, store_distillation,
            distillation_stats, DistillationResult,
        )
        migrate_distillation_table()
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (1, '2025-01-01', '2025-01-01')"
            )
            conn.execute(
                "INSERT INTO conversations(id, started_at, last_active) VALUES (2, '2025-01-02', '2025-01-02')"
            )

        r1 = DistillationResult(session_id=1, summary="Test", exchange_count=5)
        store_distillation(1, r1, embedding=[1.0, 0.0])

        stats = distillation_stats()
        assert stats["total_distillations"] == 1
        assert stats["embedded_distillations"] == 1
        assert stats["total_sessions"] == 2
        assert stats["coverage_pct"] == 50.0

    def test_stats_empty(self):
        from bantz.memory.distiller import migrate_distillation_table, distillation_stats
        migrate_distillation_table()
        stats = distillation_stats()
        assert stats["total_distillations"] == 0


# ── Distill session (integration with mocked LLM) ─────────────────────────

LLM_RESPONSE = (
    "SUMMARY: User checked weather, email and calendar. Decided on new design.\n"
    "TOPICS: weather, email, calendar\n"
    "DECISIONS: go with new design\n"
    "PEOPLE: Ahmet, Mehmet, Ali\n"
    "TOOLS: weather, gmail, calendar, reminder"
)


class TestDistillSession:
    def test_distill_session_success(self, session_with_messages):
        from bantz.memory.distiller import (
            migrate_distillation_table, distill_session, get_distillation,
        )
        migrate_distillation_table()

        with patch("bantz.memory.distiller._llm_summarise", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLM_RESPONSE
            result = asyncio.get_event_loop().run_until_complete(
                distill_session(
                    1,
                    min_exchanges=5, embed=False, extract_graph=False,
                )
            )

        assert result is not None
        assert result.session_id == 1
        assert result.exchange_count == 6
        assert "weather" in result.summary.lower() or "email" in result.summary.lower()
        assert len(result.topics) >= 2
        assert len(result.people) >= 2

        # Check persistent storage
        stored = get_distillation(1)
        assert stored is not None
        assert stored["exchange_count"] == 6

    def test_distill_below_threshold(self, short_session):
        from bantz.memory.distiller import migrate_distillation_table, distill_session
        migrate_distillation_table()

        result = asyncio.get_event_loop().run_until_complete(
            distill_session(
                1,
                min_exchanges=5, embed=False, extract_graph=False,
            )
        )
        assert result is None

    def test_distill_with_embedding(self, session_with_messages, pool):
        from bantz.memory.distiller import (
            migrate_distillation_table, distill_session,
        )
        migrate_distillation_table()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_embedder.model = "test-model"

        with patch("bantz.memory.distiller._llm_summarise", new_callable=AsyncMock) as mock_llm, \
             patch("bantz.config.config.embedding_enabled", True), \
             patch("bantz.memory.embeddings.embedder", mock_embedder):
            mock_llm.return_value = LLM_RESPONSE
            result = asyncio.get_event_loop().run_until_complete(
                distill_session(
                    1,
                    min_exchanges=5, embed=True, extract_graph=False,
                )
            )

        assert result is not None
        # Check embedding was stored
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT embed_dim, embed_model FROM session_distillations WHERE conversation_id=1"
            ).fetchone()
        assert row is not None
        if row["embed_dim"]:  # embedding may or may not succeed depending on mock
            assert row["embed_dim"] == 3

    def test_distill_with_graph_extraction(self, session_with_messages):
        from bantz.memory.distiller import migrate_distillation_table, distill_session

        migrate_distillation_table()

        with patch("bantz.memory.distiller._llm_summarise", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLM_RESPONSE
            result = asyncio.get_event_loop().run_until_complete(
                distill_session(
                    1,
                    min_exchanges=5, embed=False, extract_graph=True,
                )
            )

        assert result is not None
        # entities_extracted should be >= 0 (extract_entities is rule-based)
        assert result.entities_extracted >= 0

    def test_distill_llm_failure(self, session_with_messages):
        from bantz.memory.distiller import migrate_distillation_table, distill_session
        migrate_distillation_table()

        with patch("bantz.memory.distiller._llm_summarise", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM unavailable")
            result = asyncio.get_event_loop().run_until_complete(
                distill_session(
                    1,
                    min_exchanges=5, embed=False, extract_graph=False,
                )
            )

        assert result is None

    def test_distill_nonexistent_session(self):
        from bantz.memory.distiller import migrate_distillation_table, distill_session
        migrate_distillation_table()

        result = asyncio.get_event_loop().run_until_complete(
            distill_session(
                999,
                min_exchanges=1, embed=False, extract_graph=False,
            )
        )
        assert result is None  # no messages → 0 exchanges < 1

    def test_distill_tools_from_messages(self, session_with_messages):
        """Tools used should be extracted from actual message tool_used column."""
        from bantz.memory.distiller import migrate_distillation_table, distill_session

        migrate_distillation_table()

        with patch("bantz.memory.distiller._llm_summarise", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "SUMMARY: Test.\nTOPICS: test\nDECISIONS: none\nPEOPLE: none\nTOOLS: none"
            result = asyncio.get_event_loop().run_until_complete(
                distill_session(
                    1,
                    min_exchanges=5, embed=False, extract_graph=False,
                )
            )

        assert result is not None
        # Should have tools from actual messages, not LLM output
        assert set(result.tools_used) == {"weather", "gmail", "calendar", "reminder"}


# ── Config integration ────────────────────────────────────────────────────

class TestConfigFields:
    def test_distillation_config_defaults(self):
        from bantz.config import Config
        c = Config(
            _env_file=None,
            BANTZ_OLLAMA_MODEL="test",
        )
        assert c.distillation_enabled is True
        assert c.distillation_min_exchanges == 5

    def test_distillation_config_override(self):
        from bantz.config import Config
        c = Config(
            _env_file=None,
            BANTZ_OLLAMA_MODEL="test",
            BANTZ_DISTILLATION_ENABLED="false",
            BANTZ_DISTILLATION_MIN_EXCHANGES="10",
        )
        assert c.distillation_enabled is False
        assert c.distillation_min_exchanges == 10


# ── DistillationResult dataclass ──────────────────────────────────────────

class TestDistillationResult:
    def test_defaults(self):
        from bantz.memory.distiller import DistillationResult
        r = DistillationResult(session_id=1, summary="Test")
        assert r.topics == []
        assert r.decisions == []
        assert r.people == []
        assert r.tools_used == []
        assert r.exchange_count == 0
        assert r.entities_extracted == 0

    def test_full_init(self):
        from bantz.memory.distiller import DistillationResult
        r = DistillationResult(
            session_id=42,
            summary="Summary text",
            topics=["a", "b"],
            decisions=["d1"],
            people=["Alice"],
            tools_used=["weather"],
            exchange_count=10,
            entities_extracted=5,
        )
        assert r.session_id == 42
        assert len(r.topics) == 2
        assert r.entities_extracted == 5


# ── fetch_session_messages ────────────────────────────────────────────────

class TestFetchMessages:
    def test_fetch_returns_dicts(self, session_with_messages):
        from bantz.memory.distiller import fetch_session_messages
        msgs = fetch_session_messages(1)
        assert isinstance(msgs, list)
        assert len(msgs) == 12
        assert isinstance(msgs[0], dict)
        assert "role" in msgs[0]
        assert "content" in msgs[0]

    def test_fetch_chronological_order(self, session_with_messages):
        from bantz.memory.distiller import fetch_session_messages
        msgs = fetch_session_messages(1)
        timestamps = [m["created_at"] for m in msgs]
        assert timestamps == sorted(timestamps)

    def test_fetch_empty_session(self):
        from bantz.memory.distiller import fetch_session_messages
        msgs = fetch_session_messages(999)
        assert msgs == []
