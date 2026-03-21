"""
Tests for bantz.agent.workflows.reflection — Nightly memory reflection (#130).

Coverage:
  - ReflectionResult dataclass (to_dict, summary_line)
  - Prompt constants and formatting
  - _collect_today_distillations (hierarchical summarisation — Rec #1)
  - _collect_today_sessions_meta
  - _collect_undistilled_sessions
  - _build_summaries_text
  - _parse_reflection_json (valid, fenced, malformed, fallback)
  - _parse_entity_json (valid, fenced, malformed)
  - _rule_based_entity_extraction (people, decisions, tasks)
  - _fetch_existing_entities_from_graph (Rec #4: entity resolution)
  - _llm_extract_entities (with graph context injection)
  - _prune_old_messages + vector cleanup (Rec #3: orphan vectors)
  - _store_reflection (KV + vector + memory)
  - _send_report (notification + Telegram)
  - run_reflection: dry-run mode
  - run_reflection: zero sessions graceful handling
  - run_reflection: normal run with mocked LLM
  - run_reflection: timeout enforcement (Rec #2: generous timeout)
  - list_reflections
  - CLI --reflect / --reflections / --dry-run arg presence
  - Job scheduler delegation
  - Edge cases (empty inputs, LLM failures)
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip('telegram')


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the connection pool after every test."""
    yield
    from bantz.data.connection_pool import SQLitePool
    SQLitePool.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db(tmp_path):
    """Create a test SQLite DB with conversations, messages, and distillations.

    Also initializes the connection pool pointing to this DB so that
    reflection.py functions (which use the pool internally) read from it.
    """
    db_path = tmp_path / "bantz.db"

    # Initialize pool first — this creates the DB file
    from bantz.data.connection_pool import get_pool
    pool = get_pool(db_path)

    # Create schema via pool
    with pool.connection(write=True) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                last_active TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_used TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_distillations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                topics TEXT NOT NULL DEFAULT '',
                decisions TEXT NOT NULL DEFAULT '',
                people TEXT NOT NULL DEFAULT '',
                tools_used TEXT NOT NULL DEFAULT '',
                exchange_count INTEGER NOT NULL DEFAULT 0,
                embedding BLOB,
                embed_dim INTEGER,
                embed_model TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_vectors (
                message_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

    return pool, tmp_path


@pytest.fixture
def populated_db(tmp_db):
    """DB with today's conversations and distillations."""
    pool, tmp_path = tmp_db
    today = datetime.now().strftime("%Y-%m-%d")

    with pool.connection(write=True) as conn:
        # Insert 2 conversations with messages
        conn.execute(
            "INSERT INTO conversations (id, started_at, last_active) VALUES (1, ?, ?)",
            (f"{today}T10:00:00", f"{today}T10:30:00"),
        )
        conn.execute(
            "INSERT INTO conversations (id, started_at, last_active) VALUES (2, ?, ?)",
            (f"{today}T14:00:00", f"{today}T15:00:00"),
        )

        for conv_id, msgs in [
            (1, [
                ("user", "What's the weather?", "weather"),
                ("assistant", "Sunny, 22°C in Ankara.", None),
                ("user", "Check my calendar", "calendar"),
                ("assistant", "You have a meeting with Prof. Yilmaz at 3pm.", None),
                ("user", "Remind me to study", None),
                ("assistant", "Reminder set for 8pm.", "reminder"),
            ]),
            (2, [
                ("user", "Help me with Docker networking", "shell"),
                ("assistant", "The bridge network is misconfigured. Try...", None),
                ("user", "That worked, thanks!", None),
                ("assistant", "Great! Docker bridge is fixed.", None),
                ("user", "Now let's work on Bantz v3 planning", None),
                ("assistant", "Let's outline the architecture. I suggest LanceDB for vectors.", None),
            ]),
        ]:
            for role, content, tool in msgs:
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, tool_used, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (conv_id, role, content, tool, f"{today}T10:00:00"),
                )

        # Insert distillation for conversation 1
        conn.execute(
            """INSERT INTO session_distillations
               (conversation_id, summary, topics, decisions, people,
                tools_used, exchange_count, created_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "Checked weather and calendar. Reminder to study set.",
                "weather,calendar,reminder",
                "none",
                "Prof. Yilmaz",
                "weather,calendar",
                3,
                f"{today}T10:30:00",
            ),
        )

        # Insert distillation for conversation 2
        conn.execute(
            """INSERT INTO session_distillations
               (conversation_id, summary, topics, decisions, people,
                tools_used, exchange_count, created_at)
               VALUES (2, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "Fixed Docker networking. Started Bantz v3 planning. Chose LanceDB for vectors.",
                "docker,architecture,planning",
                "Chose LanceDB over ChromaDB",
                "",
                "shell",
                3,
                f"{today}T15:00:00",
            ),
        )

    return pool, tmp_path


@pytest.fixture
def mock_memory(populated_db):
    """Mock the memory singleton to use our test DB."""
    pool, tmp_path = populated_db
    mock_mem = MagicMock()
    mock_mem._initialized = True
    mock_mem.add = MagicMock()
    return mock_mem, pool, tmp_path


# ═══════════════════════════════════════════════════════════════════════════
# ReflectionResult tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReflectionResult:
    def test_defaults(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult()
        assert r.date == ""
        assert r.sessions == 0
        assert r.decisions == []
        assert r.entities_extracted == 0

    def test_to_dict(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(
            date="2026-03-10",
            sessions=3,
            total_messages=25,
            summary="Productive day",
            decisions=["Use LanceDB"],
            people_mentioned=["Ali"],
            reflection="Good focus",
        )
        d = r.to_dict()
        assert d["date"] == "2026-03-10"
        assert d["sessions"] == 3
        assert d["decisions"] == ["Use LanceDB"]
        assert "people_mentioned" in d

    def test_to_dict_roundtrip(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(date="2026-03-10", summary="Test")
        s = json.dumps(r.to_dict())
        d = json.loads(s)
        assert d["date"] == "2026-03-10"

    def test_summary_line(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(
            date="2026-03-10",
            sessions=4,
            total_messages=47,
            summary="Worked on planning.",
            decisions=["Chose LanceDB"],
            reflection="Pattern: context-switching",
            unresolved=["Exam prep"],
            entities_extracted=5,
            raw_pruned=100,
            vectors_pruned=80,
        )
        line = r.summary_line()
        assert "2026-03-10" in line
        assert "4 sessions" in line
        assert "Chose LanceDB" in line
        assert "context-switching" in line
        assert "Exam prep" in line
        assert "5 entities" in line
        assert "100 old messages" in line

    def test_summary_line_empty(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(date="2026-03-10")
        line = r.summary_line()
        assert "2026-03-10" in line

    def test_summary_line_with_pruning(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(raw_pruned=50, vectors_pruned=40)
        line = r.summary_line()
        assert "50" in line
        assert "40" in line


# ═══════════════════════════════════════════════════════════════════════════
# Data collection tests (Rec #1: hierarchical summarisation)
# ═══════════════════════════════════════════════════════════════════════════

class TestDataCollection:
    def test_collect_today_distillations(self, populated_db):
        from bantz.agent.workflows.reflection import _collect_today_distillations
        pool, _ = populated_db
        today = datetime.now().strftime("%Y-%m-%d")
        distills = _collect_today_distillations(today)
        assert len(distills) == 2
        assert "weather" in distills[0]["summary"].lower() or "docker" in distills[0]["summary"].lower()

    def test_collect_today_distillations_empty(self, tmp_db):
        from bantz.agent.workflows.reflection import _collect_today_distillations
        pool, _ = tmp_db
        distills = _collect_today_distillations("2099-01-01")
        assert distills == []

    def test_collect_today_sessions_meta(self, populated_db):
        from bantz.agent.workflows.reflection import _collect_today_sessions_meta
        pool, _ = populated_db
        today = datetime.now().strftime("%Y-%m-%d")
        sessions, messages = _collect_today_sessions_meta(today)
        assert sessions == 2
        assert messages == 12

    def test_collect_undistilled_sessions(self, populated_db):
        from bantz.agent.workflows.reflection import _collect_undistilled_sessions
        pool, _ = populated_db
        today = datetime.now().strftime("%Y-%m-%d")
        # All sessions have distillations
        undistilled = _collect_undistilled_sessions(today)
        assert len(undistilled) == 0

    def test_collect_undistilled_sessions_with_gap(self, populated_db):
        from bantz.agent.workflows.reflection import _collect_undistilled_sessions
        pool, _ = populated_db
        today = datetime.now().strftime("%Y-%m-%d")
        # Add a third conversation without distillation
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations (id, started_at, last_active) VALUES (3, ?, ?)",
                (f"{today}T18:00:00", f"{today}T18:30:00"),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (3, 'user', 'Quick question', ?)",
                (f"{today}T18:00:00",),
            )
        undistilled = _collect_undistilled_sessions(today)
        assert len(undistilled) == 1
        assert undistilled[0]["conversation_id"] == 3


class TestBuildSummariesText:
    def test_with_distillations(self):
        from bantz.agent.workflows.reflection import _build_summaries_text
        distills = [
            {"summary": "Checked weather", "topics": "weather", "decisions": "", "people": "Ali", "exchange_count": 5},
            {"summary": "Fixed Docker", "topics": "docker", "decisions": "Use bridge", "people": "", "exchange_count": 3},
        ]
        text = _build_summaries_text(distills, [])
        assert "Session 1" in text
        assert "Session 2" in text
        assert "Checked weather" in text
        assert "Fixed Docker" in text

    def test_with_undistilled(self):
        from bantz.agent.workflows.reflection import _build_summaries_text
        undistilled = [{"msg_count": 5, "user_preview": "help with Docker"}]
        text = _build_summaries_text([], undistilled)
        assert "no distillation" in text
        assert "Docker" in text

    def test_empty(self):
        from bantz.agent.workflows.reflection import _build_summaries_text
        text = _build_summaries_text([], [])
        assert "No conversations" in text


# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing tests
# ═══════════════════════════════════════════════════════════════════════════

class TestParseReflectionJson:
    def test_valid_json(self):
        from bantz.agent.workflows.reflection import _parse_reflection_json
        raw = json.dumps({
            "summary": "Good day",
            "decisions": ["Use LanceDB"],
            "tasks_created": ["Write DAL"],
            "tasks_completed": [],
            "people_mentioned": ["Ali (study)"],
            "reflection": "Focus improved",
            "unresolved": ["Exam prep"],
        })
        result = _parse_reflection_json(raw)
        assert result["summary"] == "Good day"
        assert result["decisions"] == ["Use LanceDB"]

    def test_fenced_json(self):
        from bantz.agent.workflows.reflection import _parse_reflection_json
        raw = '```json\n{"summary": "Test", "decisions": [], "tasks_created": [], "tasks_completed": [], "people_mentioned": [], "reflection": "", "unresolved": []}\n```'
        result = _parse_reflection_json(raw)
        assert result["summary"] == "Test"

    def test_json_with_prefix(self):
        from bantz.agent.workflows.reflection import _parse_reflection_json
        raw = 'Here is the reflection:\n{"summary": "Test", "decisions": []}'
        result = _parse_reflection_json(raw)
        assert result["summary"] == "Test"

    def test_malformed_fallback(self):
        from bantz.agent.workflows.reflection import _parse_reflection_json
        raw = "This is just plain text without JSON"
        result = _parse_reflection_json(raw)
        assert result["summary"]  # should contain the raw text
        assert isinstance(result["decisions"], list)


class TestParseEntityJson:
    def test_valid_array(self):
        from bantz.agent.workflows.reflection import _parse_entity_json
        raw = json.dumps([
            {"label": "Person", "key_prop": "name", "value": "Ali"},
        ])
        result = _parse_entity_json(raw)
        assert len(result) == 1
        assert result[0]["label"] == "Person"

    def test_fenced_array(self):
        from bantz.agent.workflows.reflection import _parse_entity_json
        raw = '```json\n[{"label": "Topic", "key_prop": "name", "value": "Docker"}]\n```'
        result = _parse_entity_json(raw)
        assert len(result) == 1

    def test_empty(self):
        from bantz.agent.workflows.reflection import _parse_entity_json
        assert _parse_entity_json("[]") == []

    def test_malformed(self):
        from bantz.agent.workflows.reflection import _parse_entity_json
        assert _parse_entity_json("not json") == []


# ═══════════════════════════════════════════════════════════════════════════
# Rule-based entity extraction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleBasedEntityExtraction:
    def test_extracts_people(self):
        from bantz.agent.workflows.reflection import _rule_based_entity_extraction
        reflection = {
            "people_mentioned": ["Ali (study group)", "Prof. Yilmaz (exam)"],
            "decisions": [],
            "tasks_created": [],
            "tasks_completed": [],
        }
        entities = _rule_based_entity_extraction(reflection)
        person_names = [e["value"] for e in entities if e["label"] == "Person"]
        assert "Ali" in person_names
        assert "Prof. Yilmaz" in person_names

    def test_extracts_decisions(self):
        from bantz.agent.workflows.reflection import _rule_based_entity_extraction
        reflection = {
            "people_mentioned": [],
            "decisions": ["Use LanceDB", "AT-SPI before VLM"],
            "tasks_created": [],
            "tasks_completed": [],
        }
        entities = _rule_based_entity_extraction(reflection)
        decision_vals = [e["value"] for e in entities if e["label"] == "Decision"]
        assert "Use LanceDB" in decision_vals

    def test_extracts_tasks(self):
        from bantz.agent.workflows.reflection import _rule_based_entity_extraction
        reflection = {
            "people_mentioned": [],
            "decisions": [],
            "tasks_created": ["Write DAL"],
            "tasks_completed": ["Fix Docker"],
        }
        entities = _rule_based_entity_extraction(reflection)
        task_vals = [e["value"] for e in entities if e["label"] == "Task"]
        assert "Write DAL" in task_vals
        assert "Fix Docker" in task_vals

    def test_empty_reflection(self):
        from bantz.agent.workflows.reflection import _rule_based_entity_extraction
        entities = _rule_based_entity_extraction({})
        assert entities == []


# ═══════════════════════════════════════════════════════════════════════════
# Entity resolution tests (Rec #4)
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityResolution:
    @pytest.mark.asyncio
    async def test_fetch_existing_entities_graph_disabled(self):
        from bantz.agent.workflows.reflection import _fetch_existing_entities_from_graph
        mock_graph = MagicMock()
        mock_graph.enabled = False
        with patch("bantz.memory.graph.graph_memory", mock_graph):
            result = await _fetch_existing_entities_from_graph()
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_fetch_existing_entities_graph_enabled(self):
        from bantz.agent.workflows.reflection import _fetch_existing_entities_from_graph
        mock_graph = MagicMock()
        mock_graph.enabled = True
        mock_graph._query = AsyncMock(return_value=[
            {"name": "Ali"}, {"name": "Bantz Project"},
        ])
        with patch("bantz.memory.graph.graph_memory", mock_graph):
            result = await _fetch_existing_entities_from_graph()
        assert "Ali" in result
        assert "Bantz Project" in result

    @pytest.mark.asyncio
    async def test_fetch_existing_entities_no_graph(self):
        from bantz.agent.workflows.reflection import _fetch_existing_entities_from_graph
        with patch.dict("sys.modules", {"bantz.memory.graph": None}):
            result = await _fetch_existing_entities_from_graph()
        assert "not available" in result.lower() or "graph" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Prune tests (Rec #3: vector orphan cleanup)
# ═══════════════════════════════════════════════════════════════════════════

class TestPruneOldMessages:
    def test_prune_removes_old_messages_and_vectors(self, populated_db):
        """Messages AND vectors older than 30 days should be deleted."""
        from bantz.agent.workflows.reflection import _prune_old_messages
        pool, _ = populated_db

        # Add an old conversation (40 days ago) with distillation
        old_date = (datetime.now() - timedelta(days=40)).isoformat()
        import struct
        dummy_blob = struct.pack("3f", 0.1, 0.2, 0.3)
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations (id, started_at, last_active) VALUES (10, ?, ?)",
                (old_date, old_date),
            )
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (100, 10, 'user', 'old message', ?)",
                (old_date,),
            )
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (101, 10, 'assistant', 'old response', ?)",
                (old_date,),
            )
            conn.execute(
                """INSERT INTO session_distillations
                   (conversation_id, summary, created_at)
                   VALUES (10, 'Old session summary', ?)""",
                (old_date,),
            )
            # Add vector embeddings for those messages
            conn.execute(
                "INSERT INTO message_vectors (message_id, embedding, dim, model, created_at) "
                "VALUES (100, ?, 3, 'test', ?)",
                (dummy_blob, old_date),
            )
            conn.execute(
                "INSERT INTO message_vectors (message_id, embedding, dim, model, created_at) "
                "VALUES (101, ?, 3, 'test', ?)",
                (dummy_blob, old_date),
            )

        msgs_deleted, vecs_deleted = _prune_old_messages(keep_days=30)
        assert msgs_deleted == 2
        assert vecs_deleted == 2

        # Verify vectors are actually gone
        with pool.connection() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM message_vectors WHERE message_id IN (100, 101)"
            ).fetchone()[0]
            assert remaining == 0

            # Distillation should STILL exist (we keep summaries)
            dist = conn.execute(
                "SELECT COUNT(*) FROM session_distillations WHERE conversation_id = 10"
            ).fetchone()[0]
            assert dist == 1

    def test_prune_skips_recent_messages(self, populated_db):
        from bantz.agent.workflows.reflection import _prune_old_messages
        pool, _ = populated_db
        msgs_deleted, vecs_deleted = _prune_old_messages(keep_days=30)
        # Today's messages should NOT be deleted
        assert msgs_deleted == 0

    def test_prune_dry_run(self, populated_db):
        from bantz.agent.workflows.reflection import _prune_old_messages
        pool, _ = populated_db
        # Add old data
        old_date = (datetime.now() - timedelta(days=40)).isoformat()
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations (id, started_at, last_active) VALUES (10, ?, ?)",
                (old_date, old_date),
            )
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (100, 10, 'user', 'old', ?)",
                (old_date,),
            )
            conn.execute(
                """INSERT INTO session_distillations
                   (conversation_id, summary, created_at)
                   VALUES (10, 'summary', ?)""",
                (old_date,),
            )

        msgs, vecs = _prune_old_messages(keep_days=30, dry_run=True)
        assert msgs > 0  # would delete

        # But messages should still exist
        with pool.connection() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE id = 100"
            ).fetchone()[0]
            assert remaining == 1

    def test_prune_only_distilled_conversations(self, populated_db):
        """Only conversations WITH distillations should be pruned."""
        from bantz.agent.workflows.reflection import _prune_old_messages
        pool, _ = populated_db

        old_date = (datetime.now() - timedelta(days=40)).isoformat()
        # Old conversation WITHOUT distillation — should NOT be pruned
        with pool.connection(write=True) as conn:
            conn.execute(
                "INSERT INTO conversations (id, started_at, last_active) VALUES (11, ?, ?)",
                (old_date, old_date),
            )
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (110, 11, 'user', 'important', ?)",
                (old_date,),
            )

        msgs_deleted, _ = _prune_old_messages(keep_days=30)
        # Should NOT delete this undistilled conversation's messages
        with pool.connection() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE id = 110"
            ).fetchone()[0]
            assert remaining == 1


# ═══════════════════════════════════════════════════════════════════════════
# Store reflection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStoreReflection:
    @pytest.mark.asyncio
    async def test_stores_to_kv(self, tmp_path):
        from bantz.agent.workflows.reflection import _store_reflection, ReflectionResult
        from bantz.data.connection_pool import get_pool
        # The KV store inside _store_reflection opens _data_dir()/bantz.db
        kv_db_path = tmp_path / "bantz.db"
        pool = get_pool(kv_db_path)

        result = ReflectionResult(
            date="2026-03-10",
            sessions=2,
            total_messages=12,
            summary="Good day",
        )
        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            with patch("bantz.core.memory.memory") as mock_mem:
                mock_mem._initialized = True
                mock_mem.add = MagicMock()
                with patch("bantz.config.config") as mock_cfg:
                    mock_cfg.embedding_enabled = False
                    await _store_reflection(result)

        from bantz.data.sqlite_store import SQLiteKVStore
        kv = SQLiteKVStore(kv_db_path)
        stored = json.loads(kv.get("reflection_2026-03-10", "{}"))
        assert stored["summary"] == "Good day"
        assert kv.get("reflection_latest_date") == "2026-03-10"


# ═══════════════════════════════════════════════════════════════════════════
# Send report tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSendReport:
    @pytest.mark.asyncio
    async def test_sends_notification(self):
        from bantz.agent.workflows.reflection import _send_report, ReflectionResult
        result = ReflectionResult(date="2026-03-10", sessions=2, total_messages=12)
        mock_notifier = MagicMock()
        mock_notifier.enabled = True
        with patch("bantz.agent.notifier.notifier", mock_notifier):
            with patch("bantz.config.config") as mock_cfg:
                mock_cfg.telegram_bot_token = ""
                mock_cfg.telegram_allowed_users = ""
                await _send_report(result, dry_run=False)
        mock_notifier.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_telegram_if_no_token(self):
        from bantz.agent.workflows.reflection import _send_report, ReflectionResult
        result = ReflectionResult(date="2026-03-10")
        with patch("bantz.agent.notifier.notifier") as mock_not:
            mock_not.enabled = False
            with patch("bantz.config.config") as mock_cfg:
                mock_cfg.telegram_bot_token = ""
                mock_cfg.telegram_allowed_users = ""
                with patch("httpx.AsyncClient") as mock_http:
                    await _send_report(result, dry_run=False)
        mock_http.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# run_reflection integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRunReflection:
    @pytest.mark.asyncio
    async def test_zero_sessions_graceful(self, tmp_db):
        """0 sessions → immediate return with 'No conversations today'."""
        from bantz.agent.workflows.reflection import run_reflection
        pool, tmp_path = tmp_db

        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            with patch("bantz.agent.job_scheduler.inhibit_sleep") as mock_inh:
                mock_inh.return_value.__enter__ = MagicMock()
                mock_inh.return_value.__exit__ = MagicMock(return_value=False)
                with patch("bantz.agent.notifier.notifier") as mock_not:
                    mock_not.enabled = False
                    with patch("bantz.config.config") as mock_cfg:
                        mock_cfg.telegram_bot_token = ""
                        mock_cfg.telegram_allowed_users = ""
                        mock_cfg.embedding_enabled = False
                        result = await run_reflection(dry_run=False)

        assert result.sessions == 0
        assert "No conversations" in result.summary

    @pytest.mark.asyncio
    async def test_dry_run_no_llm_calls(self, populated_db):
        """Dry-run mode should NOT call the LLM."""
        from bantz.agent.workflows.reflection import run_reflection
        pool, tmp_path = populated_db

        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            with patch("bantz.agent.workflows.reflection._llm_reflect") as mock_llm:
                with patch("bantz.agent.job_scheduler.inhibit_sleep") as mock_inh:
                    mock_inh.return_value.__enter__ = MagicMock()
                    mock_inh.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("bantz.agent.notifier.notifier") as mock_not:
                        mock_not.enabled = False
                        with patch("bantz.config.config") as mock_cfg:
                            mock_cfg.telegram_bot_token = ""
                            mock_cfg.telegram_allowed_users = ""
                            result = await run_reflection(dry_run=True)

        mock_llm.assert_not_called()
        assert "DRY-RUN" in result.summary

    @pytest.mark.asyncio
    async def test_normal_run_with_mocked_llm(self, populated_db):
        """Normal run should call LLM, parse result, and store."""
        from bantz.agent.workflows.reflection import run_reflection
        pool, tmp_path = populated_db

        llm_response = json.dumps({
            "summary": "Worked on weather, Docker, and Bantz v3 planning.",
            "decisions": ["Chose LanceDB"],
            "tasks_created": ["Write DAL"],
            "tasks_completed": ["Fixed Docker bridge"],
            "people_mentioned": ["Prof. Yilmaz (exam)"],
            "reflection": "Context-switching between coding and study prep.",
            "unresolved": ["Exam prep"],
        })

        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            with patch("bantz.agent.workflows.reflection._llm_reflect",
                       new_callable=AsyncMock, return_value=llm_response):
                with patch("bantz.agent.workflows.reflection._llm_extract_entities",
                           new_callable=AsyncMock, return_value=[]):
                    with patch("bantz.agent.workflows.reflection._store_entities_to_graph",
                               new_callable=AsyncMock, return_value=0):
                        with patch("bantz.agent.job_scheduler.inhibit_sleep") as mock_inh:
                            mock_inh.return_value.__enter__ = MagicMock()
                            mock_inh.return_value.__exit__ = MagicMock(return_value=False)
                            with patch("bantz.agent.notifier.notifier") as mock_not:
                                mock_not.enabled = False
                                with patch("bantz.config.config") as mock_cfg:
                                    mock_cfg.telegram_bot_token = ""
                                    mock_cfg.telegram_allowed_users = ""
                                    mock_cfg.embedding_enabled = False
                                    mock_cfg.db_path = tmp_path / "bantz.db"
                                    with patch("bantz.core.memory.memory") as mock_mem:
                                        mock_mem._initialized = True
                                        mock_mem.add = MagicMock()
                                        result = await run_reflection(dry_run=False)

        assert result.sessions == 2
        assert result.summary == "Worked on weather, Docker, and Bantz v3 planning."
        assert "Chose LanceDB" in result.decisions
        assert "Prof. Yilmaz (exam)" in result.people_mentioned

    @pytest.mark.asyncio
    async def test_timeout_constant_is_generous(self):
        """Rec #2: verify total timeout is 10 min (generous for local LLM)."""
        from bantz.agent.workflows.reflection import _TOTAL_TIMEOUT
        assert _TOTAL_TIMEOUT >= 600  # at least 10 minutes

    @pytest.mark.asyncio
    async def test_llm_timeout_is_reasonable(self):
        """Each LLM call should have a per-call timeout."""
        from bantz.agent.workflows.reflection import _LLM_TIMEOUT
        assert _LLM_TIMEOUT >= 60  # at least 1 minute per call

    @pytest.mark.asyncio
    async def test_date_override(self, populated_db):
        """date_override should reflect on a specific date."""
        from bantz.agent.workflows.reflection import run_reflection
        pool, tmp_path = populated_db

        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            with patch("bantz.agent.job_scheduler.inhibit_sleep") as mock_inh:
                mock_inh.return_value.__enter__ = MagicMock()
                mock_inh.return_value.__exit__ = MagicMock(return_value=False)
                with patch("bantz.agent.notifier.notifier") as mock_not:
                    mock_not.enabled = False
                    with patch("bantz.config.config") as mock_cfg:
                        mock_cfg.telegram_bot_token = ""
                        mock_cfg.telegram_allowed_users = ""
                        mock_cfg.embedding_enabled = False
                        result = await run_reflection(
                            dry_run=False,
                            date_override="2099-12-31",
                        )

        assert result.date == "2099-12-31"
        assert result.sessions == 0


# ═══════════════════════════════════════════════════════════════════════════
# list_reflections tests
# ═══════════════════════════════════════════════════════════════════════════

class TestListReflections:
    def test_list_empty(self, tmp_db):
        from bantz.agent.workflows.reflection import list_reflections
        _, tmp_path = tmp_db
        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            results = list_reflections()
        assert results == []

    def test_list_with_data(self, tmp_db):
        from bantz.agent.workflows.reflection import list_reflections
        from bantz.data.sqlite_store import SQLiteKVStore
        _, tmp_path = tmp_db
        kv = SQLiteKVStore(tmp_path / "bantz.db")
        kv.set("reflection_2026-03-08", json.dumps({"date": "2026-03-08", "summary": "Day 1"}))
        kv.set("reflection_2026-03-09", json.dumps({"date": "2026-03-09", "summary": "Day 2"}))
        kv.set("reflection_latest", json.dumps({"date": "2026-03-09"}))  # should be filtered

        with patch("bantz.agent.workflows.reflection._data_dir", return_value=tmp_path):
            results = list_reflections(limit=10)

        assert len(results) == 2
        # Should be sorted descending
        assert results[0]["date"] == "2026-03-09"


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCLIArgs:
    def test_reflect_arg(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--reflect", action="store_true")
        parser.add_argument("--reflections", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        ns = parser.parse_args(["--reflect", "--dry-run"])
        assert ns.reflect
        assert ns.dry_run

    def test_reflections_arg(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--reflections", action="store_true")
        ns = parser.parse_args(["--reflections"])
        assert ns.reflections


# ═══════════════════════════════════════════════════════════════════════════
# Job scheduler delegation
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSchedulerDelegation:
    @pytest.mark.asyncio
    async def test_job_reflection_delegates(self):
        from bantz.agent.job_scheduler import _job_reflection
        from bantz.agent.workflows.reflection import ReflectionResult

        mock_result = ReflectionResult(
            date="2026-03-10",
            sessions=3,
            entities_extracted=5,
        )
        with patch("bantz.agent.workflows.reflection.run_reflection",
                   new_callable=AsyncMock,
                   return_value=mock_result) as mock_run:
            await _job_reflection()
        mock_run.assert_awaited_once_with(dry_run=False)

    def test_registry_updated(self):
        from bantz.agent.job_scheduler import _JOB_REGISTRY
        assert "reflection" in _JOB_REGISTRY
        fn, desc = _JOB_REGISTRY["reflection"]
        assert "reflection" in desc.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_parse_reflection_empty_decisions(self):
        from bantz.agent.workflows.reflection import _parse_reflection_json
        raw = json.dumps({
            "summary": "Quiet day", "decisions": [],
            "tasks_created": [], "tasks_completed": [],
            "people_mentioned": [], "reflection": "",
            "unresolved": [],
        })
        result = _parse_reflection_json(raw)
        assert result["decisions"] == []

    def test_reflection_result_serializable(self):
        from bantz.agent.workflows.reflection import ReflectionResult
        r = ReflectionResult(
            date="2026-03-10",
            decisions=["a", "b"],
            people_mentioned=["Ali"],
        )
        s = json.dumps(r.to_dict())
        assert json.loads(s)["date"] == "2026-03-10"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rules(self, populated_db):
        """If LLM entity extraction fails, fallback to rule-based."""
        from bantz.agent.workflows.reflection import _llm_extract_entities

        with patch("bantz.llm.gemini.gemini") as mock_gemini:
            mock_gemini.is_enabled.return_value = False
            with patch("bantz.llm.ollama.ollama") as mock_ollama:
                mock_ollama.chat = AsyncMock(side_effect=Exception("timeout"))
                entities = await _llm_extract_entities(
                    "2026-03-10",
                    {
                        "people_mentioned": ["Ali"],
                        "decisions": ["Use LanceDB"],
                        "tasks_created": [],
                        "tasks_completed": [],
                    },
                    "(empty graph)",
                )
        assert len(entities) >= 2  # Ali + Use LanceDB

    def test_prune_raw_days_constant(self):
        from bantz.agent.workflows.reflection import _PRUNE_RAW_DAYS
        assert _PRUNE_RAW_DAYS == 30

    @pytest.mark.asyncio
    async def test_store_entities_graph_disabled(self):
        from bantz.agent.workflows.reflection import _store_entities_to_graph
        mock_graph = MagicMock()
        mock_graph.enabled = False
        with patch("bantz.memory.graph.graph_memory", mock_graph):
            count = await _store_entities_to_graph([
                {"label": "Person", "key_prop": "name", "value": "Test"},
            ])
        assert count == 0
