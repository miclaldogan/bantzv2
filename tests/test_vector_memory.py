"""
Tests for Issue #116 — Vector DB integration for semantic cross-trial memory.

Covers:
  - VectorStore: store, search, batch, stats, backfill helpers
  - Embedder: mock-based embed/embed_batch
  - Memory: semantic_search, hybrid_search, embed_pending, backfill
  - Config: new embedding fields
  - Cosine similarity math
"""
from __future__ import annotations

import asyncio
import math
import sqlite3
import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the Memory schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            last_active TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id),
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            tool_used TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at)
    """)
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE messages_fts
            USING fts5(content, content='messages', content_rowid='id')
        """)
        conn.execute("""
            CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
    except sqlite3.OperationalError:
        pass
    return conn


def _insert_message(conn, conv_id: int, role: str, content: str) -> int:
    """Insert a test message and return its id."""
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO messages(conversation_id, role, content, tool_used, created_at) VALUES (?,?,?,?,?)",
        (conv_id, role, content, None, now),
    )
    return cur.lastrowid


def _insert_conversation(conn) -> int:
    """Insert a test conversation and return its id."""
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO conversations(started_at, last_active) VALUES (?,?)",
        (now, now),
    )
    return cur.lastrowid


def _fake_embedding(text: str, dim: int = 8) -> list[float]:
    """Generate a deterministic pseudo-embedding from text hash."""
    h = hash(text)
    vec = []
    for i in range(dim):
        # Use bit manipulation for deterministic but varied values
        val = ((h >> (i * 4)) & 0xFF) / 255.0 - 0.5
        vec.append(val)
    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


# ══════════════════════════════════════════════════════════════════════════
# Test: Cosine Similarity
# ══════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from bantz.memory.vector_store import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from bantz.memory.vector_store import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from bantz.memory.vector_store import _cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        from bantz.memory.vector_store import _cosine_similarity
        a = [1.0, 1.0]
        b = [1.0, 0.9]
        sim = _cosine_similarity(a, b)
        assert sim > 0.99

    def test_zero_vector(self):
        from bantz.memory.vector_store import _cosine_similarity
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0


# ══════════════════════════════════════════════════════════════════════════
# Test: Vector Blob Encoding
# ══════════════════════════════════════════════════════════════════════════

class TestBlobEncoding:
    def test_roundtrip(self):
        from bantz.memory.vector_store import _vec_to_blob, _blob_to_vec
        original = [0.1, 0.2, -0.3, 0.4, 0.5]
        blob = _vec_to_blob(original)
        assert isinstance(blob, bytes)
        assert len(blob) == 5 * 4  # 5 floats × 4 bytes each

        restored = _blob_to_vec(blob, 5)
        for a, b in zip(original, restored):
            assert a == pytest.approx(b, abs=1e-6)

    def test_empty_vector(self):
        from bantz.memory.vector_store import _vec_to_blob, _blob_to_vec
        blob = _vec_to_blob([])
        assert blob == b""
        assert _blob_to_vec(blob, 0) == []


# ══════════════════════════════════════════════════════════════════════════
# Test: VectorStore
# ══════════════════════════════════════════════════════════════════════════

class TestVectorStore:
    def setup_method(self):
        self.conn = _make_memory_db()
        from bantz.memory.vector_store import VectorStore
        self.vs = VectorStore(self.conn)
        self.vs.migrate()
        self.conv_id = _insert_conversation(self.conn)

    def teardown_method(self):
        self.conn.close()

    def test_migrate_creates_table(self):
        tables = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vectors'"
        ).fetchone()
        assert tables is not None

    def test_store_and_count(self):
        msg_id = _insert_message(self.conn, self.conv_id, "user", "hello world")
        vec = _fake_embedding("hello world")
        self.vs.store(msg_id, vec, model="test-model")
        assert self.vs.count() == 1
        assert self.vs.has_embedding(msg_id)

    def test_store_overwrites(self):
        msg_id = _insert_message(self.conn, self.conv_id, "user", "test")
        vec1 = _fake_embedding("test1")
        vec2 = _fake_embedding("test2")
        self.vs.store(msg_id, vec1)
        self.vs.store(msg_id, vec2)
        assert self.vs.count() == 1  # overwrite, not duplicate

    def test_search_finds_similar(self):
        # Store messages with embeddings
        texts = [
            "how is the weather today",
            "it is sunny and warm",
            "remind me to buy groceries",
            "what time is my meeting",
        ]
        for text in texts:
            msg_id = _insert_message(self.conn, self.conv_id, "user", text)
            self.vs.store(msg_id, _fake_embedding(text))

        # Search with a similar query
        query_vec = _fake_embedding("weather forecast for tomorrow")
        results = self.vs.search(query_vec, limit=2, min_score=-1.0)
        assert len(results) <= 2
        assert all("message_id" in r for r in results)
        assert all("score" in r for r in results)
        assert all("content" in r for r in results)

    def test_search_respects_min_score(self):
        msg_id = _insert_message(self.conn, self.conv_id, "user", "some content")
        self.vs.store(msg_id, [1.0, 0.0, 0.0])

        # Orthogonal query — score should be 0
        results = self.vs.search([0.0, 1.0, 0.0], limit=5, min_score=0.5)
        assert len(results) == 0

    def test_search_empty_store(self):
        results = self.vs.search([1.0, 0.0], limit=5)
        assert results == []

    def test_store_batch(self):
        items = []
        for i in range(5):
            msg_id = _insert_message(self.conn, self.conv_id, "user", f"message {i}")
            items.append((msg_id, _fake_embedding(f"message {i}")))
        count = self.vs.store_batch(items)
        assert count == 5
        assert self.vs.count() == 5

    def test_unembedded_messages(self):
        # Insert messages but only embed some
        for i in range(5):
            msg_id = _insert_message(self.conn, self.conv_id, "user", f"long content message number {i}")
            if i < 2:  # Only embed first 2
                self.vs.store(msg_id, _fake_embedding(f"message {i}"))

        unembedded = self.vs.unembedded_messages()
        assert len(unembedded) == 3

    def test_prune_orphans(self):
        msg_id = _insert_message(self.conn, self.conv_id, "user", "real message")
        self.vs.store(msg_id, _fake_embedding("real"))

        # Store an embedding for a non-existent message
        self.vs.store(9999, _fake_embedding("orphan"))
        assert self.vs.count() == 2

        pruned = self.vs.prune_orphans()
        assert pruned == 1
        assert self.vs.count() == 1

    def test_stats(self):
        msg_id = _insert_message(self.conn, self.conv_id, "user", "test message for stats")
        _insert_message(self.conn, self.conv_id, "assistant", "reply for stats")
        self.vs.store(msg_id, _fake_embedding("test"))

        stats = self.vs.stats()
        assert stats["total_embeddings"] == 1
        assert stats["total_messages"] == 2
        assert stats["coverage_pct"] == 50.0

    def test_has_embedding_false(self):
        assert not self.vs.has_embedding(9999)


# ══════════════════════════════════════════════════════════════════════════
# Test: Embedder
# ══════════════════════════════════════════════════════════════════════════

class TestEmbedder:
    def test_embed_empty_returns_none(self):
        from bantz.memory.embeddings import Embedder
        emb = Embedder()
        result = asyncio.get_event_loop().run_until_complete(emb.embed(""))
        assert result is None

    def test_embed_whitespace_returns_none(self):
        from bantz.memory.embeddings import Embedder
        emb = Embedder()
        result = asyncio.get_event_loop().run_until_complete(emb.embed("   "))
        assert result is None

    @patch("bantz.memory.embeddings.httpx.AsyncClient")
    def test_embed_success_new_api(self, mock_client_cls):
        """Test /api/embed (newer Ollama) response format."""
        from bantz.memory.embeddings import Embedder

        fake_vec = [0.1] * 768
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"embeddings": [fake_vec]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        emb = Embedder()
        result = asyncio.get_event_loop().run_until_complete(emb.embed("hello"))
        assert result is not None
        assert len(result) == 768
        assert emb.dim == 768

    @patch("bantz.memory.embeddings.httpx.AsyncClient")
    def test_embed_fallback_old_api(self, mock_client_cls):
        """Test /api/embeddings (older Ollama) fallback."""
        from bantz.memory.embeddings import Embedder

        fake_vec = [0.2] * 384

        # First call returns 404 (/api/embed), second succeeds (/api/embeddings)
        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404

        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.raise_for_status = MagicMock()
        mock_resp_ok.json.return_value = {"embedding": fake_vec}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_resp_404, mock_resp_ok])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        emb = Embedder()
        result = asyncio.get_event_loop().run_until_complete(emb.embed("hello"))
        assert result is not None
        assert len(result) == 384
        assert emb.dim == 384


# ══════════════════════════════════════════════════════════════════════════
# Test: Memory integration (semantic + hybrid search)
# ══════════════════════════════════════════════════════════════════════════

class TestMemoryVectorIntegration:
    """Test Memory's new vector search methods using mocked embedder."""

    def setup_method(self):
        from bantz.core.memory import Memory
        self.mem = Memory()
        # Use a temp file for the DB
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.mem.init(self.db_path)
        self.mem.new_session()

    def teardown_method(self):
        self.mem.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_vector_store_initialized(self):
        """Memory should auto-create a VectorStore."""
        assert self.mem.vector_store is not None

    def test_add_queues_embedding(self):
        """add() should queue messages for embedding."""
        self.mem.add("user", "what is the weather like today")
        assert len(self.mem._embed_queue) == 1
        assert self.mem._embed_queue[0][1] == "what is the weather like today"

    def test_add_skips_short_messages(self):
        """Short messages (<=10 chars) should not be queued."""
        self.mem.add("user", "hi")
        assert len(self.mem._embed_queue) == 0

    def test_add_skips_system_role(self):
        """System messages should not be queued."""
        self.mem.add("system", "You are a helpful assistant with long content")
        assert len(self.mem._embed_queue) == 0

    @patch("bantz.memory.embeddings.embedder")
    def test_embed_pending(self, mock_embedder):
        """embed_pending should process the queue."""
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 8)
        mock_embedder.model = "test-model"

        self.mem.add("user", "this is a test message for embedding")
        assert len(self.mem._embed_queue) == 1

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.embedding_enabled = True
            count = asyncio.get_event_loop().run_until_complete(
                self.mem.embed_pending()
            )

        assert count == 1
        assert len(self.mem._embed_queue) == 0
        assert self.mem.vector_store.count() == 1

    @patch("bantz.memory.embeddings.embedder")
    def test_semantic_search(self, mock_embedder):
        """semantic_search should return results from vector store."""
        # Store some messages with embeddings
        for text in ["sunny weather today", "meeting at 3pm", "buy groceries"]:
            msg_id = self.mem.add("user", text)
            vec = _fake_embedding(text)
            self.mem.vector_store.store(msg_id, vec)

        query_vec = _fake_embedding("how is the weather")
        mock_embedder.embed = AsyncMock(return_value=query_vec)

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.embedding_enabled = True
            results = asyncio.get_event_loop().run_until_complete(
                self.mem.semantic_search("how is the weather", limit=2, min_score=-1.0)
            )

        assert len(results) <= 2
        assert all("score" in r for r in results)

    @patch("bantz.memory.embeddings.embedder")
    def test_hybrid_search(self, mock_embedder):
        """hybrid_search should combine FTS and vector results."""
        for text in [
            "the weather is beautiful today sunshine",
            "remind me to check the forecast tomorrow",
            "buy milk from the store",
        ]:
            msg_id = self.mem.add("user", text)
            vec = _fake_embedding(text)
            self.mem.vector_store.store(msg_id, vec)

        query_vec = _fake_embedding("weather forecast")
        mock_embedder.embed = AsyncMock(return_value=query_vec)

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.embedding_enabled = True
            mock_cfg.vector_search_weight = 0.5
            results = asyncio.get_event_loop().run_until_complete(
                self.mem.hybrid_search("weather", limit=3)
            )

        assert len(results) <= 3
        for r in results:
            assert "hybrid_score" in r
            assert "source" in r
            assert r["source"] in ("fts", "vector", "both")

    def test_semantic_search_disabled(self):
        """semantic_search returns [] when embeddings disabled."""
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.embedding_enabled = False
            results = asyncio.get_event_loop().run_until_complete(
                self.mem.semantic_search("anything")
            )
        assert results == []

    @patch("bantz.memory.embeddings.embedder")
    def test_backfill_embeddings(self, mock_embedder):
        """backfill should embed messages that don't have vectors yet."""
        # Add messages without embedding
        for i in range(5):
            self.mem.add("user", f"long test message number {i} for backfill testing")

        # Clear the queue to simulate old messages
        self.mem._embed_queue.clear()

        mock_embedder.embed = AsyncMock(return_value=[0.1] * 8)
        mock_embedder.model = "test-model"

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.embedding_enabled = True
            count = asyncio.get_event_loop().run_until_complete(
                self.mem.backfill_embeddings(batch_size=10)
            )

        assert count == 5
        assert self.mem.vector_store.count() == 5

    def test_vector_stats(self):
        stats = self.mem.vector_stats()
        assert stats["enabled"] is True
        assert stats["total_embeddings"] == 0

    def test_vector_stats_when_no_store(self):
        self.mem._vector_store = None
        stats = self.mem.vector_stats()
        assert stats["enabled"] is False


# ══════════════════════════════════════════════════════════════════════════
# Test: Config fields
# ══════════════════════════════════════════════════════════════════════════

class TestConfigEmbedding:
    def test_default_embedding_model(self):
        from bantz.config import Config
        cfg = Config()
        assert cfg.embedding_model == "nomic-embed-text"

    def test_default_embedding_enabled(self):
        from bantz.config import Config
        cfg = Config()
        assert cfg.embedding_enabled is True

    def test_default_vector_search_weight(self):
        from bantz.config import Config
        cfg = Config()
        assert cfg.vector_search_weight == 0.5

    def test_env_override(self):
        import os
        os.environ["BANTZ_EMBEDDING_MODEL"] = "all-minilm"
        os.environ["BANTZ_EMBEDDING_ENABLED"] = "false"
        os.environ["BANTZ_VECTOR_SEARCH_WEIGHT"] = "0.7"
        try:
            from bantz.config import Config
            cfg = Config()
            assert cfg.embedding_model == "all-minilm"
            assert cfg.embedding_enabled is False
            assert cfg.vector_search_weight == pytest.approx(0.7)
        finally:
            del os.environ["BANTZ_EMBEDDING_MODEL"]
            del os.environ["BANTZ_EMBEDDING_ENABLED"]
            del os.environ["BANTZ_VECTOR_SEARCH_WEIGHT"]
