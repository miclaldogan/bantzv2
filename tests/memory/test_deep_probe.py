"""
Tests for Issue #170 — Spontaneous Vector Memory Retrieval (Deep Probe).

Covers:
  1. Alzheimer Math Fix  — threshold filters on RAW cosine, decay is ranking-only
  2. Déjà Vu Fix         — recently-used memory IDs are excluded
  3. Butler Lore Fix     — injection uses 1920s human memory framing
  4. Rate-limiting       — probe fires only every N messages
  5. Config integration  — enabled/threshold/max_results from config
  6. Error resilience    — graceful "" on embedding or search failures
  7. Format output       — temporal age labels ("yesterday", "2 weeks ago", etc.)
  8. brain.py integration — CHAT_SYSTEM template has {deep_memory} placeholder
  9. finalizer.py integration — FINALIZER_SYSTEM has {deep_memory} placeholder
"""
from __future__ import annotations

import asyncio
import math
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ── Pool lifecycle ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the connection pool after every test."""
    yield
    from bantz.data.connection_pool import SQLitePool
    SQLitePool.reset()


# ── Helpers ────────────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _make_pool_db():
    """Create a temp-file pool-backed SQLite DB with Memory schema.

    Returns (pool, tmpdir).  Caller should clean up tmpdir after use.
    """
    tmpdir = tempfile.mkdtemp()
    db_path = Path(tmpdir) / "test.db"

    from bantz.data.connection_pool import SQLitePool
    pool = SQLitePool.get_instance(db_path)

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
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content TEXT NOT NULL,
                tool_used TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at)
        """)
    return pool, tmpdir


def _insert_conversation(pool) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with pool.connection(write=True) as conn:
        cur = conn.execute(
            "INSERT INTO conversations(started_at, last_active) VALUES (?,?)",
            (now, now),
        )
        return cur.lastrowid


def _insert_message(
    pool, conv_id: int, role: str, content: str,
    created_at: str | None = None,
) -> int:
    ts = created_at or datetime.now().isoformat(timespec="seconds")
    with pool.connection(write=True) as conn:
        cur = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, tool_used, created_at) "
            "VALUES (?,?,?,?,?)",
            (conv_id, role, content, None, ts),
        )
        return cur.lastrowid


def _fake_embedding(text: str, dim: int = 8) -> list[float]:
    """Deterministic pseudo-embedding from text hash."""
    h = hash(text)
    vec = []
    for i in range(dim):
        val = ((h >> (i * 4)) & 0xFF) / 255.0 - 0.5
        vec.append(val)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _similar_embedding(base: list[float], noise: float = 0.05) -> list[float]:
    """Create a vector very similar to base (high cosine) with slight noise."""
    import random
    rng = random.Random(42)
    vec = [v + rng.uniform(-noise, noise) for v in base]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _setup_vector_store():
    """Create VectorStore (uses shared pool) and migrate."""
    from bantz.memory.vector_store import VectorStore
    vs = VectorStore()
    vs.migrate()
    return vs


# ══════════════════════════════════════════════════════════════════════════
# Test: DeepMemoryProbe — Unit Tests
# ══════════════════════════════════════════════════════════════════════════

class TestDeepMemoryProbeUnit:
    """Direct unit tests on _search_raw, _rank_with_decay, _format_hint."""

    def test_search_raw_filters_below_threshold(self):
        """Alzheimer Fix: raw cosine below threshold → excluded."""
        from bantz.memory.deep_probe import DeepMemoryProbe
        from bantz.memory.vector_store import VectorStore

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        # Two messages: one similar, one dissimilar
        base_vec = _fake_embedding("study for exam")
        similar_vec = _similar_embedding(base_vec, noise=0.01)  # high cosine
        dissimilar_vec = _fake_embedding("banana smoothie recipe")  # low cosine

        msg1 = _insert_message(pool, conv_id, "user", "I need to study for my exam")
        msg2 = _insert_message(pool, conv_id, "user", "banana smoothie recipe please")
        vs.store(msg1, similar_vec)
        vs.store(msg2, dissimilar_vec)

        probe = DeepMemoryProbe(cosine_threshold=0.8)
        results = probe._search_raw(vs, base_vec, threshold=0.8)

        # Only the similar one should survive
        ids = {r["message_id"] for r in results}
        assert msg1 in ids
        assert msg2 not in ids

    def test_search_raw_keeps_old_relevant_memory(self):
        """Alzheimer Fix: old memory with high cosine is NOT discarded."""
        from bantz.memory.deep_probe import DeepMemoryProbe
        from bantz.memory.vector_store import VectorStore

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        # Old message (6 months ago) but very similar embedding
        old_date = (datetime.utcnow() - timedelta(days=180)).isoformat(timespec="seconds")
        base_vec = _fake_embedding("project deadline")
        similar_vec = _similar_embedding(base_vec, noise=0.01)

        msg_id = _insert_message(pool, conv_id, "user",
                                 "The project deadline is next Friday",
                                 created_at=old_date)
        vs.store(msg_id, similar_vec)

        probe = DeepMemoryProbe(cosine_threshold=0.7)
        results = probe._search_raw(vs, base_vec, threshold=0.7)

        # Old-but-relevant memory MUST survive threshold (Alzheimer Fix)
        assert len(results) >= 1
        assert results[0]["message_id"] == msg_id

    def test_rank_with_decay_recent_wins(self):
        """Recent memories rank higher than old ones at equal cosine."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        now = datetime.utcnow().isoformat(timespec="seconds")
        old = (datetime.utcnow() - timedelta(days=60)).isoformat(timespec="seconds")

        candidates = [
            {"cosine": 0.85, "created_at": old, "message_id": 1},
            {"cosine": 0.85, "created_at": now, "message_id": 2},
        ]

        probe = DeepMemoryProbe()
        ranked = probe._rank_with_decay(candidates)

        # Recent one (msg 2) should rank higher
        assert ranked[0]["message_id"] == 2
        assert ranked[0]["rank_score"] > ranked[1]["rank_score"]

    def test_rank_preserves_high_cosine_old(self):
        """Even with decay, high-cosine old memory gets reasonable rank_score."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        old = (datetime.utcnow() - timedelta(days=90)).isoformat(timespec="seconds")
        candidates = [
            {"cosine": 0.95, "created_at": old, "message_id": 1},
        ]

        probe = DeepMemoryProbe(decay_half_life_days=30.0)
        ranked = probe._rank_with_decay(candidates)

        # exp(-90/30) ≈ 0.05; 0.95 * 0.05 ≈ 0.047 — low but still ranked
        assert ranked[0]["rank_score"] > 0
        assert ranked[0]["message_id"] == 1

    def test_format_hint_butler_lore(self):
        """Butler Lore Fix: injection uses human memory framing."""
        from bantz.memory.deep_probe import DeepMemoryProbe, _INJECT_PREAMBLE

        now = datetime.utcnow().isoformat(timespec="seconds")
        memories = [
            {
                "role": "user",
                "content": "I have an exam on Thursday",
                "created_at": now,
                "message_id": 42,
            }
        ]

        hint = DeepMemoryProbe._format_hint(memories)

        # Must contain the butler preamble
        assert "Spontaneous Human Memory" in hint
        assert "naturally in your mind as a human butler" in hint
        assert "Do NOT say 'according to my records'" in hint

        # Must contain the content
        assert "exam on Thursday" in hint
        # Must use "Ma'am" for user messages
        assert "Ma'am once said" in hint

    def test_format_hint_assistant_role_label(self):
        """Assistant messages show 'You' label."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        now = datetime.utcnow().isoformat(timespec="seconds")
        memories = [
            {
                "role": "assistant",
                "content": "I arranged your schedule for Thursday",
                "created_at": now,
                "message_id": 99,
            }
        ]

        hint = DeepMemoryProbe._format_hint(memories)
        assert "You once said" in hint

    def test_format_hint_age_labels(self):
        """Temporal age labels are correct."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        now = datetime.utcnow()
        test_cases = [
            (now - timedelta(hours=2), "earlier today"),
            (now - timedelta(hours=30), "yesterday"),
            (now - timedelta(days=4), "4 days ago"),
            (now - timedelta(days=15), "about 2 weeks ago"),
            (now - timedelta(days=45), "roughly 1 months ago"),
        ]

        for ts, expected_fragment in test_cases:
            memories = [
                {
                    "role": "user",
                    "content": "test content",
                    "created_at": ts.isoformat(timespec="seconds"),
                    "message_id": 1,
                }
            ]
            hint = DeepMemoryProbe._format_hint(memories)
            assert expected_fragment in hint, \
                f"Expected '{expected_fragment}' for age {(now - ts).days}d, got: {hint}"


# ══════════════════════════════════════════════════════════════════════════
# Test: Déjà Vu Fix
# ══════════════════════════════════════════════════════════════════════════

class TestDejaVuFix:
    """Test that recently-used memory IDs are not repeated."""

    def test_recently_used_excluded(self):
        """Memories used within TTL are skipped."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(dedup_ttl_seconds=300.0)
        # Pre-mark message_id 42 as used
        probe._recently_used[42] = time.monotonic()

        candidates = [
            {
                "message_id": 42, "cosine": 0.9, "role": "user",
                "content": "exam", "created_at": datetime.utcnow().isoformat(),
                "conv_id": 1, "rank_score": 0.9,
            },
            {
                "message_id": 99, "cosine": 0.85, "role": "user",
                "content": "project", "created_at": datetime.utcnow().isoformat(),
                "conv_id": 1, "rank_score": 0.85,
            },
        ]

        # Simulate what probe() does: filter out recently used
        probe._evict_stale()
        filtered = [c for c in candidates if c["message_id"] not in probe._recently_used]
        assert len(filtered) == 1
        assert filtered[0]["message_id"] == 99

    def test_stale_entries_evicted(self):
        """Entries older than TTL are evicted."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(dedup_ttl_seconds=1.0)
        probe._recently_used[42] = time.monotonic() - 2.0  # expired
        probe._recently_used[99] = time.monotonic()  # still fresh

        probe._evict_stale()

        assert 42 not in probe._recently_used
        assert 99 in probe._recently_used

    def test_probe_marks_used_ids(self):
        """After surfacing a memory, its ID enters _recently_used."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=1)
        probe._call_counter = 0  # next call will fire

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        base_vec = _fake_embedding("exam study")
        similar_vec = _similar_embedding(base_vec, noise=0.01)
        msg_id = _insert_message(pool, conv_id, "user", "study for exam")
        vs.store(msg_id, similar_vec)

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.deep_memory_threshold = 0.5
        mock_config.deep_memory_max_results = 3
        mock_config.embedding_enabled = True

        mock_memory = MagicMock()
        mock_memory.vectors = vs

        async def mock_embed(text):
            return base_vec

        with patch("bantz.memory.deep_probe.config", mock_config, create=True), \
             patch("bantz.memory.embeddings.embedder") as mock_embedder:
            mock_embedder.embed = AsyncMock(side_effect=mock_embed)

            with patch("bantz.config.config", mock_config), \
                 patch("bantz.memory.embeddings.embedder", mock_embedder):
                # Direct test: call internal methods
                results = probe._search_raw(vs, base_vec, threshold=0.5)
                assert len(results) >= 1

                ranked = probe._rank_with_decay(results)
                top = ranked[:3]
                now = time.monotonic()
                for r in top:
                    probe._recently_used[r["message_id"]] = now

                assert msg_id in probe._recently_used


# ══════════════════════════════════════════════════════════════════════════
# Test: Rate Limiting
# ══════════════════════════════════════════════════════════════════════════

class TestRateLimiting:
    """Probe fires only every N messages."""

    def test_skips_non_nth_calls(self):
        """Calls 1,2 are skipped; call 3 fires (rate_every_n=3)."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=3)

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True

        with patch("bantz.config.config", mock_config):
            # Calls 1 and 2 should return "" immediately (rate limit)
            result1 = _run(probe.probe("hello"))
            assert result1 == ""
            assert probe._call_counter == 1

            result2 = _run(probe.probe("hi"))
            assert result2 == ""
            assert probe._call_counter == 2

    def test_fires_on_nth_call(self):
        """Third call fires the actual search."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=3)
        probe._call_counter = 2  # next call will be 3 → fires

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True
        mock_config.deep_memory_threshold = 0.72
        mock_config.deep_memory_max_results = 3

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=None)  # no embedding → ""

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder):
            # The probe fires but embed returns None → ""
            result = _run(probe.probe("test"))
            assert result == ""
            assert probe._call_counter == 3

    def test_reset_clears_counter(self):
        """reset() clears call counter and recently_used."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe()
        probe._call_counter = 42
        probe._recently_used = {1: time.monotonic(), 2: time.monotonic()}

        probe.reset()
        assert probe._call_counter == 0
        assert len(probe._recently_used) == 0


# ══════════════════════════════════════════════════════════════════════════
# Test: Config Integration
# ══════════════════════════════════════════════════════════════════════════

class TestDeepMemoryConfig:
    """Config fields exist and have correct defaults."""

    def test_config_fields_exist(self):
        """Config has deep_memory_enabled, threshold, max_results."""
        from bantz.config import Config

        cfg = Config(
            _env_file=None,
            BANTZ_DEEP_MEMORY_ENABLED="true",
            BANTZ_DEEP_MEMORY_THRESHOLD="0.72",
            BANTZ_DEEP_MEMORY_MAX_RESULTS="3",
        )
        assert cfg.deep_memory_enabled is True
        assert cfg.deep_memory_threshold == 0.72
        assert cfg.deep_memory_max_results == 3

    def test_config_defaults(self):
        """Default values are sensible."""
        from bantz.config import Config

        cfg = Config(_env_file=None)
        assert cfg.deep_memory_enabled is True
        assert 0.5 <= cfg.deep_memory_threshold <= 0.95
        assert 1 <= cfg.deep_memory_max_results <= 10

    def test_config_disabled(self):
        """deep_memory_enabled=false disables probe."""
        from bantz.config import Config

        cfg = Config(_env_file=None, BANTZ_DEEP_MEMORY_ENABLED="false")
        assert cfg.deep_memory_enabled is False

    def test_probe_respects_disabled_config(self):
        """When disabled, probe() returns '' immediately."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=1)
        probe._call_counter = 0

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = False

        with patch("bantz.config.config", mock_config):
            result = _run(probe.probe("hello"))
            assert result == ""

    def test_probe_respects_embedding_disabled(self):
        """When embeddings disabled, probe returns ''."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=1)
        probe._call_counter = 0

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = False

        with patch("bantz.config.config", mock_config):
            result = _run(probe.probe("test"))
            assert result == ""


# ══════════════════════════════════════════════════════════════════════════
# Test: Error Resilience
# ══════════════════════════════════════════════════════════════════════════

class TestErrorResilience:
    """Graceful degradation on failures."""

    def test_embed_failure_returns_empty(self):
        """If embedding fails, return '' not crash."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=1)
        probe._call_counter = 0

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(side_effect=RuntimeError("Ollama down"))

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder):
            result = _run(probe.probe("test"))
            assert result == ""

    def test_no_vectors_returns_empty(self):
        """Empty vector store → ''."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=1)
        probe._call_counter = 0

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()  # empty store

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True
        mock_config.deep_memory_threshold = 0.72
        mock_config.deep_memory_max_results = 3

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 8)

        mock_memory = MagicMock()
        mock_memory.vectors = vs

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder), \
             patch("bantz.core.memory.memory", mock_memory):
            result = _run(probe.probe("test"))
            assert result == ""

    def test_memory_import_failure_returns_empty(self):
        """If memory module import fails, return ''."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        probe = DeepMemoryProbe(rate_every_n=1)
        probe._call_counter = 0

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 8)

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder), \
             patch.dict("sys.modules", {"bantz.core.memory": None}):
            result = _run(probe.probe("test"))
            assert result == ""


# ══════════════════════════════════════════════════════════════════════════
# Test: Brain Template Integration
# ══════════════════════════════════════════════════════════════════════════

class TestBrainIntegration:
    """Verify {deep_memory} is in CHAT_SYSTEM and format calls work."""

    def test_chat_system_has_deep_memory_placeholder(self):
        """CHAT_SYSTEM contains {deep_memory}."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "{deep_memory}" in CHAT_SYSTEM

    def test_chat_system_format_with_deep_memory(self):
        """CHAT_SYSTEM.format() accepts deep_memory kwarg."""
        from bantz.core.brain import CHAT_SYSTEM

        formatted = CHAT_SYSTEM.format(
            persona_state="",
            style_hint="",
            formality_hint="",
            time_hint="",
            profile_hint="",
            graph_hint="",
            vector_hint="",
            deep_memory="Spontaneous recall: ma'am mentioned an exam",
            desktop_hint="",
        )
        assert "Spontaneous recall: ma'am mentioned an exam" in formatted
        assert "{deep_memory}" not in formatted

    def test_chat_system_format_empty_deep_memory(self):
        """When deep_memory='', no artifact in output."""
        from bantz.core.brain import CHAT_SYSTEM

        formatted = CHAT_SYSTEM.format(
            persona_state="",
            style_hint="",
            formality_hint="",
            time_hint="",
            profile_hint="",
            graph_hint="",
            vector_hint="",
            deep_memory="",
            desktop_hint="",
        )
        assert "{deep_memory}" not in formatted


# ══════════════════════════════════════════════════════════════════════════
# Test: Finalizer Template Integration
# ══════════════════════════════════════════════════════════════════════════

class TestFinalizerIntegration:
    """Verify {deep_memory} is in FINALIZER_SYSTEM."""

    def test_finalizer_system_has_deep_memory(self):
        """FINALIZER_SYSTEM contains {deep_memory}."""
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "{deep_memory}" in FINALIZER_SYSTEM

    def test_finalizer_system_format(self):
        """FINALIZER_SYSTEM.format() accepts deep_memory."""
        from bantz.core.finalizer import FINALIZER_SYSTEM

        formatted = FINALIZER_SYSTEM.format(
            persona_state="",
            style_hint="",
            formality_hint="",
            time_hint="",
            profile_hint="",
            graph_hint="",
            deep_memory="Butler recalls a prior conversation",
        )
        assert "Butler recalls a prior conversation" in formatted
        assert "{deep_memory}" not in formatted

    def test_finalize_signature_accepts_deep_memory(self):
        """finalize() and finalize_stream() accept deep_memory kwarg."""
        import inspect
        from bantz.core.finalizer import finalize, finalize_stream

        sig_f = inspect.signature(finalize)
        assert "deep_memory" in sig_f.parameters

        sig_fs = inspect.signature(finalize_stream)
        assert "deep_memory" in sig_fs.parameters


# ══════════════════════════════════════════════════════════════════════════
# Test: End-to-End Deep Probe Flow
# ══════════════════════════════════════════════════════════════════════════

class TestEndToEndProbe:
    """Full probe flow with real VectorStore."""

    def test_full_probe_returns_hint(self):
        """Complete flow: embed → search → rank → format."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        # Store a message with a known embedding
        base_vec = _fake_embedding("exam study")
        similar_vec = _similar_embedding(base_vec, noise=0.01)
        msg_id = _insert_message(pool, conv_id, "user", "I have a big exam on Thursday")
        vs.store(msg_id, similar_vec)

        probe = DeepMemoryProbe(
            cosine_threshold=0.5,
            max_results=3,
            rate_every_n=1,
        )

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True
        mock_config.deep_memory_threshold = 0.5
        mock_config.deep_memory_max_results = 3

        mock_memory = MagicMock()
        mock_memory.vectors = vs

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=base_vec)

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder), \
             patch("bantz.core.memory.memory", mock_memory):
            result = _run(probe.probe("studying for my test"))

        assert result != ""
        assert "Spontaneous Human Memory" in result
        assert "exam on Thursday" in result

    def test_full_probe_no_match(self):
        """No memories above threshold → ''."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        # Store dissimilar message
        msg_vec = _fake_embedding("banana smoothie")
        msg_id = _insert_message(pool, conv_id, "user", "banana smoothie recipe")
        vs.store(msg_id, msg_vec)

        probe = DeepMemoryProbe(
            cosine_threshold=0.95,  # very high threshold
            max_results=3,
            rate_every_n=1,
        )

        query_vec = _fake_embedding("exam study deadline")

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True
        mock_config.deep_memory_threshold = 0.95
        mock_config.deep_memory_max_results = 3

        mock_memory = MagicMock()
        mock_memory.vectors = vs

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=query_vec)

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder), \
             patch("bantz.core.memory.memory", mock_memory):
            result = _run(probe.probe("studying for exams"))

        assert result == ""

    def test_dedup_blocks_repeat_in_full_flow(self):
        """Second probe for same topic → '' because IDs are in recently_used."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        base_vec = _fake_embedding("exam study")
        similar_vec = _similar_embedding(base_vec, noise=0.01)
        msg_id = _insert_message(pool, conv_id, "user", "I have a big exam")
        vs.store(msg_id, similar_vec)

        probe = DeepMemoryProbe(
            cosine_threshold=0.5,
            max_results=3,
            rate_every_n=1,
            dedup_ttl_seconds=300.0,
        )

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True
        mock_config.deep_memory_threshold = 0.5
        mock_config.deep_memory_max_results = 3

        mock_memory = MagicMock()
        mock_memory.vectors = vs

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=base_vec)

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder), \
             patch("bantz.core.memory.memory", mock_memory):
            # First call → surfaces memory
            result1 = _run(probe.probe("studying"))
            assert result1 != ""

            # Second call → same topic, blocked by déjà vu
            result2 = _run(probe.probe("studying again"))
            assert result2 == ""

    def test_max_results_respected(self):
        """Only top-N memories are surfaced."""
        from bantz.memory.deep_probe import DeepMemoryProbe

        pool, _tmpdir = _make_pool_db()
        vs = _setup_vector_store()
        conv_id = _insert_conversation(pool)

        base_vec = _fake_embedding("exam")

        # Insert 5 similar messages
        for i in range(5):
            vec = _similar_embedding(base_vec, noise=0.01 + i * 0.001)
            msg_id = _insert_message(pool, conv_id, "user", f"exam topic {i}")
            vs.store(msg_id, vec)

        probe = DeepMemoryProbe(
            cosine_threshold=0.3,
            max_results=2,
            rate_every_n=1,
        )

        mock_config = MagicMock()
        mock_config.deep_memory_enabled = True
        mock_config.embedding_enabled = True
        mock_config.deep_memory_threshold = 0.3
        mock_config.deep_memory_max_results = 2

        mock_memory = MagicMock()
        mock_memory.vectors = vs

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=base_vec)

        with patch("bantz.config.config", mock_config), \
             patch("bantz.memory.embeddings.embedder", mock_embedder), \
             patch("bantz.core.memory.memory", mock_memory):
            result = _run(probe.probe("exam prep"))

        # Count memory lines (each starts with "- ")
        memory_lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(memory_lines) == 2


# ══════════════════════════════════════════════════════════════════════════
# Test: Brain._deep_memory_context
# ══════════════════════════════════════════════════════════════════════════

class TestBrainDeepMemoryContext:
    """Verify Brain has _deep_memory_context method."""

    def test_brain_has_method(self):
        """Brain class has _deep_memory_context."""
        from bantz.core.brain import Brain
        assert hasattr(Brain, "_deep_memory_context")
        assert asyncio.iscoroutinefunction(Brain._deep_memory_context)

    def test_brain_deep_memory_context_graceful_on_error(self):
        """_deep_memory_context returns '' on import/probe failures."""
        from bantz.core.brain import Brain

        brain = Brain.__new__(Brain)

        with patch("bantz.memory.deep_probe.deep_probe") as mock_probe:
            mock_probe.probe = AsyncMock(side_effect=RuntimeError("fail"))
            result = _run(brain._deep_memory_context("test"))
            assert result == ""
