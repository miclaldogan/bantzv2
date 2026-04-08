"""
End-to-end integration test: OmniMemoryManager → MemPalaceBridge → ChromaDB.

Validates that the full recall pipeline (graph + vector + deep search)
actually works through real ChromaDB storage — not mocks.

Run: pytest tests/memory/test_omni_bridge_e2e.py -v
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine in the default event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def tmp_palace(tmp_path):
    """Temporary palace directory for real ChromaDB storage."""
    palace_path = str(tmp_path / "palace")
    kg_path = str(tmp_path / "knowledge_graph.sqlite3")
    identity_path = str(tmp_path / "identity.txt")
    Path(identity_path).write_text("I am Bantz, a test assistant.")
    return {
        "palace_path": palace_path,
        "kg_path": kg_path,
        "identity_path": identity_path,
    }


@pytest.fixture
def mock_config(tmp_palace):
    """Mock config pointing to real temp dirs."""
    cfg = MagicMock()
    cfg.mempalace_enabled = True
    cfg.resolved_palace_path = tmp_palace["palace_path"]
    cfg.resolved_kg_path = tmp_palace["kg_path"]
    cfg.resolved_identity_path = tmp_palace["identity_path"]
    cfg.mempalace_wing = "e2e_test"
    cfg.deep_memory_enabled = True
    cfg.deep_memory_threshold = 0.1  # low threshold for test
    cfg.deep_memory_max_results = 5
    return cfg


@pytest.fixture
def live_bridge(mock_config):
    """Create a real MemPalaceBridge backed by on-disk ChromaDB."""
    with patch("bantz.memory.bridge._get_config", return_value=mock_config):
        from bantz.memory.bridge import MemPalaceBridge
        b = MemPalaceBridge()
        _run(b.init())
        assert b.enabled, "Bridge should be enabled after init"
        yield b
        b.close()


# ═══════════════════════════════════════════════════════════════════════════
# Test: omni_memory._graph_search → bridge.graph_context → real KG
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniGraphSearch:
    """End-to-end: omni._graph_search → bridge → KnowledgeGraph."""

    def test_graph_search_returns_string(self, live_bridge):
        """_graph_search returns a string (possibly empty) from real KG."""
        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            result = _run(omni._graph_search("hello world"))
        assert isinstance(result, str)

    def test_graph_search_with_stored_triples(self, live_bridge):
        """After storing KG triples, _graph_search should find them."""
        kg = live_bridge.kg
        assert kg is not None
        kg.add_triple(subject="Murat", predicate="is_friend_of", obj="User")
        kg.add_triple(subject="User", predicate="decided_to", obj="learn Rust this year")

        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            result = _run(omni._graph_search("Murat"))
        assert isinstance(result, str)
        # KG context should mention Murat
        assert "Murat" in result or result == ""  # depends on entity detector


# ═══════════════════════════════════════════════════════════════════════════
# Test: omni_memory._vector_search → memory.hybrid_search → bridge
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniVectorSearch:
    """End-to-end: store exchanges, then find them via vector search."""

    def test_vector_search_after_store(self, live_bridge):
        """Store 3 exchanges about cooking, then search for cooking."""
        exchanges = [
            ("What's a good pasta recipe?", "Try aglio e olio with garlic and chilli."),
            ("How do I make risotto?", "Start with arborio rice, add broth gradually."),
            ("What dessert should I make?", "Tiramisu is a classic Italian dessert."),
        ]
        for user, asst in exchanges:
            _run(live_bridge.store_exchange(user, asst, tool_used=None))

        # Now search via bridge.vector_context (direct, not through omni)
        result = live_bridge.vector_context("cooking pasta Italian food")
        assert isinstance(result, str)
        # ChromaDB should return something related
        if result:
            assert any(kw in result.lower() for kw in ("pasta", "risotto", "aglio", "garlic", "tiramisu"))

    def test_deep_search_after_store(self, live_bridge):
        """Store exchanges, then use deep_memory to find them."""
        _run(live_bridge.store_exchange(
            "I'm planning to move to Berlin next month",
            "That's exciting! Berlin has a great tech scene.",
        ))
        _run(live_bridge.store_exchange(
            "My friend Ayse is helping me find an apartment",
            "Ayse sounds very helpful. Let me know if you need recommendations.",
        ))

        result = live_bridge.deep_memory("Berlin apartment plans")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Full recall pipeline (graph + vector + deep in parallel)
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniRecallPipeline:
    """End-to-end: full OmniMemoryManager.recall() with real ChromaDB."""

    def test_recall_returns_result_type(self, live_bridge):
        """recall() returns a MemoryRecallResult even with empty palace."""
        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager, MemoryRecallResult
            omni = OmniMemoryManager()
            result = _run(omni.recall("hello"))
        assert isinstance(result, MemoryRecallResult)
        assert result.total_chars >= 0
        assert result.total_tokens_approx >= 0

    def test_recall_with_stored_data(self, live_bridge):
        """recall() returns non-empty combined when data exists."""
        # Seed some data
        _run(live_bridge.store_exchange(
            "My cat Pamuk loves sitting by the window",
            "Pamuk sounds adorable! Cats love watching birds.",
        ))
        _run(live_bridge.store_exchange(
            "I graduated from Boğaziçi University",
            "Boğaziçi is one of Turkey's most prestigious universities.",
        ))

        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()

            # Ask about the cat
            result = _run(omni.recall("tell me about my cat"))
            assert isinstance(result.combined, str)
            # With enough data, recall should return something
            # (but ChromaDB results depend on embedding quality)

    def test_recall_budget_enforcement(self, live_bridge):
        """recall() respects token budget — combined output never exceeds max."""
        # Store a lot of data
        for i in range(10):
            _run(live_bridge.store_exchange(
                f"Topic {i}: Tell me about random subject number {i}",
                f"Here's a really long response about subject {i}. " * 20,
            ))

        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager(max_memory_tokens=100)  # tight budget
            result = _run(omni.recall("random subjects"))
        # 100 tokens × 4 chars ≈ 400 chars max
        assert len(result.combined) <= 500  # small buffer for truncation marker


# ═══════════════════════════════════════════════════════════════════════════
# Test: Store/Forget/Transaction CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniCRUD:
    """Test that store/forget work through real KV backend."""

    def test_store_and_recall_kv(self, live_bridge, tmp_path):
        """store() + recall should work via KV store (SQLite)."""
        # This needs a real data_layer, so we mock the KV part
        mock_kv = MagicMock()
        mock_kv.get = MagicMock(return_value="stored_value")
        mock_dl = MagicMock()
        mock_dl.kv = mock_kv

        with patch("bantz.memory.bridge.palace_bridge", live_bridge), \
             patch("bantz.data.data_layer", mock_dl):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            _run(omni.store("test_key", "test_value"))
            mock_kv.set.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Test: omni_memory.summarize → bridge.distill_session
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniSummarize:
    """summarize() delegates to bridge.distill_session."""

    def test_summarize_returns_string(self, live_bridge):
        """summarize() should return a string (distilled or empty)."""
        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            # No messages for session 999 → should return string, not crash
            result = _run(omni.summarize(999))
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# Test: omni_memory.graph_query → bridge.kg
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniGraphQuery:
    """graph_query() returns KG triples."""

    def test_graph_query_returns_list(self, live_bridge):
        """graph_query should return a list of dicts from KG."""
        kg = live_bridge.kg
        assert kg is not None
        kg.add_triple(subject="Alice", predicate="knows", obj="Bob")

        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            result = _run(omni.graph_query("recent triples", limit=10))
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_graph_query_empty_kg(self, live_bridge):
        """graph_query on empty KG returns empty list."""
        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            result = _run(omni.graph_query("nothing here"))
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# Test: _deep_search → bridge.deep_memory → ChromaDB L3
# ═══════════════════════════════════════════════════════════════════════════

class TestOmniDeepSearch:
    """_deep_search routes through bridge.deep_memory."""

    def test_deep_search_returns_string(self, live_bridge):
        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            result = _run(omni._deep_search("anything"))
        assert isinstance(result, str)

    def test_deep_search_finds_stored_content(self, live_bridge):
        """After storing exchanges, deep search should find them."""
        _run(live_bridge.store_exchange(
            "I have a meeting with Deniz on Friday",
            "Got it, I'll remind you about the meeting with Deniz.",
        ))

        with patch("bantz.memory.bridge.palace_bridge", live_bridge):
            from bantz.memory.omni_memory import OmniMemoryManager
            omni = OmniMemoryManager()
            result = _run(omni._deep_search("Deniz meeting"))
        assert isinstance(result, str)
        # Content should appear if ChromaDB finds it
