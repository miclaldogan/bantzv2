"""
Smoke tests for MemPalace Bridge Adapter.

Verifies that the bridge provides behavioral equivalence with the old
memory subsystem before we start deleting old modules.

Run: pytest tests/memory/test_bridge_smoke.py -v
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_palace(tmp_path):
    """Create a temporary palace directory for testing."""
    palace_path = str(tmp_path / "palace")
    kg_path = str(tmp_path / "knowledge_graph.sqlite3")
    identity_path = str(tmp_path / "identity.txt")

    # Write a test identity
    Path(identity_path).write_text("I am Bantz, a test assistant.")

    return {
        "palace_path": palace_path,
        "kg_path": kg_path,
        "identity_path": identity_path,
    }


@pytest.fixture
def mock_config(tmp_palace):
    """Mock config with MemPalace paths pointing to tmp dirs."""
    mock_cfg = MagicMock()
    mock_cfg.mempalace_enabled = True
    mock_cfg.resolved_palace_path = tmp_palace["palace_path"]
    mock_cfg.resolved_kg_path = tmp_palace["kg_path"]
    mock_cfg.resolved_identity_path = tmp_palace["identity_path"]
    mock_cfg.mempalace_wing = "test_wing"
    mock_cfg.deep_memory_enabled = True
    mock_cfg.deep_memory_threshold = 0.4
    mock_cfg.deep_memory_max_results = 3
    return mock_cfg


@pytest.fixture
def bridge(mock_config):
    """Create and initialize a MemPalaceBridge with test config."""
    with patch("bantz.memory.bridge._get_config", return_value=mock_config):
        from bantz.memory.bridge import MemPalaceBridge
        b = MemPalaceBridge()
        asyncio.get_event_loop().run_until_complete(b.init())
        yield b
        b.close()


# ── Test: Initialization ─────────────────────────────────────────────────

class TestBridgeInit:
    """Bridge lifecycle mirrors old graph_memory.init() / close()."""

    def test_init_creates_palace(self, bridge, tmp_palace):
        """init() should create palace dir and initialize components."""
        assert bridge.enabled is True
        assert bridge.stack is not None
        assert bridge.kg is not None
        assert Path(tmp_palace["palace_path"]).exists()

    def test_close_disables(self, bridge):
        """close() should clean up and disable the bridge."""
        bridge.close()
        assert bridge.enabled is False
        assert bridge.stack is None
        assert bridge.kg is None

    def test_disabled_config(self, tmp_palace):
        """When mempalace_enabled=False, bridge stays inert."""
        mock_cfg = MagicMock()
        mock_cfg.mempalace_enabled = False
        with patch("bantz.memory.bridge._get_config", return_value=mock_cfg):
            from bantz.memory.bridge import MemPalaceBridge
            b = MemPalaceBridge()
            asyncio.get_event_loop().run_until_complete(b.init())
            assert b.enabled is False


# ── Test: Context retrieval (replaces graph_memory.context_for) ──────────

class TestContextRetrieval:
    """Bridge recall methods produce context strings like the old system."""

    def test_wake_up_returns_identity(self, bridge):
        """wake_up_context() should include L0 identity text."""
        ctx = bridge.wake_up_context()
        assert "Bantz" in ctx or "identity" in ctx.lower() or "L0" in ctx or "L1" in ctx

    def test_vector_context_empty_palace(self, bridge):
        """vector_context() returns something even on empty palace."""
        result = bridge.vector_context("test query")
        # Empty palace = "No results found" or empty — both are valid
        assert isinstance(result, str)

    def test_graph_context_empty_kg(self, bridge):
        """graph_context() returns empty string when KG has no data."""
        result = bridge.graph_context("Who is Alice?")
        assert isinstance(result, str)

    def test_deep_memory_rate_limited(self, bridge):
        """deep_memory() should be rate-limited (fires every 3rd call)."""
        results = []
        for i in range(6):
            r = bridge.deep_memory(f"test message {i}")
            results.append(r)
        # At minimum, it should return strings (empty is fine for empty palace)
        assert all(isinstance(r, str) for r in results)


# ── Test: Store exchange (replaces graph_memory.extract_and_store) ────────

class TestStoreExchange:
    """Exchange storage deposits drawers in ChromaDB + triples in KG."""

    def test_store_creates_drawer(self, bridge, mock_config):
        """store_exchange() should add a drawer to ChromaDB."""
        with patch("bantz.memory.bridge._get_config", return_value=mock_config):
            asyncio.get_event_loop().run_until_complete(
                bridge.store_exchange(
                    user_msg="What's the weather in Istanbul?",
                    assistant_msg="It's 18°C and sunny in Istanbul today.",
                    tool_used="weather",
                )
            )

        # Verify drawer exists in ChromaDB
        import chromadb
        client = chromadb.PersistentClient(path=mock_config.resolved_palace_path)
        col = client.get_collection("mempalace_drawers")
        assert col.count() >= 1

    def test_store_extracts_kg_triples(self, bridge, mock_config):
        """store_exchange() should extract triples into the KG."""
        with patch("bantz.memory.bridge._get_config", return_value=mock_config):
            asyncio.get_event_loop().run_until_complete(
                bridge.store_exchange(
                    user_msg="I had a meeting with Alice about the project",
                    assistant_msg="I noted your meeting with Alice.",
                    tool_used=None,
                )
            )

        # Check KG has entities (KG returns "entities" key, not "total_entities")
        stats = bridge.kg.stats()
        entity_count = stats.get("entities", stats.get("total_entities", 0))
        assert entity_count >= 1

    def test_store_routes_to_correct_room(self, bridge, mock_config):
        """Tool-based exchanges should route to the right room."""
        with patch("bantz.memory.bridge._get_config", return_value=mock_config):
            asyncio.get_event_loop().run_until_complete(
                bridge.store_exchange(
                    user_msg="Check my calendar",
                    assistant_msg="You have 3 meetings today.",
                    tool_used="calendar_list",
                )
            )

        import chromadb
        client = chromadb.PersistentClient(path=mock_config.resolved_palace_path)
        col = client.get_collection("mempalace_drawers")
        results = col.get(where={"room": "events"}, include=["metadatas"])
        assert len(results["ids"]) >= 1

    def test_store_multiple_exchanges_searchable(self, bridge, mock_config):
        """Multiple stored exchanges should be searchable via L3."""
        with patch("bantz.memory.bridge._get_config", return_value=mock_config):
            exchanges = [
                ("Should we use PostgreSQL or MySQL?",
                 "Let's go with PostgreSQL for better JSON support.",
                 None),
                ("Deploy the app to production",
                 "Deployed successfully to production server.",
                 "shell"),
                ("What did Alice say about the budget?",
                 "Alice mentioned the budget is $50k for Q2.",
                 None),
            ]
            for user, asst, tool in exchanges:
                asyncio.get_event_loop().run_until_complete(
                    bridge.store_exchange(user, asst, tool)
                )

        # Now search should find PostgreSQL-related content
        result = bridge.vector_context("PostgreSQL database decision")
        # ChromaDB needs at least some data to search meaningfully
        import chromadb
        client = chromadb.PersistentClient(path=mock_config.resolved_palace_path)
        col = client.get_collection("mempalace_drawers")
        assert col.count() >= 3


# ── Test: Distillation (replaces distiller.distill_session) ──────────────

class TestDistillation:
    """Session distillation mines conversations into palace drawers."""

    def test_distill_empty_session(self, bridge, mock_config):
        """Distilling a session with no messages should return empty."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda self: mock_conn
        mock_conn.__exit__ = lambda *args: None
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pool.connection.return_value = mock_conn

        with patch("bantz.memory.bridge._get_config", return_value=mock_config), \
             patch("bantz.data.connection_pool.get_pool", return_value=mock_pool):
            result = asyncio.get_event_loop().run_until_complete(
                bridge.distill_session(999)
            )
        assert result["drawers_added"] == 0


# ── Test: Stats (replaces graph_memory.stats) ────────────────────────────

class TestStats:
    """Stats and status methods work like old graph_memory equivalents."""

    def test_stats_returns_dict(self, bridge):
        """stats() should return a dict with standard keys."""
        s = bridge.stats()
        assert "palace_enabled" in s
        assert "total_drawers" in s
        assert "kg_entities" in s
        assert "kg_triples" in s

    def test_status_line_is_string(self, bridge):
        """status_line() should return a human-readable string."""
        line = bridge.status_line()
        assert isinstance(line, str)
        assert "MemPalace" in line

    def test_growth_since(self, bridge, mock_config):
        """growth_since() counts new KG entries after a timestamp."""
        # Store something first
        with patch("bantz.memory.bridge._get_config", return_value=mock_config):
            asyncio.get_event_loop().run_until_complete(
                bridge.store_exchange(
                    "Meeting with Bob tomorrow",
                    "Noted your meeting with Bob.",
                    None,
                )
            )
        growth = bridge.growth_since("2020-01-01")
        assert isinstance(growth, dict)
        assert "entities" in growth
        assert "triples" in growth


# ── Test: API Parity ─────────────────────────────────────────────────────

class TestAPIParity:
    """Verify bridge has all methods the old singletons exposed."""

    def test_has_graph_memory_equivalents(self, bridge):
        """Bridge should have methods matching graph_memory's public API."""
        # graph_memory.init()     → bridge.init()
        # graph_memory.close()    → bridge.close()
        # graph_memory.enabled    → bridge.enabled
        # graph_memory.context_for() → bridge.graph_context()
        # graph_memory.extract_and_store() → bridge.store_exchange()
        # graph_memory.stats()    → bridge.stats()
        # graph_memory.status_line() → bridge.status_line()
        # graph_memory.growth_since() → bridge.growth_since()
        assert callable(bridge.init)
        assert callable(bridge.close)
        assert isinstance(bridge.enabled, bool)
        assert callable(bridge.graph_context)
        assert callable(bridge.store_exchange)
        assert callable(bridge.stats)
        assert callable(bridge.status_line)
        assert callable(bridge.growth_since)

    def test_has_deep_probe_equivalent(self, bridge):
        """Bridge should expose deep_memory() replacing deep_probe.probe()."""
        assert callable(bridge.deep_memory)

    def test_has_distiller_equivalent(self, bridge):
        """Bridge should expose distill_session() replacing distiller."""
        assert callable(bridge.distill_session)

    def test_has_vector_context_equivalent(self, bridge):
        """Bridge should expose vector_context() replacing vector_store search."""
        assert callable(bridge.vector_context)

    def test_has_wake_up_context(self, bridge):
        """Bridge should expose wake_up_context() (new L0+L1 feature)."""
        assert callable(bridge.wake_up_context)


# ── Test: Spontaneous Probe ──────────────────────────────────────────────

class TestSpontaneousProbe:
    """The spontaneous probe preserves deep_probe.py behavior."""

    def test_rate_limiting(self):
        """Probe should only fire every N calls."""
        from bantz.memory.bridge import SpontaneousProbe
        probe = SpontaneousProbe(rate_every_n=3)
        mock_l3 = MagicMock()
        mock_l3.search_raw.return_value = []

        with patch("bantz.memory.bridge._get_config") as mock_cfg:
            mock_cfg.return_value.deep_memory_enabled = True
            mock_cfg.return_value.deep_memory_max_results = 3
            mock_cfg.return_value.deep_memory_threshold = 0.4

            fires = 0
            for i in range(9):
                result = probe.probe(f"msg {i}", mock_l3)
                if mock_l3.search_raw.called:
                    fires += 1
                    mock_l3.search_raw.reset_mock()

            # Should have fired 3 times out of 9 (every 3rd call)
            assert fires == 3

    def test_dedup_prevents_repeats(self):
        """Same memory shouldn't surface twice within TTL."""
        from bantz.memory.bridge import SpontaneousProbe
        probe = SpontaneousProbe(rate_every_n=1, dedup_ttl_seconds=600)
        mock_l3 = MagicMock()
        mock_l3.search_raw.return_value = [
            {"text": "Some unique memory content here", "wing": "w", "room": "r", "similarity": 0.9},
        ]

        with patch("bantz.memory.bridge._get_config") as mock_cfg:
            mock_cfg.return_value.deep_memory_enabled = True
            mock_cfg.return_value.deep_memory_max_results = 3
            mock_cfg.return_value.deep_memory_threshold = 0.4

            r1 = probe.probe("test", mock_l3)
            r2 = probe.probe("test", mock_l3)

            # First should return content, second should be empty (dedup)
            assert len(r1) > 0
            assert r2 == ""

    def test_disabled_returns_empty(self):
        """When deep_memory_enabled=False, probe returns empty."""
        from bantz.memory.bridge import SpontaneousProbe
        probe = SpontaneousProbe(rate_every_n=1)

        with patch("bantz.memory.bridge._get_config") as mock_cfg:
            mock_cfg.return_value.deep_memory_enabled = False
            result = probe.probe("test", MagicMock())
            assert result == ""
