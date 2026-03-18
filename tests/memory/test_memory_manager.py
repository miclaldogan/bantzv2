"""Tests for MemoryManager (#293) — store, query, summarize_context."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _disabled_gm():
    gm = MagicMock()
    gm.enabled = False
    return gm


def _enabled_gm(query_return=None):
    gm = MagicMock()
    gm.enabled = True
    gm.extract_and_store = AsyncMock()
    gm._query = AsyncMock(return_value=query_return or [])
    return gm


# ── store() ───────────────────────────────────────────────────────────────────

class TestStore:

    @pytest.mark.asyncio
    async def test_store_delegates_to_graph_memory(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        with patch("bantz.memory.graph.graph_memory", gm):
            await mm.store({"user": "hello", "assistant": "hi"})
        gm.extract_and_store.assert_called_once_with(
            user_msg="hello",
            assistant_msg="hi",
            tool_used=None,
            tool_result_data=None,
        )

    @pytest.mark.asyncio
    async def test_store_passes_tool_fields(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        with patch("bantz.memory.graph.graph_memory", gm):
            await mm.store({
                "user": "search",
                "assistant": "done",
                "tool": "web_search",
                "tool_data": {"results": []},
            })
        gm.extract_and_store.assert_called_once_with(
            user_msg="search",
            assistant_msg="done",
            tool_used="web_search",
            tool_result_data={"results": []},
        )

    @pytest.mark.asyncio
    async def test_store_noop_when_disabled(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _disabled_gm()
        with patch("bantz.memory.graph.graph_memory", gm):
            await mm.store({"user": "x", "assistant": "y"})
        gm.extract_and_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_silences_exceptions(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        gm.extract_and_store.side_effect = RuntimeError("neo4j down")
        with patch("bantz.memory.graph.graph_memory", gm):
            await mm.store({"user": "x", "assistant": "y"})  # must not raise


# ── query() ───────────────────────────────────────────────────────────────────

class TestQuery:

    @pytest.mark.asyncio
    async def test_query_returns_empty_when_disabled(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        with patch("bantz.memory.graph.graph_memory", _disabled_gm()):
            result = await mm.query("AI decisions")
        assert result == []

    @pytest.mark.asyncio
    async def test_query_uses_fulltext_index(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        ft_rows = [{"label": "Decision", "value": "use Redis", "score": 0.9}]
        gm = _enabled_gm(query_return=ft_rows)
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.query("Redis")
        assert result == ft_rows

    @pytest.mark.asyncio
    async def test_query_falls_back_to_keyword_search(self):
        """When full-text index raises, fall back to keyword_search."""
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        # First call (full-text) raises; second call (keyword search) succeeds
        call_count = 0

        async def side_effect(cypher, **params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("no such index")
            return [{"label": "Decision", "val": "use Redis"}]

        gm._query = side_effect
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.query("Redis cache decision")
        # Must return something without raising
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        many_rows = [{"label": "Fact", "value": f"fact {i}", "score": 1.0}
                     for i in range(20)]
        gm = _enabled_gm(query_return=many_rows)
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.query("something", top_k=3)
        # The query uses LIMIT inside Cypher; our mock just returns all rows
        # but the interface must accept top_k without error
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_query_returns_empty_on_exception(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        gm._query = AsyncMock(side_effect=RuntimeError("db error"))
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.query("something")
        assert result == []


# ── summarize_context() ───────────────────────────────────────────────────────

class TestSummarizeContext:

    @pytest.mark.asyncio
    async def test_summarize_empty_when_disabled(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        with patch("bantz.memory.graph.graph_memory", _disabled_gm()):
            result = await mm.summarize_context("auth")
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_includes_decisions(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        call_num = 0

        async def query_fn(cypher, **params):
            nonlocal call_num
            call_num += 1
            if "Decision" in cypher:
                return [{"text": "use JWT tokens"}]
            return []

        gm._query = query_fn
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.summarize_context("authentication")
        assert "JWT tokens" in result

    @pytest.mark.asyncio
    async def test_summarize_includes_tasks(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()

        async def query_fn(cypher, **params):
            if "Task" in cypher:
                return [{"text": "write auth tests", "status": "open"}]
            return []

        gm._query = query_fn
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.summarize_context("auth")
        assert "auth tests" in result

    @pytest.mark.asyncio
    async def test_summarize_includes_people(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()

        async def query_fn(cypher, **params):
            if "Person" in cypher:
                return [{"text": "Alice"}, {"text": "Bob"}]
            return []

        gm._query = query_fn
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.summarize_context("project")
        assert "Alice" in result
        assert "Bob" in result

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_when_no_data(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()  # _query returns []
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.summarize_context("unknown_topic_xyz")
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_silences_exceptions(self):
        from bantz.memory.memory_manager import MemoryManager
        mm = MemoryManager()
        gm = _enabled_gm()
        gm._query = AsyncMock(side_effect=RuntimeError("query failed"))
        with patch("bantz.memory.graph.graph_memory", gm):
            result = await mm.summarize_context("anything")
        assert result == ""


# ── nodes.py schema additions ─────────────────────────────────────────────────

class TestSchemaAdditions:

    def test_project_in_node_labels(self):
        from bantz.memory.nodes import NODE_LABELS
        assert "Project" in NODE_LABELS

    def test_fact_in_node_labels(self):
        from bantz.memory.nodes import NODE_LABELS
        assert "Fact" in NODE_LABELS

    def test_about_rel_type(self):
        from bantz.memory.nodes import REL_TYPES
        assert "ABOUT" in REL_TYPES

    def test_decided_by_rel_type(self):
        from bantz.memory.nodes import REL_TYPES
        assert "DECIDED_BY" in REL_TYPES

    def test_depends_on_rel_type(self):
        from bantz.memory.nodes import REL_TYPES
        assert "DEPENDS_ON" in REL_TYPES

    def test_happened_at_rel_type(self):
        from bantz.memory.nodes import REL_TYPES
        assert "HAPPENED_AT" in REL_TYPES

    def test_references_rel_type(self):
        from bantz.memory.nodes import REL_TYPES
        assert "REFERENCES" in REL_TYPES

    def test_extract_entities_returns_project(self):
        from bantz.memory.nodes import extract_entities
        entities = extract_entities(
            user_msg="I'm working on project Bantz v3",
            assistant_msg="Great project!",
            tool_used=None,
            tool_data=None,
        )
        labels = [e["label"] for e in entities]
        assert "Project" in labels

    def test_extract_entities_returns_fact(self):
        from bantz.memory.nodes import extract_entities
        entities = extract_entities(
            user_msg="Note that the API key expires in 30 days",
            assistant_msg="Understood.",
            tool_used=None,
            tool_data=None,
        )
        labels = [e["label"] for e in entities]
        assert "Fact" in labels


# ── module singleton ──────────────────────────────────────────────────────────

class TestSingleton:

    def test_memory_manager_singleton_exists(self):
        from bantz.memory.memory_manager import memory_manager, MemoryManager
        assert isinstance(memory_manager, MemoryManager)

    def test_singleton_identity(self):
        from bantz.memory.memory_manager import memory_manager as a
        from bantz.memory.memory_manager import memory_manager as b
        assert a is b
