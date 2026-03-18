"""Tests for OmniMemoryManager CRUD methods + context_builder token cap (#219)."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── store() ───────────────────────────────────────────────────────────────────

class TestStore:

    def test_store_calls_kv_set(self):
        """store() writes to data_layer.kv."""
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        mock_kv = MagicMock()
        mock_dl = MagicMock(kv=mock_kv)

        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.store("key1", "hello"))

        mock_kv.set.assert_called_once_with("key1", "hello")

    def test_store_with_ttl_encodes_expiry(self):
        """store() with ttl wraps value in JSON with _exp field."""
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        captured = {}
        mock_kv = MagicMock()

        def fake_set(key, value):
            captured["key"] = key
            captured["value"] = value

        mock_kv.set.side_effect = fake_set
        mock_dl = MagicMock(kv=mock_kv)

        before = time.time()
        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.store("ttl_key", "ttl_value", ttl=60))
        after = time.time()

        assert captured["key"] == "ttl_key"
        payload = json.loads(captured["value"])
        assert payload["_v"] == "ttl_value"
        assert before + 60 <= payload["_exp"] <= after + 61

    def test_store_no_kv_silently_noop(self):
        """store() silently no-ops when data_layer.kv is None."""
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        mock_dl = MagicMock(kv=None)
        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.store("k", "v"))  # must not raise

    def test_store_exception_silently_logged(self):
        """store() logs and swallows exceptions."""
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        mock_kv = MagicMock()
        mock_kv.set.side_effect = RuntimeError("disk full")
        mock_dl = MagicMock(kv=mock_kv)
        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.store("k", "v"))  # must not raise


# ── forget() ──────────────────────────────────────────────────────────────────

class TestForget:

    def test_forget_calls_kv_delete(self):
        """forget() removes the key from data_layer.kv."""
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        mock_kv = MagicMock()
        mock_dl = MagicMock(kv=mock_kv)
        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.forget("key1"))
        mock_kv.delete.assert_called_once_with("key1")

    def test_forget_no_kv_silently_noop(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        mock_dl = MagicMock(kv=None)
        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.forget("k"))  # must not raise

    def test_forget_exception_silently_logged(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        mock_kv = MagicMock()
        mock_kv.delete.side_effect = RuntimeError("locked")
        mock_dl = MagicMock(kv=mock_kv)
        with patch("bantz.data.data_layer", mock_dl):
            _run(omm.forget("k"))  # must not raise


# ── summarize() ───────────────────────────────────────────────────────────────

class TestSummarize:

    def test_summarize_returns_summary_from_distiller(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        async def fake_distill(conv_id):
            return {"summary": "This conversation was about AI."}

        with patch("bantz.memory.distiller.distill_session", side_effect=fake_distill):
            result = _run(omm.summarize(42))

        assert result == "This conversation was about AI."

    def test_summarize_returns_empty_on_exception(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        async def bad_distill(conv_id):
            raise RuntimeError("DB error")

        with patch("bantz.memory.distiller.distill_session", side_effect=bad_distill):
            result = _run(omm.summarize(99))

        assert result == ""

    def test_summarize_returns_empty_when_no_summary_key(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        async def fake_distill(conv_id):
            return {}

        with patch("bantz.memory.distiller.distill_session", side_effect=fake_distill):
            result = _run(omm.summarize(1))

        assert result == ""


# ── graph_query() ─────────────────────────────────────────────────────────────

class TestGraphQuery:

    def test_graph_query_delegates_to_graph_memory(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        mock_gm = MagicMock()
        mock_gm.enabled = True
        mock_gm.query = AsyncMock(return_value=[{"name": "Alice"}])

        with patch("bantz.memory.graph.graph_memory", mock_gm):
            result = _run(omm.graph_query("MATCH (p:Person) RETURN p.name AS name LIMIT 1"))

        assert result == [{"name": "Alice"}]

    def test_graph_query_returns_empty_when_disabled(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        mock_gm = MagicMock()
        mock_gm.enabled = False

        with patch("bantz.memory.graph.graph_memory", mock_gm):
            result = _run(omm.graph_query("MATCH (n) RETURN n"))

        assert result == []

    def test_graph_query_returns_empty_on_exception(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        mock_gm = MagicMock()
        mock_gm.enabled = True
        mock_gm.query = AsyncMock(side_effect=RuntimeError("connection refused"))

        with patch("bantz.memory.graph.graph_memory", mock_gm):
            result = _run(omm.graph_query("MATCH (n) RETURN n"))

        assert result == []

    def test_graph_query_passes_params(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        mock_gm = MagicMock()
        mock_gm.enabled = True
        mock_gm.query = AsyncMock(return_value=[])

        with patch("bantz.memory.graph.graph_memory", mock_gm):
            _run(omm.graph_query("MATCH (p:Person {name: $name}) RETURN p", name="Bob"))

        mock_gm.query.assert_called_once_with(
            "MATCH (p:Person {name: $name}) RETURN p", name="Bob"
        )


# ── transaction() ─────────────────────────────────────────────────────────────

class TestTransaction:

    def test_transaction_returns_context_manager(self):
        from bantz.memory.omni_memory import OmniMemoryManager, _OmniTransaction
        omm = OmniMemoryManager()
        tx = omm.transaction()
        assert isinstance(tx, _OmniTransaction)

    @pytest.mark.asyncio
    async def test_transaction_store_and_forget_buffered(self):
        """Operations buffered during transaction, applied on exit."""
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()

        applied_ops = []

        async def fake_aenter(self_):
            return self_

        async def fake_aexit(self_, exc_type, exc, tb):
            applied_ops.extend(self_._ops)

        with patch("bantz.memory.omni_memory._OmniTransaction.__aenter__", fake_aenter), \
             patch("bantz.memory.omni_memory._OmniTransaction.__aexit__", fake_aexit):
            async with omm.transaction() as tx:
                await tx.store("k1", "v1")
                await tx.forget("k2")

        assert ("k1", "v1") in applied_ops
        assert ("k2", None) in applied_ops

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_exception(self):
        """Exception inside transaction → ops NOT applied to SQLite."""
        from bantz.memory.omni_memory import _OmniTransaction

        tx = _OmniTransaction()
        mock_kv = MagicMock()
        mock_dl = MagicMock(kv=mock_kv)

        try:
            async with tx:
                await tx.store("key", "value")
                raise ValueError("something went wrong")
        except ValueError:
            pass

        # __aexit__ received an exception → _apply never called → kv.set never called
        # (The ops are buffered but _apply is skipped because exc_type is not None)
        mock_kv.set.assert_not_called()


# ── context_builder token cap ─────────────────────────────────────────────────

class TestContextBuilderTokenCap:

    def test_max_context_tokens_constant(self):
        from bantz.memory.context_builder import MAX_CONTEXT_TOKENS, _MAX_CONTEXT_CHARS
        assert MAX_CONTEXT_TOKENS == 2000
        assert _MAX_CONTEXT_CHARS == 2000 * 4

    @pytest.mark.asyncio
    async def test_build_context_truncates_at_limit(self):
        """build_context() output must not exceed MAX_CONTEXT_CHARS."""
        from bantz.memory.context_builder import build_context, _MAX_CONTEXT_CHARS

        # Mock query_fn that returns many results
        async def fat_query(cypher, **params):
            return [{"name": "Person " + str(i), "seen": "2024-01-01"} for i in range(200)]

        result = await build_context("tell me everything", fat_query)

        assert len(result) <= _MAX_CONTEXT_CHARS + 5  # +5 for ellipsis/newline

    @pytest.mark.asyncio
    async def test_build_context_short_output_unchanged(self):
        """Short output not truncated."""
        from bantz.memory.context_builder import build_context

        async def minimal_query(cypher, **params):
            if "Person" in cypher:
                return [{"name": "Alice", "seen": "2024-01-01"}]
            return []

        result = await build_context("hi", minimal_query)
        assert "Alice" in result
        assert "…" not in result or len(result) < 8005  # well under limit
