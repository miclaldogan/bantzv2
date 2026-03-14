"""
Tests for ``bantz.core.memory_injector`` (#227).

Covers:
  - Individual hint helpers (style_hint, persona_hint, formality_hint)
  - Individual async fetchers (graph_context, vector_context, deep_memory_context)
  - desktop_context builder
  - inject() concurrent enrichment
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ── Minimal BantzContext stand-in for isolation ──────────────────────
# We import the real one but keep a fallback so the test file is self-contained.
try:
    from bantz.core.context import BantzContext
except ImportError:

    @dataclass
    class BantzContext:  # type: ignore[no-redef]
        en_input: str = ""
        graph_context: str = ""
        vector_context: str = ""
        deep_memory: str = ""
        desktop_context: str = ""
        persona_state: str = ""
        formality_hint: str = ""
        style_hint: str = ""
        profile_hint: str = ""
        feedback_hint: str = ""


from bantz.core.memory_injector import (
    style_hint,
    persona_hint,
    formality_hint,
    desktop_context,
    graph_context,
    vector_context,
    deep_memory_context,
    inject,
)


# ═══════════════════════════════════════════════════════════════════════════
# style_hint
# ═══════════════════════════════════════════════════════════════════════════


class TestStyleHint:
    """Exercise style_hint with various profile configurations."""

    def test_casual_default(self):
        mock_profile = MagicMock()
        mock_profile.response_style = "casual"
        mock_profile.get = lambda key, default="": {"pronoun": "casual", "preferred_address": ""}.get(key, default)
        with patch("bantz.core.profile.profile", mock_profile):
            result = style_hint()
        assert "casual" in result.lower() or "friendly" in result.lower()
        assert "boss" in result

    def test_formal_siz(self):
        mock_profile = MagicMock()
        mock_profile.response_style = "formal"
        mock_profile.get = lambda key, default="": {"pronoun": "siz", "preferred_address": ""}.get(key, default)
        with patch("bantz.core.profile.profile", mock_profile):
            result = style_hint()
        assert "professional" in result.lower() or "respectful" in result.lower()
        assert "ma'am" in result

    def test_custom_address(self):
        mock_profile = MagicMock()
        mock_profile.response_style = "casual"
        mock_profile.get = lambda key, default="": {"pronoun": "casual", "preferred_address": "chief"}.get(key, default)
        with patch("bantz.core.profile.profile", mock_profile):
            result = style_hint()
        assert "chief" in result


# ═══════════════════════════════════════════════════════════════════════════
# persona_hint
# ═══════════════════════════════════════════════════════════════════════════


class TestPersonaHint:
    def test_returns_build_output(self):
        mock_builder = MagicMock()
        mock_builder.build.return_value = "Curious and playful"
        with patch("bantz.core.memory_injector.persona_builder", mock_builder, create=True):
            # persona_hint imports persona_builder lazily, so we patch the import path
            with patch("bantz.personality.persona.persona_builder", mock_builder):
                result = persona_hint()
        # If the lazy import succeeds it returns the build output
        assert isinstance(result, str)

    def test_returns_empty_on_import_error(self):
        with patch.dict("sys.modules", {"bantz.personality.persona": None}):
            result = persona_hint()
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# formality_hint
# ═══════════════════════════════════════════════════════════════════════════


class TestFormalityHint:
    def test_returns_bonding_hint(self):
        mock_meter = MagicMock()
        mock_meter.get_formality_hint.return_value = "warm and familiar"
        with patch("bantz.personality.bonding.bonding_meter", mock_meter):
            result = formality_hint()
        assert "Bonding level" in result
        assert "warm and familiar" in result

    def test_returns_empty_on_no_hint(self):
        mock_meter = MagicMock()
        mock_meter.get_formality_hint.return_value = ""
        with patch("bantz.personality.bonding.bonding_meter", mock_meter):
            result = formality_hint()
        assert result == ""

    def test_returns_empty_on_error(self):
        with patch.dict("sys.modules", {"bantz.personality.bonding": None}):
            result = formality_hint()
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# desktop_context
# ═══════════════════════════════════════════════════════════════════════════


class TestDesktopContext:
    def test_returns_empty_when_not_initialized(self):
        mock_det = MagicMock()
        mock_det.initialized = False
        with patch("bantz.agent.app_detector.app_detector", mock_det):
            result = desktop_context()
        assert result == ""

    def test_builds_context_string(self):
        mock_det = MagicMock()
        mock_det.initialized = True
        mock_det.get_workspace_context.return_value = {
            "active_window": {"name": "Firefox", "title": "GitHub"},
            "activity": "browsing",
            "apps": ["Firefox", "Terminal"],
            "ide": {"ide": "VSCode", "file": "brain.py", "project": "bantz"},
            "docker": [{"state": "running", "name": "redis"}],
        }
        with patch("bantz.agent.app_detector.app_detector", mock_det):
            result = desktop_context()
        assert "Firefox" in result
        assert "browsing" in result
        assert "VSCode" in result
        assert "redis" in result

    def test_returns_empty_on_exception(self):
        with patch.dict("sys.modules", {"bantz.agent.app_detector": None}):
            result = desktop_context()
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# Async fetchers
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphContext:
    @pytest.mark.asyncio
    async def test_returns_context_when_enabled(self):
        mock_gm = MagicMock()
        mock_gm.enabled = True
        mock_gm.context_for = AsyncMock(return_value="Entity: Alice → knows → Bob")
        with patch("bantz.memory.graph.graph_memory", mock_gm):
            result = await graph_context("tell me about Alice")
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_returns_empty_on_import_error(self):
        with patch.dict("sys.modules", {"bantz.memory.graph": None}):
            result = await graph_context("hello")
        assert result == ""


class TestVectorContext:
    @pytest.mark.asyncio
    async def test_returns_past_messages(self):
        mock_mem = MagicMock()
        mock_mem.hybrid_search = AsyncMock(return_value=[
            {"source": "chat", "hybrid_score": 0.85, "role": "user", "content": "Hello there!"},
        ])
        mock_mem.search_distillations = AsyncMock(return_value=[])
        with patch("bantz.core.memory.memory", mock_mem):
            result = await vector_context("hello")
        assert "Relevant past context" in result
        assert "Hello there" in result

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_results(self):
        mock_mem = MagicMock()
        mock_mem.hybrid_search = AsyncMock(return_value=[])
        with patch("bantz.core.memory.memory", mock_mem):
            result = await vector_context("xyz")
        assert result == ""


class TestDeepMemoryContext:
    @pytest.mark.asyncio
    async def test_returns_probe_output(self):
        mock_probe = MagicMock()
        mock_probe.probe = AsyncMock(return_value="You mentioned a cat named Whiskers")
        with patch("bantz.memory.deep_probe.deep_probe", mock_probe):
            result = await deep_memory_context("do I have a pet?")
        assert "Whiskers" in result

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        with patch.dict("sys.modules", {"bantz.memory.deep_probe": None}):
            result = await deep_memory_context("hello")
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# inject() — concurrent enrichment
# ═══════════════════════════════════════════════════════════════════════════


class TestInject:
    @pytest.mark.asyncio
    async def test_populates_all_fields(self):
        ctx = BantzContext()
        mock_profile = MagicMock()
        mock_profile.response_style = "casual"
        mock_profile.get = lambda k, d="": d
        mock_profile.prompt_hint.return_value = "User: TestUser"

        with (
            patch("bantz.core.memory_injector.graph_context", new_callable=AsyncMock, return_value="graph data"),
            patch("bantz.core.memory_injector.vector_context", new_callable=AsyncMock, return_value="vector data"),
            patch("bantz.core.memory_injector.deep_memory_context", new_callable=AsyncMock, return_value="deep data"),
            patch("bantz.core.memory_injector.desktop_context", return_value="desktop data"),
            patch("bantz.core.memory_injector.persona_hint", return_value="persona data"),
            patch("bantz.core.memory_injector.formality_hint", return_value="formality data"),
            patch("bantz.core.memory_injector.style_hint", return_value="style data"),
            patch("bantz.core.profile.profile", mock_profile),
        ):
            await inject(ctx, "test input")

        assert ctx.graph_context == "graph data"
        assert ctx.vector_context == "vector data"
        assert ctx.deep_memory == "deep data"
        assert ctx.desktop_context == "desktop data"
        assert ctx.persona_state == "persona data"
        assert ctx.formality_hint == "formality data"
        assert ctx.style_hint == "style data"
        assert ctx.profile_hint == "User: TestUser"

    @pytest.mark.asyncio
    async def test_handles_async_exceptions_gracefully(self):
        """If one memory source fails, others still populate."""
        ctx = BantzContext()
        mock_profile = MagicMock()
        mock_profile.response_style = "casual"
        mock_profile.get = lambda k, d="": d
        mock_profile.prompt_hint.return_value = ""

        async def failing_graph(_msg):
            raise RuntimeError("neo4j down")

        with (
            patch("bantz.core.memory_injector.graph_context", side_effect=failing_graph),
            patch("bantz.core.memory_injector.vector_context", new_callable=AsyncMock, return_value="ok"),
            patch("bantz.core.memory_injector.deep_memory_context", new_callable=AsyncMock, return_value="ok"),
            patch("bantz.core.memory_injector.desktop_context", return_value=""),
            patch("bantz.core.memory_injector.persona_hint", return_value=""),
            patch("bantz.core.memory_injector.formality_hint", return_value=""),
            patch("bantz.core.memory_injector.style_hint", return_value=""),
            patch("bantz.core.profile.profile", mock_profile),
        ):
            await inject(ctx, "test input")

        # graph failed → empty string, others OK
        assert ctx.graph_context == ""
        assert ctx.vector_context == "ok"
        assert ctx.deep_memory == "ok"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Verify that async fetchers run concurrently, not sequentially."""
        call_order = []

        async def slow_graph(msg):
            call_order.append("graph_start")
            await asyncio.sleep(0.05)
            call_order.append("graph_end")
            return "g"

        async def slow_vector(msg):
            call_order.append("vector_start")
            await asyncio.sleep(0.05)
            call_order.append("vector_end")
            return "v"

        async def slow_deep(msg):
            call_order.append("deep_start")
            await asyncio.sleep(0.05)
            call_order.append("deep_end")
            return "d"

        ctx = BantzContext()
        mock_profile = MagicMock()
        mock_profile.response_style = "casual"
        mock_profile.get = lambda k, d="": d
        mock_profile.prompt_hint.return_value = ""

        with (
            patch("bantz.core.memory_injector.graph_context", side_effect=slow_graph),
            patch("bantz.core.memory_injector.vector_context", side_effect=slow_vector),
            patch("bantz.core.memory_injector.deep_memory_context", side_effect=slow_deep),
            patch("bantz.core.memory_injector.desktop_context", return_value=""),
            patch("bantz.core.memory_injector.persona_hint", return_value=""),
            patch("bantz.core.memory_injector.formality_hint", return_value=""),
            patch("bantz.core.memory_injector.style_hint", return_value=""),
            patch("bantz.core.profile.profile", mock_profile),
        ):
            await inject(ctx, "test")

        # All _start events should come before all _end events (concurrent)
        starts = [i for i, e in enumerate(call_order) if e.endswith("_start")]
        ends = [i for i, e in enumerate(call_order) if e.endswith("_end")]
        assert max(starts) < min(ends), f"Not concurrent: {call_order}"
