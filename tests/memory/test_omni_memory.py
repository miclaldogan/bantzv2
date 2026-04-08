"""
Tests for Issue #211 — OmniMemoryManager: Context Bloat & GraphRAG Integration.

Covers:
  - OmniMemoryManager.recall(): parallel hybrid search pipeline
  - Token budget enforcement (MAX_MEMORY_TOKENS / per-section budgets)
  - _truncate(): char-based truncation with newline-aware breaks
  - _extract_entity_names(): graph entity extraction
  - _rerank_vector_with_graph(): graph-informed vector re-ranking
  - _merge_sections(): combining non-empty sections
  - Budget redistribution (unused slack → other sections)
  - Hybrid fallback: graph empty → pure vector results
  - MemoryRecallResult container
  - memory_injector.inject() integration with OmniMemory
  - prompt_builder uses memory_combined via OmniMemory
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch



# ── Helpers ──────────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: _truncate
# ═══════════════════════════════════════════════════════════════════════════


class TestTruncate:
    """Test the _truncate helper for budget enforcement."""

    def test_empty_string(self):
        from bantz.memory.omni_memory import _truncate
        assert _truncate("", 100) == ""

    def test_short_string_unchanged(self):
        from bantz.memory.omni_memory import _truncate
        text = "Hello world"
        assert _truncate(text, 100) == text

    def test_exact_length_unchanged(self):
        from bantz.memory.omni_memory import _truncate
        text = "x" * 50
        assert _truncate(text, 50) == text

    def test_truncates_with_ellipsis(self):
        from bantz.memory.omni_memory import _truncate
        text = "a" * 200
        result = _truncate(text, 100)
        assert len(result) <= 102  # 100 chars + "…"
        assert result.endswith("…")

    def test_breaks_at_newline_when_possible(self):
        from bantz.memory.omni_memory import _truncate
        text = "Line 1: some content\nLine 2: more content\nLine 3: final content"
        result = _truncate(text, 45)
        # Should break at a newline boundary
        assert "\n…" in result or result.endswith("…")
        assert len(result) <= 50

    def test_none_returns_empty(self):
        from bantz.memory.omni_memory import _truncate
        assert _truncate(None, 100) == ""  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: _extract_entity_names
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractEntityNames:
    """Test entity extraction from graph context text."""

    def test_empty_graph(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        assert omm._extract_entity_names("") == set()

    def test_known_people_line(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = "Known people: Alice, Bob, Charlie"
        names = omm._extract_entity_names(text)
        assert "alice" in names
        assert "bob" in names
        assert "charlie" in names

    def test_bracket_tags(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = "[Person] Alice\n[Task] Build the widget\n[Event] Team meeting"
        names = omm._extract_entity_names(text)
        assert "alice" in names
        assert "build the widget" in names
        assert "team meeting" in names

    def test_dash_items(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = "- Alice (colleague)\n- Project X\n- Bob"
        names = omm._extract_entity_names(text)
        assert "alice" in names
        assert "project x" in names
        assert "bob" in names

    def test_mixed_format(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = "Known people: Eve\n[Task] Deploy app\n- Frank"
        names = omm._extract_entity_names(text)
        assert "eve" in names
        assert "deploy app" in names
        assert "frank" in names


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: _rerank_vector_with_graph
# ═══════════════════════════════════════════════════════════════════════════


class TestRerankVectorWithGraph:
    """Test graph-informed vector result re-ranking."""

    def test_empty_vector(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        result = omm._rerank_vector_with_graph("", {"alice"})
        assert result == ""

    def test_empty_entities(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = "Relevant past context:\n[vec 0.9] user: hello"
        result = omm._rerank_vector_with_graph(text, set())
        assert result == text

    def test_matching_lines_promoted(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = (
            "Relevant past context:\n"
            "[vec 0.5] user: weather is nice\n"
            "[vec 0.7] user: alice said hello\n"
            "[vec 0.3] user: random stuff"
        )
        result = omm._rerank_vector_with_graph(text, {"alice"})
        lines = result.splitlines()
        # Header stays first
        assert "Relevant past context" in lines[0]
        # "alice" line should come before non-matching lines
        alice_idx = next(i for i, l in enumerate(lines) if "alice" in l.lower())
        weather_idx = next(i for i, l in enumerate(lines) if "weather" in l.lower())
        assert alice_idx < weather_idx

    def test_no_match_order_preserved(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        omm = OmniMemoryManager()
        text = (
            "Relevant past context:\n"
            "[vec 0.9] user: first\n"
            "[vec 0.8] user: second\n"
            "[vec 0.7] user: third"
        )
        result = omm._rerank_vector_with_graph(text, {"nonexistent"})
        # Order should be preserved (stable sort, all score 0)
        lines = [l for l in result.splitlines() if l and "Relevant" not in l]
        assert "first" in lines[0]
        assert "second" in lines[1]
        assert "third" in lines[2]


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: _merge_sections
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeSections:
    """Test section merging."""

    def test_all_empty(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        assert OmniMemoryManager._merge_sections("", "", "") == ""

    def test_one_section(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        assert OmniMemoryManager._merge_sections("graph data", "", "") == "graph data"

    def test_all_sections(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        result = OmniMemoryManager._merge_sections("graph", "vector", "deep")
        assert "graph" in result
        assert "vector" in result
        assert "deep" in result

    def test_two_sections(self):
        from bantz.memory.omni_memory import OmniMemoryManager
        result = OmniMemoryManager._merge_sections("", "vector", "deep")
        assert "vector" in result
        assert "deep" in result
        assert result.startswith("vector")


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: MemoryRecallResult
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryRecallResult:
    """Test the result container."""

    def test_empty_result(self):
        from bantz.memory.omni_memory import MemoryRecallResult
        r = MemoryRecallResult()
        assert r.is_empty
        assert r.total_chars == 0
        assert r.total_tokens_approx == 0
        assert "0 tokens" in repr(r)

    def test_populated_result(self):
        from bantz.memory.omni_memory import MemoryRecallResult
        r = MemoryRecallResult(
            graph_context="graph",
            vector_context="vector",
            deep_memory="deep",
            combined="graph\nvector\ndeep",
            total_chars=22,
            total_tokens_approx=5,
        )
        assert not r.is_empty
        assert r.graph_context == "graph"
        assert r.vector_context == "vector"
        assert r.deep_memory == "deep"
        assert "5 tokens" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: OmniMemoryManager.recall()
# ═══════════════════════════════════════════════════════════════════════════


class TestOmniMemoryRecall:
    """Test the full recall() pipeline with mocked I/O."""

    def test_all_sources_return_data(self):
        """When all three sources return data, all appear in combined."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager()

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = "[Person] Alice\nKnown people: Alice"
            v.return_value = "Relevant past context:\n[vec 0.9] user: alice said hi"
            d.return_value = "[deep] You once discussed Alice's project"

            result = _run(omm.recall("tell me about alice"))

        assert not result.is_empty
        assert "alice" in result.combined.lower()
        assert result.graph_context  # graph section populated
        assert result.vector_context  # vector section populated
        assert result.deep_memory  # deep section populated

    def test_graph_empty_uses_pure_vector(self):
        """When graph returns nothing, vector results are used as-is."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager()

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = ""
            v.return_value = "Relevant past context:\n[vec 0.8] user: hello world"
            d.return_value = ""

            result = _run(omm.recall("hello"))

        assert not result.is_empty
        assert result.graph_context == ""
        assert "hello world" in result.vector_context

    def test_all_empty(self):
        """When all sources return empty, result is empty."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager()

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = ""
            v.return_value = ""
            d.return_value = ""

            result = _run(omm.recall("hello"))

        assert result.is_empty
        assert result.combined == ""

    def test_exception_in_one_source_doesnt_crash(self):
        """If one search raises, it's treated as empty string."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager()

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.side_effect = RuntimeError("Graph search down")
            v.return_value = "Relevant past context:\n[vec 0.8] user: working"
            d.return_value = ""

            result = _run(omm.recall("test"))

        assert not result.is_empty
        assert result.graph_context == ""  # failed source → empty
        assert "working" in result.vector_context

    def test_budget_enforcement(self):
        """Combined output must not exceed MAX_MEMORY_CHARS."""
        from bantz.memory.omni_memory import OmniMemoryManager, _CHARS_PER_TOKEN

        # Use a very small budget to force truncation
        omm = OmniMemoryManager(max_memory_tokens=50)  # 200 chars max

        long_graph = "G" * 300
        long_vector = "V" * 300
        long_deep = "D" * 300

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = long_graph
            v.return_value = long_vector
            d.return_value = long_deep

            result = _run(omm.recall("test"))

        max_allowed = 50 * _CHARS_PER_TOKEN + 10  # small overhead for join/ellipsis
        assert len(result.combined) <= max_allowed

    def test_graph_reranking_applied(self):
        """When graph finds entities, vector results are re-ranked."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager()

        graph_text = "Known people: Alice"
        vector_text = (
            "Relevant past context:\n"
            "[vec 0.3] user: weather today\n"
            "[vec 0.5] user: alice email check"
        )

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = graph_text
            v.return_value = vector_text
            d.return_value = ""

            result = _run(omm.recall("what about alice"))

        # "alice" line should be promoted above "weather" line
        lines = result.vector_context.splitlines()
        data_lines = [l for l in lines if l and "Relevant" not in l]
        if len(data_lines) >= 2:
            alice_idx = next(
                (i for i, l in enumerate(data_lines) if "alice" in l.lower()), -1
            )
            weather_idx = next(
                (i for i, l in enumerate(data_lines) if "weather" in l.lower()), -1
            )
            assert alice_idx < weather_idx

    def test_budget_redistribution(self):
        """Unused budget from empty section is given to others."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager(max_memory_tokens=100)  # 400 chars max

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = ""  # graph empty → 35% budget freed
            v.return_value = "V" * 350  # needs more than 40% of 400 = 160 chars
            d.return_value = ""

            result = _run(omm.recall("test"))

        # Vector should get more than its normal 40% allocation
        assert len(result.vector_context) > 160  # would be ~160 at 40%

    def test_custom_budget_allocation(self):
        """OmniMemoryManager respects custom budget percentages."""
        from bantz.memory.omni_memory import OmniMemoryManager

        omm = OmniMemoryManager(
            max_memory_tokens=100,
            graph_budget_pct=0.10,
            vector_budget_pct=0.80,
            deep_budget_pct=0.10,
        )

        with patch.object(omm, "_graph_search", new_callable=AsyncMock) as g, \
             patch.object(omm, "_vector_search", new_callable=AsyncMock) as v, \
             patch.object(omm, "_deep_search", new_callable=AsyncMock) as d:
            g.return_value = "G" * 200
            v.return_value = "V" * 200
            d.return_value = "D" * 200

            result = _run(omm.recall("test"))

        # Vector should get the lion's share
        assert len(result.vector_context) > len(result.graph_context)
        assert len(result.vector_context) > len(result.deep_memory)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: memory_injector.inject() with OmniMemory
# ═══════════════════════════════════════════════════════════════════════════


class TestInjectIntegration:
    """Test that inject() correctly uses OmniMemory for historical memory."""

    def test_inject_populates_memory_combined(self):
        """inject() should populate ctx.memory_combined via OmniMemory."""
        from bantz.core.context import BantzContext
        from bantz.memory.omni_memory import MemoryRecallResult

        mock_result = MemoryRecallResult(
            graph_context="[Person] Alice",
            vector_context="[vec 0.9] user: alice",
            deep_memory="[deep] alice project",
            combined="[Person] Alice\n[vec 0.9] user: alice\n[deep] alice project",
            total_chars=60,
            total_tokens_approx=15,
        )

        ctx = BantzContext(en_input="tell me about alice")

        with patch("bantz.core.memory_injector.desktop_context", return_value=""), \
             patch("bantz.core.memory_injector.persona_hint", return_value="persona"), \
             patch("bantz.core.memory_injector.formality_hint", return_value="formal"), \
             patch("bantz.core.memory_injector.style_hint", return_value="casual"), \
             patch("bantz.core.profile.profile") as mock_profile, \
             patch("bantz.memory.omni_memory.omni_memory") as mock_omni:

            mock_omni.recall = AsyncMock(return_value=mock_result)
            mock_profile.prompt_hint.return_value = "profile hint"

            # Patch the import inside inject()
            with patch(
                "bantz.core.memory_injector.omni_memory", mock_omni,
                create=True,
            ):
                # Need to patch the dynamic import
                import bantz.core.memory_injector as mi
                _run(mi.inject(ctx, "tell me about alice"))

        assert ctx.memory_combined  # should be populated
        assert ctx.graph_context == "[Person] Alice"
        assert ctx.vector_context == "[vec 0.9] user: alice"

    def test_inject_realtime_always_included(self):
        """Real-time context (desktop, persona) is always populated."""
        from bantz.core.context import BantzContext
        from bantz.memory.omni_memory import MemoryRecallResult

        mock_result = MemoryRecallResult()  # empty memory

        ctx = BantzContext(en_input="hi")

        with patch("bantz.core.memory_injector.desktop_context", return_value="Desktop: Firefox"), \
             patch("bantz.core.memory_injector.persona_hint", return_value="butler persona"), \
             patch("bantz.core.memory_injector.formality_hint", return_value="[Bonding level] warm"), \
             patch("bantz.core.memory_injector.style_hint", return_value="Tone: casual"), \
             patch("bantz.core.profile.profile") as mock_profile, \
             patch("bantz.memory.omni_memory.omni_memory") as mock_omni:

            mock_omni.recall = AsyncMock(return_value=mock_result)
            mock_profile.prompt_hint.return_value = "profile"

            import bantz.core.memory_injector as mi
            _run(mi.inject(ctx, "hi"))

        # Real-time context should always be populated
        assert ctx.desktop_context == "Desktop: Firefox"
        assert ctx.persona_state == "butler persona"
        assert ctx.formality_hint == "[Bonding level] warm"
        assert ctx.style_hint == "Tone: casual"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: prompt_builder with memory_combined
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptBuilderMemoryCombined:
    """Test that prompt_builder uses memory_combined when available."""

    def test_memory_combined_in_system_prompt(self):
        """When memory_combined is set, it appears in the system prompt."""
        from bantz.core.context import BantzContext
        from bantz.core.prompt_builder import build_chat_system

        ctx = BantzContext(
            memory_combined="[Memory] Alice is a colleague who works on Project X",
            desktop_context="Desktop: VSCode",
            persona_state="",
            style_hint="",
            formality_hint="",
            profile_hint="",
        )

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.input_control_enabled = False
            result = build_chat_system(ctx, {"prompt_hint": ""})

        assert "Alice is a colleague" in result
        assert "Project X" in result

    def test_legacy_fallback_when_no_combined(self):
        """When memory_combined is empty, individual fields are used."""
        from bantz.core.context import BantzContext
        from bantz.core.prompt_builder import build_chat_system

        ctx = BantzContext(
            memory_combined="",
            graph_context="[Graph] Bob is a friend",
            vector_context="[Vec] mentioned Bob yesterday",
            deep_memory="[Deep] Bob's birthday is March 5",
            desktop_context="",
            persona_state="",
            style_hint="",
            formality_hint="",
            profile_hint="",
        )

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.input_control_enabled = False
            result = build_chat_system(ctx, {"prompt_hint": ""})

        # All three legacy fields should appear in the prompt
        assert "Bob is a friend" in result
        assert "mentioned Bob yesterday" in result
        assert "birthday is March 5" in result

    def test_combined_preferred_over_legacy(self):
        """When both combined and individual fields exist, combined is used."""
        from bantz.core.context import BantzContext
        from bantz.core.prompt_builder import build_chat_system

        ctx = BantzContext(
            memory_combined="COMBINED MEMORY BLOCK",
            graph_context="SHOULD NOT APPEAR AS SEPARATE",
            vector_context="SHOULD NOT APPEAR AS SEPARATE",
            deep_memory="SHOULD NOT APPEAR AS SEPARATE",
            desktop_context="",
            persona_state="",
            style_hint="",
            formality_hint="",
            profile_hint="",
        )

        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.input_control_enabled = False
            result = build_chat_system(ctx, {"prompt_hint": ""})

        assert "COMBINED MEMORY BLOCK" in result
        # The template has a single {memory_context} placeholder,
        # which gets the combined value — legacy fields are NOT
        # rendered separately.


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════


class TestModuleSingleton:
    """Test the module-level omni_memory singleton."""

    def test_singleton_exists(self):
        from bantz.memory.omni_memory import omni_memory, OmniMemoryManager
        assert isinstance(omni_memory, OmniMemoryManager)

    def test_singleton_default_budget(self):
        from bantz.memory.omni_memory import omni_memory, MAX_MEMORY_TOKENS, _CHARS_PER_TOKEN
        assert omni_memory._max_chars == MAX_MEMORY_TOKENS * _CHARS_PER_TOKEN


# ═══════════════════════════════════════════════════════════════════════════
# context.py field tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBantzContextMemoryField:
    """Test the new memory_combined field on BantzContext."""

    def test_memory_combined_default_empty(self):
        from bantz.core.context import BantzContext
        ctx = BantzContext()
        assert ctx.memory_combined == ""

    def test_has_memory_with_combined(self):
        from bantz.core.context import BantzContext
        ctx = BantzContext(memory_combined="some memory")
        assert ctx.has_memory

    def test_has_memory_with_legacy_fields(self):
        from bantz.core.context import BantzContext
        ctx = BantzContext(graph_context="graph data")
        assert ctx.has_memory

    def test_has_memory_false_when_all_empty(self):
        from bantz.core.context import BantzContext
        ctx = BantzContext()
        assert not ctx.has_memory
