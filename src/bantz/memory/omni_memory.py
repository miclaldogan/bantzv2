"""
Bantz — OmniMemoryManager (#211)

Unified memory orchestrator that replaces the scatter-shot context gathering
with a **budget-aware, parallel hybrid search** pipeline.

Architecture (Architect's Revision):
    ┌───────────────────────────────────────────────────────────────┐
    │  OmniMemoryManager.recall(user_msg)  [ASYNC HYBRID]         │
    │       │                                                      │
    │       ├─ Task A: Neo4j Graph Search   ─┐                    │
    │       │                                 ├─ Merge + Re-rank  │
    │       ├─ Task B: Vector Semantic Search ┘                    │
    │       │                                                      │
    │       ├─ Task C: Deep Memory Probe                           │
    │       │                                                      │
    │       └─ Budget Enforcer → trim to ~MAX_MEMORY_TOKENS        │
    └───────────────────────────────────────────────────────────────┘

Design decisions:
  💡1: Real-time context (desktop, persona) is NOT managed here.
       Those are short, ephemeral signals injected directly by
       memory_injector — never filtered or removed.

  💡2: Hybrid search runs Graph + Vector in PARALLEL (asyncio.gather).
       If Graph finds entities → vector results are re-ranked by graph
       relevance.  If Graph is empty → pure vector results are used.
       This prevents the "Amnesia by Over-Filtering" syndrome.

  💡3: All I/O runs via asyncio.gather — zero sequential waiting.
       Latency ≈ max(graph_latency, vector_latency) not the sum.

  💡4: Token budget (~400 tokens for memory) prevents context bloat.
       Uses a simple char-based proxy (1 token ≈ 4 chars) for speed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("bantz.omni_memory")

# ── Token budget constants ──────────────────────────────────────────────
# Approximate: 1 token ≈ 4 chars for English text.  Using chars avoids
# importing a tokenizer (fast-path).
MAX_MEMORY_TOKENS: int = 400          # hard cap for combined memory text
_CHARS_PER_TOKEN: int = 4
_MAX_MEMORY_CHARS: int = MAX_MEMORY_TOKENS * _CHARS_PER_TOKEN  # 1600

# Section budget allocation (approximate percentages)
_GRAPH_BUDGET_PCT: float = 0.35       # 35% → graph context
_VECTOR_BUDGET_PCT: float = 0.40      # 40% → vector (semantic search + distillations)
_DEEP_BUDGET_PCT: float = 0.25        # 25% → deep memory probe


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, breaking at last newline if possible."""
    if not text or len(text) <= max_chars:
        return text or ""
    # Try to break at a newline
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars // 2:
        return cut[:last_nl] + "\n…"
    return cut + "…"


class OmniMemoryManager:
    """Unified memory orchestrator with parallel hybrid search.

    Merges graph, vector, and deep memory results into a single,
    budget-constrained context block for the system prompt.
    """

    def __init__(
        self,
        *,
        max_memory_tokens: int = MAX_MEMORY_TOKENS,
        graph_budget_pct: float = _GRAPH_BUDGET_PCT,
        vector_budget_pct: float = _VECTOR_BUDGET_PCT,
        deep_budget_pct: float = _DEEP_BUDGET_PCT,
    ) -> None:
        self._max_chars = max_memory_tokens * _CHARS_PER_TOKEN
        self._graph_pct = graph_budget_pct
        self._vector_pct = vector_budget_pct
        self._deep_pct = deep_budget_pct

    async def recall(self, user_msg: str) -> MemoryRecallResult:
        """Run parallel hybrid search and return budget-trimmed results.

        Fires Graph + Vector + DeepMemory concurrently.  If Graph finds
        entities, vector results can be enriched.  If Graph is empty,
        pure vector results are used (no data loss).

        Returns a ``MemoryRecallResult`` with individual sections +
        a pre-formatted combined block.
        """
        # ── Fire all searches in parallel ─────────────────────────────
        graph_coro = self._graph_search(user_msg)
        vector_coro = self._vector_search(user_msg)
        deep_coro = self._deep_search(user_msg)

        graph_raw, vector_raw, deep_raw = await asyncio.gather(
            graph_coro, vector_coro, deep_coro,
            return_exceptions=True,
        )

        # Gracefully handle exceptions → empty string
        graph_text = graph_raw if isinstance(graph_raw, str) else ""
        vector_text = vector_raw if isinstance(vector_raw, str) else ""
        deep_text = deep_raw if isinstance(deep_raw, str) else ""

        # ── Re-rank: if Graph found entities, note it for vector ──────
        graph_entities = self._extract_entity_names(graph_text)
        if graph_entities and vector_text:
            vector_text = self._rerank_vector_with_graph(
                vector_text, graph_entities
            )

        # ── Apply per-section token budgets ───────────────────────────
        graph_chars = int(self._max_chars * self._graph_pct)
        vector_chars = int(self._max_chars * self._vector_pct)
        deep_chars = int(self._max_chars * self._deep_pct)

        graph_trimmed = _truncate(graph_text, graph_chars)
        vector_trimmed = _truncate(vector_text, vector_chars)
        deep_trimmed = _truncate(deep_text, deep_chars)

        # ── Redistribute unused budget ────────────────────────────────
        # If one section is short, give its slack to the others
        used = len(graph_trimmed) + len(vector_trimmed) + len(deep_trimmed)
        slack = self._max_chars - used
        if slack > 100:
            # Give slack to whichever section was truncated most
            sections = [
                ("graph", graph_text, graph_trimmed, graph_chars),
                ("vector", vector_text, vector_trimmed, vector_chars),
                ("deep", deep_text, deep_trimmed, deep_chars),
            ]
            for name, full, trimmed, budget in sections:
                if len(trimmed) >= budget and len(full) > len(trimmed):
                    extra = min(slack, len(full) - len(trimmed))
                    if name == "graph":
                        graph_trimmed = _truncate(graph_text, budget + extra)
                    elif name == "vector":
                        vector_trimmed = _truncate(vector_text, budget + extra)
                    elif name == "deep":
                        deep_trimmed = _truncate(deep_text, budget + extra)
                    slack -= extra
                    if slack <= 100:
                        break

        # ── Build combined block ──────────────────────────────────────
        combined = self._merge_sections(
            graph_trimmed, vector_trimmed, deep_trimmed
        )

        # Final safety trim
        if len(combined) > self._max_chars:
            combined = combined[:self._max_chars] + "…"

        total_tokens_approx = len(combined) // _CHARS_PER_TOKEN

        if combined:
            log.debug(
                "OmniMemory recall: ~%d tokens "
                "(graph=%d, vector=%d, deep=%d chars)",
                total_tokens_approx,
                len(graph_trimmed), len(vector_trimmed), len(deep_trimmed),
            )

        return MemoryRecallResult(
            graph_context=graph_trimmed,
            vector_context=vector_trimmed,
            deep_memory=deep_trimmed,
            combined=combined,
            total_chars=len(combined),
            total_tokens_approx=total_tokens_approx,
        )

    # ── Private: individual search methods ────────────────────────────

    @staticmethod
    async def _graph_search(user_msg: str) -> str:
        """Query Neo4j graph memory (empty string if disabled/error)."""
        try:
            from bantz.memory.graph import graph_memory
        except ImportError:
            return ""
        if graph_memory and graph_memory.enabled:
            try:
                return await graph_memory.context_for(user_msg)
            except Exception as exc:
                log.debug("Graph search failed: %s", exc)
        return ""

    @staticmethod
    async def _vector_search(user_msg: str, limit: int = 3) -> str:
        """Semantic search over past conversations + distillations."""
        try:
            from bantz.core.memory import memory
            results = await memory.hybrid_search(user_msg, limit=limit)
            if not results:
                return ""
            lines = []
            for r in results:
                src = r.get("source", "?")
                score = r.get("hybrid_score", 0)
                lines.append(f"[{src} {score:.2f}] {r['role']}: {r['content'][:200]}")

            # Append distillation context (#118)
            try:
                distills = await memory.search_distillations(user_msg, limit=2)
                for d in distills:
                    lines.append(
                        f"[session-summary {d['score']:.2f}] {d['summary'][:200]}"
                    )
            except Exception:
                pass

            return "Relevant past context:\n" + "\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    async def _deep_search(user_msg: str) -> str:
        """Spontaneous deep memory recall (#170)."""
        try:
            from bantz.memory.deep_probe import deep_probe
            return await deep_probe.probe(user_msg)
        except Exception:
            return ""

    # ── Re-ranking: graph-informed vector boost ───────────────────────

    @staticmethod
    def _extract_entity_names(graph_text: str) -> set[str]:
        """Extract entity names from graph context for re-ranking.

        Looks for patterns like "Known people: Alice, Bob" or
        "[Person] Alice" or "[Task] Build the widget".
        """
        if not graph_text:
            return set()
        names: set[str] = set()
        for line in graph_text.splitlines():
            line = line.strip()
            # "Known people: Alice, Bob, Charlie"
            if line.startswith("Known people:"):
                for name in line.split(":")[1].split(","):
                    name = name.strip()
                    if name:
                        names.add(name.lower())
            # "[Person] Alice" or "[Task] Build widget"
            elif line.startswith("[") and "]" in line:
                val = line.split("]", 1)[1].strip()
                if val:
                    names.add(val.lower())
            # "  - Something Something"
            elif line.startswith("- "):
                val = line[2:].strip()
                # Extract first phrase (up to " (" or end)
                paren = val.find(" (")
                if paren > 0:
                    val = val[:paren]
                if val and len(val) < 60:
                    names.add(val.lower())
        return names

    @staticmethod
    def _rerank_vector_with_graph(
        vector_text: str, graph_entities: set[str]
    ) -> str:
        """Re-order vector search lines: graph-matching lines first.

        If a vector result mentions an entity found in the graph,
        it gets promoted to the top of the list.
        """
        if not vector_text or not graph_entities:
            return vector_text

        lines = vector_text.splitlines()
        header = ""
        data_lines: list[str] = []

        for line in lines:
            if line.startswith("Relevant past") or line.startswith("==="):
                header = line
            else:
                data_lines.append(line)

        if not data_lines:
            return vector_text

        # Score each line: 1 if it mentions a graph entity, 0 otherwise
        scored: list[tuple[int, str]] = []
        for line in data_lines:
            low = line.lower()
            boost = 1 if any(ent in low for ent in graph_entities) else 0
            scored.append((boost, line))

        # Stable sort: boosted lines first, original order preserved
        scored.sort(key=lambda x: -x[0])
        result_lines = [s[1] for s in scored]

        if header:
            return header + "\n" + "\n".join(result_lines)
        return "\n".join(result_lines)

    # ── Merge into final block ────────────────────────────────────────

    @staticmethod
    def _merge_sections(
        graph: str, vector: str, deep: str,
    ) -> str:
        """Combine non-empty sections into a single context block."""
        parts: list[str] = []
        if graph:
            parts.append(graph)
        if vector:
            parts.append(vector)
        if deep:
            parts.append(deep)
        return "\n".join(parts)


class MemoryRecallResult:
    """Container for OmniMemoryManager.recall() results."""

    __slots__ = (
        "graph_context", "vector_context", "deep_memory",
        "combined", "total_chars", "total_tokens_approx",
    )

    def __init__(
        self,
        *,
        graph_context: str = "",
        vector_context: str = "",
        deep_memory: str = "",
        combined: str = "",
        total_chars: int = 0,
        total_tokens_approx: int = 0,
    ) -> None:
        self.graph_context = graph_context
        self.vector_context = vector_context
        self.deep_memory = deep_memory
        self.combined = combined
        self.total_chars = total_chars
        self.total_tokens_approx = total_tokens_approx

    def __repr__(self) -> str:
        return (
            f"MemoryRecallResult(~{self.total_tokens_approx} tokens, "
            f"{self.total_chars} chars)"
        )

    @property
    def is_empty(self) -> bool:
        return not self.combined


# Module singleton
omni_memory = OmniMemoryManager()
