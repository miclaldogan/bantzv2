"""
Tests for the smarter SQLite Knowledge-Graph traversal (pure SQLite, no Neo4j).

Covers the four enhancements added to bridge.graph_search / omni_memory:
  1. Relation traversal — follow an entity's triples 2 hops deep.
  2. Time-aware recency boost (1.5x / 1.2x / 0.8x).
  3. Importance decay — access_count increments on recall; log boost.
  4. Topic clustering — vector chunks sharing KG entities boosted 1.3x.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from bantz.memory.bridge import (
    MemPalaceBridge,
    _importance_multiplier,
    _recency_multiplier,
)
from bantz.memory.omni_memory import OmniMemoryManager


@pytest.fixture
def kg_bridge():
    """A MemPalaceBridge wired to a throwaway on-disk KnowledgeGraph."""
    from mempalace.knowledge_graph import KnowledgeGraph

    d = tempfile.mkdtemp()
    kg = KnowledgeGraph(db_path=os.path.join(d, "kg.sqlite3"))
    b = MemPalaceBridge()
    b._kg = kg
    b._registry = None  # rely on capitalised-word detection only
    return b, kg


# ── 1. Relation traversal (2 hops) ──────────────────────────────────────────

class TestRelationTraversal:
    def test_two_hop_traversal_surfaces_indirect_facts(self, kg_bridge):
        b, kg = kg_bridge
        # Baki → works_at → FÜ → located_in → Elazığ  (the user's example)
        kg.add_triple(subject="Baki", predicate="works_at", obj="FÜ")
        kg.add_triple(subject="FÜ", predicate="located_in", obj="Elazığ")
        kg.add_triple(subject="Baki", predicate="mentioned_in", obj="ProjectX")

        res = b.graph_search("Tell me about Baki")
        texts = [f["text"] for f in res["facts"]]

        # Direct (hop-1) facts present
        assert "Baki → works_at → FÜ" in texts
        assert "Baki → mentioned_in → ProjectX" in texts
        # Indirect (hop-2) fact reached by following FÜ
        assert "FÜ → located_in → Elazığ" in texts

        # Entities set spans both hops (for topic clustering downstream)
        assert {"baki", "fü", "elazığ"} <= res["entities"]

    def test_direct_facts_outrank_indirect(self, kg_bridge):
        b, kg = kg_bridge
        kg.add_triple(subject="Baki", predicate="works_at", obj="FÜ")
        kg.add_triple(subject="FÜ", predicate="located_in", obj="Elazığ")

        res = b.graph_search("Baki")
        hops = {f["text"]: f["hop"] for f in res["facts"]}
        # Results are score-sorted; hop-1 fact ranks above the hop-2 fact.
        order = [f["text"] for f in res["facts"]]
        assert order.index("Baki → works_at → FÜ") < order.index(
            "FÜ → located_in → Elazığ"
        )
        assert hops["Baki → works_at → FÜ"] == 1
        assert hops["FÜ → located_in → Elazığ"] == 2

    def test_no_entities_returns_empty(self, kg_bridge):
        b, kg = kg_bridge
        kg.add_triple(subject="Baki", predicate="works_at", obj="FÜ")
        # Query names no known entity (lowercase, undetected)
        res = b.graph_search("what is the weather like")
        assert res["facts"] == []


# ── 2. Time-aware recency boost ──────────────────────────────────────────────

class TestRecencyBoost:
    def test_multiplier_tiers(self):
        now = datetime.now(timezone.utc)
        fmt = "%Y-%m-%d %H:%M:%S"
        assert _recency_multiplier((now - timedelta(days=2)).strftime(fmt)) == 1.5
        assert _recency_multiplier((now - timedelta(days=20)).strftime(fmt)) == 1.2
        assert _recency_multiplier((now - timedelta(days=90)).strftime(fmt)) == 0.8

    def test_unknown_timestamp_demoted_not_dropped(self):
        # Missing/garbage timestamp → 0.8 (demoted) but never 0 (kept).
        assert _recency_multiplier(None) == 0.8
        assert _recency_multiplier("not-a-date") == 0.8

    def test_recent_fact_scores_higher_than_old(self, kg_bridge):
        b, kg = kg_bridge
        kg.add_triple(subject="Baki", predicate="likes", obj="Tea")
        kg.add_triple(subject="Baki", predicate="liked", obj="Coffee")
        # Age the "Coffee" triple by 100 days via direct SQL.
        old = (datetime.now(timezone.utc) - timedelta(days=100)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        conn = sqlite3.connect(kg.db_path)
        conn.execute(
            "UPDATE triples SET extracted_at = ? WHERE object = "
            "(SELECT id FROM entities WHERE name = 'Coffee')",
            (old,),
        )
        conn.commit()
        conn.close()

        res = b.graph_search("Baki")
        scores = {f["text"]: f["score"] for f in res["facts"]}
        assert scores["Baki → likes → Tea"] > scores["Baki → liked → Coffee"]


# ── 3. Importance decay (access_count) ───────────────────────────────────────

class TestImportanceDecay:
    def test_multiplier_monotonic_and_floored(self):
        assert _importance_multiplier(0) == 1.0          # never-accessed → floor
        assert _importance_multiplier(1) > _importance_multiplier(0)
        assert _importance_multiplier(50) > _importance_multiplier(5)

    def test_access_count_increments_on_recall(self, kg_bridge):
        b, kg = kg_bridge
        kg.add_triple(subject="Baki", predicate="works_at", obj="FÜ")

        def count() -> int:
            conn = sqlite3.connect(kg.db_path)
            row = conn.execute(
                "SELECT access_count FROM entities WHERE lower(name) = 'baki'"
            ).fetchone()
            conn.close()
            return row[0]

        b.graph_search("Baki")
        assert count() == 1
        b.graph_search("Baki")
        assert count() == 2


# ── 4. Topic clustering ──────────────────────────────────────────────────────

class TestTopicClustering:
    def test_shared_entity_chunk_boosted(self):
        vector_text = (
            "Relevant past context:\n"
            "[vec 0.30] user: weather is nice today\n"
            "[vec 0.20] user: notes about FÜ campus"
        )
        boosted = OmniMemoryManager._apply_topic_boost(
            vector_text, {"fü", "baki"}
        )
        data = [l for l in boosted.splitlines() if "Relevant" not in l]
        # The FÜ-mentioning chunk (entity-linked) rises above the weather one,
        # despite a lower semantic score.
        fu_idx = next(i for i, l in enumerate(data) if "fü" in l.lower())
        weather_idx = next(i for i, l in enumerate(data) if "weather" in l.lower())
        assert fu_idx < weather_idx

    def test_no_shared_entity_preserves_order(self):
        vector_text = (
            "Relevant past context:\n"
            "[vec 0.9] user: first\n"
            "[vec 0.8] user: second"
        )
        boosted = OmniMemoryManager._apply_topic_boost(vector_text, {"nobody"})
        data = [l for l in boosted.splitlines() if "Relevant" not in l]
        assert "first" in data[0]
        assert "second" in data[1]

    def test_merge_results_applies_boost(self):
        # _merge_results with entities should reorder the vector section.
        graph = "=== Knowledge Graph ===\n  Baki → works_at → FÜ"
        vector = (
            "Relevant past context:\n"
            "[vec 0.30] user: weather\n"
            "[vec 0.20] user: FÜ meeting notes"
        )
        merged = OmniMemoryManager._merge_results(
            graph, vector, "", graph_entities={"fü"}
        )
        assert merged.index("FÜ meeting notes") < merged.index("weather")
