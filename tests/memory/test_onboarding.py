"""Tests for MemPalace onboarding + KG write verification."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_bridge(tmp_path):
    """Create a fresh bridge with isolated paths."""
    import os

    os.environ["BANTZ_MEMPALACE_ENABLED"] = "true"
    os.environ["BANTZ_MEMPALACE_PALACE_PATH"] = str(tmp_path / "palace")
    os.environ["BANTZ_MEMPALACE_KG_PATH"] = str(tmp_path / "kg.sqlite3")
    os.environ["BANTZ_MEMPALACE_IDENTITY_PATH"] = str(tmp_path / "identity.txt")

    # Ensure onboarding doesn't trigger interactively
    (tmp_path / ".bantz_onboarding_done").write_text("skip")

    from bantz.memory.bridge import MemPalaceBridge

    bridge = MemPalaceBridge()
    return bridge


def _kg_rows(kg_path: str, table: str) -> list[dict]:
    db = sqlite3.connect(kg_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(f"SELECT * FROM {table}").fetchall()]
    db.close()
    return rows


# ═════════════════════════════════════════════════════════════════════════
# Task 1 — KG Write Tests
# ═════════════════════════════════════════════════════════════════════════


class TestKGExtraction:
    """Verify _extract_kg_triples writes entities + triples to KG."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Create a fresh bridge with its own KG per test."""
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.entity_registry import EntityRegistry
        from mempalace.layers import MemoryStack
        from bantz.memory.bridge import MemPalaceBridge

        palace_path = str(tmp_path / "palace")
        self.kg_path = str(tmp_path / "kg.sqlite3")
        identity_path = str(tmp_path / "identity.txt")
        Path(palace_path).mkdir(parents=True, exist_ok=True)
        Path(identity_path).write_text("## L0\ntest identity\n")

        self.bridge = MemPalaceBridge()
        self.bridge._stack = MemoryStack(
            palace_path=palace_path, identity_path=identity_path,
        )
        self.bridge._kg = KnowledgeGraph(db_path=self.kg_path)
        self.bridge._registry = EntityRegistry.load(config_dir=tmp_path)
        self.bridge._initialized = True

        yield
        self.bridge.close()

    def test_turkish_name_extracted(self):
        """'Benim adım İclal' → entity İclal + triple user→name_is→İclal."""
        self.bridge._extract_kg_triples(
            "Benim adım İclal",
            "Merhaba İclal!",
            None, None,
        )
        entities = _kg_rows(self.kg_path, "entities")
        names = {e["name"] for e in entities}
        assert "İclal" in names

        triples = _kg_rows(self.kg_path, "triples")
        name_triples = [t for t in triples if t["predicate"] == "name_is"]
        assert any(t["object"] == "i̇clal" for t in name_triples)

    def test_english_name_extracted(self):
        """'My name is Alice' → entity Alice + triple."""
        self.bridge._extract_kg_triples(
            "My name is Alice",
            "Hi Alice!",
            None, None,
        )
        entities = _kg_rows(self.kg_path, "entities")
        names = {e["name"] for e in entities}
        assert "Alice" in names

    def test_turkish_profession_extracted(self):
        """'yazılımcıyım' → works_as triple."""
        self.bridge._extract_kg_triples(
            "Benim adım İclal, yazılımcıyım",
            "Harika!",
            None, None,
        )
        triples = _kg_rows(self.kg_path, "triples")
        works_as = [t for t in triples if t["predicate"] == "works_as"]
        assert len(works_as) >= 1
        assert any("yazılımcı" in t["object"] for t in works_as)

    def test_person_after_preposition(self):
        """'meeting with Bob' → entity Bob."""
        self.bridge._extract_kg_triples(
            "Check calendar",
            "You have a meeting with Bob at 3pm",
            None, None,
        )
        entities = _kg_rows(self.kg_path, "entities")
        names = {e["name"] for e in entities}
        assert "Bob" in names

    def test_tool_triple(self):
        """Using a tool → used_<tool> triple."""
        self.bridge._extract_kg_triples(
            "Check my calendar",
            "Here are your events",
            "calendar_list", None,
        )
        triples = _kg_rows(self.kg_path, "triples")
        tool_triples = [t for t in triples if "used_" in t["predicate"]]
        assert len(tool_triples) >= 1

    def test_decision_triple(self):
        """'Let's go with PostgreSQL' → decided triple."""
        self.bridge._extract_kg_triples(
            "Let's go with PostgreSQL for this project",
            "Good choice!",
            None, None,
        )
        triples = _kg_rows(self.kg_path, "triples")
        decisions = [t for t in triples if t["predicate"] == "decided"]
        assert len(decisions) >= 1
        assert any("postgresql" in t["object"].lower() for t in decisions)

    def test_preference_triple(self):
        """'I prefer dark mode' → prefers triple."""
        self.bridge._extract_kg_triples(
            "I prefer dark mode",
            "Noted!",
            None, None,
        )
        triples = _kg_rows(self.kg_path, "triples")
        prefs = [t for t in triples if t["predicate"] == "prefers"]
        assert len(prefs) >= 1
        assert any("dark_mode" in t["object"] for t in prefs)

    def test_no_false_positive_stop_words(self):
        """Common words should NOT appear as person entities."""
        self.bridge._extract_kg_triples(
            "Hello, how are you?",
            "I'm fine!",
            None, None,
        )
        entities = _kg_rows(self.kg_path, "entities")
        bad = {"Hello", "How", "Are", "You", "Fine"}
        names = {e["name"] for e in entities}
        assert names.isdisjoint(bad), f"False positives found: {names & bad}"

    async def test_store_exchange_writes_kg(self):
        """Full store_exchange → both ChromaDB + KG populated."""
        await self.bridge.store_exchange(
            user_msg="Benim adım İclal, yazılımcıyım",
            assistant_msg="Merhaba İclal!",
        )
        triples = _kg_rows(self.kg_path, "triples")
        assert len(triples) > 0, "store_exchange should write KG triples"


# ═════════════════════════════════════════════════════════════════════════
# Task 2 — Onboarding Tests
# ═════════════════════════════════════════════════════════════════════════


class TestOnboarding:
    """Verify onboarding flow seeds identity, KG, and EntityRegistry."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = tmp_path
        self.identity_path = str(tmp_path / "identity.txt")
        self.kg_path = str(tmp_path / "kg.sqlite3")
        self.palace_parent = str(tmp_path)

        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.entity_registry import EntityRegistry

        self.kg = KnowledgeGraph(db_path=self.kg_path)
        self.registry = EntityRegistry.load(config_dir=tmp_path)

    def _sample_answers(self) -> dict[str, str]:
        return {
            "name": "İclal",
            "profession": "Yazılım mühendisi",
            "language": "tr",
            "interests": "backend, AI, distributed systems",
            "tools": "Python, Neovim, Docker",
        }

    def test_seed_identity_writes_file(self):
        """Identity file is created with correct content."""
        from bantz.memory.onboarding import seed_identity

        answers = self._sample_answers()
        seed_identity(answers, self.identity_path)

        content = Path(self.identity_path).read_text(encoding="utf-8")
        assert "İclal" in content
        assert "Yazılım mühendisi" in content
        assert "backend" in content
        assert "Python" in content

    def test_seed_kg_writes_triples(self):
        """KG should have name, profession, language, interests, tools triples."""
        from bantz.memory.onboarding import seed_knowledge_graph

        answers = self._sample_answers()
        count = seed_knowledge_graph(answers, self.kg)

        assert count >= 8  # name + profession + language + 3 interests + 3 tools

        triples = _kg_rows(self.kg_path, "triples")
        predicates = {t["predicate"] for t in triples}
        assert "name_is" in predicates
        assert "works_as" in predicates
        assert "speaks" in predicates
        assert "interested_in" in predicates
        assert "uses_tool" in predicates

    def test_seed_kg_entities_created(self):
        """User name should be an entity in KG."""
        from bantz.memory.onboarding import seed_knowledge_graph

        answers = self._sample_answers()
        seed_knowledge_graph(answers, self.kg)

        entities = _kg_rows(self.kg_path, "entities")
        names = {e["name"] for e in entities}
        assert "İclal" in names

    def test_seed_entity_registry(self):
        """EntityRegistry should know the user as a person."""
        from bantz.memory.onboarding import seed_entity_registry

        answers = self._sample_answers()
        seed_entity_registry(answers, self.registry)

        result = self.registry.lookup("İclal")
        assert result["type"] == "person"
        assert result["source"] == "onboarding"

    def test_flag_set_after_onboarding(self):
        """After onboarding completes, flag file should exist."""
        from bantz.memory.onboarding import (
            is_onboarding_done,
            run_onboarding_noninteractive,
        )

        assert not is_onboarding_done(self.palace_parent)

        run_onboarding_noninteractive(
            answers=self._sample_answers(),
            identity_path=self.identity_path,
            kg=self.kg,
            registry=self.registry,
            palace_parent=self.palace_parent,
        )

        assert is_onboarding_done(self.palace_parent)

    def test_flag_prevents_rerun(self):
        """Once flag is set, is_onboarding_done returns True."""
        from bantz.memory.onboarding import (
            is_onboarding_done,
            run_onboarding_noninteractive,
        )

        run_onboarding_noninteractive(
            answers=self._sample_answers(),
            identity_path=self.identity_path,
            kg=self.kg,
            registry=self.registry,
            palace_parent=self.palace_parent,
        )

        assert is_onboarding_done(self.palace_parent)

    def test_noninteractive_returns_triple_count(self):
        """run_onboarding_noninteractive returns the number of triples."""
        from bantz.memory.onboarding import run_onboarding_noninteractive

        count = run_onboarding_noninteractive(
            answers=self._sample_answers(),
            identity_path=self.identity_path,
            kg=self.kg,
            registry=self.registry,
            palace_parent=self.palace_parent,
        )
        assert count >= 8

    def test_empty_name_returns_zero(self):
        """If name is empty, onboarding should bail out."""
        from bantz.memory.onboarding import run_onboarding_noninteractive

        count = run_onboarding_noninteractive(
            answers={"profession": "dev"},
            identity_path=self.identity_path,
            kg=self.kg,
            registry=self.registry,
            palace_parent=self.palace_parent,
        )
        assert count == 0

    async def test_full_e2e_with_bridge(self):
        """Bridge + noninteractive onboarding → KG + identity populated."""
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.entity_registry import EntityRegistry
        from mempalace.layers import MemoryStack
        from bantz.memory.bridge import MemPalaceBridge
        from bantz.memory.onboarding import run_onboarding_noninteractive

        palace_path = str(self.tmp / "e2e_palace")
        kg_path = str(self.tmp / "e2e_kg.sqlite3")
        identity_path = str(self.tmp / "e2e_identity.txt")
        palace_parent = str(self.tmp)
        Path(palace_path).mkdir(parents=True, exist_ok=True)

        bridge = MemPalaceBridge()
        bridge._stack = MemoryStack(
            palace_path=palace_path, identity_path=identity_path,
        )
        bridge._kg = KnowledgeGraph(db_path=kg_path)
        bridge._registry = EntityRegistry.load(config_dir=self.tmp)
        bridge._initialized = True

        # Run onboarding
        count = run_onboarding_noninteractive(
            answers=self._sample_answers(),
            identity_path=identity_path,
            kg=bridge._kg,
            registry=bridge._registry,
            palace_parent=palace_parent,
        )

        # Verify KG
        triples = _kg_rows(kg_path, "triples")
        assert len(triples) >= 8

        # Verify identity
        identity = Path(identity_path).read_text(encoding="utf-8")
        assert "İclal" in identity

        # Now store an exchange — should also add to KG
        await bridge.store_exchange(
            user_msg="Python ile backend geliştiriyorum",
            assistant_msg="Harika bir seçim!",
        )

        # Should have more triples now
        triples_after = _kg_rows(kg_path, "triples")
        assert len(triples_after) >= len(triples)

        bridge.close()


# ═════════════════════════════════════════════════════════════════════════
# Task 3 — Heuristic Extraction Tests
# ═════════════════════════════════════════════════════════════════════════


class TestHeuristicExtraction:
    """Verify _extract_without_llm extracts structured data from natural responses."""

    def test_name_turkish_diyebilirsin(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("name", "İclal diyebilirsin") == "İclal"

    def test_name_english_call_me(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("name", "call me Bob") == "Bob"

    def test_name_english_im(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("name", "I'm Alice") == "Alice"

    def test_name_short_single_word(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("name", "Mehmet") == "Mehmet"

    def test_name_bana_pattern(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("name", "bana İclal de") == "İclal"

    def test_language_turkish(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("language", "Türkçe konuşalım") == "tr"

    def test_language_english(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("language", "English please") == "en"

    def test_language_code_direct(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("language", "tr") == "tr"

    def test_profession_passthrough(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("profession", "yazılımcıyım") == "yazılımcıyım"

    def test_empty_returns_empty(self):
        from bantz.memory.onboarding import _extract_without_llm
        assert _extract_without_llm("name", "") == ""
        assert _extract_without_llm("tools", "  ") == ""