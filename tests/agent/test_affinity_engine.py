"""Tests for bantz.agent.affinity_engine (#221).

Covers:
  - Lifecycle: init / close / re-init
  - Score persistence via SQLiteKVStore
  - Clamping to [-100, 100]
  - get_persona_state() threshold boundaries
  - Thread-safety of add_reward
  - status_line / cumulative_reward compat
  - Source audit: rl_engine must not be imported anywhere in src/
"""
from __future__ import annotations

import ast
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bantz.agent.affinity_engine import AffinityEngine, _PERSONA_TIERS


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def engine(tmp_path: Path) -> AffinityEngine:
    """Fresh AffinityEngine wired to a temp DB."""
    from bantz.data.connection_pool import SQLitePool
    SQLitePool.reset()
    ae = AffinityEngine()
    ae.init(tmp_path / "test.db")
    yield ae
    ae.close()
    SQLitePool.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_init_sets_initialized(self, engine: AffinityEngine):
        assert engine.initialized is True

    def test_close_clears_initialized(self, engine: AffinityEngine):
        engine.close()
        assert engine.initialized is False

    def test_default_score_is_zero(self, engine: AffinityEngine):
        assert engine.get_score() == 0.0

    def test_reinit_restores_persisted_score(self, tmp_path: Path):
        from bantz.data.connection_pool import SQLitePool
        SQLitePool.reset()

        ae = AffinityEngine()
        ae.init(tmp_path / "test.db")
        ae.add_reward(25.0)
        assert ae.get_score() == 25.0
        ae.close()

        # Re-init on same DB — score should survive
        ae2 = AffinityEngine()
        ae2.init(tmp_path / "test.db")
        assert ae2.get_score() == 25.0
        ae2.close()
        SQLitePool.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════════


class TestCoreLogic:
    def test_add_positive_reward(self, engine: AffinityEngine):
        result = engine.add_reward(5.0)
        assert result == 5.0
        assert engine.get_score() == 5.0

    def test_add_negative_reward(self, engine: AffinityEngine):
        result = engine.add_reward(-10.0)
        assert result == -10.0
        assert engine.get_score() == -10.0

    def test_cumulative(self, engine: AffinityEngine):
        engine.add_reward(10.0)
        engine.add_reward(5.0)
        engine.add_reward(-3.0)
        assert engine.get_score() == pytest.approx(12.0)

    def test_clamp_upper(self, engine: AffinityEngine):
        engine.add_reward(200.0)
        assert engine.get_score() == 100.0

    def test_clamp_lower(self, engine: AffinityEngine):
        engine.add_reward(-200.0)
        assert engine.get_score() == -100.0

    def test_clamp_stays_at_boundary(self, engine: AffinityEngine):
        engine.add_reward(100.0)
        engine.add_reward(1.0)
        assert engine.get_score() == 100.0

        engine.add_reward(-201.0)
        assert engine.get_score() == -100.0

    def test_cumulative_reward_alias(self, engine: AffinityEngine):
        engine.add_reward(42.0)
        assert engine.cumulative_reward() == 42.0


# ═══════════════════════════════════════════════════════════════════════════
# Persona state
# ═══════════════════════════════════════════════════════════════════════════


class TestPersonaState:
    def test_very_cold(self, engine: AffinityEngine):
        engine.add_reward(-80.0)
        state = engine.get_persona_state()
        assert "cold" in state.lower() or "distant" in state.lower()

    def test_cold(self, engine: AffinityEngine):
        engine.add_reward(-40.0)
        state = engine.get_persona_state()
        assert "cold" in state.lower() or "guarded" in state.lower()

    def test_neutral(self, engine: AffinityEngine):
        # Score 0 should be neutral
        state = engine.get_persona_state()
        assert "neutral" in state.lower() or "professional" in state.lower()

    def test_warm(self, engine: AffinityEngine):
        engine.add_reward(40.0)
        state = engine.get_persona_state()
        assert "warm" in state.lower() or "friendly" in state.lower()

    def test_deeply_bonded(self, engine: AffinityEngine):
        engine.add_reward(80.0)
        state = engine.get_persona_state()
        assert "bonded" in state.lower() or "best friend" in state.lower()

    def test_boundary_minus_60(self, engine: AffinityEngine):
        engine.add_reward(-60.0)
        # At exactly -60, should transition from "very cold" to "cold"
        state = engine.get_persona_state()
        assert isinstance(state, str) and len(state) > 5

    def test_boundary_minus_20(self, engine: AffinityEngine):
        engine.add_reward(-20.0)
        state = engine.get_persona_state()
        assert isinstance(state, str) and len(state) > 5

    def test_boundary_plus_20(self, engine: AffinityEngine):
        engine.add_reward(20.0)
        state = engine.get_persona_state()
        assert isinstance(state, str) and len(state) > 5

    def test_boundary_plus_60(self, engine: AffinityEngine):
        engine.add_reward(60.0)
        state = engine.get_persona_state()
        assert isinstance(state, str) and len(state) > 5

    def test_all_tiers_return_strings(self):
        """Every tier definition must be a (float, str) pair."""
        for threshold, desc in _PERSONA_TIERS:
            assert isinstance(threshold, float)
            assert isinstance(desc, str)
            assert len(desc) > 10


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestPersistence:
    def test_score_written_to_kv(self, engine: AffinityEngine):
        engine.add_reward(33.0)
        # Read directly from KV store
        raw = engine._kv.get("affinity_score", "0.0")
        assert float(raw) == pytest.approx(33.0)

    def test_persist_failure_does_not_crash(self, engine: AffinityEngine):
        """If KV store write fails, add_reward still works in-memory."""
        engine._kv = MagicMock()
        engine._kv.set.side_effect = RuntimeError("disk full")
        # Should NOT raise
        engine.add_reward(10.0)
        assert engine.get_score() == 10.0

    def test_corrupted_kv_defaults_to_zero(self, tmp_path: Path):
        """If KV value is garbage, init defaults to 0.0."""
        from bantz.data.connection_pool import SQLitePool
        SQLitePool.reset()

        ae = AffinityEngine()
        ae.init(tmp_path / "test.db")
        ae._kv.set("affinity_score", "not_a_number")
        ae.close()

        ae2 = AffinityEngine()
        ae2.init(tmp_path / "test.db")
        assert ae2.get_score() == 0.0
        ae2.close()
        SQLitePool.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Thread safety
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    def test_concurrent_rewards(self, engine: AffinityEngine):
        """Hammer add_reward from 20 threads — score must be consistent."""
        errors: list[Exception] = []

        def _add():
            try:
                for _ in range(100):
                    engine.add_reward(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_add) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 20 threads × 100 iterations × 0.01 = 20.0 (but clamped to 100)
        expected = min(100.0, 20 * 100 * 0.01)
        assert engine.get_score() == pytest.approx(expected, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Status line
# ═══════════════════════════════════════════════════════════════════════════


class TestStatusLine:
    def test_contains_score(self, engine: AffinityEngine):
        engine.add_reward(15.0)
        line = engine.status_line()
        assert "15.0" in line

    def test_contains_persona(self, engine: AffinityEngine):
        line = engine.status_line()
        assert "persona=" in line


# ═══════════════════════════════════════════════════════════════════════════
# Source audit: ensure rl_engine is truly dead in src/
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceAudit:
    """AST-level verification that no source file imports rl_engine."""

    def test_no_rl_engine_imports_in_src(self):
        src_root = Path(__file__).resolve().parents[2] / "src" / "bantz"
        violations: list[str] = []

        for py_file in src_root.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if "rl_engine" in mod:
                        violations.append(
                            f"{py_file.relative_to(src_root.parent.parent)}"
                            f":{node.lineno}: from {mod} import ..."
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "rl_engine" in alias.name:
                            violations.append(
                                f"{py_file.relative_to(src_root.parent.parent)}"
                                f":{node.lineno}: import {alias.name}"
                            )

        assert not violations, (
            "rl_engine imports still found in src/:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )

    def test_rl_engine_module_deleted(self):
        """The old rl_engine.py file must not exist."""
        rl_path = Path(__file__).resolve().parents[2] / "src" / "bantz" / "agent" / "rl_engine.py"
        assert not rl_path.exists(), f"rl_engine.py still exists at {rl_path}"
