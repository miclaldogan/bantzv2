"""
Tests for Issue #125 — RL framework for routine optimization.

Covers:
  - State: encoding, key round-trip, normalisation
  - QTable: get/set, persist/load, stats
  - Blacklist: block, is_blocked, global block
  - EpisodeLog: record, recent, total, avg_reward
  - RLEngine: suggest (epsilon-greedy), reward (Q-learning update),
              threshold gate, blacklist integration, seed_from_habits
  - Config: new RL fields
"""
from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bantz.agent.rl_engine import (
    ALL_ACTIONS,
    Action,
    Blacklist,
    EpisodeLog,
    QTable,
    RLEngine,
    Reward,
    State,
    encode_state,
)


# ═══════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════


class TestState:
    def test_key_format(self):
        s = State(time_segment="morning", day="monday", location="home", recent_tool="shell")
        assert s.key == "morning|monday|home|shell"

    def test_from_key_round_trip(self):
        s = State(time_segment="evening", day="friday", location="work", recent_tool="web_search")
        s2 = State.from_key(s.key)
        assert s2.time_segment == s.time_segment
        assert s2.day == s.day
        assert s2.location == s.location
        assert s2.recent_tool == s.recent_tool

    def test_to_dict(self):
        s = State(time_segment="morning", day="monday", location="home", recent_tool="")
        d = s.to_dict()
        assert d == {"time_segment": "morning", "day": "monday", "location": "home", "recent_tool": ""}

    def test_frozen(self):
        s = State()
        with pytest.raises(AttributeError):
            s.time_segment = "night"  # type: ignore

    def test_encode_state_normalises(self):
        s = encode_state(time_segment="INVALID", day="Invalid", location="MARS", recent_tool="x" * 50)
        assert s.time_segment == "morning"  # default fallback
        assert s.day == "monday"            # default fallback
        assert s.location == "other"        # default fallback
        assert len(s.recent_tool) == 30     # truncated

    def test_encode_state_valid(self):
        s = encode_state(time_segment="evening", day="Friday", location="Work", recent_tool="shell")
        assert s.time_segment == "evening"
        assert s.day == "friday"
        assert s.location == "work"
        assert s.recent_tool == "shell"

    def test_encode_empty_tool(self):
        s = encode_state(recent_tool="")
        assert s.recent_tool == ""

    def test_from_key_short(self):
        s = State.from_key("morning")
        assert s.time_segment == "morning"
        assert s.day == "monday"


# ═══════════════════════════════════════════════════════════════════════════
# QTable
# ═══════════════════════════════════════════════════════════════════════════


class TestQTable:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.qt = QTable()
        self.qt.init(self.db_path)

    def teardown_method(self):
        self.qt.close()
        self.db_path.unlink(missing_ok=True)

    def test_get_default(self):
        s = State()
        assert self.qt.get(s, Action.LAUNCH_DOCKER) == 0.0

    def test_update_and_get(self):
        s = State()
        self.qt.update(s, Action.LAUNCH_DOCKER, 1.5)
        assert self.qt.get(s, Action.LAUNCH_DOCKER) == 1.5

    def test_get_all(self):
        s = State()
        self.qt.update(s, Action.LAUNCH_DOCKER, 1.0)
        self.qt.update(s, Action.OPEN_BROWSER, 0.5)
        all_q = self.qt.get_all(s)
        assert all_q["launch_docker"] == 1.0
        assert all_q["open_browser"] == 0.5

    def test_visits_increments(self):
        s = State()
        assert self.qt.visits(s, Action.LAUNCH_DOCKER) == 0
        self.qt.update(s, Action.LAUNCH_DOCKER, 1.0)
        assert self.qt.visits(s, Action.LAUNCH_DOCKER) == 1
        self.qt.update(s, Action.LAUNCH_DOCKER, 2.0)
        assert self.qt.visits(s, Action.LAUNCH_DOCKER) == 2

    def test_persist_and_reload(self):
        s = State(time_segment="evening", day="friday")
        self.qt.update(s, Action.FOCUS_MUSIC, 0.8)
        self.qt.persist()

        # Reload from same DB
        qt2 = QTable()
        qt2.init(self.db_path)
        assert qt2.get(s, Action.FOCUS_MUSIC) == 0.8
        qt2.close()

    def test_total_entries(self):
        assert self.qt.total_entries() == 0
        self.qt.update(State(), Action.LAUNCH_DOCKER, 1.0)
        assert self.qt.total_entries() == 1

    def test_total_states(self):
        assert self.qt.total_states() == 0
        self.qt.update(State(day="monday"), Action.LAUNCH_DOCKER, 1.0)
        self.qt.update(State(day="tuesday"), Action.LAUNCH_DOCKER, 1.0)
        assert self.qt.total_states() == 2


# ═══════════════════════════════════════════════════════════════════════════
# Blacklist
# ═══════════════════════════════════════════════════════════════════════════


class TestBlacklist:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.bl = Blacklist()
        self.bl.init(self.db_path)

    def teardown_method(self):
        self.bl.close()
        self.db_path.unlink(missing_ok=True)

    def test_not_blocked_by_default(self):
        assert not self.bl.is_blocked(State(), Action.LAUNCH_DOCKER)

    def test_block_and_check(self):
        s = State()
        self.bl.block(s, Action.LAUNCH_DOCKER, "test")
        assert self.bl.is_blocked(s, Action.LAUNCH_DOCKER)
        # Different state should not be blocked
        s2 = State(day="friday")
        assert not self.bl.is_blocked(s2, Action.LAUNCH_DOCKER)

    def test_global_block(self):
        self.bl.block_global(Action.RUN_MAINTENANCE, "never")
        assert self.bl.is_blocked(State(), Action.RUN_MAINTENANCE)
        assert self.bl.is_blocked(State(day="friday"), Action.RUN_MAINTENANCE)

    def test_count(self):
        assert self.bl.count() == 0
        self.bl.block(State(), Action.LAUNCH_DOCKER)
        assert self.bl.count() == 1

    def test_all_blocked(self):
        self.bl.block(State(), Action.LAUNCH_DOCKER)
        entries = self.bl.all_blocked()
        assert len(entries) == 1
        assert entries[0]["action"] == "launch_docker"

    def test_persist_across_reload(self):
        self.bl.block(State(), Action.FOCUS_MUSIC, "test")
        self.bl.close()

        bl2 = Blacklist()
        bl2.init(self.db_path)
        assert bl2.is_blocked(State(), Action.FOCUS_MUSIC)
        bl2.close()


# ═══════════════════════════════════════════════════════════════════════════
# EpisodeLog
# ═══════════════════════════════════════════════════════════════════════════


class TestEpisodeLog:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.ep = EpisodeLog()
        self.ep.init(self.db_path)

    def teardown_method(self):
        self.ep.close()
        self.db_path.unlink(missing_ok=True)

    def test_record_and_total(self):
        assert self.ep.total_episodes() == 0
        self.ep.record(State(), Action.LAUNCH_DOCKER, 1.0)
        assert self.ep.total_episodes() == 1

    def test_recent(self):
        self.ep.record(State(), Action.LAUNCH_DOCKER, 1.0)
        self.ep.record(State(), Action.OPEN_BROWSER, -0.5)
        recent = self.ep.recent(2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["action"] == "open_browser"
        assert recent[1]["action"] == "launch_docker"

    def test_avg_reward(self):
        self.ep.record(State(), Action.LAUNCH_DOCKER, 1.0)
        self.ep.record(State(), Action.OPEN_BROWSER, -1.0)
        avg = self.ep.avg_reward(days=7)
        assert avg == 0.0

    def test_avg_reward_empty(self):
        assert self.ep.avg_reward() == 0.0

    def test_no_connection(self):
        ep2 = EpisodeLog()  # no init
        assert ep2.total_episodes() == 0
        assert ep2.recent() == []
        ep2.record(State(), Action.LAUNCH_DOCKER, 1.0)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# RLEngine — Q-learning
# ═══════════════════════════════════════════════════════════════════════════


class TestRLEngine:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.rl = RLEngine()
        self.rl.init(self.db_path)

    def teardown_method(self):
        self.rl.close()
        self.db_path.unlink(missing_ok=True)

    def test_init(self):
        assert self.rl.initialized is True

    def test_double_init(self):
        self.rl.init(self.db_path)  # should not raise
        assert self.rl.initialized is True

    def test_suggest_empty_qtable_exploration(self):
        """With empty Q-table, exploration should still return an action."""
        self.rl.epsilon = 1.0  # force exploration
        s = State()
        action = self.rl.suggest(s)
        assert action is not None
        assert action in ALL_ACTIONS

    def test_suggest_exploitation(self):
        """With a seeded Q-value, exploitation should return the best action."""
        s = State()
        self.rl.epsilon = 0.0  # force exploitation
        self.rl.q_table.update(s, Action.LAUNCH_DOCKER, 1.5)
        self.rl.confidence_threshold = 0.5
        action = self.rl.suggest(s)
        assert action == Action.LAUNCH_DOCKER

    def test_suggest_below_threshold(self):
        """Action with Q < threshold should not be suggested in exploitation."""
        s = State()
        self.rl.epsilon = 0.0
        self.rl.confidence_threshold = 2.0
        self.rl.q_table.update(s, Action.LAUNCH_DOCKER, 1.5)
        action = self.rl.suggest(s)
        assert action is None

    def test_suggest_blacklisted_skipped(self):
        """Blacklisted actions should be skipped."""
        s = State()
        self.rl.epsilon = 0.0
        self.rl.confidence_threshold = 0.5
        self.rl.q_table.update(s, Action.LAUNCH_DOCKER, 5.0)
        self.rl.blacklist.block(s, Action.LAUNCH_DOCKER)
        action = self.rl.suggest(s)
        # Should pick something else or None
        assert action != Action.LAUNCH_DOCKER

    def test_reward_updates_q(self):
        s = State()
        self.rl.epsilon = 0.0
        self.rl.confidence_threshold = 0.0
        self.rl.q_table.update(s, Action.LAUNCH_DOCKER, 0.0)
        self.rl.suggest(s)

        # Force internal state to use this action
        self.rl._current_state = s
        self.rl._current_action = Action.LAUNCH_DOCKER
        self.rl.reward(1.0)

        new_q = self.rl.q_table.get(s, Action.LAUNCH_DOCKER)
        assert new_q > 0.0  # Q should increase with positive reward

    def test_reward_negative_decreases_q(self):
        s = State()
        self.rl.q_table.update(s, Action.LAUNCH_DOCKER, 1.0)
        self.rl._current_state = s
        self.rl._current_action = Action.LAUNCH_DOCKER
        self.rl.reward(-1.0)

        new_q = self.rl.q_table.get(s, Action.LAUNCH_DOCKER)
        assert new_q < 1.0

    def test_reward_blacklists_on_severe(self):
        s = State()
        self.rl._current_state = s
        self.rl._current_action = Action.LAUNCH_DOCKER
        self.rl.reward(Reward.BLACKLIST)

        assert self.rl.blacklist.is_blocked(s, Action.LAUNCH_DOCKER)

    def test_epsilon_decays(self):
        initial_eps = self.rl.epsilon
        self.rl._current_state = State()
        self.rl._current_action = Action.LAUNCH_DOCKER
        self.rl.reward(1.0)
        assert self.rl.epsilon < initial_eps

    def test_force_reward(self):
        s = State()
        self.rl.force_reward(s, Action.LAUNCH_DOCKER, 1.0)
        assert self.rl.q_table.get(s, Action.LAUNCH_DOCKER) > 0.0
        assert self.rl.episodes.total_episodes() == 1

    def test_stats(self):
        stats = self.rl.stats()
        assert stats["initialized"] is True
        assert stats["q_entries"] == 0
        assert stats["episodes"] == 0
        assert stats["blacklisted"] == 0

    def test_top_actions(self):
        s = State()
        self.rl.q_table.update(s, Action.LAUNCH_DOCKER, 1.5)
        self.rl.q_table.update(s, Action.OPEN_BROWSER, 0.5)
        top = self.rl.top_actions(s, n=2)
        assert len(top) == 2
        assert top[0]["action"] == "launch_docker"
        assert top[0]["q_value"] == 1.5

    def test_status_line(self):
        line = self.rl.status_line()
        assert "Q-entries" in line
        assert "episodes" in line

    def test_not_initialized(self):
        rl2 = RLEngine()
        assert rl2.suggest(State()) is None
        assert rl2.status_line() == "not initialized"

    def test_reward_without_suggest(self):
        """Reward without a prior suggest should be a no-op."""
        self.rl.reward(1.0)  # should not raise
        assert self.rl.episodes.total_episodes() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Seed from habits (mocked)
# ═══════════════════════════════════════════════════════════════════════════


class TestSeedFromHabits:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.rl = RLEngine()
        self.rl.init(self.db_path)

    def teardown_method(self):
        self.rl.close()
        self.db_path.unlink(missing_ok=True)

    def test_seed_with_patterns(self):
        patterns = [
            {"tool": "shell", "hour": 9, "days": 5},
            {"tool": "calendar", "hour": 8, "days": 4},
        ]
        mock_habits_mod = MagicMock()
        mock_habits_mod.habits.recurring_patterns.return_value = patterns
        with patch.dict("sys.modules", {"bantz.core.habits": mock_habits_mod}):
            count = self.rl.seed_from_habits()
            assert count == 10  # 2 tools × 5 weekdays

        # Verify Q-table has entries
        assert self.rl.q_table.total_entries() > 0
        assert self.rl.episodes.total_episodes() > 0

    def test_seed_no_habits(self):
        """Seed should handle import failure gracefully."""
        with patch.dict("sys.modules", {"bantz.core.habits": None}):
            # Direct call — should return 0 or handle error
            count = self.rl.seed_from_habits()
            # Will get ImportError or 0 patterns
            assert count >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestRLConfig:
    def test_default_rl_disabled(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.rl_enabled is False

    def test_default_learning_rate(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.rl_learning_rate == 0.3

    def test_default_discount_factor(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.rl_discount_factor == 0.9

    def test_default_exploration_rate(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.rl_exploration_rate == 0.15

    def test_default_confidence_threshold(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.rl_confidence_threshold == 0.7

    def test_default_suggestion_interval(self):
        from bantz.config import Config
        cfg = Config(BANTZ_OLLAMA_MODEL="test", _env_file=None)
        assert cfg.rl_suggestion_interval == 1800

    def test_env_override(self):
        from bantz.config import Config
        cfg = Config(
            BANTZ_OLLAMA_MODEL="test",
            BANTZ_RL_ENABLED="true",
            BANTZ_RL_LEARNING_RATE="0.5",
            BANTZ_RL_CONFIDENCE_THRESHOLD="0.9",
            _env_file=None,
        )
        assert cfg.rl_enabled is True
        assert cfg.rl_learning_rate == 0.5
        assert cfg.rl_confidence_threshold == 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Action enum
# ═══════════════════════════════════════════════════════════════════════════


class TestAction:
    def test_all_actions(self):
        assert len(ALL_ACTIONS) == 8

    def test_action_values(self):
        assert Action.LAUNCH_DOCKER.value == "launch_docker"
        assert Action.PREPARE_BRIEFING.value == "prepare_briefing"

    def test_action_from_string(self):
        assert Action("launch_docker") == Action.LAUNCH_DOCKER


class TestReward:
    def test_reward_values(self):
        assert float(Reward.THANKS) == 2.0
        assert float(Reward.ACCEPT) == 1.0
        assert float(Reward.IGNORE) == 0.0
        assert float(Reward.DISMISS) == -0.5
        assert float(Reward.REVERT) == -1.0
        assert float(Reward.BLACKLIST) == -2.0
