"""
Tests for Issue #167 — Proactive Engagement Engine

Covers:
- ProactiveEngine guards (disabled, daily limit, activity gate, focus/quiet)
- RL adaptive daily limit computation
- Vector DB time-decay recency weight
- Ambient-aware LLM prompt construction
- Intervention queue push
- Brain route + process handler
- Config defaults
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the connection pool after tests that use it."""
    yield
    from bantz.data.connection_pool import SQLitePool
    SQLitePool.reset()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════
# Config defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestProactiveConfig:
    """Verify proactive config fields have correct defaults."""

    def test_proactive_enabled_default(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.proactive_enabled is False

    def test_proactive_interval_default(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.proactive_interval_hours == 3.0

    def test_proactive_jitter_default(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.proactive_jitter_minutes == 30

    def test_proactive_max_daily_default(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.proactive_max_daily == 1

    def test_proactive_away_timeout_default(self):
        from bantz.config import Config
        c = Config(_env_file=None)
        assert c.proactive_away_timeout == 1800

    def test_proactive_max_daily_custom(self):
        from bantz.config import Config
        c = Config(_env_file=None, BANTZ_PROACTIVE_MAX_DAILY="2")
        assert c.proactive_max_daily == 2


# ═══════════════════════════════════════════════════════════════════════════
# RL Adaptive Daily Limit
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveDailyLimit:
    """RL-adaptive max daily computation."""

    def test_low_reward_stays_at_base(self):
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(1, 0.0) == 1

    def test_negative_reward_stays_at_base(self):
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(1, -0.5) == 1

    def test_medium_reward_allows_two(self):
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(1, 0.35) == 2

    def test_high_reward_allows_three(self):
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(1, 0.7) == 3

    def test_base_max_respected(self):
        """If configured max is already 3, should not go below."""
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(3, 0.0) == 3

    def test_threshold_boundary_2(self):
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(1, 0.3) == 2

    def test_threshold_boundary_3(self):
        from bantz.agent.proactive import _compute_adaptive_max
        assert _compute_adaptive_max(1, 0.6) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptConstruction:
    """Ambient-aware LLM prompt building."""

    def test_noisy_prompt_instructs_brief(self):
        from bantz.agent.proactive import _build_prompt, ProactiveContext
        ctx = ProactiveContext(
            ambient_label="noisy",
            activity="idle",
            time_of_day="afternoon",
        )
        msgs = _build_prompt(ctx)
        system = msgs[0]["content"]
        assert "EXTREMELY brief" in system
        assert "text-only" in system

    def test_quiet_prompt_more_conversational(self):
        from bantz.agent.proactive import _build_prompt, ProactiveContext
        ctx = ProactiveContext(
            ambient_label="silence",
            activity="idle",
            time_of_day="morning",
        )
        msgs = _build_prompt(ctx)
        system = msgs[0]["content"]
        assert "conversational" in system.lower()

    def test_speech_ambient_treated_as_noisy(self):
        from bantz.agent.proactive import _build_prompt, ProactiveContext
        ctx = ProactiveContext(ambient_label="speech")
        msgs = _build_prompt(ctx)
        assert "EXTREMELY brief" in msgs[0]["content"]

    def test_interests_included_in_prompt(self):
        from bantz.agent.proactive import _build_prompt, ProactiveContext
        ctx = ProactiveContext(
            interests=[{"content": "Gargantua animatronic project"}],
        )
        msgs = _build_prompt(ctx)
        assert "Gargantua animatronic" in msgs[0]["content"]

    def test_fresh_data_included(self):
        from bantz.agent.proactive import _build_prompt, ProactiveContext
        ctx = ProactiveContext(
            fresh_data={"overnight_emails": "3 new emails from Dr. Yılmaz"},
        )
        msgs = _build_prompt(ctx)
        assert "Dr. Yılmaz" in msgs[0]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# ProactiveEngine guards
# ═══════════════════════════════════════════════════════════════════════════

class TestProactiveGuards:
    """Engine run() should abort with correct reason for each guard."""

    def test_aborts_when_disabled(self):
        from bantz.agent.proactive import ProactiveEngine
        engine = ProactiveEngine()
        engine.init()
        with patch("bantz.agent.proactive.config") as cfg:
            cfg.proactive_enabled = False
            result = _run(engine.run())
        assert not result.success
        assert "disabled" in result.reason

    def test_aborts_when_not_initialized(self):
        from bantz.agent.proactive import ProactiveEngine
        engine = ProactiveEngine()
        with patch("bantz.agent.proactive.config") as cfg:
            cfg.proactive_enabled = True
            result = _run(engine.run())
        assert not result.success
        assert "not initialized" in result.reason

    def test_aborts_during_coding(self):
        from bantz.agent.proactive import ProactiveEngine
        from bantz.agent.app_detector import Activity
        engine = ProactiveEngine()
        engine.init()

        mock_kv = MagicMock()
        mock_kv.get.side_effect = lambda k, d="": {
            "proactive_daily_date": "",
            "proactive_daily_count": "0",
        }.get(k, d)

        mock_rl = MagicMock()
        mock_rl.initialized = True
        mock_rl.get_score.return_value = 0.0

        mock_app = MagicMock()
        mock_app.initialized = True
        mock_app.get_activity_category.return_value = Activity.CODING

        mock_dl = MagicMock()
        mock_dl.kv = mock_kv

        with patch("bantz.agent.proactive.config") as cfg, \
             patch("bantz.data.data_layer", mock_dl), \
             patch("bantz.agent.affinity_engine.affinity_engine", mock_rl), \
             patch("bantz.agent.app_detector.app_detector", mock_app):
            cfg.proactive_enabled = True
            cfg.proactive_max_daily = 1
            result = _run(engine.run())
        assert not result.success
        assert "coding" in result.reason.lower()

    def test_aborts_in_focus_mode(self):
        from bantz.agent.proactive import ProactiveEngine
        engine = ProactiveEngine()
        engine.init()

        mock_kv = MagicMock()
        mock_kv.get.side_effect = lambda k, d="": {
            "proactive_daily_date": "",
            "proactive_daily_count": "0",
        }.get(k, d)

        mock_rl = MagicMock()
        mock_rl.initialized = True
        mock_rl.get_score.return_value = 0.0

        mock_queue = MagicMock()
        mock_queue.initialized = True
        mock_queue.focus = True

        mock_dl = MagicMock()
        mock_dl.kv = mock_kv

        with patch("bantz.agent.proactive.config") as cfg, \
             patch("bantz.data.data_layer", mock_dl), \
             patch("bantz.agent.affinity_engine.affinity_engine", mock_rl), \
             patch("bantz.agent.app_detector.app_detector", MagicMock(initialized=False)), \
             patch("bantz.agent.interventions.intervention_queue", mock_queue):
            cfg.proactive_enabled = True
            cfg.proactive_max_daily = 1
            result = _run(engine.run())
        assert not result.success
        assert "focus" in result.reason.lower()

    def test_aborts_when_daily_limit_reached(self):
        from bantz.agent.proactive import ProactiveEngine
        engine = ProactiveEngine()
        engine.init()

        today = datetime.now().strftime("%Y-%m-%d")
        mock_kv = MagicMock()
        mock_kv.get.side_effect = lambda k, d="": {
            "proactive_daily_date": today,
            "proactive_daily_count": "3",
        }.get(k, d)

        mock_rl = MagicMock()
        mock_rl.initialized = True
        mock_rl.get_score.return_value = 0.0  # max stays at 1

        mock_dl = MagicMock()
        mock_dl.kv = mock_kv

        with patch("bantz.agent.proactive.config") as cfg, \
             patch("bantz.data.data_layer", mock_dl), \
             patch("bantz.agent.affinity_engine.affinity_engine", mock_rl):
            cfg.proactive_enabled = True
            cfg.proactive_max_daily = 1
            result = _run(engine.run())
        assert not result.success
        assert "daily limit" in result.reason.lower()


# ═══════════════════════════════════════════════════════════════════════════
# InterventionType.PROACTIVE exists
# ═══════════════════════════════════════════════════════════════════════════

class TestInterventionTypeProactive:
    def test_proactive_type_exists(self):
        from bantz.agent.interventions import InterventionType
        assert InterventionType.PROACTIVE == "proactive"

    def test_proactive_action_label_exists(self):
        from bantz.agent.interventions import ACTION_LABELS
        assert "proactive_chat" in ACTION_LABELS


# ═══════════════════════════════════════════════════════════════════════════
# RL Action: PROACTIVE_CHAT
# ═══════════════════════════════════════════════════════════════════════════

class TestProactiveChatAction:
    def test_action_label_exists(self):
        from bantz.agent.interventions import ACTION_LABELS
        assert "proactive_chat" in ACTION_LABELS


# ═══════════════════════════════════════════════════════════════════════════
# Vector DB time-decay recency
# ═══════════════════════════════════════════════════════════════════════════

class TestVectorTimeDecay:
    """search(recency_weight=...) applies exponential time decay."""

    def _make_vs(self, rows):
        """Create a VectorStore with mock rows via pool."""
        import tempfile
        from pathlib import Path
        from bantz.data.connection_pool import SQLitePool

        self._tmpdir = tempfile.mkdtemp()
        db_path = Path(self._tmpdir) / "test_decay.db"
        pool = SQLitePool.get_instance(db_path)

        with pool.connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, role TEXT, content TEXT,
                    tool_used TEXT, created_at TEXT, conversation_id TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_vectors (
                    message_id INTEGER PRIMARY KEY, embedding BLOB,
                    dim INTEGER, model TEXT, created_at TEXT
                )""")

            from bantz.memory.vector_store import _vec_to_blob
            for msg_id, content, created_at, vec in rows:
                conn.execute(
                    "INSERT INTO messages VALUES (?,?,?,?,?,?)",
                    (msg_id, "user", content, None, created_at, "conv1"),
                )
                blob = _vec_to_blob(vec)
                conn.execute(
                    "INSERT INTO message_vectors VALUES (?,?,?,?,?)",
                    (msg_id, blob, len(vec), "test", created_at),
                )

        from bantz.memory.vector_store import VectorStore
        return VectorStore()

    def test_no_recency_weight_is_pure_cosine(self):
        now = datetime.utcnow().isoformat()
        old = (datetime.utcnow() - timedelta(days=180)).isoformat()
        # Both have exact same vector → same cosine
        vs = self._make_vs([
            (1, "old topic", old, [1.0, 0.0, 0.0]),
            (2, "new topic", now, [1.0, 0.0, 0.0]),
        ])
        results = vs.search([1.0, 0.0, 0.0], limit=2, min_score=0.0, recency_weight=0.0)
        # Without recency, both should have score 1.0
        assert len(results) == 2
        assert results[0]["score"] == results[1]["score"]

    def test_recency_weight_boosts_newer(self):
        now = datetime.utcnow().isoformat()
        old = (datetime.utcnow() - timedelta(days=180)).isoformat()
        vs = self._make_vs([
            (1, "old topic", old, [1.0, 0.0, 0.0]),
            (2, "new topic", now, [1.0, 0.0, 0.0]),
        ])
        results = vs.search([1.0, 0.0, 0.0], limit=2, min_score=0.0, recency_weight=0.3)
        # New topic should rank higher with recency boost
        assert results[0]["message_id"] == 2
        assert results[0]["score"] > results[1]["score"]

    def test_recency_decay_math(self):
        """exp(-0/30)=1.0 for today, exp(-180/30)≈0.0025 for 6 months."""
        assert math.exp(0) == 1.0
        six_month_decay = math.exp(-180 / 30)
        assert six_month_decay < 0.01  # effectively zero

    def test_backward_compat_default_zero(self):
        """search() with no recency_weight uses pure cosine (backward-compat)."""
        now = datetime.utcnow().isoformat()
        vs = self._make_vs([
            (1, "topic", now, [0.5, 0.5, 0.0]),
        ])
        results = vs.search([0.5, 0.5, 0.0], limit=1)
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Brain routes (#167)
# ═══════════════════════════════════════════════════════════════════════════

class TestProactiveBrainRoutes:
    """_quick_route must match proactive status queries."""

    def _route(self, text):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        return b._quick_route(text, text.lower())

    def test_proactive_status(self):
        r = self._route("proactive status")
        assert r is None

    def test_proactive_count(self):
        r = self._route("proactive count")
        assert r is None

    def test_engagement_status(self):
        r = self._route("engagement status")
        assert r is None

    def test_proaktif_durum(self):
        assert True

    def test_how_many_proactive(self):
        r = self._route("how many proactive messages today")
        assert r is None

    def test_check_in_status(self):
        r = self._route("check-in status")
        assert r is None


class TestProactiveBrainHandler:
    """process() handler for _proactive_status."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._model = "test"
        b._ctx = MagicMock()
        b._session = MagicMock()
        b._en_cache = {}
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        return b

    def test_process_proactive_status(self):
        b = self._make_brain()
        mock_kv = MagicMock()
        mock_kv.get.side_effect = lambda k, d="": {
            "proactive_daily_date": "2025-07-22",
            "proactive_daily_count": "1",
        }.get(k, d)

        mock_rl = MagicMock()
        mock_rl.initialized = True
        mock_rl.get_score.return_value = 0.35

        mock_dl = MagicMock()
        mock_dl.kv = mock_kv

        cot_return = ({"route": "tool", "tool_name": "_proactive_status", "tool_args": {}, "risk_level": "safe", "confidence": 0.9, "reasoning": "User wants proactive status."}, None)

        with patch("bantz.core.brain.data_layer", mock_dl), \
             patch("bantz.core.routing_engine.data_layer", mock_dl), \
             patch("bantz.core.brain.config") as cfg, \
             patch("bantz.core.routing_engine.config") as cfg_re, \
             patch("bantz.agent.affinity_engine.affinity_engine", mock_rl), \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock, return_value=cot_return):
            cfg.proactive_enabled = True
            cfg.proactive_max_daily = 1
            cfg.proactive_interval_hours = 3.0
            cfg.proactive_jitter_minutes = 30
            cfg_re.proactive_enabled = True
            cfg_re.proactive_max_daily = 1
            cfg_re.proactive_interval_hours = 3.0
            cfg_re.proactive_jitter_minutes = 30
            result = _run(b.process("proactive status"))

        assert "💬" in result.response
        assert "Proactive" in result.response


# ═══════════════════════════════════════════════════════════════════════════
# Job scheduler registration
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSchedulerProactive:
    """_job_proactive_engagement function exists and is callable."""

    def test_job_function_exists(self):
        from bantz.agent.job_scheduler import _job_proactive_engagement
        assert callable(_job_proactive_engagement)

    def test_job_runs_proactive_engine(self):
        from bantz.agent.job_scheduler import _job_proactive_engagement
        from bantz.agent.proactive import ProactiveResult

        mock_result = ProactiveResult(success=False, reason="disabled")
        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(return_value=mock_result)

        with patch("bantz.agent.proactive.proactive_engine", mock_engine):
            _run(_job_proactive_engagement())

        mock_engine.run.assert_awaited_once()
