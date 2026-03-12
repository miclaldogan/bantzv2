"""
Bantz — Reinforcement Learning engine for routine optimisation (#125).

Replaces static rule-based habit analysis with an adaptive Q-learning agent
that learns from user feedback.

State Space  (~1680 discrete states)
    time_segment  × day_of_week × location_bucket × recent_tool

Action Space
    Proactive suggestions: launch docker, open workspace, open browser tabs,
    start focus music, run maintenance, prepare briefing, …

Reward Signal
    +2  user says "thanks"
    +1  user accepts suggestion
     0  user ignores (60 s)
    -0.5 user dismisses
    -1  user reverts
    -2  user says "never do this" → blacklist

Persistence
    Q-table + blacklist + episode log stored in SQLite (bantz.db).

Usage
    from bantz.agent.rl_engine import rl_engine

    rl_engine.init(db_path)              # call in DataLayer.init()
    action = rl_engine.suggest(state)    # returns best action or None
    rl_engine.reward(action, +1)         # after user feedback
"""
from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Enums ────────────────────────────────────────────────────────────────

TIME_SEGMENTS = ("late_night", "morning", "afternoon", "evening", "night")
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
LOCATION_BUCKETS = ("home", "work", "school", "other")


class Action(str, Enum):
    """Proactive actions the agent can suggest."""
    LAUNCH_DOCKER = "launch_docker"
    OPEN_WORKSPACE = "open_workspace"
    OPEN_BROWSER = "open_browser"
    FOCUS_MUSIC = "focus_music"
    RUN_MAINTENANCE = "run_maintenance"
    PREPARE_BRIEFING = "prepare_briefing"
    SUGGEST_BREAK = "suggest_break"
    DAILY_REVIEW = "daily_review"
    PROACTIVE_CHAT = "proactive_chat"  # #167: idle conversation initiation
    HEALTH_BREAK = "health_break"  # #168: health & break nudge
    FEEDBACK_CHAT = "feedback_chat"  # #180: direct RLHF via sentiment


ALL_ACTIONS = list(Action)


class Reward(float, Enum):
    """Standard reward values."""
    THANKS = 2.0
    ACCEPT = 1.0
    IGNORE = 0.0
    DISMISS = -0.5
    REVERT = -1.0
    BLACKLIST = -2.0


# ── State ────────────────────────────────────────────────────────────────

# Ambient buckets for state encoding (#166)
AMBIENT_BUCKETS = ("silence", "speech", "noisy", "unknown")


@dataclass(frozen=True)
class State:
    """Discrete state representation."""
    time_segment: str = "morning"
    day: str = "monday"
    location: str = "home"
    recent_tool: str = ""
    ambient: str = "unknown"  # #166: ambient environment label

    @property
    def key(self) -> str:
        """String key for Q-table lookup."""
        return f"{self.time_segment}|{self.day}|{self.location}|{self.recent_tool}|{self.ambient}"

    @classmethod
    def from_key(cls, key: str) -> "State":
        parts = key.split("|", 4)
        return cls(
            time_segment=parts[0] if len(parts) > 0 else "morning",
            day=parts[1] if len(parts) > 1 else "monday",
            location=parts[2] if len(parts) > 2 else "home",
            recent_tool=parts[3] if len(parts) > 3 else "",
            ambient=parts[4] if len(parts) > 4 else "unknown",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "time_segment": self.time_segment,
            "day": self.day,
            "location": self.location,
            "recent_tool": self.recent_tool,
            "ambient": self.ambient,
        }


def encode_state(
    time_segment: str = "morning",
    day: str = "monday",
    location: str = "home",
    recent_tool: str = "",
    ambient: str = "unknown",
) -> State:
    """Build a State from raw context values, normalising buckets."""
    seg = time_segment if time_segment in TIME_SEGMENTS else "morning"
    d = day.lower() if day.lower() in DAYS else "monday"
    loc = location.lower() if location.lower() in LOCATION_BUCKETS else "other"
    tool = recent_tool[:30] if recent_tool else ""
    amb = ambient.lower() if ambient.lower() in AMBIENT_BUCKETS else "unknown"
    return State(time_segment=seg, day=d, location=loc, recent_tool=tool, ambient=amb)


# ── Q-Table ──────────────────────────────────────────────────────────────

class QTable:
    """In-memory Q-table backed by SQLite for persistence.

    Schema:
        rl_qtable(state TEXT, action TEXT, q_value REAL, visits INT,
                  updated_at REAL, PRIMARY KEY(state, action))
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._table: dict[str, dict[str, float]] = {}   # state_key → {action → q}
        self._visits: dict[str, dict[str, int]] = {}     # state_key → {action → n}
        self._lock = threading.Lock()

    def init(self, db_path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_qtable (
                state   TEXT NOT NULL,
                action  TEXT NOT NULL,
                q_value REAL NOT NULL DEFAULT 0.0,
                visits  INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (state, action)
            )
        """)
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        """Load persisted Q-values into memory."""
        if not self._conn:
            return
        rows = self._conn.execute(
            "SELECT state, action, q_value, visits FROM rl_qtable"
        ).fetchall()
        with self._lock:
            for state_key, action, q_val, visits in rows:
                self._table.setdefault(state_key, {})[action] = q_val
                self._visits.setdefault(state_key, {})[action] = visits

    def get(self, state: State, action: Action) -> float:
        with self._lock:
            return self._table.get(state.key, {}).get(action.value, 0.0)

    def get_all(self, state: State) -> dict[str, float]:
        """Return Q-values for all actions in this state."""
        with self._lock:
            return dict(self._table.get(state.key, {}))

    def update(self, state: State, action: Action, value: float) -> None:
        with self._lock:
            self._table.setdefault(state.key, {})[action.value] = value
            v = self._visits.setdefault(state.key, {})
            v[action.value] = v.get(action.value, 0) + 1

    def visits(self, state: State, action: Action) -> int:
        with self._lock:
            return self._visits.get(state.key, {}).get(action.value, 0)

    def persist(self) -> None:
        """Write in-memory Q-table to SQLite."""
        if not self._conn:
            return
        now = time.time()
        with self._lock:
            rows = []
            for state_key, actions in self._table.items():
                for action, q_val in actions.items():
                    visits = self._visits.get(state_key, {}).get(action, 0)
                    rows.append((state_key, action, q_val, visits, now))
        self._conn.executemany(
            """INSERT OR REPLACE INTO rl_qtable
               (state, action, q_value, visits, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def total_entries(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._table.values())

    def total_states(self) -> int:
        with self._lock:
            return len(self._table)

    def close(self) -> None:
        self.persist()
        if self._conn:
            self._conn.close()
            self._conn = None


# ── Blacklist ────────────────────────────────────────────────────────────

class Blacklist:
    """Permanently rejected actions per state (or globally).

    Schema:
        rl_blacklist(state TEXT, action TEXT, reason TEXT, created_at REAL,
                     PRIMARY KEY(state, action))
    State='*' means globally blacklisted.
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._entries: set[tuple[str, str]] = set()  # (state_key, action)
        self._lock = threading.Lock()

    def init(self, db_path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_blacklist (
                state      TEXT NOT NULL,
                action     TEXT NOT NULL,
                reason     TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                PRIMARY KEY (state, action)
            )
        """)
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        if not self._conn:
            return
        rows = self._conn.execute("SELECT state, action FROM rl_blacklist").fetchall()
        with self._lock:
            self._entries = {(r[0], r[1]) for r in rows}

    def is_blocked(self, state: State, action: Action) -> bool:
        with self._lock:
            return (
                (state.key, action.value) in self._entries
                or ("*", action.value) in self._entries
            )

    def block(self, state: State, action: Action, reason: str = "") -> None:
        with self._lock:
            self._entries.add((state.key, action.value))
        if self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO rl_blacklist
                   (state, action, reason, created_at) VALUES (?, ?, ?, ?)""",
                (state.key, action.value, reason, time.time()),
            )
            self._conn.commit()

    def block_global(self, action: Action, reason: str = "") -> None:
        with self._lock:
            self._entries.add(("*", action.value))
        if self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO rl_blacklist
                   (state, action, reason, created_at) VALUES (?, ?, ?, ?)""",
                ("*", action.value, reason, time.time()),
            )
            self._conn.commit()

    def all_blocked(self) -> list[dict[str, str]]:
        with self._lock:
            return [{"state": s, "action": a} for s, a in self._entries]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ── Episode Log ──────────────────────────────────────────────────────────

class EpisodeLog:
    """Records state→action→reward episodes for analysis.

    Schema:
        rl_episodes(id INTEGER PRIMARY KEY, state TEXT, action TEXT,
                    reward REAL, timestamp REAL)
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None

    def init(self, db_path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_episodes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                state     TEXT NOT NULL,
                action    TEXT NOT NULL,
                reward    REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.commit()

    def record(self, state: State, action: Action, reward: float) -> None:
        if not self._conn:
            return
        self._conn.execute(
            "INSERT INTO rl_episodes (state, action, reward, timestamp) VALUES (?, ?, ?, ?)",
            (state.key, action.value, reward, time.time()),
        )
        self._conn.commit()

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        if not self._conn:
            return []
        rows = self._conn.execute(
            "SELECT state, action, reward, timestamp FROM rl_episodes "
            "ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [
            {"state": r[0], "action": r[1], "reward": r[2], "timestamp": r[3]}
            for r in rows
        ]

    def total_episodes(self) -> int:
        if not self._conn:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM rl_episodes").fetchone()
        return row[0] if row else 0

    def avg_reward(self, days: int = 7) -> float:
        if not self._conn:
            return 0.0
        cutoff = time.time() - days * 86400
        row = self._conn.execute(
            "SELECT AVG(reward) FROM rl_episodes WHERE timestamp > ?",
            (cutoff,),
        ).fetchone()
        return row[0] if row and row[0] is not None else 0.0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ── RL Engine ────────────────────────────────────────────────────────────

class RLEngine:
    """Q-learning agent for proactive routine suggestions.

    Algorithm: Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

    Parameters tuned for low-data scenarios (a few interactions/day):
        α  = 0.3  (high learning rate — few samples)
        γ  = 0.9  (long-horizon: morning patterns affect afternoon)
        ε  = 0.15 (exploration) decaying to ε_min = 0.02
        confidence_threshold = 0.7 (only suggest when Q > threshold)
    """

    def __init__(self) -> None:
        self.q_table = QTable()
        self.blacklist = Blacklist()
        self.episodes = EpisodeLog()

        # Hyperparameters (configurable)
        self.alpha: float = 0.3
        self.gamma: float = 0.9
        self.epsilon: float = 0.15
        self.epsilon_min: float = 0.02
        self.epsilon_decay: float = 0.995
        self.confidence_threshold: float = 0.7

        self._current_state: Optional[State] = None
        self._current_action: Optional[Action] = None
        self._initialized = False
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init(self, db_path) -> None:
        """Initialise Q-table, blacklist, and episode log from SQLite."""
        if self._initialized:
            return
        self.q_table.init(db_path)
        self.blacklist.init(db_path)
        self.episodes.init(db_path)
        self._initialized = True
        log.info(
            "RL engine initialised: %d Q-entries, %d episodes, %d blacklisted",
            self.q_table.total_entries(),
            self.episodes.total_episodes(),
            self.blacklist.count(),
        )

    def close(self) -> None:
        """Persist and close all stores."""
        self.q_table.close()
        self.blacklist.close()
        self.episodes.close()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── Core RL ───────────────────────────────────────────────────────

    def suggest(self, state: State) -> Optional[Action]:
        """Select an action for the given state using ε-greedy policy.

        Returns None if no action exceeds the confidence threshold or
        all actions are blacklisted.
        """
        if not self._initialized:
            return None

        # ε-greedy exploration
        if random.random() < self.epsilon:
            candidates = [
                a for a in ALL_ACTIONS
                if not self.blacklist.is_blocked(state, a)
            ]
            if not candidates:
                return None
            action = random.choice(candidates)
            with self._lock:
                self._current_state = state
                self._current_action = action
            return action

        # Exploitation: pick best Q-value action
        q_values = self.q_table.get_all(state)
        best_action = None
        best_q = -math.inf

        for action in ALL_ACTIONS:
            if self.blacklist.is_blocked(state, action):
                continue
            q = q_values.get(action.value, 0.0)
            if q > best_q:
                best_q = q
                best_action = action

        # Confidence gate
        if best_action is None or best_q < self.confidence_threshold:
            return None

        with self._lock:
            self._current_state = state
            self._current_action = best_action
        return best_action

    def reward(self, reward_value: float, next_state: Optional[State] = None) -> None:
        """Apply reward for the last suggested action and update Q-table.

        Uses Q-learning update:
            Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
        """
        with self._lock:
            state = self._current_state
            action = self._current_action

        if state is None or action is None:
            return

        old_q = self.q_table.get(state, action)

        # Max future Q-value
        if next_state:
            future_qs = self.q_table.get_all(next_state)
            max_future = max(future_qs.values()) if future_qs else 0.0
        else:
            max_future = 0.0

        # Q-learning update
        new_q = old_q + self.alpha * (reward_value + self.gamma * max_future - old_q)
        self.q_table.update(state, action, new_q)

        # Log episode
        self.episodes.record(state, action, reward_value)

        # Blacklist on severe negative reward
        if reward_value <= Reward.BLACKLIST:
            self.blacklist.block(state, action, reason="user_blacklisted")
            log.info("Blacklisted %s in state %s", action.value, state.key)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Persist periodically (every 10 episodes)
        if self.episodes.total_episodes() % 10 == 0:
            self.q_table.persist()

        with self._lock:
            self._current_state = None
            self._current_action = None

    def force_reward(self, state: State, action: Action, reward_value: float,
                     next_state: Optional[State] = None) -> None:
        """Apply reward for an explicit state-action pair (for seeding / batch training)."""
        old_q = self.q_table.get(state, action)
        if next_state:
            future_qs = self.q_table.get_all(next_state)
            max_future = max(future_qs.values()) if future_qs else 0.0
        else:
            max_future = 0.0

        new_q = old_q + self.alpha * (reward_value + self.gamma * max_future - old_q)
        self.q_table.update(state, action, new_q)
        self.episodes.record(state, action, reward_value)

    # ── Seeding from HabitEngine ──────────────────────────────────────

    def seed_from_habits(self) -> int:
        """Seed Q-table from existing HabitEngine recurring patterns.

        Maps recurring tool patterns to corresponding RL actions with
        positive reward, bootstrapping the Q-table for faster learning.
        """
        _TOOL_TO_ACTION: dict[str, Action] = {
            "shell": Action.LAUNCH_DOCKER,
            "web_search": Action.OPEN_BROWSER,
            "news": Action.OPEN_BROWSER,
            "calendar": Action.PREPARE_BRIEFING,
            "reminder": Action.PREPARE_BRIEFING,
            "weather": Action.PREPARE_BRIEFING,
            "gmail": Action.PREPARE_BRIEFING,
            "mail": Action.PREPARE_BRIEFING,
        }

        try:
            from bantz.core.habits import habits
            patterns = habits.recurring_patterns(days=14)
        except Exception:
            return 0

        count = 0
        for pat in patterns:
            tool = pat.get("tool", "")
            hour = pat.get("hour", 12)
            action = _TOOL_TO_ACTION.get(tool)
            if action is None:
                continue

            # Map hour to segment
            if hour < 6:
                seg = "late_night"
            elif hour < 12:
                seg = "morning"
            elif hour < 17:
                seg = "afternoon"
            elif hour < 21:
                seg = "evening"
            else:
                seg = "night"

            # Create a representative state for all weekdays
            for day in DAYS[:5]:  # weekdays
                state = State(time_segment=seg, day=day, location="home", recent_tool=tool)
                self.force_reward(state, action, Reward.ACCEPT)
                count += 1

        if count:
            self.q_table.persist()
            log.info("Seeded Q-table with %d entries from habit patterns", count)
        return count

    # ── Stats & Reporting ─────────────────────────────────────────────

    def cumulative_reward(self) -> float:
        """Sum of all episode rewards, cached for 60s (#172)."""
        now = time.time()
        if (
            hasattr(self, "_cum_reward_cache")
            and now - self._cum_reward_cache_ts < 60
        ):
            return self._cum_reward_cache
        if not self._initialized or not self.episodes._conn:
            return 0.0
        row = self.episodes._conn.execute(
            "SELECT COALESCE(SUM(reward), 0) FROM rl_episodes"
        ).fetchone()
        total = row[0] if row else 0.0
        self._cum_reward_cache = total
        self._cum_reward_cache_ts = now
        return total

    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "q_entries": self.q_table.total_entries(),
            "q_states": self.q_table.total_states(),
            "episodes": self.episodes.total_episodes(),
            "avg_reward_7d": round(self.episodes.avg_reward(7), 3),
            "blacklisted": self.blacklist.count(),
            "epsilon": round(self.epsilon, 4),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "confidence_threshold": self.confidence_threshold,
        }

    def top_actions(self, state: State, n: int = 3) -> list[dict[str, Any]]:
        """Return top-N actions for a state, sorted by Q-value."""
        q_values = self.q_table.get_all(state)
        items = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "action": a,
                "q_value": round(q, 3),
                "visits": self.q_table.visits(state, Action(a)),
                "blocked": self.blacklist.is_blocked(state, Action(a)),
            }
            for a, q in items[:n]
        ]

    def status_line(self) -> str:
        if not self._initialized:
            return "not initialized"
        s = self.stats()
        return (
            f"{s['q_entries']} Q-entries, {s['episodes']} episodes, "
            f"ε={s['epsilon']}, avg_reward={s['avg_reward_7d']}"
        )


# ── Module singleton ─────────────────────────────────────────────────────

rl_engine = RLEngine()
