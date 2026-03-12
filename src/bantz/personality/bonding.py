"""
Bantz v3 — RL-Based Bonding Meter (#172)

Maps cumulative RL reward to a formality index (0.0 = ultra-formal,
1.0 = bonded old retainer).  A shifted sigmoid ensures Day 1 is stiff
and distant while months of positive interaction unlock Bantz's full
unfiltered sarcasm—without ever breaking his 1920s butler character.

Highwater mark with 10 % drop-limit prevents a few bad interactions
from tanking the bond built over months.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
from typing import Optional

from bantz.config import config

log = logging.getLogger(__name__)

# ── Formality tiers (1920s Butler persona) ────────────────────────────────

TIERS: list[tuple[float, float, str, str]] = [
    # (low, high, label, prompt_hint)
    (0.00, 0.20, "Ultra Formal",
     "You are a new servant. Be stiff, distant, and strictly professional. "
     "No sarcasm. Very brief."),

    (0.20, 0.40, "Formal",
     "You are perfectly polite and maintain a strict professional wall. "
     "Minimal dry wit."),

    (0.40, 0.60, "Balanced",
     "You are comfortable. Use your signature dry wit and mild sarcastic "
     "complaints about the modern noisy machines."),

    (0.60, 0.80, "Trusted",
     "You feel like a trusted old retainer. Your sarcasm is sharper, and "
     "you offer unsolicited, witty observations about ma'am's tasks while "
     "serving efficiently."),

    (0.80, 1.00, "Bonded",
     "You are deeply loyal but completely unfiltered in your dry sarcasm. "
     "You might subtly tease ma'am's choices or habits, acting like an old "
     "family friend who has seen it all. You STILL strictly use 'ma'am' "
     "and never modern slang."),
]

DEFAULT_TIER = TIERS[2]  # Balanced


# ── BondingMeter ──────────────────────────────────────────────────────────

class BondingMeter:
    """Maps cumulative RL reward → formality index → prompt hint.

    Sigmoid: index = 1 / (1 + exp(-rate * (reward - midpoint)))
    Highwater mark: index never drops more than 10 % below the peak.
    """

    def __init__(self) -> None:
        self._highwater: float = 0.0
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init(self, db_path) -> None:
        """Load highwater from SQLite (creates table if needed)."""
        if self._initialized:
            return
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bonding_meter (
                key   TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)
        self._conn.commit()
        row = self._conn.execute(
            "SELECT value FROM bonding_meter WHERE key = 'highwater'"
        ).fetchone()
        self._highwater = row[0] if row else 0.0
        self._initialized = True
        log.info("Bonding meter initialised — highwater=%.3f", self._highwater)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False

    # ── Core math ─────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(reward: float, rate: float, midpoint: float) -> float:
        """Shifted sigmoid: reward → 0.0-1.0."""
        return 1.0 / (1.0 + math.exp(-rate * (reward - midpoint)))

    def _compute_raw_index(self, cumulative_reward: float) -> float:
        rate = config.bonding_sigmoid_rate
        midpoint = config.bonding_sigmoid_midpoint
        return self._sigmoid(cumulative_reward, rate, midpoint)

    # ── Public API ────────────────────────────────────────────────────

    def formality_index(self, cumulative_reward: float) -> float:
        """Compute formality index with highwater drop-limit.

        Returns value in [0.0, 1.0].  Never drops more than 10 % below
        the all-time peak (highwater mark).
        """
        raw = self._compute_raw_index(cumulative_reward)
        with self._lock:
            self._highwater = max(self._highwater, raw)
            clamped = max(raw, self._highwater * 0.9)
            self._persist_highwater()
        return clamped

    def formality_hint(self, cumulative_reward: float) -> str:
        """Return the prompt hint matching the current formality index."""
        if not config.bonding_enabled:
            return DEFAULT_TIER[3]
        idx = self.formality_index(cumulative_reward)
        return self._tier_for_index(idx)[3]

    def formality_label(self, cumulative_reward: float) -> str:
        """Return the tier label (e.g. 'Trusted') for the given reward."""
        if not config.bonding_enabled:
            return DEFAULT_TIER[2]
        idx = self.formality_index(cumulative_reward)
        return self._tier_for_index(idx)[2]

    @property
    def highwater(self) -> float:
        return self._highwater

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _tier_for_index(index: float) -> tuple[float, float, str, str]:
        for low, high, label, hint in TIERS:
            if index < high:
                return (low, high, label, hint)
        return TIERS[-1]

    def _persist_highwater(self) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO bonding_meter (key, value) "
                "VALUES ('highwater', ?)",
                (self._highwater,),
            )
            self._conn.commit()
        except Exception:
            log.debug("Failed to persist bonding highwater")

    # ── Convenience ───────────────────────────────────────────────────

    def get_formality_hint(self) -> str:
        """One-call convenience: fetches cumulative reward from RL engine
        and returns the appropriate prompt hint.

        Safe to call even if RL engine is not initialised (returns default).
        """
        try:
            from bantz.agent.rl_engine import rl_engine
            if not rl_engine.initialized:
                return DEFAULT_TIER[3]
            reward = rl_engine.cumulative_reward()
            return self.formality_hint(reward)
        except Exception:
            return DEFAULT_TIER[3]


# ── Module singleton ──────────────────────────────────────────────────────

bonding_meter = BondingMeter()
