"""
Bantz — Affinity Engine (#221)

Replaces the 670-LOC Q-learning ``RLEngine`` with a simple cumulative
score that persists across restarts via ``SQLiteKVStore``.

The score represents how the user treats Bantz over time:
  - Positive interactions (thank-you, accepted suggestions) → add_reward(+N)
  - Negative interactions (dismissals, anger)                → add_reward(-N)

Clamped to [-100, 100].  ``get_persona_state()`` translates the raw
number into a human-readable mood directive that the LLM can understand.

Usage::

    from bantz.agent.affinity_engine import affinity_engine

    affinity_engine.init(db_path)
    affinity_engine.add_reward(+5.0)
    print(affinity_engine.get_score())         # e.g. 5.0
    print(affinity_engine.get_persona_state())  # "Warm, friendly and playful."

Closes #221.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.agent.affinity")

# ── Persona state thresholds ─────────────────────────────────────────────

_PERSONA_TIERS: list[tuple[float, str]] = [
    (-60.0, "Very cold, distant and resentful. Speak in clipped, minimal sentences."),
    (-20.0, "Cold and guarded. Keep responses brief and impersonal."),
    (20.0, "Neutral and professional. Be helpful but keep emotional distance."),
    (60.0, "Warm, friendly and playful. Use casual language and light humour."),
    (100.1, "Deeply bonded — speak like a trusted best friend. Be affectionate and proactive."),
]

_KV_KEY = "affinity_score"


class AffinityEngine:
    """Cumulative affinity score persisted in SQLite KV store.

    Thread-safe: all mutations go through ``_lock``.
    """

    def __init__(self) -> None:
        self._score: float = 0.0
        self._lock = threading.Lock()
        self._initialized = False
        self._kv: Optional[object] = None  # SQLiteKVStore once init'd

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init(self, db_path: Path) -> None:
        """Initialise the engine, loading persisted score from KV store."""
        from bantz.data.sqlite_store import SQLiteKVStore

        self._kv = SQLiteKVStore(db_path)
        raw = self._kv.get(_KV_KEY, "0.0")
        try:
            self._score = max(-100.0, min(100.0, float(raw)))
        except (ValueError, TypeError):
            self._score = 0.0
        self._initialized = True
        log.info(
            "AffinityEngine ready  score=%.1f  persona=%s",
            self._score,
            self.get_persona_state(),
        )

    def close(self) -> None:
        """Flush and mark as closed (idempotent)."""
        self._initialized = False
        log.debug("AffinityEngine closed")

    # ── Public API ────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        return self._initialized

    def get_score(self) -> float:
        """Return the current affinity score (0.0 if uninitialised)."""
        return self._score

    def add_reward(self, value: float) -> float:
        """Add *value* to the cumulative score and persist.

        The score is clamped to [-100, 100].  Returns the new score.
        """
        with self._lock:
            self._score = max(-100.0, min(100.0, self._score + value))
            self._persist()
            return self._score

    def get_persona_state(self) -> str:
        """Translate the numeric score into an LLM-friendly mood directive."""
        score = self._score
        for threshold, description in _PERSONA_TIERS:
            if score < threshold:
                return description
        # Fallback (should not happen due to 100.1 sentinel)
        return _PERSONA_TIERS[-1][1]

    def status_line(self) -> str:
        """Human-readable one-liner for boot diagnostics."""
        return (
            f"score={self._score:.1f}  "
            f"persona=\"{self.get_persona_state()[:40]}…\""
        )

    def cumulative_reward(self) -> float:
        """Backward-compat alias used by bonding meter."""
        return self._score

    # ── Internal ──────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Write score to KV store.  Must be called under ``_lock``."""
        if self._kv is not None:
            try:
                self._kv.set(_KV_KEY, str(self._score))
            except Exception:
                log.debug("Failed to persist affinity score", exc_info=True)


# ── Module singleton ──────────────────────────────────────────────────────

affinity_engine = AffinityEngine()
