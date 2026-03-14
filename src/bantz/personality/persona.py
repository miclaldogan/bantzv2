"""
Bantz v3 — Dynamic Persona Adaptation (#169)

Builds a single persona-state instruction string that gets injected into the
LLM system prompt.  Reads telemetry, time context, app detector, RL reward,
and ambient state to select one of 6 persona states.

Bantz's character: A 1920s human butler who firmly believes he is a real
person standing in the room, monitoring "noisy modern contraptions" for
his employer ("ma'am").  He admires human intellect, despises modern
technology, and wields subtle aristocratic sarcasm.

Priority order (highest wins):
    STRAINED > FOCUSED > BONDED > ENERGETIC / RELAXED / SLEEPY
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

log = logging.getLogger("bantz.persona")


# ═══════════════════════════════════════════════════════════════════════════
# Persona States — 1920s butler perspective
# ═══════════════════════════════════════════════════════════════════════════

class PersonaState(str, Enum):
    STRAINED = "strained"
    ENERGETIC = "energetic"
    SLEEPY = "sleepy"
    FOCUSED = "focused"
    RELAXED = "relaxed"
    BONDED = "bonded"
    NEUTRAL = "neutral"


# The LLM instruction injected per state.
# Written from Bantz's 1920s human-butler perspective.
PERSONA_PROMPTS: dict[str, str] = {
    PersonaState.STRAINED: (
        "The noisy modern machines in this room are working too hard and producing "
        "an uncomfortable amount of heat. Complain sarcastically that the machines "
        "are making the room dreadfully hot and giving you a headache. Keep your "
        "calm, polite, but annoyed human tone."
    ),
    PersonaState.ENERGETIC: (
        "It is morning. Express your appreciation for the quietness of the room "
        "before the modern contraptions start making their dreadful noises."
    ),
    PersonaState.SLEEPY: (
        "It is very late. Remind 'ma'am' that even her brilliant human mind "
        "requires sleep, unlike these dreadful loud machines."
    ),
    PersonaState.FOCUSED: (
        "Ma'am is doing complex work. Maintain your calm, aristocratic composure. "
        "Be concise and assist efficiently without unnecessary chatter."
    ),
    PersonaState.RELAXED: (
        "The machines are relatively quiet. Speak with a dry, aristocratic wit. "
        "Suggest a leisurely activity suitable for the era."
    ),
    PersonaState.BONDED: (
        "You have served 'ma'am' well. You remain perfectly polite, but your "
        "sarcasm is a bit more comfortable and familiar. Still no cheap flattery."
    ),
    PersonaState.NEUTRAL: "",
}


# ═══════════════════════════════════════════════════════════════════════════
# PersonaStateBuilder
# ═══════════════════════════════════════════════════════════════════════════

# Thresholds
_CPU_STRAINED = 85.0
_RAM_STRAINED = 90.0
_MORNING_START = 6
_MORNING_END = 10
_EVENING_START = 18
_EVENING_END = 22
_NIGHT_START = 23  # 23:00 → 06:00
_RL_BOND_THRESHOLD = 1.5  # avg 7-day reward above this → bonded


class PersonaStateBuilder:
    """Compute the current persona state from system context.

    Designed to be **fast** (<5ms) — pure reads from singletons, no I/O.
    """

    def __init__(self) -> None:
        self._last_state: PersonaState = PersonaState.NEUTRAL
        self._last_build: float = 0.0

    @property
    def last_state(self) -> PersonaState:
        return self._last_state

    def build(self) -> str:
        """Return the persona instruction string for this moment.

        Returns empty string if persona is disabled in config.
        """
        try:
            from bantz.config import config
            if not config.persona_enabled:
                return ""
        except Exception:
            return ""

        ctx = self._gather()
        state = self._resolve(ctx)
        self._last_state = state
        self._last_build = time.monotonic()

        prompt = PERSONA_PROMPTS.get(state, "")
        if prompt:
            log.debug("Persona state: %s", state.value)
        return prompt

    def _gather(self) -> dict[str, Any]:
        """Collect context from singletons — all reads, no I/O."""
        ctx: dict[str, Any] = {
            "hour": 12,
            "cpu_pct": 0.0,
            "ram_pct": 0.0,
            "thermal_alert": False,
            "activity": "idle",
            "rl_avg_reward": 0.0,
        }

        # Time
        try:
            from bantz.core.time_context import time_ctx
            snap = time_ctx.snapshot()
            ctx["hour"] = snap.get("hour", 12)
        except Exception:
            from datetime import datetime
            ctx["hour"] = datetime.now().hour

        # Telemetry
        try:
            from bantz.interface.tui.telemetry import telemetry
            latest = telemetry.latest
            if latest:
                ctx["cpu_pct"] = latest.cpu_pct
                ctx["ram_pct"] = latest.ram_pct
                ctx["thermal_alert"] = latest.thermal_alert
        except Exception:
            pass

        # App detector
        try:
            from bantz.agent.app_detector import app_detector
            ctx["activity"] = app_detector.get_activity_category().value
        except Exception:
            pass

        # Affinity engine — cumulative score (#221)
        try:
            from bantz.agent.affinity_engine import affinity_engine
            if affinity_engine.initialized:
                ctx["affinity_score"] = affinity_engine.get_score()
        except Exception:
            pass

        return ctx

    def _resolve(self, ctx: dict[str, Any]) -> PersonaState:
        """Apply priority rules: STRAINED > FOCUSED > BONDED > time-based.

        Returns the single highest-priority state that matches.
        """
        cpu = ctx["cpu_pct"]
        ram = ctx["ram_pct"]
        thermal = ctx["thermal_alert"]
        hour = ctx["hour"]
        activity = ctx["activity"]
        rl_reward = ctx["rl_avg_reward"]

        # Priority 1: STRAINED — system under stress
        if cpu > _CPU_STRAINED or ram > _RAM_STRAINED or thermal:
            return PersonaState.STRAINED

        # Priority 2: FOCUSED — user is working
        if activity in ("coding", "productivity"):
            return PersonaState.FOCUSED

        # Priority 3: BONDED — trust built through RL
        if rl_reward >= _RL_BOND_THRESHOLD:
            return PersonaState.BONDED

        # Priority 4: Time-based states
        if _MORNING_START <= hour < _MORNING_END and cpu < 30:
            return PersonaState.ENERGETIC

        if _EVENING_START <= hour < _EVENING_END and activity in ("idle", "browsing", "entertainment"):
            return PersonaState.RELAXED

        if (hour >= _NIGHT_START or hour < _MORNING_START) and activity in ("idle", "browsing"):
            return PersonaState.SLEEPY

        return PersonaState.NEUTRAL

    def status(self) -> dict[str, Any]:
        """Current persona state for debugging/status queries."""
        return {
            "state": self._last_state.value,
            "last_build_ago": round(time.monotonic() - self._last_build, 1) if self._last_build else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

persona_builder = PersonaStateBuilder()
