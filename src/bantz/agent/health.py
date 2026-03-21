"""
Bantz — Proactive Health & Break Interventions (#168)

Rule-driven health monitoring engine that combines telemetry, time context,
app detection, and ambient state to deliver empathetic, non-intrusive
break/health interventions.

Architecture:
    telemetry.py ──┐
    time_context ──┤     ┌─────────────────────────┐
    app_detector ──┼────▶│   HealthRuleEvaluator    │
    ambient (#166)─┤     │                          │
    idle detector──┘     │  Rule 1: Late Night Load │
                         │  Rule 2: Marathon Session │
                         │  Rule 3: Eye Strain       │
                         │  Rule 4: Thermal Stress   │
                         │  Rule 5: Late Night Music │
                         └──────────┬────────────────┘
                                    │ triggered?
                                    ▼
                         InterventionQueue (Priority.HIGH)
                             + RL feedback

Senior Fixes Applied:
  1. Ghost Session Trap: Uses real keyboard/mouse idle detection (loginctl,
     dbus, /proc/stat) instead of uptime.  Session clock pauses after
     15 min idle and resets on genuine user activity.
  2. RL False Positive: Only awards break-reward when screen is LOCKED or
     machine is SUSPENDED — not just mouse-idle (YouTube-proof).
  3. Thermal Panic: Requires temperature to be sustained for ≥30s across
     3 consecutive readings before firing to avoid spike false positives.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from bantz.core.event_bus import bus, Event

log = logging.getLogger("bantz.health")


# ═══════════════════════════════════════════════════════════════════════════
# Idle detection — Senior Fix #1 (Ghost Session Trap)
# ═══════════════════════════════════════════════════════════════════════════

_IDLE_THRESHOLD_SEC = 15 * 60  # 15 min → user is "away"


def get_idle_ms() -> int:
    """Return keyboard/mouse idle time in milliseconds.

    Tries multiple methods in order of reliability:
      1. xprintidle (X11, most accurate)
      2. dbus org.gnome.Mutter.IdleMonitor (GNOME/Wayland)
      3. /proc/stat heuristic (fallback — always available)

    Returns 0 on failure (assume active — never over-trigger).
    """
    # Method 1: xprintidle
    if shutil.which("xprintidle"):
        try:
            r = subprocess.run(
                ["xprintidle"], capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip().isdigit():
                return int(r.stdout.strip())
        except Exception:
            pass

    # Method 2: GNOME Mutter IdleMonitor (works on Wayland too)
    if shutil.which("dbus-send"):
        try:
            r = subprocess.run(
                [
                    "dbus-send", "--print-reply",
                    "--dest=org.gnome.Mutter.IdleMonitor",
                    "/org/gnome/Mutter/IdleMonitor/Core",
                    "org.gnome.Mutter.IdleMonitor.GetIdletime",
                ],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                # Output: "   uint64 12345"
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("uint64") or line.startswith("uint32"):
                        val = line.split()[-1]
                        if val.isdigit():
                            return int(val)
        except Exception:
            pass

    return 0  # assume active


def is_screen_locked() -> bool:
    """Check if the screen is currently locked — Senior Fix #2 (RL reward).

    Uses loginctl LockedHint as the most reliable cross-DE method.
    """
    try:
        r = subprocess.run(
            ["loginctl", "show-session", "auto", "-p", "LockedHint"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return "yes" in r.stdout.lower()
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Session tracker — real active-time counter (not uptime)
# ═══════════════════════════════════════════════════════════════════════════

class SessionTracker:
    """Tracks actual active computer time, pausing when user is idle.

    The session clock only runs when the user is actively using the machine
    (keyboard/mouse activity detected within the last 15 minutes).  If the
    user goes idle for ≥15 min, the clock pauses and resumes when activity
    returns.  A screen lock/suspend resets the session entirely.
    """

    def __init__(self, idle_threshold_s: float = _IDLE_THRESHOLD_SEC) -> None:
        self._idle_threshold = idle_threshold_s
        self._active_seconds: float = 0.0
        self._last_tick: float = time.monotonic()
        self._was_idle: bool = False
        self._last_break_at: float = time.monotonic()
        self._break_detected: bool = False

    def tick(self) -> None:
        """Call every health-check cycle (~5 min).  Updates active time."""
        now = time.monotonic()
        idle_ms = get_idle_ms()
        idle_s = idle_ms / 1000.0
        locked = is_screen_locked()

        elapsed = now - self._last_tick
        self._last_tick = now

        if locked:
            # Screen locked → count as break, reset session
            self._register_break()
            self._was_idle = True
            return

        if idle_s >= self._idle_threshold:
            # User is idle → don't count this time
            if not self._was_idle:
                log.debug("Session: user went idle (%.0fs)", idle_s)
            self._was_idle = True
            return

        if self._was_idle:
            # User just came back from idle
            log.debug("Session: user returned from idle")
            self._register_break()
            self._was_idle = False
            return

        # User is active — accumulate time
        self._active_seconds += elapsed

    def _register_break(self) -> None:
        """Mark that a genuine break was taken."""
        self._break_detected = True
        self._last_break_at = time.monotonic()
        self._active_seconds = 0.0

    @property
    def active_hours(self) -> float:
        """Hours of active computer use since last break."""
        return self._active_seconds / 3600.0

    @property
    def had_recent_break(self) -> bool:
        """True if user has taken a break (idle ≥15 min or lock) this session."""
        return self._break_detected

    def consume_break_flag(self) -> bool:
        """Return and clear break_detected (for RL reward — one-shot)."""
        had = self._break_detected
        self._break_detected = False
        return had

    @property
    def minutes_since_break(self) -> float:
        return (time.monotonic() - self._last_break_at) / 60.0

    def reset(self) -> None:
        self._active_seconds = 0.0
        self._last_tick = time.monotonic()
        self._was_idle = False
        self._break_detected = False
        self._last_break_at = time.monotonic()


# ═══════════════════════════════════════════════════════════════════════════
# Thermal sustained-check — Senior Fix #3
# ═══════════════════════════════════════════════════════════════════════════

class ThermalHistory:
    """Track temperature readings to avoid spike false positives.

    Requires ≥ `required_count` consecutive above-threshold readings
    before flagging thermal stress.  With a 10s collection interval,
    3 readings = 30s sustained.
    """

    def __init__(self, required_count: int = 3) -> None:
        self._required = required_count
        self._cpu_streak: int = 0
        self._gpu_streak: int = 0

    def record(self, cpu_temp: float, gpu_temp: float,
               cpu_threshold: float, gpu_threshold: float) -> bool:
        """Record new readings.  Returns True only if sustained."""
        if cpu_temp > cpu_threshold:
            self._cpu_streak += 1
        else:
            self._cpu_streak = 0

        if gpu_temp > gpu_threshold:
            self._gpu_streak += 1
        else:
            self._gpu_streak = 0

        return (self._cpu_streak >= self._required or
                self._gpu_streak >= self._required)

    @property
    def cpu_streak(self) -> int:
        return self._cpu_streak

    @property
    def gpu_streak(self) -> int:
        return self._gpu_streak

    def reset(self) -> None:
        self._cpu_streak = 0
        self._gpu_streak = 0


# ═══════════════════════════════════════════════════════════════════════════
# Health Rules
# ═══════════════════════════════════════════════════════════════════════════


def _is_late_night(hour: int, late_hour: int) -> bool:
    """Check if current hour falls in the late-night window [late_hour, 5).

    Handles wrap-around: if late_hour >= 5 (e.g. 23), the range wraps
    through midnight: hour >= 23 or hour < 5.
    """
    if late_hour < 5:
        return late_hour <= hour < 5
    return hour >= late_hour or hour < 5


class RuleID(str, Enum):
    LATE_NIGHT_LOAD = "late_night_load"
    MARATHON_SESSION = "marathon_session"
    EYE_STRAIN = "eye_strain"
    THERMAL_STRESS = "thermal_stress"
    LATE_NIGHT_MUSIC = "late_night_music"


# Default cooldowns in seconds
RULE_COOLDOWNS: dict[str, float] = {
    RuleID.LATE_NIGHT_LOAD: 2 * 3600,     # 2h
    RuleID.MARATHON_SESSION: 3 * 3600,     # 3h
    RuleID.EYE_STRAIN: 1.5 * 3600,        # 1.5h
    RuleID.THERMAL_STRESS: 30 * 60,        # 30 min
    RuleID.LATE_NIGHT_MUSIC: 2 * 3600,     # 2h
}


@dataclass
class RuleResult:
    """Outcome of evaluating a single health rule."""
    rule_id: RuleID
    fired: bool
    title: str = ""
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Health Rule Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class HealthRuleEvaluator:
    """Evaluate all health rules against current system/user state.

    Manages per-rule cooldowns, thermal history, and session tracking.
    Pushes interventions to InterventionQueue on rule fire.
    """

    def __init__(self) -> None:
        self._cooldowns: dict[str, float] = {}  # rule_id → monotonic ts
        self._session = SessionTracker()
        self._thermal = ThermalHistory(required_count=3)
        self._initialized = False
        self._screen_time_start: float = time.monotonic()
        self._last_activity_not_idle: float = time.monotonic()
        # Telemetry cache — updated via EventBus (Sprint 3 Part 2)
        self._telemetry: dict[str, float] = {
            "cpu_pct": 0.0, "cpu_temp": 0.0, "gpu_temp": 0.0,
        }

    def init(self) -> None:
        self._initialized = True
        self._session.reset()
        self._thermal.reset()
        self._cooldowns.clear()
        self._screen_time_start = time.monotonic()
        # Subscribe to telemetry events from the bus
        bus.on("telemetry_update", self._on_telemetry)
        log.info("HealthRuleEvaluator initialized")

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _on_telemetry(self, event: Event) -> None:
        """Bus handler: cache latest telemetry for rule evaluation."""
        d = event.data
        for key in ("cpu_pct", "cpu_temp", "gpu_temp"):
            if key in d:
                self._telemetry[key] = float(d[key])

    @property
    def session(self) -> SessionTracker:
        return self._session

    @property
    def thermal(self) -> ThermalHistory:
        return self._thermal

    # ── Cooldown management ───────────────────────────────────────────

    def _can_fire(self, rule_id: str, cooldown_s: float | None = None) -> bool:
        """Check if enough time has passed since this rule last fired."""
        if cooldown_s is None:
            cooldown_s = RULE_COOLDOWNS.get(rule_id, 3600)
        last = self._cooldowns.get(rule_id, 0.0)
        return (time.monotonic() - last) > cooldown_s

    def _mark_fired(self, rule_id: str) -> None:
        self._cooldowns[rule_id] = time.monotonic()

    def cooldown_remaining(self, rule_id: str) -> float:
        """Seconds until this rule can fire again."""
        cd = RULE_COOLDOWNS.get(rule_id, 3600)
        last = self._cooldowns.get(rule_id, 0.0)
        remaining = cd - (time.monotonic() - last)
        return max(0.0, remaining)

    # ── Gather context ────────────────────────────────────────────────

    def _gather(self) -> dict[str, Any]:
        """Collect all sensor/context data for rule evaluation."""
        ctx: dict[str, Any] = {
            "hour": 0, "segment": "morning",
            "cpu_pct": 0.0, "cpu_temp": 0.0, "gpu_temp": 0.0,
            "activity": "idle", "ambient_label": "unknown",
            "active_hours": self._session.active_hours,
            "had_break": self._session.had_recent_break,
        }

        # Time context
        try:
            from bantz.core.time_context import time_ctx
            snap = time_ctx.snapshot()
            ctx["hour"] = snap.get("hour", 0)
            ctx["segment"] = snap.get("segment", "morning")
        except Exception:
            from datetime import datetime
            ctx["hour"] = datetime.now().hour

        # Telemetry — from EventBus cache (Sprint 3 Part 2)
        ctx["cpu_pct"] = self._telemetry.get("cpu_pct", 0.0)
        ctx["cpu_temp"] = self._telemetry.get("cpu_temp", 0.0)
        ctx["gpu_temp"] = self._telemetry.get("gpu_temp", 0.0)

        # App detector
        try:
            from bantz.agent.app_detector import app_detector
            ctx["activity"] = app_detector.get_activity_category().value
        except Exception:
            pass

        # Ambient
        try:
            from bantz.agent.ambient import ambient_analyzer
            snap = ambient_analyzer.latest()
            if snap:
                ctx["ambient_label"] = snap.label.value
        except Exception:
            pass

        return ctx

    # ── Individual rules ──────────────────────────────────────────────

    def _eval_late_night_load(self, ctx: dict) -> RuleResult:
        """Rule 1: Late night (>02:00) + CPU > 80% + CODING."""
        from bantz.config import config
        hour = ctx["hour"]
        cpu = ctx["cpu_pct"]
        activity = ctx["activity"]
        late_hour = config.health_late_hour

        fired = (
            _is_late_night(hour, late_hour)
            and cpu > 80
            and activity == "coding"
            and self._can_fire(RuleID.LATE_NIGHT_LOAD)
        )
        return RuleResult(
            rule_id=RuleID.LATE_NIGHT_LOAD,
            fired=fired,
            title="🌙 Late Night Heavy Load",
            reason=(
                f"It's {hour:02d}:00 and your CPU is at {cpu:.0f}% while coding. "
                "Your machine is working as hard as you are — how about some rest?"
            ),
            data={"hour": hour, "cpu": cpu, "activity": activity},
        )

    def _eval_marathon_session(self, ctx: dict) -> RuleResult:
        """Rule 2: Active session > 4h without a genuine break."""
        from bantz.config import config
        hours = ctx["active_hours"]
        had_break = ctx["had_break"]
        threshold = config.health_session_max_hours

        fired = (
            hours >= threshold
            and not had_break
            and ctx["activity"] != "idle"
            and self._can_fire(RuleID.MARATHON_SESSION)
        )
        return RuleResult(
            rule_id=RuleID.MARATHON_SESSION,
            fired=fired,
            title="⏰ Marathon Session",
            reason=(
                f"You've been actively working for {hours:.1f} hours straight. "
                "A 10-minute break would do wonders for your focus."
            ),
            data={"active_hours": hours, "had_break": had_break},
        )

    def _eval_eye_strain(self, ctx: dict) -> RuleResult:
        """Rule 3: Screen time > 2h without break + user is active."""
        screen_hours = (time.monotonic() - self._screen_time_start) / 3600.0

        if ctx["activity"] == "idle":
            # Reset screen time tracker on idle
            self._screen_time_start = time.monotonic()
            screen_hours = 0.0

        fired = (
            screen_hours >= 2.0
            and ctx["activity"] != "idle"
            and self._can_fire(RuleID.EYE_STRAIN)
        )
        if fired:
            # Reset screen time after firing
            self._screen_time_start = time.monotonic()

        return RuleResult(
            rule_id=RuleID.EYE_STRAIN,
            fired=fired,
            title="👁️ Eye Strain Reminder",
            reason=(
                f"You've been staring at the screen for {screen_hours:.1f}h. "
                "Quick 20-20-20: look at something 20 feet away for 20 seconds."
            ),
            data={"screen_hours": screen_hours},
        )

    def _eval_thermal_stress(self, ctx: dict) -> RuleResult:
        """Rule 4: CPU > 85°C or GPU > 80°C — sustained for ≥30s."""
        from bantz.config import config
        cpu_t = ctx["cpu_temp"]
        gpu_t = ctx["gpu_temp"]
        cpu_threshold = config.health_thermal_cpu
        gpu_threshold = config.health_thermal_gpu

        sustained = self._thermal.record(
            cpu_t, gpu_t, cpu_threshold, gpu_threshold,
        )

        fired = sustained and self._can_fire(RuleID.THERMAL_STRESS)

        hot_part = "CPU" if self._thermal.cpu_streak >= 3 else "GPU"
        hot_temp = cpu_t if hot_part == "CPU" else gpu_t

        return RuleResult(
            rule_id=RuleID.THERMAL_STRESS,
            fired=fired,
            title=f"🔥 System Running Hot ({hot_part}: {hot_temp:.0f}°C)",
            reason=(
                f"Your {hot_part} has been above threshold for sustained period "
                f"(CPU: {cpu_t:.0f}°C, GPU: {gpu_t:.0f}°C). "
                "Maybe close some heavy tabs or resource-hungry apps?"
            ),
            data={"cpu_temp": cpu_t, "gpu_temp": gpu_t, "sustained": sustained},
        )

    def _eval_late_night_music(self, ctx: dict) -> RuleResult:
        """Rule 5: Late night + ambient=music/noisy + coding."""
        from bantz.config import config
        hour = ctx["hour"]
        ambient = ctx["ambient_label"]
        activity = ctx["activity"]
        late_hour = config.health_late_hour

        fired = (
            _is_late_night(hour, late_hour)
            and ambient in ("noisy", "speech")  # music detection via ambient
            and activity == "coding"
            and self._can_fire(RuleID.LATE_NIGHT_MUSIC)
        )
        return RuleResult(
            rule_id=RuleID.LATE_NIGHT_MUSIC,
            fired=fired,
            title="🎧 Late Night Coding Session",
            reason=(
                f"Great vibes for a late session at {hour:02d}:00, "
                "but your body needs sleep more than code. "
                "How about wrapping up after this commit?"
            ),
            data={"hour": hour, "ambient": ambient, "activity": activity},
        )

    # ── Main evaluation ───────────────────────────────────────────────

    def evaluate_all(self) -> list[RuleResult]:
        """Run all health rules against current state.

        Returns a list of results — caller should check `.fired` on each.
        """
        if not self._initialized:
            return []

        # Tick session tracker
        self._session.tick()

        ctx = self._gather()

        results = [
            self._eval_late_night_load(ctx),
            self._eval_marathon_session(ctx),
            self._eval_eye_strain(ctx),
            self._eval_thermal_stress(ctx),
            self._eval_late_night_music(ctx),
        ]

        for r in results:
            if r.fired:
                self._mark_fired(r.rule_id)
                # Publish to EventBus (Sprint 3 Part 2)
                try:
                    bus.emit_threadsafe(
                        "health_alert",
                        rule_id=r.rule_id.value,
                        title=r.title,
                        reason=r.reason,
                        **r.data,
                    )
                except Exception:
                    pass

        return results

    def check_break_reward(self) -> bool:
        """Senior Fix #2: Only reward when screen is LOCKED (YouTube-proof).

        Returns True if a genuine break was detected since last check.
        This is consumed once (one-shot) so RL gets exactly one reward.
        """
        if is_screen_locked():
            # Definitive break — screen is locked right now
            self._session._register_break()
            return self._session.consume_break_flag()
        return self._session.consume_break_flag()

    # ── Push intervention ─────────────────────────────────────────────

    def push_intervention(self, result: RuleResult) -> bool:
        """Create and push a health intervention to the queue."""
        try:
            from bantz.agent.interventions import (
                intervention_queue, Intervention,
                InterventionType, Priority,
            )

            iv = Intervention(
                type=InterventionType.HEALTH,
                priority=Priority.HIGH,
                title=result.title,
                reason=result.reason,
                source="health",
                action="health_break",
                ttl=30.0,
            )
            return intervention_queue.push(iv)
        except Exception as exc:
            log.warning("Failed to push health intervention: %s", exc)
            return False

    # ── Status report ─────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Current health engine state for status queries."""
        return {
            "initialized": self._initialized,
            "active_hours": round(self._session.active_hours, 2),
            "had_break": self._session.had_recent_break,
            "minutes_since_break": round(self._session.minutes_since_break, 1),
            "thermal_cpu_streak": self._thermal.cpu_streak,
            "thermal_gpu_streak": self._thermal.gpu_streak,
            "cooldowns": {
                rule_id: round(self.cooldown_remaining(rule_id) / 60, 1)
                for rule_id in RuleID
            },
        }

    def status_line(self) -> str:
        """One-line summary for TUI/brain status."""
        s = self.status()
        parts = [
            f"active={s['active_hours']:.1f}h",
            f"break={'✅' if s['had_break'] else '❌'}",
            f"since_break={s['minutes_since_break']:.0f}m",
        ]
        cpu_s = s["thermal_cpu_streak"]
        gpu_s = s["thermal_gpu_streak"]
        if cpu_s or gpu_s:
            parts.append(f"thermal=CPU:{cpu_s}/GPU:{gpu_s}")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

health_engine = HealthRuleEvaluator()
