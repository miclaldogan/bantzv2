"""
Bantz v2 — Greeting & Scheduled Morning Briefing (#80)

Handles:
1. Boot greeting — delegates to butler for rich proactive greeting on launch
2. Scheduled morning briefing — fires at configured time (default 08:00)

Usage:
    from bantz.personality.greeting import greeting_manager

    # Boot greeting (called from app.py on_mount)
    text = await greeting_manager.boot_greeting(session_info)

    # Check if morning briefing should fire now
    text = await greeting_manager.morning_briefing_if_due()
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Optional

from bantz.config import config
from bantz.core.briefing import briefing as _briefing

log = logging.getLogger("bantz.greeting")


class GreetingManager:

    def __init__(self) -> None:
        self._last_briefing_date: date | None = None

    async def boot_greeting(self, session_info: dict[str, Any]) -> str:
        """
        Rich boot greeting via butler.
        Already wired in app.py via _enrich_butler_greeting — this is an
        alternative entry point if callers want it directly.
        """
        try:
            from bantz.core.butler import butler
            return await butler.greet(session_info)
        except Exception as exc:
            log.debug("Boot greeting failed: %s", exc)
            return ""

    async def morning_briefing_if_due(self) -> Optional[str]:
        """
        Check if it's time for the morning briefing.
        Returns briefing text if due, None otherwise.

        Fires once per day at the configured hour:minute (default 08:00).
        Won't fire if:
        - morning_briefing_enabled is False
        - already fired today
        - current time is outside the [hour:minute, hour:minute+15) window
        """
        if not config.morning_briefing_enabled:
            return None

        now = datetime.now()
        today = now.date()

        # Already fired today?
        if self._last_briefing_date == today:
            return None

        # Are we in the right time window? (configured time ± 15 min)
        target_hour = config.morning_briefing_hour
        target_minute = config.morning_briefing_minute
        now_minutes = now.hour * 60 + now.minute
        target_minutes = target_hour * 60 + target_minute

        if not (target_minutes <= now_minutes < target_minutes + 15):
            return None

        # Fire the briefing
        self._last_briefing_date = today
        log.info("Morning briefing triggered at %02d:%02d", now.hour, now.minute)

        try:
            text = await _briefing.generate()
            return text
        except Exception as exc:
            log.warning("Morning briefing failed: %s", exc)
            return None


greeting_manager = GreetingManager()
