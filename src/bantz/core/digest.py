"""
Bantz v3 — Daily & Weekly Digest (#88)

Evening daily digest and Sunday weekly digest combining all data sources.
Gemini Flash synthesises the raw data into a natural language summary.

Usage:
    from bantz.core.digest import digest_manager

    text = await digest_manager.daily_digest()
    text = await digest_manager.weekly_digest()

Config (env / .env):
    BANTZ_DAILY_DIGEST_ENABLED=true
    BANTZ_DAILY_DIGEST_HOUR=20
    BANTZ_DAILY_DIGEST_MINUTE=0
    BANTZ_WEEKLY_DIGEST_ENABLED=true
    BANTZ_WEEKLY_DIGEST_DAY=sunday
    BANTZ_WEEKLY_DIGEST_HOUR=20
    BANTZ_WEEKLY_DIGEST_MINUTE=0
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("bantz.digest")

_DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

DAILY_SYSTEM_PROMPT = """\
You are Bantz, a personal AI assistant. Write a concise evening Daily Digest
for the user (3rd person is fine). Use bullet points. Keep it under 300 words.
Tone: friendly, informative, slightly witty — like a personal news anchor.
Sections: Tasks & Reminders, Email, Calendar, Weather Tomorrow, Commitments.
Skip any section with no data. Don't add data the user didn't provide.
Write in English."""

WEEKLY_SYSTEM_PROMPT = """\
You are Bantz, a personal AI assistant. Write a Weekly Digest (Sunday review)
for the user. Use bullet points. Keep it under 500 words.
Tone: reflective, helpful — highlight wins, flag overdue items, preview the week.
Sections: Week Review (tasks/reminders), Email Volume, Calendar Utilization,
Knowledge Graph Growth, Open Commitments, Upcoming Week Preview.
Skip any section with no data. Don't add data the user didn't provide.
Write in English."""


class DigestManager:
    """Generates daily and weekly digests."""

    def __init__(self) -> None:
        self._last_daily: Optional[str] = None  # date string
        self._last_weekly: Optional[str] = None  # week number string

    # ── Public API ────────────────────────────────────────────────────────

    async def daily_digest(self) -> str:
        """Generate the evening daily digest."""
        raw = await self._collect_daily_data()
        summary = await self._synthesize(raw, DAILY_SYSTEM_PROMPT)
        now = datetime.now()
        header = f"📊 Daily Digest — {now.strftime('%A, %d %B %Y')}"
        return f"{header}\n\n{summary}"

    async def weekly_digest(self) -> str:
        """Generate the Sunday weekly digest."""
        raw = await self._collect_weekly_data()
        summary = await self._synthesize(raw, WEEKLY_SYSTEM_PROMPT)
        now = datetime.now()
        week_num = now.isocalendar()[1]
        header = f"📊 Weekly Digest — Week {week_num}, {now.year}"
        return f"{header}\n\n{summary}"

    async def daily_if_due(self) -> Optional[str]:
        """Return digest text if it's time, else None. Call from polling loop."""
        from bantz.config import config
        if not config.daily_digest_enabled:
            return None

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        if self._last_daily == today:
            return None

        if now.hour == config.daily_digest_hour and now.minute >= config.daily_digest_minute:
            self._last_daily = today
            try:
                return await self.daily_digest()
            except Exception as exc:
                log.warning("Daily digest failed: %s", exc)
                return None
        return None

    async def weekly_if_due(self) -> Optional[str]:
        """Return weekly digest if it's the right day+time, else None."""
        from bantz.config import config
        if not config.weekly_digest_enabled:
            return None

        now = datetime.now()
        target_day = _DAY_MAP.get(config.weekly_digest_day.lower(), 6)

        if now.weekday() != target_day:
            return None

        week_key = f"{now.year}-W{now.isocalendar()[1]}"
        if self._last_weekly == week_key:
            return None

        if now.hour == config.weekly_digest_hour and now.minute >= config.weekly_digest_minute:
            self._last_weekly = week_key
            try:
                return await self.weekly_digest()
            except Exception as exc:
                log.warning("Weekly digest failed: %s", exc)
                return None
        return None

    # ── Data collection — daily ───────────────────────────────────────────

    async def _collect_daily_data(self) -> str:
        """Gather raw data from all sources for daily digest."""
        parts: list[str] = []

        results = await asyncio.gather(
            self._get_reminders_today(),
            self._get_email_stats_today(),
            self._get_calendar_today(),
            self._get_tomorrow_weather(),
            self._get_commitments(),
            return_exceptions=True,
        )

        labels = [
            "REMINDERS & TASKS TODAY",
            "EMAIL TODAY",
            "CALENDAR TODAY",
            "WEATHER TOMORROW",
            "COMMITMENTS DUE SOON",
        ]

        for label, result in zip(labels, results):
            if isinstance(result, str) and result:
                parts.append(f"[{label}]\n{result}")

        if not parts:
            return "No data available for today's digest."

        return "\n\n".join(parts)

    # ── Data collection — weekly ──────────────────────────────────────────

    async def _collect_weekly_data(self) -> str:
        """Gather raw data for weekly digest."""
        parts: list[str] = []

        results = await asyncio.gather(
            self._get_reminders_week(),
            self._get_email_stats_week(),
            self._get_calendar_week(),
            self._get_graph_growth(),
            self._get_commitments(),
            self._get_next_week_preview(),
            return_exceptions=True,
        )

        labels = [
            "TASKS & REMINDERS THIS WEEK",
            "EMAIL THIS WEEK",
            "CALENDAR THIS WEEK",
            "KNOWLEDGE GRAPH GROWTH",
            "OPEN COMMITMENTS",
            "NEXT WEEK PREVIEW",
        ]

        for label, result in zip(labels, results):
            if isinstance(result, str) and result:
                parts.append(f"[{label}]\n{result}")

        if not parts:
            return "No data available for this week's digest."

        return "\n\n".join(parts)

    # ── Individual data fetchers ──────────────────────────────────────────

    async def _get_reminders_today(self) -> Optional[str]:
        try:
            from bantz.core.scheduler import scheduler
            upcoming = scheduler.list_upcoming(limit=20)
            fired_today = scheduler.due_today()
            total_active = scheduler.count_active()

            lines = [f"Active reminders: {total_active}"]
            if fired_today:
                lines.append(f"Due today: {len(fired_today)}")
                for r in fired_today[:5]:
                    lines.append(f"  • {r['title']}")
            if upcoming:
                lines.append("Upcoming:")
                for r in upcoming[:5]:
                    fire = datetime.fromisoformat(r["fire_at"])
                    place = r.get("trigger_place")
                    if place:
                        lines.append(f"  • {r['title']} (at {place})")
                    else:
                        lines.append(f"  • {r['title']} — {fire.strftime('%d %b %H:%M')}")
            return "\n".join(lines)
        except Exception:
            return None

    async def _get_reminders_week(self) -> Optional[str]:
        try:
            from bantz.core.scheduler import scheduler
            upcoming = scheduler.list_upcoming(limit=20)
            all_items = scheduler.list_all(limit=50)

            # Count fired this week
            week_ago = datetime.now() - timedelta(days=7)
            fired_this_week = [
                r for r in all_items
                if r.get("fired") and r.get("created_at", "") >= week_ago.isoformat()
            ]

            lines = [
                f"Completed this week: {len(fired_this_week)}",
                f"Still active: {scheduler.count_active()}",
            ]
            if upcoming:
                lines.append("Next up:")
                for r in upcoming[:5]:
                    lines.append(f"  • {r['title']}")
            return "\n".join(lines)
        except Exception:
            return None

    async def _get_email_stats_today(self) -> Optional[str]:
        try:
            from bantz.tools.gmail import email_stats_today
            stats = await email_stats_today()
            if all(v == 0 for v in stats.values()):
                return None
            return (
                f"Received: {stats['received']}\n"
                f"Sent: {stats['sent']}\n"
                f"Currently unread: {stats['unread']}"
            )
        except Exception:
            return None

    async def _get_email_stats_week(self) -> Optional[str]:
        try:
            from bantz.tools.gmail import email_stats_week
            stats = await email_stats_week()
            if all(v == 0 for v in stats.values()):
                return None
            return (
                f"Received this week: {stats['received']}\n"
                f"Sent this week: {stats['sent']}\n"
                f"Currently unread: {stats['unread']}"
            )
        except Exception:
            return None

    async def _get_calendar_today(self) -> Optional[str]:
        try:
            from bantz.tools.calendar import CalendarTool
            c = CalendarTool()
            result = await c.execute(action="today")
            if result.success:
                text = result.output.strip()
                if "no events" in text.lower():
                    return "No events today"
                return text
        except Exception:
            pass
        return None

    async def _get_calendar_week(self) -> Optional[str]:
        try:
            from bantz.tools.calendar import CalendarTool
            c = CalendarTool()
            result = await c.execute(action="week")
            if result.success:
                text = result.output.strip()
                # Count events
                event_lines = [line for line in text.splitlines() if line.strip().startswith("•") or "—" in line]
                return f"Events this week: {len(event_lines)}\n{text}"
        except Exception:
            pass
        return None

    async def _get_tomorrow_weather(self) -> Optional[str]:
        try:
            from bantz.tools.weather import tomorrow_forecast
            forecast = await tomorrow_forecast()
            if forecast:
                return f"Tomorrow: {forecast}"
        except Exception:
            pass
        return None

    async def _get_graph_growth(self) -> Optional[str]:
        try:
            from bantz.memory.bridge import palace_bridge
            if not palace_bridge or not palace_bridge.enabled:
                return None

            stats = palace_bridge.stats()
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            growth = palace_bridge.growth_since(week_ago)

            lines = [
                f"Total: {stats.get('entities', 0)} entities, "
                f"{stats.get('triples', 0)} triples, "
                f"{stats.get('drawers', 0)} drawers",
            ]
            if growth.get("new_entities", 0) > 0:
                lines.append(f"New this week: +{growth['new_entities']} entities, +{growth['new_triples']} triples")
            else:
                lines.append("No new entries this week")
            return "\n".join(lines)
        except Exception:
            return None

    async def _get_commitments(self) -> Optional[str]:
        try:
            from bantz.memory.bridge import palace_bridge
            if not palace_bridge or not palace_bridge.enabled:
                return None

            # Query KnowledgeGraph for commitment/decision triples
            kg = palace_bridge.kg
            if kg is None:
                return None

            triples = kg.recent(limit=30)
            commitments = [
                t for t in triples
                if any(kw in (t.get("relation", "") or "").lower()
                       for kw in ("committed", "decided", "promised", "will"))
            ]
            if not commitments:
                return None

            lines = [f"Active commitments: {len(commitments)}"]
            for c in commitments:
                subj = c.get("subject", "")
                rel = c.get("relation", "")
                obj = c.get("object", "")
                lines.append(f"  • {subj} {rel} {obj}".strip())
            return "\n".join(lines)
        except Exception:
            return None

    async def _get_next_week_preview(self) -> Optional[str]:
        """Calendar events for the upcoming week."""
        try:
            from bantz.tools.calendar import CalendarTool
            c = CalendarTool()
            result = await c.execute(action="week")
            if result.success:
                return result.output.strip() or None
        except Exception:
            pass
        return None

    # ── LLM synthesis ─────────────────────────────────────────────────────

    @staticmethod
    async def _synthesize(raw_data: str, system_prompt: str) -> str:
        """Pass raw data through Gemini Flash for natural language summary."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_data},
        ]

        # Try Gemini first (preferred for digest — faster, cheaper)
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                result = await gemini.chat(messages, temperature=0.3)
                if result and len(result) > 20:
                    return result
        except Exception:
            pass

        # Fallback to Ollama
        try:
            from bantz.llm.ollama import ollama
            return await ollama.chat(messages)
        except Exception:
            pass

        # Final fallback: return raw data if no LLM available
        return raw_data


# Singleton
digest_manager = DigestManager()
