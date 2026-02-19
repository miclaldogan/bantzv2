"""
Bantz v2 — Broadcaster Greeting

Context-aware theatrical greeting on app launch.
Bantz is "The Broadcaster" — a 1930s radio showman who treats the terminal as his studio.
Combines absence awareness, time-of-day, and live service summaries.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from bantz.core.time_context import get_segment
from bantz.core.profile import profile


class Butler:

    async def greet(self, session_info: dict[str, Any]) -> str:
        """
        Compose a butler-style greeting:
        1. Time-of-day + absence-aware opener
        2. Live data summaries (mail, calendar, classroom)
        3. Return 2-4 natural sentences
        """
        now = datetime.now()
        seg = get_segment(now.hour)

        opener = self._opener(
            absence_label=session_info["absence_label"],
            absence_hours=session_info["absence_hours"],
            is_first=session_info.get("is_first", False),
            segment=seg,
        )

        # Fire live data fetchers in parallel (each isolated)
        summaries = await asyncio.gather(
            self._mail_summary(),
            self._calendar_summary(now),
            self._classroom_summary(),
            return_exceptions=True,
        )
        mail_str, cal_str, class_str = [
            r if isinstance(r, str) else None for r in summaries
        ]

        return self._format(opener, mail_str, cal_str, class_str)

    # ── Opener generation ─────────────────────────────────────────────────

    def _opener(
        self,
        absence_label: str,
        absence_hours: float,
        is_first: bool,
        segment: str,
    ) -> str:
        if is_first:
            # Very first launch ever
            return (
                "Sinyal güçlü, kahve acı, ve ben buradayım. "
                "Selamlar eski dostum — yayın başlasın."
            )

        if absence_hours < 1:
            return "Aa, tekrar sen! Sahneyi terk etmemişsin bile."

        if absence_hours < 6:
            return "Hoş geldin dostum, sesin güzel geliyor bu frekanstan."

        if absence_hours < 20:
            # Same day, been away for a while
            return "Eski arkadaşım, geri döndün! Stüdyo sensiz sessizdi."

        if absence_hours < 30:
            # Closed last night, opened this morning
            if segment == "sabah":
                return (
                    "Günaydın dostum! Umarım rüyalar güzeldi — "
                    "çünkü gerçeklik burada beni bekliyor."
                )
            return "Dün gece kapattın perdeyi, ama şov devam ediyor."

        if absence_hours < 72:
            return (
                "Eski arkadaşım! Birkaç gündür stüdyo karanlıktı. "
                "Ama sinyal hiç kesilmedi."
            )

        if absence_hours < 168:
            return (
                "Dostum, bir haftadır yayında yoktun! "
                "Frekanslar seni arıyordu."
            )

        return (
            "Sevgili dinleyicim... uzun zamandır ses yoktu bu kanalda. "
            "Ama Bantz her zaman burada, her zaman yayında."
        )

    @staticmethod
    def _time_greeting(segment: str) -> str:
        return {
            "gece_erken": "İyi geceler",
            "sabah": "Günaydın",
            "oglen": "İyi öğlenler",
            "aksam": "İyi akşamlar",
            "gece_gec": "İyi geceler",
        }.get(segment, "Merhaba")

    # ── Formatters ────────────────────────────────────────────────────────

    def _format(
        self,
        opener: str,
        mail: Optional[str],
        calendar: Optional[str],
        classroom: Optional[str],
    ) -> str:
        parts = [opener]

        details: list[str] = []
        if calendar:
            details.append(calendar)
        if mail:
            details.append(mail)
        if classroom:
            details.append(classroom)

        if details:
            parts.append(" ".join(details))

        return " ".join(parts)

    # ── Live data fetchers (isolated, never crash) ────────────────────────

    async def _mail_summary(self) -> Optional[str]:
        """Count unread messages if Gmail is configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("gmail") != "ok":
                return None

            from bantz.tools.gmail import GmailTool
            g = GmailTool()
            result = await g.execute(action="filter", q="is:unread", max_results=10)
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "sonuç yok" in output.lower() or "bulunamadı" in output.lower():
                return None
            # Count lines that look like mail entries (start with emoji or number)
            lines = [l for l in output.splitlines() if l.strip()]
            count = len(lines)
            if count == 0:
                return None
            if count == 1:
                return "Posta kutusunda 1 okunmamış mektup bekliyor."
            return f"Posta kutusunda {count} okunmamış mektup birikmiş."
        except Exception:
            return None

    async def _calendar_summary(self, now: datetime) -> Optional[str]:
        """Get next calendar event if configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("calendar") != "ok":
                return None

            from bantz.tools.calendar import CalendarTool
            c = CalendarTool()
            result = await c.execute(action="today")
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "etkinlik yok" in output.lower() or "boş" in output.lower():
                return None
            # Find first meaningful event line (skip date headers like "Bugün:")
            import re
            for line in output.splitlines():
                stripped = line.strip()
                if stripped and not stripped.endswith(":") and len(stripped) > 5:
                    # Remove [id:...] suffixes from calendar output
                    cleaned = re.sub(r"\s*\[id:[^\]]+\]", "", stripped).strip()
                    if cleaned:
                        return f"Bugünün programında: {cleaned}."
            return None
        except Exception:
            return None

    async def _classroom_summary(self) -> Optional[str]:
        """Get upcoming assignment deadlines if configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("classroom") != "ok":
                return None

            from bantz.tools.classroom import ClassroomTool
            c = ClassroomTool()
            result = await c.execute(action="upcoming")
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "ödev yok" in output.lower() or "bulunamadı" in output.lower():
                return None
            lines = [l for l in output.splitlines() if l.strip()]
            count = len(lines)
            if count == 0:
                return None
            return f"Sahne arkasında {count} ödev teslim bekliyor."
        except Exception:
            return None


butler = Butler()
