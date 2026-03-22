"""
Bantz v3 — Google Calendar Tool
Full CRUD with smart time parsing, recurring events, attendees,
conflict detection, and morning-briefing integration.

Actions:
  today      — today's events
  week       — this week's events
  date       — events for a specific date
  upcoming   — next N events regardless of range
  create     — create event (natural time: "5pm", "noon", "at 3")
  delete     — delete by title or event ID
  update     — reschedule, rename, change duration
  conflicts  — check for conflicts at a given time

Scopes: calendar.readonly + calendar.events (already in token_store)
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.core.location import location_service
from bantz.core.time_parser import normalize_time
from bantz.tools import BaseTool, ToolResult, registry


# ── Recurrence helpers ────────────────────────────────────────────────────────

_RECURRENCE_MAP = {
    "daily":   "RRULE:FREQ=DAILY",
    "weekly":  "RRULE:FREQ=WEEKLY",
    "biweekly": "RRULE:FREQ=WEEKLY;INTERVAL=2",
    "monthly": "RRULE:FREQ=MONTHLY",
    "yearly":  "RRULE:FREQ=YEARLY",
    "weekdays": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
}


class CalendarTool(BaseTool):
    name = "calendar"
    description = (
        "Shows, creates, updates and deletes Google Calendar events. "
        "Supports recurring events, attendees, conflict detection. "
        "Use for: calendar, meeting, event, what's today, this week, "
        "add appointment, schedule, reschedule, delete event, upcoming. "
        "Always include date/time when creating events. "
        "NEVER invent event titles or times — ask if not provided."
    )
    risk_level = "safe"

    async def execute(
        self,
        action: str = "today",
        # today | week | date | upcoming | create | delete | update | conflicts
        title: str = "",
        date: str = "",           # "2026-03-03" or natural ("tomorrow")
        time: str = "",           # "15:00" or "5pm" or "noon"
        duration: int = 60,       # minutes
        event_id: str = "",       # for delete/update
        new_title: str = "",      # for update
        new_date: str = "",       # for update
        new_time: str = "",       # for update
        new_duration: int = 0,    # for update (0 = keep existing)
        recurrence: str = "",     # "daily", "weekly", "monthly", "yearly", "weekdays"
        attendees: str = "",      # comma-separated emails
        limit: int = 10,          # for upcoming
        **kwargs: Any,
    ) -> ToolResult:
        try:
            creds = token_store.get("calendar")
        except TokenNotFoundError as e:
            return ToolResult(success=False, output="", error=str(e))

        loc = await location_service.get()
        tz_name = loc.timezone

        # Resolve natural date references
        date = self._resolve_date_param(date)
        new_date = self._resolve_date_param(new_date)

        if action == "week":
            return await self._get_events(creds, tz_name, days=7, anchor=date)
        elif action == "date":
            return await self._get_events(creds, tz_name, days=1, anchor=date)
        elif action == "upcoming":
            return await self._upcoming(creds, tz_name, limit)
        elif action == "create":
            return await self._create(
                creds, tz_name, title, date, time, duration,
                recurrence, attendees,
            )
        elif action == "delete":
            return await self._delete(creds, event_id, title)
        elif action == "update":
            return await self._update(
                creds, tz_name, event_id, title,
                new_title, new_date, new_time, new_duration,
            )
        elif action == "conflicts":
            return await self._check_conflicts(creds, tz_name, date, time, duration)
        else:
            return await self._get_events(creds, tz_name, days=1, anchor=date)

    # ── Date param resolver ───────────────────────────────────────────────

    @staticmethod
    def _resolve_date_param(date_str: str) -> str:
        """Resolve natural date references to ISO format."""
        if not date_str:
            return ""
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str
        try:
            from bantz.core.date_parser import resolve_date
            dt = resolve_date(date_str)
            if dt:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        return date_str

    # ── List events ───────────────────────────────────────────────────────

    async def _get_events(
        self, creds, tz_name: str, days: int = 1, anchor: str = "",
    ) -> ToolResult:
        events = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_events_sync, creds, tz_name, days, anchor
        )
        if not events:
            label = self._period_label(days, anchor)
            return ToolResult(
                success=True,
                output=f"No events on your calendar for {label}.",
            )

        lines = []
        for ev in events:
            loc_str = f"  @ {ev['location']}" if ev.get("location") else ""
            recur_str = "  (recurring)" if ev.get("recurring") else ""
            lines.append(
                f"  {ev['start_local']}  {ev['summary']}{loc_str}{recur_str}"
                f"  [id:{ev['id'][:8]}]"
            )

        label = self._period_label(days, anchor)
        return ToolResult(
            success=True,
            output=f"{label}:\n" + "\n".join(lines),
            data={"count": len(events), "events": events},
        )

    async def _upcoming(self, creds, tz_name: str, limit: int) -> ToolResult:
        """Next N events regardless of date range."""
        events = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_upcoming_sync, creds, tz_name, limit
        )
        if not events:
            return ToolResult(success=True, output="No upcoming events.")

        lines = []
        for ev in events:
            loc_str = f"  @ {ev['location']}" if ev.get("location") else ""
            lines.append(f"  {ev['start_local']}  {ev['summary']}{loc_str}")

        return ToolResult(
            success=True,
            output=f"Next {len(events)} events:\n" + "\n".join(lines),
            data={"count": len(events), "events": events},
        )

    @staticmethod
    def _period_label(days: int, anchor: str) -> str:
        if not anchor:
            return "Today" if days == 1 else f"Next {days} days"
        try:
            dt = datetime.strptime(anchor, "%Y-%m-%d")
            today = datetime.now().date()
            delta = (dt.date() - today).days
            if delta == 0:
                return "Today"
            elif delta == 1:
                return "Tomorrow"
            elif delta == -1:
                return "Yesterday"
            else:
                return dt.strftime("%d %b %A")
        except Exception:
            return anchor

    # ── Create ────────────────────────────────────────────────────────────

    async def _create(
        self, creds, tz_name: str,
        title: str, date: str, time: str, duration: int,
        recurrence: str = "", attendees: str = "",
    ) -> ToolResult:
        if not title:
            return ToolResult(
                success=False, output="", error="Event title is required.",
            )
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # Smart time parsing — "5pm" -> "17:00", "noon" -> "12:00"
        time = normalize_time(time) if time else "09:00"

        # Check for conflicts before creating
        conflicts = await asyncio.get_event_loop().run_in_executor(
            None, self._find_conflicts_sync, creds, tz_name, date, time, duration
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._create_sync, creds, tz_name, title, date, time,
            duration, recurrence, attendees,
        )
        if not result:
            return ToolResult(
                success=False, output="", error="Could not create event.",
            )

        output = f"Event added ✓\n  {title}  {date} {time} ({duration} min)"
        if recurrence:
            output += f"\n  Recurring: {recurrence}"
        if attendees:
            output += f"\n  Attendees: {attendees}"
        if conflicts:
            names = ", ".join(c["summary"] for c in conflicts[:3])
            output += f"\n  ⚠ Overlaps with: {names}"

        return ToolResult(
            success=True, output=output,
            data={"event_id": result, "date": date, "time": time},
        )

    # ── Delete ────────────────────────────────────────────────────────────

    async def _delete(self, creds, event_id: str, title: str) -> ToolResult:
        if not event_id and title:
            events = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_events_sync, creds, "Europe/Istanbul", 30
            )
            matches = [
                e for e in events if title.lower() in e["summary"].lower()
            ]
            if not matches:
                return ToolResult(
                    success=False, output="",
                    error=f"No event found named '{title}'.",
                )
            if len(matches) > 1:
                lines = [
                    f"  [{e['id'][:8]}] {e['start_local']}  {e['summary']}"
                    for e in matches
                ]
                return ToolResult(
                    success=False, output="",
                    error="Multiple matches:\n" + "\n".join(lines)
                    + "\n\nPlease specify which one to delete.",
                )
            event_id = matches[0]["id"]
            title = matches[0]["summary"]

        if not event_id:
            return ToolResult(
                success=False, output="",
                error="No event specified for deletion.",
            )

        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._delete_sync, creds, event_id
        )
        if ok:
            return ToolResult(
                success=True,
                output=f"Event deleted ✓  [{title or event_id[:8]}]",
            )
        return ToolResult(
            success=False, output="", error="Could not delete event.",
        )

    # ── Update ────────────────────────────────────────────────────────────

    async def _update(
        self, creds, tz_name: str,
        event_id: str, title: str,
        new_title: str, new_date: str, new_time: str,
        new_duration: int = 0,
    ) -> ToolResult:
        if not event_id and title:
            events = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_events_sync, creds, tz_name, 30
            )
            matches = [
                e for e in events if title.lower() in e["summary"].lower()
            ]
            if not matches:
                return ToolResult(
                    success=False, output="",
                    error=f"'{title}' not found.",
                )
            event_id = matches[0]["id"]
            title = matches[0]["summary"]

        if not event_id:
            return ToolResult(
                success=False, output="",
                error="No event specified for update.",
            )

        # Normalize new_time if provided
        if new_time:
            new_time = normalize_time(new_time)

        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._update_sync, creds, tz_name, event_id,
            new_title, new_date, new_time, new_duration,
        )
        if ok:
            changes = []
            if new_title:
                changes.append(f"title → {new_title}")
            if new_date:
                changes.append(f"date → {new_date}")
            if new_time:
                changes.append(f"time → {new_time}")
            if new_duration:
                changes.append(f"duration → {new_duration} min")
            return ToolResult(
                success=True,
                output=f"Event updated ✓  [{title}]\n  "
                + ", ".join(changes),
            )
        return ToolResult(
            success=False, output="", error="Could not update event.",
        )

    # ── Conflict detection ────────────────────────────────────────────────

    async def _check_conflicts(
        self, creds, tz_name: str,
        date: str, time: str, duration: int,
    ) -> ToolResult:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        time = normalize_time(time) if time else "09:00"

        conflicts = await asyncio.get_event_loop().run_in_executor(
            None, self._find_conflicts_sync, creds, tz_name, date, time, duration
        )
        if not conflicts:
            return ToolResult(
                success=True,
                output=f"No conflicts at {date} {time} ({duration} min). You're clear!",
            )
        lines = [
            f"  {c['start_local']}  {c['summary']}" for c in conflicts
        ]
        return ToolResult(
            success=True,
            output=f"Conflicts at {date} {time}:\n" + "\n".join(lines),
            data={"conflicts": conflicts},
        )

    # ── Sync helpers ──────────────────────────────────────────────────────

    def _build_service(self, creds):
        from googleapiclient.discovery import build
        return build("calendar", "v3", credentials=creds)

    def _fetch_events_sync(
        self, creds, tz_name: str, days: int, anchor: str = "",
    ) -> list[dict]:
        import pytz

        tz = pytz.timezone(tz_name)
        if anchor:
            try:
                base = datetime.strptime(anchor, "%Y-%m-%d")
                base = tz.localize(base)
            except Exception:
                base = datetime.now(tz)
        else:
            base = datetime.now(tz)

        time_min = base.replace(
            hour=0, minute=0, second=0, microsecond=0,
        ).isoformat()
        time_max = (base + timedelta(days=days - 1)).replace(
            hour=23, minute=59, second=59,
        ).isoformat()

        svc = self._build_service(creds)
        result = svc.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=25,
        ).execute()

        return self._parse_event_list(result.get("items", []), tz)

    def _fetch_upcoming_sync(
        self, creds, tz_name: str, limit: int,
    ) -> list[dict]:
        import pytz

        tz = pytz.timezone(tz_name)
        now = datetime.now(tz).isoformat()

        svc = self._build_service(creds)
        result = svc.events().list(
            calendarId="primary",
            timeMin=now,
            singleEvents=True,
            orderBy="startTime",
            maxResults=limit,
        ).execute()

        return self._parse_event_list(result.get("items", []), tz)

    def _parse_event_list(self, items: list, tz) -> list[dict]:
        events = []
        for ev in items:
            start_raw = ev["start"].get("dateTime") or ev["start"].get("date")
            end_raw = ev["end"].get("dateTime") or ev["end"].get("date", "")
            try:
                if "T" in start_raw:
                    dt = datetime.fromisoformat(start_raw).astimezone(tz)
                    start_local = dt.strftime("%d %b %H:%M")
                else:
                    start_local = start_raw
            except Exception:
                start_local = start_raw

            attendee_list = [
                a.get("email", "") for a in ev.get("attendees", [])
            ]

            events.append({
                "id": ev.get("id", ""),
                "summary": ev.get("summary", "(untitled)"),
                "start_raw": start_raw,
                "end_raw": end_raw,
                "start_local": start_local,
                "location": ev.get("location", ""),
                "recurring": bool(ev.get("recurringEventId")),
                "attendees": attendee_list,
            })
        return events

    def _create_sync(
        self, creds, tz_name: str,
        title: str, date: str, time_str: str, duration: int,
        recurrence: str = "", attendees: str = "",
    ) -> str:
        """Create event, return event ID or empty string on failure."""
        h, m = int(time_str.split(":")[0]), int(time_str.split(":")[1])
        end_total = h * 60 + m + duration
        end_h, end_m = divmod(end_total, 60)

        body: dict[str, Any] = {
            "summary": title,
            "start": {
                "dateTime": f"{date}T{time_str}:00",
                "timeZone": tz_name,
            },
            "end": {
                "dateTime": f"{date}T{end_h:02d}:{end_m:02d}:00",
                "timeZone": tz_name,
            },
        }

        # Recurrence
        if recurrence:
            rule = _RECURRENCE_MAP.get(recurrence.lower())
            if rule:
                body["recurrence"] = [rule]

        # Attendees
        if attendees:
            emails = [e.strip() for e in attendees.split(",") if e.strip()]
            if emails:
                body["attendees"] = [{"email": e} for e in emails]

        try:
            svc = self._build_service(creds)
            result = svc.events().insert(
                calendarId="primary", body=body,
            ).execute()
            return result.get("id", "")
        except Exception:
            return ""

    def _delete_sync(self, creds, event_id: str) -> bool:
        try:
            svc = self._build_service(creds)
            svc.events().delete(
                calendarId="primary", eventId=event_id,
            ).execute()
            return True
        except Exception:
            return False

    def _update_sync(
        self, creds, tz_name: str,
        event_id: str, new_title: str,
        new_date: str, new_time: str,
        new_duration: int = 0,
    ) -> bool:
        try:
            svc = self._build_service(creds)
            event = svc.events().get(
                calendarId="primary", eventId=event_id,
            ).execute()

            if new_title:
                event["summary"] = new_title

            if new_date or new_time or new_duration:
                existing_start = event["start"].get("dateTime", "")
                existing_end = event["end"].get("dateTime", "")

                try:
                    existing_dt = datetime.fromisoformat(existing_start)
                    existing_end_dt = datetime.fromisoformat(existing_end)
                    old_duration = int(
                        (existing_end_dt - existing_dt).total_seconds() / 60
                    )

                    date_part = new_date or existing_dt.strftime("%Y-%m-%d")
                    time_part = new_time or existing_dt.strftime("%H:%M")
                    dur = new_duration if new_duration > 0 else old_duration

                    h, m = int(time_part.split(":")[0]), int(time_part.split(":")[1])
                    end_total = h * 60 + m + dur
                    end_h, end_m = divmod(end_total, 60)

                    event["start"] = {
                        "dateTime": f"{date_part}T{time_part}:00",
                        "timeZone": tz_name,
                    }
                    event["end"] = {
                        "dateTime": f"{date_part}T{end_h:02d}:{end_m:02d}:00",
                        "timeZone": tz_name,
                    }
                except Exception:
                    pass

            svc.events().update(
                calendarId="primary", eventId=event_id, body=event,
            ).execute()
            return True
        except Exception:
            return False

    def _find_conflicts_sync(
        self, creds, tz_name: str,
        date: str, time_str: str, duration: int,
    ) -> list[dict]:
        """Check if any existing events overlap with the proposed time slot."""
        import pytz

        tz = pytz.timezone(tz_name)
        h, m = int(time_str.split(":")[0]), int(time_str.split(":")[1])
        start_dt = tz.localize(
            datetime.strptime(f"{date} {h:02d}:{m:02d}", "%Y-%m-%d %H:%M")
        )
        end_dt = start_dt + timedelta(minutes=duration)

        svc = self._build_service(creds)
        result = svc.events().list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=10,
        ).execute()

        return self._parse_event_list(result.get("items", []), tz)

    # ── Briefing integration ──────────────────────────────────────────────

    async def today_briefing(self) -> Optional[str]:
        """Return a short summary of today's events for the morning briefing."""
        try:
            creds = token_store.get("calendar")
        except TokenNotFoundError:
            return None

        loc = await location_service.get()
        events = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_events_sync, creds, loc.timezone, 1
        )
        if not events:
            return None

        lines = []
        for ev in events:
            loc_str = f" @ {ev['location']}" if ev.get("location") else ""
            lines.append(f"  {ev['start_local']}  {ev['summary']}{loc_str}")

        return f"Today's calendar ({len(events)} events):\n" + "\n".join(lines)


registry.register(CalendarTool())
