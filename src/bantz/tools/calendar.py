"""
Bantz v2 — Google Calendar Tool
Today/week view, create, update, delete events.
Shares gmail_token.json.

Triggers: calendar, meeting, event, what's today, add/delete/update appointment
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.core.location import location_service
from bantz.tools import BaseTool, ToolResult, registry


class CalendarTool(BaseTool):
    name = "calendar"
    description = (
        "Shows, creates, updates and deletes Google Calendar events. "
        "Use for: calendar, meeting, event, what's today, this week, "
        "add appointment, delete event, update meeting, move appointment."
    )
    risk_level = "safe"

    async def execute(
        self,
        action: str = "today",
        # "today" | "week" | "create" | "delete" | "update"
        title: str = "",
        date: str = "",          # "2026-02-20"
        time: str = "",          # "15:00"
        duration: int = 60,
        event_id: str = "",      # for delete/update
        new_title: str = "",     # for update
        new_date: str = "",      # for update
        new_time: str = "",      # for update
        **kwargs: Any,
    ) -> ToolResult:
        try:
            creds = token_store.get("calendar")
        except TokenNotFoundError as e:
            return ToolResult(success=False, output="", error=str(e))

        loc = await location_service.get()
        tz_name = loc.timezone

        if action == "week":
            return await self._get_events(creds, tz_name, days=7, anchor=date)
        elif action == "create":
            return await self._create(creds, tz_name, title, date, time, duration)
        elif action == "delete":
            return await self._delete(creds, event_id, title)
        elif action == "update":
            return await self._update(creds, tz_name, event_id, title, new_title, new_date, new_time)
        else:
            return await self._get_events(creds, tz_name, days=1, anchor=date)

    # ── List events ───────────────────────────────────────────────────────

    async def _get_events(
        self, creds, tz_name: str, days: int = 1, anchor: str = "",
    ) -> ToolResult:
        events = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_events_sync, creds, tz_name, days, anchor
        )
        if not events:
            label = self._period_label(days, anchor)
            return ToolResult(success=True, output=f"No events on your calendar for {label}.")

        lines = []
        for ev in events:
            loc_str = f"  📍 {ev['location']}" if ev.get("location") else ""
            ev_id = f"  [id:{ev['id'][:8]}]"
            lines.append(f"  {ev['start_local']}  {ev['summary']}{loc_str}{ev_id}")

        label = self._period_label(days, anchor)
        return ToolResult(
            success=True,
            output=f"{label}:\n" + "\n".join(lines),
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
            return "Today"

    # ── Create ────────────────────────────────────────────────────────────

    async def _create(
        self, creds, tz_name: str,
        title: str, date: str, time: str, duration: int,
    ) -> ToolResult:
        if not title:
            return ToolResult(success=False, output="", error="Event title is required.")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        if not time:
            time = "09:00"

        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._create_sync, creds, tz_name, title, date, time, duration
        )
        if ok:
            return ToolResult(
                success=True,
                output=f"Event added ✓\n  📅 {title}  {date} {time} ({duration} min)",
            )
        return ToolResult(success=False, output="", error="Could not add event.")

    # ── Delete ────────────────────────────────────────────────────────────

    async def _delete(self, creds, event_id: str, title: str) -> ToolResult:
        # If no ID, try to find by title
        if not event_id and title:
            events = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_events_sync, creds, "Europe/Istanbul", 30
            )
            matches = [e for e in events if title.lower() in e["summary"].lower()]
            if not matches:
                return ToolResult(
                    success=False, output="",
                    error=f"No event found named '{title}'."
                )
            if len(matches) > 1:
                lines = [f"  [{e['id'][:8]}] {e['start_local']}  {e['summary']}" for e in matches]
                return ToolResult(
                    success=False, output="",
                    error=f"Multiple matches:\n" + "\n".join(lines) +
                          "\n\nPlease specify which one to delete."
                )
            event_id = matches[0]["id"]
            title = matches[0]["summary"]

        if not event_id:
            return ToolResult(success=False, output="", error="No event specified for deletion.")

        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._delete_sync, creds, event_id
        )
        if ok:
            return ToolResult(success=True, output=f"Event deleted ✓  [{title or event_id[:8]}]")
        return ToolResult(success=False, output="", error="Could not delete event.")

    # ── Update ────────────────────────────────────────────────────────────

    async def _update(
        self, creds, tz_name: str,
        event_id: str, title: str,
        new_title: str, new_date: str, new_time: str,
    ) -> ToolResult:
        # Find by title if no ID
        if not event_id and title:
            events = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_events_sync, creds, tz_name, 30
            )
            matches = [e for e in events if title.lower() in e["summary"].lower()]
            if not matches:
                return ToolResult(success=False, output="", error=f"'{title}' not found.")
            event_id = matches[0]["id"]
            title = matches[0]["summary"]

        if not event_id:
            return ToolResult(success=False, output="", error="No event specified for update.")

        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._update_sync, creds, tz_name, event_id, new_title, new_date, new_time
        )
        if ok:
            changes = []
            if new_title: changes.append(f"title → {new_title}")
            if new_date:  changes.append(f"date → {new_date}")
            if new_time:  changes.append(f"time → {new_time}")
            return ToolResult(
                success=True,
                output=f"Event updated ✓  [{title}]\n  " + ", ".join(changes),
            )
        return ToolResult(success=False, output="", error="Could not update event.")

    # ── Sync helpers ──────────────────────────────────────────────────────

    def _fetch_events_sync(self, creds, tz_name: str, days: int, anchor: str = "") -> list[dict]:
        from googleapiclient.discovery import build
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
        time_min = base.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        time_max = (base + timedelta(days=days - 1)).replace(hour=23, minute=59, second=59).isoformat()

        svc = build("calendar", "v3", credentials=creds)
        result = svc.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
        ).execute()

        events = []
        for ev in result.get("items", []):
            start_raw = ev["start"].get("dateTime") or ev["start"].get("date")
            try:
                if "T" in start_raw:
                    dt = datetime.fromisoformat(start_raw).astimezone(tz)
                    start_local = dt.strftime("%d %b %H:%M")
                else:
                    start_local = start_raw
            except Exception:
                start_local = start_raw

            events.append({
                "id": ev.get("id", ""),
                "summary": ev.get("summary", "(untitled)"),
                "start_local": start_local,
                "location": ev.get("location", ""),
            })
        return events

    def _create_sync(
        self, creds, tz_name: str,
        title: str, date: str, time_str: str, duration: int,
    ) -> bool:
        from googleapiclient.discovery import build
        h, m = int(time_str.split(":")[0]), int(time_str.split(":")[1])
        end_total = h * 60 + m + duration
        end_h, end_m = divmod(end_total, 60)
        svc = build("calendar", "v3", credentials=creds)
        svc.events().insert(
            calendarId="primary",
            body={
                "summary": title,
                "start": {"dateTime": f"{date}T{time_str}:00", "timeZone": tz_name},
                "end":   {"dateTime": f"{date}T{end_h:02d}:{end_m:02d}:00", "timeZone": tz_name},
            },
        ).execute()
        return True

    def _delete_sync(self, creds, event_id: str) -> bool:
        from googleapiclient.discovery import build
        svc = build("calendar", "v3", credentials=creds)
        svc.events().delete(calendarId="primary", eventId=event_id).execute()
        return True

    def _update_sync(
        self, creds, tz_name: str,
        event_id: str, new_title: str, new_date: str, new_time: str,
    ) -> bool:
        from googleapiclient.discovery import build
        svc = build("calendar", "v3", credentials=creds)

        # Fetch existing event first
        event = svc.events().get(calendarId="primary", eventId=event_id).execute()

        if new_title:
            event["summary"] = new_title
        if new_date or new_time:
            existing_start = event["start"].get("dateTime", "")
            try:
                existing_dt = datetime.fromisoformat(existing_start)
                date_part = new_date or existing_dt.strftime("%Y-%m-%d")
                time_part = new_time or existing_dt.strftime("%H:%M")
                existing_end = event["end"].get("dateTime", "")
                existing_end_dt = datetime.fromisoformat(existing_end)
                duration = int((existing_end_dt - existing_dt).total_seconds() / 60)
                h, m = int(time_part.split(":")[0]), int(time_part.split(":")[1])
                end_total = h * 60 + m + duration
                end_h, end_m = divmod(end_total, 60)
                event["start"] = {"dateTime": f"{date_part}T{time_part}:00", "timeZone": tz_name}
                event["end"]   = {"dateTime": f"{date_part}T{end_h:02d}:{end_m:02d}:00", "timeZone": tz_name}
            except Exception:
                pass

        svc.events().update(calendarId="primary", eventId=event_id, body=event).execute()
        return True


registry.register(CalendarTool())