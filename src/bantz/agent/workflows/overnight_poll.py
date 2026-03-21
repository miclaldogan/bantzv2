"""
Bantz v2 — Overnight Poll Workflow (#132)
Polls Gmail, Calendar, Classroom every 2 h overnight and stores
results in **KV Store** (not a new table — Rec #1) so the morning
briefing has data ready without extra API calls.

Key design decisions (from user's strategic recommendations):
  Rec #1  KV Store upsert instead of a new briefing_data table.
  Rec #2  Deduplication via `last_poll_time` → `after:TIMESTAMP` in Gmail queries.
  Rec #3  Configurable urgent keywords from `config.urgent_keywords` (not hardcoded).
  Rec #4  Auth-error payloads written to KV so the briefing can report them gracefully.

KV key layout:
  overnight:gmail        → JSON {status, unread, urgent, summaries, …}
  overnight:calendar     → JSON {status, events, added_since_last, …}
  overnight:classroom    → JSON {status, assignments, due_today, …}
  overnight:last_poll    → ISO timestamp of the last successful poll cycle
  overnight:poll_error   → JSON list of error payloads if any
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_POLL_TIMEOUT = 120          # seconds per source (generous for API calls)
_TOTAL_TIMEOUT = 300         # 5 min ceiling for a full poll cycle
_MAX_EMAILS_FETCH = 15       # caps the number of emails we fetch metadata for
_MAX_EVENTS_FETCH = 20       # caps calendar events
_MAX_ASSIGNMENTS_FETCH = 15  # caps classroom assignments


# ── Result datatypes ──────────────────────────────────────────────────────────

@dataclass
class PollSourceResult:
    """Result from polling one source (gmail / calendar / classroom)."""
    source: str
    status: str = "ok"       # "ok" | "auth_error" | "error" | "skipped"
    data: dict = field(default_factory=dict)
    error_message: str = ""
    poll_time: str = ""      # ISO timestamp

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "status": self.status,
            "poll_time": self.poll_time or datetime.now(timezone.utc).isoformat(),
        }
        if self.status == "ok":
            d["data"] = self.data
        elif self.status == "auth_error":
            d["error"] = self.error_message or "Token expired or revoked"
        elif self.status == "error":
            d["error"] = self.error_message
        return d


@dataclass
class OvernightPollResult:
    """Aggregate result of one overnight poll cycle."""
    gmail: Optional[PollSourceResult] = None
    calendar: Optional[PollSourceResult] = None
    classroom: Optional[PollSourceResult] = None
    poll_time: str = ""
    errors: int = 0

    def summary_line(self) -> str:
        parts = []
        if self.gmail and self.gmail.status == "ok":
            u = self.gmail.data.get("unread", 0)
            urg = self.gmail.data.get("urgent_count", 0)
            parts.append(f"📬 {u} unread" + (f" ({urg} urgent)" if urg else ""))
        if self.calendar and self.calendar.status == "ok":
            n = len(self.calendar.data.get("events", []))
            parts.append(f"📅 {n} events")
        if self.classroom and self.classroom.status == "ok":
            n = self.classroom.data.get("assignment_count", 0)
            parts.append(f"📚 {n} assignments")
        if self.errors:
            parts.append(f"⚠️ {self.errors} errors")
        return " | ".join(parts) if parts else "No data collected"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_kv():
    """Get the KV store, create if needed."""
    from bantz.data.sqlite_store import SQLiteKVStore
    from bantz.config import config
    return SQLiteKVStore(config.db_path)


def _get_last_poll_time(kv) -> Optional[str]:
    """Return ISO timestamp from last successful poll, or None."""
    raw = kv.get("overnight:last_poll", "")
    return raw if raw else None


def _get_urgent_keywords() -> list[str]:
    """
    Configurable urgent keywords (Rec #3).
    Reads from config.urgent_keywords — not hardcoded.
    """
    from bantz.config import config
    raw: str = getattr(config, "urgent_keywords", "")
    if not raw:
        return []
    return [k.strip().lower() for k in raw.split(",") if k.strip()]


def _is_urgent(subject: str, sender: str, keywords: list[str]) -> bool:
    """Check if an email is urgent based on configurable keyword matching."""
    text = f"{subject} {sender}".lower()
    return any(kw in text for kw in keywords)


def _gmail_after_timestamp(last_poll: Optional[str]) -> str:
    """
    Deduplication (Rec #2): Build a Gmail `after:EPOCH` filter
    based on the last successful poll time.
    Falls back to 8 hours ago if no previous poll.
    """
    if last_poll:
        try:
            dt = datetime.fromisoformat(last_poll)
            return str(int(dt.timestamp()))
        except Exception:
            pass
    # Default: 8 hours ago (covers overnight gap)
    return str(int((datetime.now(timezone.utc) - timedelta(hours=8)).timestamp()))


# ── Source pollers ────────────────────────────────────────────────────────────

async def _poll_gmail(last_poll: Optional[str]) -> PollSourceResult:
    """
    Poll Gmail for unread emails, classify as urgent/normal.
    Uses `after:TIMESTAMP` for deduplication (Rec #2).
    """
    result = PollSourceResult(source="gmail", poll_time=datetime.now(timezone.utc).isoformat())
    try:
        from bantz.auth.token_store import token_store, TokenNotFoundError

        try:
            creds = token_store.get("gmail")
        except TokenNotFoundError as exc:
            # Rec #4: auth error payload
            result.status = "auth_error"
            result.error_message = str(exc)
            return result

        from bantz.tools.gmail import GmailTool

        gmail = GmailTool()
        loop = asyncio.get_event_loop()

        # Count total unread
        count = await loop.run_in_executor(None, gmail._count_sync, creds)

        # Fetch recent unread messages with dedup timestamp (Rec #2)
        after_ts = _gmail_after_timestamp(last_poll)
        query = f"label:unread after:{after_ts}"
        messages = await loop.run_in_executor(
            None, gmail._fetch_messages_sync, creds, query, _MAX_EMAILS_FETCH,
        )

        # Classify urgent/normal (Rec #3: configurable keywords)
        keywords = _get_urgent_keywords()
        urgent_msgs = []
        normal_msgs = []

        for msg in messages:
            summary = {
                "id": msg.get("id", ""),
                "from": msg.get("from", ""),
                "subject": msg.get("subject", ""),
                "snippet": msg.get("snippet", "")[:120],
                "date": msg.get("date", ""),
            }
            if keywords and _is_urgent(msg.get("subject", ""), msg.get("from", ""), keywords):
                urgent_msgs.append(summary)
            else:
                normal_msgs.append(summary)

        result.status = "ok"
        result.data = {
            "unread": count,
            "new_since_last_poll": len(messages),
            "urgent_count": len(urgent_msgs),
            "urgent": urgent_msgs,
            "normal": normal_msgs[:10],  # cap normal to save space
        }

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        log.debug("Gmail poll error: %s", exc)

    return result


async def _poll_calendar() -> PollSourceResult:
    """Poll Google Calendar for today's events."""
    result = PollSourceResult(source="calendar", poll_time=datetime.now(timezone.utc).isoformat())
    try:
        from bantz.auth.token_store import token_store, TokenNotFoundError

        try:
            creds = token_store.get("calendar")
        except TokenNotFoundError as exc:
            result.status = "auth_error"
            result.error_message = str(exc)
            return result

        from bantz.tools.calendar import CalendarTool
        from bantz.core.location import location_service

        cal = CalendarTool()
        loc = await location_service.get()
        tz_name = loc.timezone

        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            None, cal._fetch_events_sync, creds, tz_name, 1,  # today
        )

        # Also check tomorrow for early morning awareness
        tomorrow_events = await loop.run_in_executor(
            None, cal._fetch_events_sync, creds, tz_name, 1,
            (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        event_summaries = []
        for ev in events[:_MAX_EVENTS_FETCH]:
            event_summaries.append({
                "id": ev.get("id", ""),
                "summary": ev.get("summary", "(untitled)"),
                "start": ev.get("start_local", ""),
                "location": ev.get("location", ""),
                "attendees": ev.get("attendees", [])[:5],
            })

        tomorrow_summaries = []
        for ev in tomorrow_events[:10]:
            tomorrow_summaries.append({
                "summary": ev.get("summary", "(untitled)"),
                "start": ev.get("start_local", ""),
            })

        result.status = "ok"
        result.data = {
            "event_count": len(events),
            "events": event_summaries,
            "tomorrow_count": len(tomorrow_events),
            "tomorrow": tomorrow_summaries,
            "timezone": tz_name,
        }

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        log.debug("Calendar poll error: %s", exc)

    return result


async def _poll_classroom() -> PollSourceResult:
    """Poll Google Classroom for active assignments."""
    result = PollSourceResult(source="classroom", poll_time=datetime.now(timezone.utc).isoformat())
    try:
        from bantz.auth.token_store import token_store, TokenNotFoundError

        try:
            creds = token_store.get("classroom")
        except TokenNotFoundError as exc:
            result.status = "auth_error"
            result.error_message = str(exc)
            return result

        from bantz.tools.classroom import ClassroomTool

        cr = ClassroomTool()
        loop = asyncio.get_event_loop()

        courses, assignments = await loop.run_in_executor(
            None, cr._fetch_assignments_sync, creds,
        )

        now = datetime.now(timezone.utc)
        due_today = []
        due_tomorrow = []
        overdue = []
        upcoming = []

        for a in assignments[:_MAX_ASSIGNMENTS_FETCH]:
            due_dt = a.get("due_dt")
            entry = {
                "title": a.get("title", ""),
                "course": a.get("course", ""),
            }
            if due_dt:
                entry["due_date"] = due_dt.strftime("%Y-%m-%d")
                delta = (due_dt.date() - now.date()).days
                if delta < 0:
                    overdue.append(entry)
                elif delta == 0:
                    due_today.append(entry)
                elif delta == 1:
                    due_tomorrow.append(entry)
                else:
                    upcoming.append(entry)
            else:
                upcoming.append(entry)

        result.status = "ok"
        result.data = {
            "assignment_count": len(assignments),
            "due_today": due_today,
            "due_tomorrow": due_tomorrow,
            "overdue": overdue,
            "upcoming": upcoming[:5],
            "course_count": len(courses),
        }

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        log.debug("Classroom poll error: %s", exc)

    return result


# ── KV Store persistence (Rec #1: upsert, no new table) ──────────────────────

def _store_poll_results(result: OvernightPollResult) -> None:
    """
    Rec #1: Write poll results to KV store (upsert).
    Each source gets its own key; the briefing reads the latest snapshot.
    """
    kv = _get_kv()

    if result.gmail:
        kv.set("overnight:gmail", json.dumps(result.gmail.to_dict(), ensure_ascii=False))

    if result.calendar:
        kv.set("overnight:calendar", json.dumps(result.calendar.to_dict(), ensure_ascii=False))

    if result.classroom:
        kv.set("overnight:classroom", json.dumps(result.classroom.to_dict(), ensure_ascii=False))

    # Update last poll timestamp for deduplication (Rec #2)
    kv.set("overnight:last_poll", result.poll_time)

    # Collect any error payloads for briefing (Rec #4)
    errors = []
    for src in (result.gmail, result.calendar, result.classroom):
        if src and src.status in ("auth_error", "error"):
            errors.append(src.to_dict())
    if errors:
        kv.set("overnight:poll_errors", json.dumps(errors, ensure_ascii=False))
    else:
        # Clear old errors on success
        kv.delete("overnight:poll_errors")


def _send_urgent_notification(result: OvernightPollResult) -> None:
    """Send desktop notification if urgent emails were found."""
    if not result.gmail or result.gmail.status != "ok":
        return
    urgent = result.gmail.data.get("urgent", [])
    if not urgent:
        return

    try:
        from bantz.agent.notifier import notifier
        subjects = [u.get("subject", "?")[:50] for u in urgent[:3]]
        body = "\n".join(f"• {s}" for s in subjects)
        if len(urgent) > 3:
            body += f"\n  … and {len(urgent) - 3} more"
        notifier.send(
            f"🚨 {len(urgent)} Urgent Email{'s' if len(urgent) > 1 else ''}",
            body,
            urgency="critical",
            expire_ms=0,  # persistent notification
        )
    except Exception as exc:
        log.debug("Urgent notification failed: %s", exc)


# ── Public API: reading overnight data ────────────────────────────────────────

def read_overnight_data(source: str = "") -> dict:
    """
    Read cached overnight poll data from KV store.
    Used by briefing.py to get pre-fetched data.

    Args:
        source: 'gmail', 'calendar', 'classroom', or '' for all.

    Returns:
        dict with the source data, or all sources if source=''.
    """
    kv = _get_kv()

    if source:
        raw = kv.get(f"overnight:{source}", "")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    # Return all sources
    result = {}
    for s in ("gmail", "calendar", "classroom"):
        raw = kv.get(f"overnight:{s}", "")
        if raw:
            try:
                result[s] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    # Include errors if any
    err_raw = kv.get("overnight:poll_errors", "")
    if err_raw:
        try:
            result["errors"] = json.loads(err_raw)
        except json.JSONDecodeError:
            pass
    result["last_poll"] = kv.get("overnight:last_poll", "")
    return result


def clear_overnight_data() -> None:
    """Clear overnight data after briefing consumes it."""
    kv = _get_kv()
    for key in ("overnight:gmail", "overnight:calendar", "overnight:classroom",
                "overnight:last_poll", "overnight:poll_errors"):
        kv.delete(key)


# ── Main entry point ─────────────────────────────────────────────────────────

async def run_overnight_poll(
    *,
    dry_run: bool = False,
    sources: tuple[str, ...] = ("gmail", "calendar", "classroom"),
) -> OvernightPollResult:
    """
    Run one cycle of overnight polling.

    Called by job_scheduler every 2 h between midnight and 7 AM.
    Also available via CLI: `bantz --overnight-poll`.

    Args:
        dry_run: If True, poll but don't write to KV store.
        sources: Which sources to poll (default: all three).
    """
    from bantz.agent.job_scheduler import inhibit_sleep

    t0 = time.monotonic()
    poll_time = datetime.now(timezone.utc).isoformat()
    log.info("📬 Overnight poll starting (sources=%s, dry_run=%s)", sources, dry_run)

    result = OvernightPollResult(poll_time=poll_time)

    # Get last poll time for deduplication (Rec #2)
    last_poll = None
    if not dry_run:
        try:
            kv = _get_kv()
            last_poll = _get_last_poll_time(kv)
        except Exception:
            pass

    with inhibit_sleep("overnight poll"):
        # Fire source polls in parallel (each with its own timeout)
        coros = {}
        if "gmail" in sources:
            coros["gmail"] = asyncio.wait_for(
                _poll_gmail(last_poll), timeout=_POLL_TIMEOUT,
            )
        if "calendar" in sources:
            coros["calendar"] = asyncio.wait_for(
                _poll_calendar(), timeout=_POLL_TIMEOUT,
            )
        if "classroom" in sources:
            coros["classroom"] = asyncio.wait_for(
                _poll_classroom(), timeout=_POLL_TIMEOUT,
            )

        # Gather with exception isolation
        if coros:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*coros.values(), return_exceptions=True),
                    timeout=_TOTAL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.warning("Overnight poll total timeout (%ds)", _TOTAL_TIMEOUT)
                results = []

            keys = list(coros.keys())
            for i, key in enumerate(keys):
                if i < len(results):
                    val = results[i]
                    if isinstance(val, PollSourceResult):
                        setattr(result, key, val)
                    elif isinstance(val, asyncio.TimeoutError):
                        setattr(result, key, PollSourceResult(
                            source=key, status="error",
                            error_message=f"Timeout after {_POLL_TIMEOUT}s",
                            poll_time=poll_time,
                        ))
                    elif isinstance(val, Exception):
                        setattr(result, key, PollSourceResult(
                            source=key, status="error",
                            error_message=str(val),
                            poll_time=poll_time,
                        ))

    # Count errors
    for src in (result.gmail, result.calendar, result.classroom):
        if src and src.status != "ok":
            result.errors += 1

    elapsed = time.monotonic() - t0

    if not dry_run:
        # Rec #1: Store in KV (upsert, not a new table)
        try:
            _store_poll_results(result)
        except Exception as exc:
            log.warning("Failed to store poll results: %s", exc)

        # Send urgent notification for critical emails
        _send_urgent_notification(result)

    log.info(
        "📬 Overnight poll complete in %.1fs: %s",
        elapsed, result.summary_line(),
    )
    return result
