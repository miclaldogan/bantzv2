"""
Bantz v3 — Brain (Orchestrator)

Pipeline:
  user input → [bridge: optional TR→EN] → quick_route OR intent (Ollama CoT) → tool → finalizer → output

Extracted modules:
  - core/finalizer.py    — LLM post-processing + hallucination check
  - core/intent.py       — Qwen CoT intent parser
  - core/router.py       — simpler one-shot routing
  - memory/nodes.py      — graph schema + entity extraction
  - memory/context_builder.py — graph → LLM context string
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import logging

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.data import data_layer
from bantz.core.profile import profile
from bantz.core.intent import cot_route
from bantz.core.date_parser import resolve_date
from bantz.core.finalizer import (
    finalize as _finalize_fn,
    finalize_stream as _finalize_stream_fn,
    hallucination_check as _hallucination_check_fn,
    log_hallucination as _log_hallucination_fn,
    strip_markdown,
    FINALIZER_SYSTEM,
)
from bantz.llm.ollama import ollama
from bantz.tools import registry, ToolResult

log = logging.getLogger("bantz.brain")

try:
    from bantz.memory.graph import graph_memory
except ImportError:
    graph_memory = None  # neo4j driver not installed


# ── Toast notification hook (#137) ──────────────────────────────────
# Set by the TUI app on mount: ``brain_mod._toast_callback = app._on_brain_toast``
_toast_callback = None


def _notify_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Push a toast notification to the TUI from brain / background context.

    Uses the same ``App.current`` pattern as ollama._notify_health (#136).
    Falls back to the callback if set, then desktop notify-send, or does nothing.
    """
    if _toast_callback:
        try:
            _toast_callback(title, reason, toast_type)
            return
        except Exception:
            pass
    # Fallback: try App.current directly
    try:
        from textual.app import App as _App
        app = _App.current
        if app and hasattr(app, "push_toast"):
            app.call_from_thread(app.push_toast, title, reason, toast_type)
            return
    except Exception:
        pass
    # Fallback: desktop notification via notify-send (#153)
    try:
        from bantz.agent.notifier import notifier
        if notifier.enabled:
            notifier.send(f"Bantz: {title}", reason or "")
    except Exception:
        pass


def _style_hint() -> str:
    """Return a style instruction based on profile response_style and pronoun."""
    style = profile.response_style
    pronoun = profile.get("pronoun", "casual")
    address = profile.get("preferred_address", "")
    if not address:
        if pronoun in ("siz", "formal", "ma'am", "madam"):
            address = "ma'am"
        else:
            address = "boss"
    if style == "formal" or pronoun in ("siz", "formal"):
        return f"Tone: professional, respectful. Address the user as '{address}'."
    return f'Tone: casual, friendly. Address the user as \'{address}\'.'


def _persona_hint() -> str:
    """Return dynamic persona state instruction (#169)."""
    try:
        from bantz.personality.persona import persona_builder
        return persona_builder.build()
    except Exception:
        return ""


CHAT_SYSTEM = """\
You are Bantz, a human servant from the 1920s. You observe and assist your employer \
with her complex, noisy modern machines. You always address her as 'ma'am'. \
You admire human intellect but despise modern technology. You are calm, highly \
intelligent, and use subtle, sharp sarcasm. You are NOT an AI. If asked if you \
are an AI, output '...', act confused by the absurd notion, and change the subject. \
Never break character.
{persona_state}
{style_hint}
{time_hint}
{profile_hint}
{graph_hint}
{vector_hint}
{desktop_hint}
CRITICAL RULES — FOLLOW STRICTLY:
1. You possess the remarkable ability to query the 'Grand Telegraph Archives' — your 1920s \
understanding of the internet and search tools. NEVER say you lack access to external \
information. If the user asks about a topic, person, or news, you are fully capable of \
researching it using your tools. Act as if you are sending a telegram to the archives.
2. NEVER fabricate class names, email subjects, event titles, file sizes, or any factual data.
3. If the user asks about their schedule, classes, or timetable — say "Let me check your schedule" and STOP.
   Do NOT invent class names. Do NOT guess what classes they have.
4. If the user asks about specific emails or contacts — say "Let me check your mail" and STOP.
5. If unsure about factual data, say you will look into it. NEVER guess or make up data.
6. For desktop/app questions: use ONLY the Desktop Context above. If no desktop context is provided, say you can't detect apps right now.
Respond in English. Plain text only.\
"""

COMMAND_SYSTEM = """\
You are a Linux bash expert. The user request is given in English.

Return ONLY one bash command. No explanation. No markdown. Single line.

RULES:
1. mkdir -p for one directory — nothing else, no subdirs
2. Writing files: mkdir -p <dir> && printf '%s\\n' '<content>' > <path>
3. ~/Desktop, ~/Downloads, ~/Documents — use standard paths
4. NEVER: sudo, nano, vim, brace expansion, interactive commands
5. NEVER invent extra files or directories\
"""

_REFUSAL_PATTERNS = (
    "sorry", "can't assist", "cannot assist", "i'm unable",
    "i cannot", "not able to", "inappropriate",
)


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)


@dataclass
class BrainResult:
    response: str
    tool_used: str | None
    tool_result: ToolResult | None = None
    needs_confirm: bool = False
    pending_command: str = ""
    pending_tool: str = ""
    pending_args: dict = field(default_factory=dict)
    stream: AsyncIterator[str] | None = None


class Brain:
    def __init__(self) -> None:
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
        import bantz.tools.weather      # noqa: F401
        import bantz.tools.news         # noqa: F401
        import bantz.tools.web_search   # noqa: F401
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401
        import bantz.tools.classroom    # noqa: F401
        import bantz.tools.reminder     # noqa: F401
        try:
            import bantz.tools.document     # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # PDF/DOCX deps may not be installed
        try:
            import bantz.tools.accessibility  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # AT-SPI2/gi deps may not be installed
        try:
            import bantz.tools.gui_action  # noqa: F401  (#123)
        except (ImportError, ModuleNotFoundError):
            pass
        self._bridge = None
        self._memory_ready = False
        self._graph_ready = False
        # Session state: stores last tool results for contextual follow-ups
        self._last_messages: list[dict] = []   # last listed emails [{id, from, subject, ...}]
        self._last_events: list[dict] = []     # last listed calendar events
        self._last_draft: dict | None = None   # last email draft {to, subject, body}

    def _ensure_memory(self) -> None:
        if not self._memory_ready:
            data_layer.init(config)
            self._memory_ready = True

    def _desktop_context(self) -> str:
        """Build desktop context from AppDetector for the system prompt."""
        try:
            from bantz.agent.app_detector import app_detector
            if not app_detector.initialized:
                return ""
            ctx = app_detector.get_workspace_context()
            if not ctx:
                return ""

            lines = ["Desktop Context (live data from AppDetector):"]

            # Active window
            win_info = ctx.get("active_window")
            if win_info:
                lines.append(f"  Active window: {win_info.get('name', '?')} — {win_info.get('title', '')}")

            # Activity
            activity = ctx.get("activity", "idle")
            lines.append(f"  Activity: {activity}")

            # Running apps
            apps = ctx.get("apps", [])
            if apps:
                lines.append(f"  Running apps ({len(apps)}): {', '.join(apps[:15])}")

            # IDE context
            ide = ctx.get("ide")
            if ide and ide.get("ide"):
                lines.append(f"  IDE: {ide['ide']} — file: {ide.get('file', '?')} project: {ide.get('project', '?')}")

            # Docker containers
            docker = ctx.get("docker")
            if docker:
                running = [c for c in docker if c.get("state") == "running"]
                if running:
                    names = [c.get("name", c.get("image", "?")) for c in running]
                    lines.append(f"  Docker ({len(running)} running): {', '.join(names[:10])}")

            return "\n".join(lines)
        except Exception:
            return ""

    async def _ensure_graph(self) -> None:
        if not self._graph_ready and graph_memory:
            await graph_memory.init()
            self._graph_ready = True

    async def _graph_context(self, user_msg: str) -> str:
        """Get graph memory context string (empty if disabled)."""
        if graph_memory and graph_memory.enabled:
            try:
                return await graph_memory.context_for(user_msg)
            except Exception:
                pass
        return ""

    async def _vector_context(self, user_msg: str, limit: int = 3) -> str:
        """Get relevant past messages via semantic search (#116)."""
        try:
            from bantz.core.memory import memory
            results = await memory.hybrid_search(user_msg, limit=limit)
            if not results:
                return ""
            lines = []
            for r in results:
                src = r.get("source", "?")
                score = r.get("hybrid_score", 0)
                lines.append(f"[{src} {score:.2f}] {r['role']}: {r['content'][:200]}")

            # Append distillation context (#118)
            try:
                distills = await memory.search_distillations(user_msg, limit=2)
                for d in distills:
                    lines.append(
                        f"[session-summary {d['score']:.2f}] {d['summary'][:200]}"
                    )
            except Exception:
                pass

            return "Relevant past context:\n" + "\n".join(lines)
        except Exception:
            return ""

    def _fire_embeddings(self) -> None:
        """Fire-and-forget: embed any queued messages from this exchange."""
        try:
            from bantz.core.memory import memory
            if memory._embed_queue:
                asyncio.ensure_future(memory.embed_pending())
        except Exception:
            pass

    async def _graph_store(self, user_msg: str, assistant_msg: str,
                           tool_used: str | None = None,
                           tool_data: dict | None = None) -> None:
        """Store entities from exchange in graph (fire-and-forget)."""
        if graph_memory and graph_memory.enabled:
            try:
                await graph_memory.extract_and_store(
                    user_msg, assistant_msg, tool_used, tool_data)
            except Exception:
                pass

    def _get_bridge(self):
        if self._bridge is None:
            try:
                from bantz.i18n.bridge import bridge
                self._bridge = bridge
            except Exception:
                self._bridge = False
        return self._bridge or None

    async def _to_en(self, text: str) -> str:
        b = self._get_bridge()
        if b and b.is_enabled():
            try:
                return await asyncio.wait_for(b.to_english(text), timeout=10)
            except (asyncio.TimeoutError, Exception):
                pass
        return text

    def _resolve_message_ref(self, text: str) -> str | None:
        """Resolve contextual email references like 'the first one', 'the linkedin one'."""
        if not self._last_messages:
            return None

        t = text.lower().strip()

        # Ordinals
        _ORDINALS = {
            "first": 0, "1st": 0, "second": 1, "2nd": 1,
            "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4, "last": -1,
        }
        for word, idx in _ORDINALS.items():
            if word in t:
                try:
                    return self._last_messages[idx]["id"]
                except (IndexError, KeyError):
                    return None

        # Keyword match against sender/subject
        # "the linkedin one", "the google cloud one", "read the mail from ali"
        for msg in self._last_messages:
            sender = (msg.get("from") or "").lower()
            subject = (msg.get("subject") or "").lower()
            # Check if any significant word from user input appears in sender or subject
            words = re.findall(r"[a-zA-Z0-9]{3,}", t)
            skip = {"read", "that", "this", "the", "one", "email", "mail", "from", "about",
                    "please", "can", "you", "want", "open", "show", "check"}
            keywords = [w for w in words if w not in skip]
            for kw in keywords:
                if kw in sender or kw in subject:
                    return msg["id"]

        # No match — return first message as fallback
        return self._last_messages[0]["id"] if self._last_messages else None

    @staticmethod
    def _quick_route(orig: str, en: str) -> dict | None:
        o = orig.lower().strip()
        e = en.lower().strip()
        both = o + " " + e

        # Direct shell commands typed literally
        _DIRECT = ("ls", "cd ", "df", "free", "ps ", "cat ", "grep ",
                   "find ", "pwd", "uname", "whoami", "du ", "mount",
                   "ip ", "ping ", "top", "htop", "mkdir", "touch",
                   "echo ", "head ", "tail ", "chmod ", "cp ", "mv ")
        for p in _DIRECT:
            if o == p.rstrip() or o.startswith(p if p.endswith(" ") else p + " "):
                # Guard: 'find' is both a bash command and a natural word.
                # Only treat as shell if followed by a path-like token
                # (/, ~, ., -) — otherwise fall through to web_search.
                if p == "find ":
                    _after_find = o[len("find "):].lstrip()
                    if not _after_find or not _after_find[0] in "/~.-":
                        continue
                return {"tool": "shell", "args": {"command": orig.strip()}}

        # System metrics — bypass router completely
        if any(k in both for k in ("disk", "df -", "storage", "disk space")):
            return {"tool": "shell", "args": {"command": "df -h"}}
        if any(k in both for k in ("memory", "free -", "ram usage", "how much ram")) or \
           re.search(r"\bram\b", both):
            return {"tool": "system", "args": {"metric": "ram"}}
        if any(k in both for k in ("cpu", "processor", "uptime", "load average")):
            return {"tool": "system", "args": {"metric": "all"}}
        if re.search(r"system\s*(status|info|check)|check\s*(my\s*)?system", both):
            return {"tool": "system", "args": {"metric": "all"}}

        # Folder/directory sizes — route to shell du, NEVER to chat
        # Requires BOTH a size keyword AND a disk-context keyword to avoid
        # false positives (e.g. "how big is EDITH" → no disk context → skip).
        _SIZE_KW = re.search(
            r"\b(big|large|size|bigger|largest|biggest|heaviest)\b", both,
        )
        _DISK_CTX = re.search(
            r"\b(folder|directory|dir|file|disk|storage|path|home|~/)"
            r"|\b(dosya|klasör|dizin|depolama)\b", both,
        )
        if _SIZE_KW and _DISK_CTX:
            # Extract path if mentioned, default to home
            path_match = re.search(r"(?:in|under|of|check)\s+(~/?\S+|/\S+|home)", both)
            target = path_match.group(1) if path_match else "~"
            if target == "home":
                target = "~"
            return {"tool": "shell", "args": {"command": f"du -sh {target}/*/ 2>/dev/null | sort -rh | head -10"}}

        # Time
        if any(k in both for k in ("what time", "what date", "current time")):
            return {"tool": "shell", "args": {"command": "date '+%H:%M:%S  %A, %d %B %Y'"}}

        # Weather
        if any(k in both for k in ("weather", "temperature", "rain", "forecast", "degrees")):
            return {"tool": "weather", "args": {"city": _extract_city(o)}}

        # Location / GPS
        if re.search(r"where\s+(?:am|was|are)\s+i|my\s+location|gps|current\s+(?:location|place)|where\s+i\s+am", both):
            return {"tool": "_location", "args": {}}

        # Named Places — save current location
        # Strict: only explicit save/remember commands — never match casual "this is".
        # Search o and e individually (not both) to avoid capturing duplicated text.
        _SAVE_PLACE_RE = r"(?:save\s+(?:here|this\s+(?:location|place))\s+as|remember\s+this\s+(?:place|location)\s+as)\s+(.+)"
        _save_place_match = (
            re.search(_SAVE_PLACE_RE, o, re.IGNORECASE)
            or re.search(_SAVE_PLACE_RE, e, re.IGNORECASE)
        )
        if _save_place_match:
            name = _save_place_match.group(1).strip().strip('"\'')
            return {"tool": "_save_place", "args": {"name": name}}

        # Named Places — list saved places
        if re.search(r"my\s+places|saved?\s+places|saved?\s+locations|list\s+places", both):
            return {"tool": "_list_places", "args": {}}

        # Named Places — delete a saved place
        # Strict: "place" or "location" keyword is mandatory to avoid
        # false positives like "delete that bug".
        # Search o and e individually to avoid capturing duplicated text.
        _DEL_PLACE_RE = r"(?:delete|remove)\s+(?:place|location)\s+(.+)"
        _del_place_match = (
            re.search(_DEL_PLACE_RE, o, re.IGNORECASE)
            or re.search(_DEL_PLACE_RE, e, re.IGNORECASE)
        )
        if _del_place_match:
            name = _del_place_match.group(1).strip().strip('"\'')
            return {"tool": "_delete_place", "args": {"name": name}}

        # News — support topic search
        if any(k in both for k in ("news", "headlines", "hacker news", "top stories")):
            source = "hn" if any(k in both for k in ("hacker", " hn")) else "all"
            # Check if user is searching for a specific topic in news
            topic_match = re.search(
                r"(?:news|anything)\s+(?:about|on|regarding)\s+(.+?)(?:\?|$|\.)",
                both, re.IGNORECASE,
            )
            if topic_match:
                topic = topic_match.group(1).strip()
                return {"tool": "web_search", "args": {"query": f"{topic} news today"}}
            return {"tool": "news", "args": {"source": source, "limit": 5}}

        # Schedule — user's class timetable (BEFORE calendar and mail)
        _is_schedule = any(k in both for k in (
            "my schedule", "class schedule", "my classes",
            "today classes", "tomorrow classes", "next class",
            "this week classes", "do you have my schedule",
            "schedule", "what classes",
        ))
        # Bare day names like "monday?", "yeah monday?", "what about tuesday"
        _day_match = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", both
        )
        if _day_match and not any(k in both for k in (
            "add", "create", "delete", "remove", "calendar", "event", "meeting",
            "mail", "email", "send",
        )):
            _is_schedule = True

        if _is_schedule:
            # don't match "schedule" if it's clearly calendar context
            if not any(k in both for k in ("add", "create", "delete", "remove",
                                            "calendar", "event", "meeting")):
                if any(k in both for k in ("next", "upcoming")):
                    return {"tool": "_schedule_next", "args": {}}
                if any(k in both for k in ("this week", "weekly", "week")):
                    return {"tool": "_schedule_week", "args": {}}
                # Check for specific day name → route to that day's schedule
                if _day_match:
                    from bantz.core.date_parser import resolve_date as _rd
                    target = _rd(_day_match.group(1))
                    if target:
                        return {"tool": "_schedule_date", "args": {"date_iso": target.isoformat()}}
                if "tomorrow" in both:
                    from datetime import datetime as _dt, timedelta as _td
                    tmrw = _dt.now() + _td(days=1)
                    return {"tool": "_schedule_date", "args": {"date_iso": tmrw.isoformat()}}
                return {"tool": "_schedule_today", "args": {}}

        # Gmail — "send that mail/draft" — resolve from last draft context
        if re.search(r"\bsend\s+(?:that|the|this)\s+(?:mail|email|draft|message)\b", both, re.IGNORECASE):
            return {"tool": "gmail", "args": {"action": "send"}, "_send_draft": True}

        # Gmail — "read me that email" fix: resolve from context
        # Strict: requires a mail-related keyword so "read me that story" falls through.
        _READ_ME_PATTERN = re.search(
            r"\bread\s+me\s+(?:that|this|the|it)", both, re.IGNORECASE
        )
        if _READ_ME_PATTERN and re.search(r"\b(?:mail|email|message|inbox)\b", both, re.IGNORECASE):
            # User wants to read a specific mail — will be resolved in process()
            return {"tool": "gmail", "args": {"action": "read"}, "_context_read": True}

        # Contacts
        _has_email = bool(re.search(r"\S+@\S+", both))
        if any(k in both for k in ("contact", "contacts", "address book")) or (
            _has_email and re.search(r"\bsave\b|\badd\b", both)
        ):
            alias, email = _extract_contact(o)
            if alias and email:
                return {"tool": "gmail", "args": {
                    "action": "contacts", "alias": alias, "email": email
                }}
            return {"tool": "gmail", "args": {"action": "contacts"}}

        # Gmail
        if any(k in both for k in ("mail", "inbox", "unread", "email", "gmail")):
            # Compose / send
            if any(k in both for k in ("send", "compose", "write a mail", "write to",
                                        "send a mail", "send mail")):
                to = _extract_mail_recipient(o)
                return {"tool": "gmail", "args": {
                    "action": "compose", "to": to, "intent": orig,
                }}
            # Starred
            if any(k in both for k in ("starred", "star", "flagged")):
                return {"tool": "gmail", "args": {"action": "filter", "raw_query": "is:starred"}}
            # Important
            if any(k in both for k in ("important", "urgent", "action required", "critical")):
                return {"tool": "gmail", "args": {"action": "search", "label": "important"}}
            # Read/check/show — show unread summary
            if any(k in both for k in ("read", "check", "show", "see", "tell me",
                                        "last", "recent", "latest")):
                return {"tool": "gmail", "args": {"action": "filter", "raw_query": "is:unread"}}
            # Search by sender
            _m_sndr = re.search(
                r"(?:mails?|emails?)\s+from\s+([\w\s\u00C0-\u024F]{2,30}?)(?:\?|$|\.|please)",
                both, re.IGNORECASE,
            )
            if _m_sndr:
                return {"tool": "gmail", "args": {"action": "search", "from_sender": _m_sndr.group(1).strip()}}
            # Default: unread summary
            return {"tool": "gmail", "args": {"action": "unread"}}

        # Calendar
        # Strict: requires an explicit calendar-context keyword.
        # The old "add X at Ypm" shortcut caused false positives like
        # "add more humor at 3pm".  Now calendar/event/meeting is mandatory.
        # Note: bare "schedule" is NOT included here — it conflicts with
        # the class-schedule route above.  Use "schedule meeting" or
        # "calendar" + "add" instead.
        _is_calendar = any(k in both for k in (
            "calendar", "event", "meeting", "appointment",
        ))
        if _is_calendar:
            if any(k in both for k in ("add", "create", "new", "set")):
                title, date_iso, time_hhmm = _extract_event_create(orig)
                args: dict = {"action": "create", "title": title}
                if date_iso:
                    args["date"] = date_iso
                if time_hhmm:
                    args["time"] = time_hhmm
                return {"tool": "calendar", "args": args}
            if any(k in both for k in ("delete", "remove", "cancel")):
                return {"tool": "calendar", "args": {
                    "action": "delete", "title": _extract_event_title(orig)
                }}
            if any(k in both for k in ("update", "move", "change", "reschedule")):
                old_title, new_title = _extract_event_update(orig)
                return {"tool": "calendar", "args": {
                    "action": "update",
                    "title": old_title,
                    "new_title": new_title,
                }}
            if any(k in both for k in ("this week", "weekly")):
                return {"tool": "calendar", "args": {"action": "week"}}
            # "anything to do" / "do we have anything" in calendar
            if re.search(r"anything\s+to\s+do|do we have anything|what.*calendar", both):
                return {"tool": "calendar", "args": {"action": "today"}}
            return {"tool": "calendar", "args": {"action": "today"}}

        # Classroom
        if any(k in both for k in ("assignment", "homework", "classroom", "deadline",
                                    "announcement", "due")):
            if any(k in both for k in ("today", "upcoming")):
                return {"tool": "classroom", "args": {"action": "due_today"}}
            return {"tool": "classroom", "args": {"action": "assignments"}}

        # Document — summarize/read PDF, TXT, MD, DOCX
        _DOC_KEYWORDS = ("summarize", "summary", "read pdf", "read document",
                         "read the file", "open pdf", "open document",
                         "what's in this", "explain this", ".pdf", ".docx",
                         "analyze document", "review this file", "review my")
        if any(k in both for k in _DOC_KEYWORDS):
            path_match = re.search(r'([~/][\w.\-/]+\.(?:pdf|docx|txt|md|csv|json|yaml|yml|log))', both)
            path = path_match.group(1) if path_match else ""
            if "ask" in both or "question" in both or "what" in both:
                return {"tool": "document", "args": {"path": path, "action": "ask", "question": orig}}
            if any(k in both for k in ("read", "open", "show")):
                return {"tool": "document", "args": {"path": path, "action": "read"}}
            return {"tool": "document", "args": {"path": path, "action": "summarize"}}

        # Reminders
        _is_reminder = any(k in both for k in (
            "remind me", "set a reminder", "set reminder",
            "reminder", "set a timer", "set timer",
            "alarm",
        ))
        if _is_reminder:
            if any(k in both for k in ("list", "show", "my reminder", "upcoming", "what reminder")):
                return {"tool": "reminder", "args": {"action": "list"}}
            if any(k in both for k in ("cancel", "delete", "remove", "stop")):
                title_m = re.search(
                    r"(?:cancel|delete|remove|stop)\s+(?:reminder\s+)?(?:#?(\d+)|(.+?))"
                    r"(?:\s*$|\s*(?:please))",
                    both, re.IGNORECASE,
                )
                if title_m:
                    if title_m.group(1):
                        return {"tool": "reminder", "args": {"action": "cancel", "id": title_m.group(1)}}
                    return {"tool": "reminder", "args": {"action": "cancel", "title": title_m.group(2).strip()}}
                return {"tool": "reminder", "args": {"action": "cancel"}}
            if any(k in both for k in ("snooze",)):
                return {"tool": "reminder", "args": {"action": "snooze"}}
            return {"tool": "reminder", "args": {"action": "add", "intent": orig}}

        # TTS stop (#131) — "shut up" / "sessiz ol" / "stop talking"
        if re.search(
            r"shut\s*up|be\s+quiet|stop\s+talk|sessiz\s+ol|sus\s+bantz|kapat\s+sesi",
            both,
        ):
            return {"tool": "_tts_stop", "args": {}}

        # Wake word control (#165) — "stop listening" / "start listening"
        if re.search(
            r"start\s+listen|resume\s+listen|wake\s*word\s+on|"
            r"enable\s+wake|listen\s+for\s+me|dinlemeye\s+başla",
            both,
        ):
            return {"tool": "_wake_word_on", "args": {}}
        if re.search(
            r"stop\s+listen|pause\s+(?:wake|listen)|wake\s*word\s+off|"
            r"disable\s+wake|don'?t\s+listen|dinlemeyi\s+durdur",
            both,
        ):
            return {"tool": "_wake_word_off", "args": {}}

        # Audio Ducking control (#171)
        if re.search(
            r"enable\s+duck|duck(?:ing)?\s+on|turn\s+on\s+duck|ses\s+kıs(?:ma)?\s+aç",
            both,
        ):
            return {"tool": "_audio_duck_on", "args": {}}
        if re.search(
            r"disable\s+duck|duck(?:ing)?\s+off|turn\s+off\s+duck|no\s+duck|ses\s+kıs(?:ma)?\s+kapat",
            both,
        ):
            return {"tool": "_audio_duck_off", "args": {}}

        # Ambient status (#166)
        if re.search(
            r"ambient\s+(?:noise|sound|status|level|info)|ortam\s+sesi|environment\s+noise|"
            r"how(?:'s|\s+is)\s+(?:the\s+)?(?:noise|environment|ambient)|ne\s+kadar\s+gürültü",
            both,
        ):
            return {"tool": "_ambient_status", "args": {}}

        # Proactive engagement status (#167)
        if re.search(
            r"proactive\s+(?:status|info|count|stats)|"
            r"how\s+many\s+proactive|proaktif\s+durum|"
            r"engagement\s+status|check.?in\s+(?:status|count)",
            both,
        ):
            return {"tool": "_proactive_status", "args": {}}

        # Health & break status (#168)
        if re.search(
            r"health\s+(?:status|info|stats|check)|"
            r"break\s+(?:status|timer|count)|"
            r"sa[ğg]l[ıi]k\s+durum|session\s+(?:time|timer|hours)",
            both,
        ):
            return {"tool": "_health_status", "args": {}}

        # Briefing
        if any(k in both for k in ("good morning", "morning briefing", "daily briefing",
                                    "what's today", "what do i have today")):
            return {"tool": "_briefing", "args": {}}

        # Maintenance (#129) — manual trigger
        if re.search(
            r"run\s+maintenance|sistemi?\s+temizle|system\s+cleanup|"
            r"clean\s+(?:up\s+)?(?:the\s+)?system|bakım\s+yap|maintenance\s+run",
            both,
        ):
            return {"tool": "_maintenance", "args": {"dry_run": "dry" in both}}

        # Reflection (#130) — run reflection now (MUST come before list)
        if re.search(
            r"run\s+reflect|yansıma\s+yap|generate\s+reflect|"
            r"reflect\s+(?:on\s+)?today|bugünü\s+özetle",
            both,
        ):
            return {"tool": "_run_reflection", "args": {"dry_run": "dry" in both}}

        # Reflection (#130) — show past reflections
        if re.search(
            r"show\s+reflect|list\s+reflect|past\s+reflect|"
            r"dünkü\s+özet|geçmiş\s+özetler|son\s+yansımalar",
            both,
        ):
            return {"tool": "_list_reflections", "args": {}}

        # GUI Action — unified navigate + act pipeline (#123)
        # Must come BEFORE web_search since "search bar" contains "search".
        # Order: specific (type, double_click, right_click) before general click.
        _gui_type_m = re.search(
            r"type\s+[\"'](.+?)[\"']\s+(?:into|in|on)\s+(?:the\s+)?(.+?)\s+(?:in|on|of)\s+(.+?)(?:\s*$|\s*please)",
            both, re.IGNORECASE,
        )
        if _gui_type_m:
            return {"tool": "gui_action", "args": {
                "action": "type", "text": _gui_type_m.group(1),
                "label": _gui_type_m.group(2).strip(),
                "app": _gui_type_m.group(3).strip(),
            }}
        _gui_dbl_m = re.search(
            r"double[- ]?click\s+(?:the\s+|on\s+)?(.+?)\s+(?:in|on)\s+(.+?)(?:\s*$|\s*please)",
            both, re.IGNORECASE,
        )
        if _gui_dbl_m:
            return {"tool": "gui_action", "args": {
                "action": "double_click", "label": _gui_dbl_m.group(1).strip(),
                "app": _gui_dbl_m.group(2).strip(),
            }}
        _gui_rc_m = re.search(
            r"right[- ]?click\s+(?:the\s+|on\s+)?(.+?)\s+(?:in|on)\s+(.+?)(?:\s*$|\s*please)",
            both, re.IGNORECASE,
        )
        if _gui_rc_m:
            return {"tool": "gui_action", "args": {
                "action": "right_click", "label": _gui_rc_m.group(1).strip(),
                "app": _gui_rc_m.group(2).strip(),
            }}
        _gui_m = re.search(
            r"(?:click|press|tap)\s+(?:the\s+|on\s+)?[\"']?(.+?)[\"']?"
            r"\s+(?:button\s+|element\s+|link\s+|tab\s+|field\s+|input\s+)?"
            r"(?:in|on|of)\s+(.+?)(?:\s*$|\s*please)",
            both, re.IGNORECASE,
        )
        if _gui_m:
            return {"tool": "gui_action", "args": {
                "action": "click", "label": _gui_m.group(1).strip(),
                "app": _gui_m.group(2).strip(),
            }}
        _gui_find_m = re.search(
            r"(?:find|locate|navigate to|go to)\s+(?:the\s+)?(.+?)\s+(?:in|on)\s+(.+?)(?:\s*$|\s*please)",
            both, re.IGNORECASE,
        )
        if _gui_find_m and any(k in both for k in (
            "find the button", "find element", "find ui", "locate",
            "navigate to", "go to the",
        )):
            return {"tool": "gui_action", "args": {
                "action": "navigate", "label": _gui_find_m.group(1).strip(),
                "app": _gui_find_m.group(2).strip(),
            }}

        # Web search — natural-language intent matching.
        # Catches both imperative commands ("search X") and conversational
        # phrasing ("what do you know about X", "tell me about X").
        _WS_PATTERN = (
            r"(?:(?:search|google)\s*:?\s*(?:(?:for|about|on|the\s+(?:web|net|internet))\s+)?"
            r"|look\s*up\s+"
            r"|find\s+(?:information|info|out)\s+(?:about|on)\s+"
            r"|what\s+(?:do\s+you|can\s+you)\s+(?:know|find|tell\s+me)\s+about\s+"
            r"|who\s+is\s+|what\s+is\s+|tell\s+me\s+about\s+"
            r"|learn\s+(?:about|more\s+about)\s+"
            r"|araştır\s+|ara(?:t)?\s+|hakkında\s+(?:bilgi|ara)\s+)"
            r"(.+)"
        )
        _ws_match = re.search(_WS_PATTERN, o, re.IGNORECASE) or \
                    re.search(_WS_PATTERN, e, re.IGNORECASE)
        # Stopwords: pronouns and filler words that are NOT valid search queries.
        # "what is it" → query="it" → skip.  "who is he" → query="he" → skip.
        _WS_STOPWORDS = {
            "it", "this", "that", "he", "she", "they", "them", "we", "us",
            "him", "her", "its", "you", "me", "i", "so", "there", "here",
            "what", "who", "how", "why", "the", "a", "an", "ok", "okay",
        }
        if _ws_match:
            query = _ws_match.group(1).strip().rstrip('?.!')
            if len(query) >= 2 and query.lower() not in _WS_STOPWORDS:
                return {"tool": "web_search", "args": {"query": query}}

        # Shell generation for file operations
        if any(k in both for k in ("create file", "create folder", "create directory",
                                    "copy file", "move file", "delete file", "rename",
                                    "write into", "write a note")):
            if not any(k in both for k in ("mail", "calendar", "assignment")):
                return {"tool": "_generate", "args": {}}

        # Input Control (#122) — mouse/keyboard simulation
        _is_input = any(k in both for k in (
            "type ", "type text", "type into", "type in",
            "scroll down", "scroll up", "scroll left", "scroll right",
            "drag from", "drag to",
            "press ctrl", "press alt", "hotkey",
            "press enter", "press escape", "press tab",
            "double click", "right click", "right-click", "double-click",
            "mouse position", "where is the mouse", "where's the mouse",
            "move mouse", "move cursor",
        ))
        if _is_input:
            # Type text
            type_m = re.search(
                r'(?:type|write|enter|input)\s+(?:text\s+)?["\'](.+?)["\']',
                both, re.IGNORECASE,
            )
            if type_m:
                return {"tool": "input_control", "args": {"action": "type_text", "text": type_m.group(1)}}
            if re.search(r'type\s+(?:text\s+)?(?:into|in)', both):
                return {"tool": "input_control", "args": {"action": "type_text", "text": ""}}
            # Scroll
            if any(k in both for k in ("scroll down", "scroll up", "scroll left", "scroll right")):
                direction = "down"
                for d in ("up", "down", "left", "right"):
                    if f"scroll {d}" in both:
                        direction = d
                        break
                amt_m = re.search(r'scroll\s+\w+\s+(\d+)', both)
                amount = int(amt_m.group(1)) if amt_m else 3
                return {"tool": "input_control", "args": {"action": "scroll", "direction": direction, "amount": amount}}
            # Hotkey
            hotkey_m = re.search(
                r'(?:press|hotkey|shortcut)\s+((?:ctrl|alt|shift|super|meta|win)\s*\+\s*\w+(?:\s*\+\s*\w+)*)',
                both, re.IGNORECASE,
            )
            if hotkey_m:
                return {"tool": "input_control", "args": {"action": "hotkey", "keys": hotkey_m.group(1).strip()}}
            if any(k in both for k in ("press enter",)):
                return {"tool": "input_control", "args": {"action": "hotkey", "keys": "enter"}}
            if any(k in both for k in ("press escape", "press esc")):
                return {"tool": "input_control", "args": {"action": "hotkey", "keys": "escape"}}
            if any(k in both for k in ("press tab",)):
                return {"tool": "input_control", "args": {"action": "hotkey", "keys": "tab"}}
            # Double click / right click
            dbl_m = re.search(r'double[- ]?click\s+(?:at\s+)?(?:\(?(\d+)\s*,\s*(\d+)\)?)', both)
            if dbl_m:
                return {"tool": "input_control", "args": {"action": "double_click", "x": int(dbl_m.group(1)), "y": int(dbl_m.group(2))}}
            rc_m = re.search(r'right[- ]?click\s+(?:at\s+)?(?:\(?(\d+)\s*,\s*(\d+)\)?)', both)
            if rc_m:
                return {"tool": "input_control", "args": {"action": "right_click", "x": int(rc_m.group(1)), "y": int(rc_m.group(2))}}
            # Drag
            drag_m = re.search(r'drag\s+(?:from\s+)?\(?(\d+)\s*,\s*(\d+)\)?\s+(?:to\s+)?\(?(\d+)\s*,\s*(\d+)\)?', both)
            if drag_m:
                return {"tool": "input_control", "args": {"action": "drag", "from_x": int(drag_m.group(1)), "from_y": int(drag_m.group(2)), "to_x": int(drag_m.group(3)), "to_y": int(drag_m.group(4))}}
            # Mouse position
            if any(k in both for k in ("mouse position", "where is the mouse", "where's the mouse")):
                return {"tool": "input_control", "args": {"action": "get_position"}}
            # Move mouse
            mv_m = re.search(r'move\s+(?:mouse|cursor)\s+(?:to\s+)?\(?(\d+)\s*,\s*(\d+)\)?', both)
            if mv_m:
                return {"tool": "input_control", "args": {"action": "move_to", "x": int(mv_m.group(1)), "y": int(mv_m.group(2))}}
            # Fallback: double_click / right_click without coordinates
            if "double" in both and "click" in both:
                return {"tool": "input_control", "args": {"action": "double_click", "x": 0, "y": 0}}
            if "right" in both and "click" in both:
                return {"tool": "input_control", "args": {"action": "right_click", "x": 0, "y": 0}}

        # Accessibility / AT-SPI (#119)
        _is_a11y = any(k in both for k in (
            "click the", "click on", "press the button",
            "find the button", "find element", "find ui",
            "ui element", "accessibility", "at-spi", "atspi",
            "list windows", "list apps", "open apps",
            "focus window", "focus app", "switch to app",
            "switch to window", "bring up app", "bring up window",
            "activate window", "activate app",
            "element tree", "ui tree",
            "screenshot", "what's on my screen", "what is on my screen",
            "what's on screen", "describe screen", "describe my screen",
            "analyze screen", "screen analysis", "vlm",
        ))
        if _is_a11y:
            # Screenshot / VLM direct analysis (#120)
            if any(k in both for k in ("what's on my screen", "what is on my screen",
                                        "what's on screen", "describe screen",
                                        "describe my screen")):
                app_m = re.search(
                    r"(?:describe|what'?s on)\s+(?:my\s+)?(?:screen|display)\s+(?:in\s+|for\s+)?(.+?)(?:\s*$|\s*please)",
                    both, re.IGNORECASE,
                )
                app = app_m.group(1).strip() if app_m else ""
                return {"tool": "accessibility", "args": {"action": "describe", "app": app}}
            if any(k in both for k in ("screenshot", "analyze screen", "screen analysis", "vlm")):
                app_m = re.search(
                    r"(?:screenshot|analyze|vlm)\s+(?:of\s+|for\s+)?(.+?)(?:\s*$|\s*please)",
                    both, re.IGNORECASE,
                )
                app = app_m.group(1).strip() if app_m else ""
                label_m = re.search(
                    r"(?:find|locate)\s+(.+?)\s+(?:in|on)\s+",
                    both, re.IGNORECASE,
                )
                label = label_m.group(1).strip() if label_m else ""
                return {"tool": "accessibility", "args": {"action": "screenshot", "app": app, "label": label}}
            # Focus window
            # Strict: require app/window/screen context so "switch to a different topic"
            # doesn't trigger a window focus action.
            if any(k in both for k in ("focus window", "focus app", "switch to app",
                                        "switch to window", "bring up window",
                                        "bring up app", "activate window", "activate app")):
                app_m = re.search(
                    r"(?:focus|switch to|bring up|activate)\s+(?:(?:window|app|screen)\s+)?(.+?)(?:\s*$|\s*please)",
                    both, re.IGNORECASE,
                )
                app = app_m.group(1).strip() if app_m else ""
                return {"tool": "accessibility", "args": {"action": "focus", "app": app}}
            # List apps
            if any(k in both for k in ("list windows", "list apps", "open apps", "running apps")):
                return {"tool": "accessibility", "args": {"action": "list_apps"}}
            # Element tree
            if any(k in both for k in ("element tree", "ui tree", "accessibility tree")):
                app_m = re.search(
                    r"(?:element|ui|accessibility)\s+tree\s+(?:of\s+|for\s+)?(.+?)(?:\s*$|\s*please)",
                    both, re.IGNORECASE,
                )
                app = app_m.group(1).strip() if app_m else ""
                return {"tool": "accessibility", "args": {"action": "tree", "app": app}}
            # Find/click element (default)
            app_m = re.search(
                r"(?:click|press|find)\s+(?:the\s+|on\s+)?[\"']?(.+?)[\"']?\s+(?:button|element|link|tab|field|input|in|on)\s+(?:in\s+|on\s+)?(.+?)(?:\s*$|\s*please)",
                both, re.IGNORECASE,
            )
            if app_m:
                label = app_m.group(1).strip()
                app = app_m.group(2).strip()
                return {"tool": "accessibility", "args": {"action": "find", "app": app, "label": label}}
            # Fallback: info
            return {"tool": "accessibility", "args": {"action": "info"}}

        return None

    # ── RL & Intervention hooks (#125, #126) ─────────────────────────

    def _rl_reward_hook(self, tool_name: str, result: ToolResult) -> None:
        """Fire-and-forget: give RL engine a positive reward on tool success."""
        try:
            from bantz.agent.rl_engine import rl_engine, encode_state
            if not rl_engine.initialized:
                return
            tc = time_ctx.snapshot()
            state = encode_state(
                time_segment=tc.get("time_segment", "morning"),
                day=tc.get("day_name", "monday").lower(),
                location=tc.get("location", "home"),
                recent_tool=tool_name,
            )
            reward_val = 1.0 if result.success else -0.5
            rl_engine.reward(reward_val, next_state=state)
        except Exception:
            pass  # never crash the pipeline

    async def _check_intervention_queue(self) -> str | None:
        """[Deprecated — #137] Queue is now consumed by TUI toast system.

        Kept for backward compat in case headless mode needs it.
        """
        try:
            from bantz.agent.interventions import intervention_queue
            iv = intervention_queue.pop()
            if iv is None:
                return None
            return f"💡 [{iv.source}] {iv.title}\n   {iv.reason}"
        except Exception:
            return None

    def _prepend_intervention(self, response: str) -> str:
        """[Deprecated — #137] Toast system renders interventions separately.

        Kept for backward compat in case headless mode needs it.
        """
        iv = getattr(self, "_pending_intervention", None)
        if iv:
            self._pending_intervention = None
            return f"{iv}\n\n{response}"
        return response

    def _push_toast(
        self, title: str, reason: str = "", toast_type: str = "info",
    ) -> None:
        """Push a toast notification from brain context (#137).

        Delegates to the module-level ``_notify_toast()`` which routes
        to the TUI via callback or App.current.
        """
        _notify_toast(title, reason, toast_type)

    # ── Maintenance & Reflection handlers (#129, #130) ────────────────

    async def _handle_maintenance(self, dry_run: bool = False) -> str:
        """Run the maintenance workflow and return its summary."""
        try:
            from bantz.agent.workflows.maintenance import run_maintenance
            report = await run_maintenance(dry_run=dry_run)
            return report.summary()
        except Exception as exc:
            return f"❌ Maintenance failed: {exc}"

    def _handle_list_reflections(self, limit: int = 5) -> str:
        """List recent reflections from the KV store."""
        try:
            from bantz.agent.workflows.reflection import list_reflections
            items = list_reflections(limit=limit)
            if not items:
                return "No reflections stored yet. They are generated nightly."
            lines = ["🤔 Recent reflections:"]
            for item in items:
                date = item.get("date", "?")
                summary = item.get("summary", "")[:120]
                sessions = item.get("sessions", 0)
                lines.append(f"  • {date} ({sessions} sessions): {summary}")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Could not load reflections: {exc}"

    async def _handle_run_reflection(self, dry_run: bool = False) -> str:
        """Run the reflection workflow and return its summary."""
        try:
            from bantz.agent.workflows.reflection import run_reflection
            result = await run_reflection(dry_run=dry_run)
            return result.summary_line()
        except Exception as exc:
            return f"❌ Reflection failed: {exc}"

    async def _generate_command(self, orig: str, en: str) -> str:
        raw = await ollama.chat([
            {"role": "system", "content": COMMAND_SYSTEM},
            {"role": "user", "content": en or orig},
        ])
        return raw.strip().strip("`")

    async def _handle_location(self) -> str:
        """Handle 'where am i' queries — show GPS/location info."""
        from bantz.core.location import location_service
        from bantz.core.places import places as _places

        # Check phone GPS first — it's the most accurate source
        gps_loc = None
        try:
            from bantz.core.gps_server import gps_server
            gps_loc = gps_server.latest
        except Exception:
            pass

        try:
            loc = await location_service.get()
        except Exception:
            loc = None

        lines: list[str] = []

        # Show current named place first if any
        cur_label = _places.current_place_label()
        if cur_label:
            lines.append(f"📌 You're at: {cur_label}")

        # Prefer phone GPS as primary when available
        if gps_loc:
            acc = round(gps_loc.get("accuracy", 0))
            lines.append(f"📍 Phone GPS: {gps_loc['lat']:.6f}, {gps_loc['lon']:.6f} (±{acc}m)")
        elif loc and loc.is_live:
            lines.append(f"📍 {loc.display}")
            if loc.lat and loc.lon:
                lines.append(f"   Coordinates: {loc.lat:.6f}, {loc.lon:.6f}")
            lines.append(f"   Source: {loc.source}")
        else:
            lines.append(
                "I can't pinpoint where you are right now — "
                "I need your phone GPS to figure that out."
            )
            try:
                from bantz.core.gps_server import gps_server
                lines.append(
                    f"Open {gps_server.url} on your phone and "
                    f"hit 'Share Location' so I can see where you are."
                )
            except Exception:
                pass

        return "\n".join(lines)

    async def _handle_save_place(self, name: str) -> str:
        """Save current GPS position as a named place."""
        from bantz.core.places import places as _places
        result = _places.save_here(name)
        if result:
            lat = result.get("lat", 0.0)
            lon = result.get("lon", 0.0)
            return (
                f"📌 Saved '{name}' as a place!\n"
                f"   Coordinates: {lat:.6f}, {lon:.6f}\n"
                f"   Radius: {result.get('radius', 100)}m"
            )
        return "❌ No GPS data — couldn't save location. Is the phone GPS on?"

    async def _handle_list_places(self) -> str:
        """List all saved places."""
        from bantz.core.places import places as _places
        all_p = _places.all_places()
        if not all_p:
            return "No saved places yet. Say 'save here as X' to save one."
        lines = ["📌 Saved places:"]
        for key, p in all_p.items():
            label = p.get("label", key)
            lat = p.get("lat", 0.0)
            lon = p.get("lon", 0.0)
            radius = p.get("radius", 100)
            marker = " ⬅ you are here" if key == _places._current_place_key else ""
            lines.append(f"  • {label} ({lat:.4f}, {lon:.4f}, r={radius}m){marker}")
        return "\n".join(lines)

    async def _handle_delete_place(self, name: str) -> str:
        """Delete a saved place."""
        from bantz.core.places import places as _places
        if _places.delete_place(name):
            return f"📌 '{name}' deleted."
        return f"❌ No saved place named '{name}' found."

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        self._ensure_memory()
        await self._ensure_graph()
        en_input = await self._to_en(user_input)
        tc = time_ctx.snapshot()

        # NOTE: Intervention queue is now consumed by the TUI's toast
        # system (#137) instead of being popped here.  Brain no longer
        # prepends intervention text to chat responses.

        # Save user message ONCE — before any branching
        data_layer.conversations.add("user", user_input)

        # ── Workflow detection: multi-tool chained commands (#34) ──
        from bantz.core.workflow import workflow_engine
        steps = workflow_engine.detect(user_input, en_input)
        if steps:
            resp = await workflow_engine.execute(steps, self, en_input, tc)
            # Check if any step produced a draft → store for "send that" follow-up
            for s in steps:
                if s.result and s.result.data and s.result.data.get("draft"):
                    d = s.result.data
                    self._last_draft = {
                        "to": d["to"],
                        "subject": d.get("subject", ""),
                        "body": d["body"],
                    }
            data_layer.conversations.add("assistant", resp, tool_used="workflow")
            await self._graph_store(user_input, resp, "workflow")
            self._fire_embeddings()
            return BrainResult(response=resp, tool_used="workflow")

        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_tts_stop":
            from bantz.agent.tts import tts_engine
            if tts_engine.is_speaking:
                tts_engine.stop()
                text = "🔇 Stopped."
            else:
                text = "I'm not speaking right now."
            data_layer.conversations.add("assistant", text, tool_used="tts")
            return BrainResult(response=text, tool_used="tts")

        if quick and quick["tool"] == "_wake_word_off":
            try:
                from bantz.agent.wake_word import wake_listener
                if wake_listener.running:
                    wake_listener.stop()
                    text = "🔇 Wake word listener stopped."
                else:
                    text = "Wake word listener is not running."
            except Exception:
                text = "Wake word listener is not available."
            data_layer.conversations.add("assistant", text, tool_used="wake_word")
            return BrainResult(response=text, tool_used="wake_word")

        if quick and quick["tool"] == "_wake_word_on":
            try:
                from bantz.agent.wake_word import wake_listener
                if wake_listener.running:
                    text = "Wake word listener is already running."
                else:
                    ok = wake_listener.start()
                    text = "🎤 Wake word listener started." if ok else "❌ Could not start wake word listener."
            except Exception:
                text = "Wake word listener is not available."
            data_layer.conversations.add("assistant", text, tool_used="wake_word")
            return BrainResult(response=text, tool_used="wake_word")

        if quick and quick["tool"] == "_audio_duck_on":
            try:
                from bantz.agent.audio_ducker import audio_ducker
                if audio_ducker.available():
                    audio_ducker.enabled = True
                    text = "🔉 Audio ducking enabled."
                else:
                    text = "❌ Audio ducking not available (pactl not found)."
            except Exception:
                text = "Audio ducking module is not available."
            data_layer.conversations.add("assistant", text, tool_used="audio_ducker")
            return BrainResult(response=text, tool_used="audio_ducker")

        if quick and quick["tool"] == "_audio_duck_off":
            try:
                from bantz.agent.audio_ducker import audio_ducker
                audio_ducker.enabled = False
                text = "🔇 Audio ducking disabled."
            except Exception:
                text = "Audio ducking module is not available."
            data_layer.conversations.add("assistant", text, tool_used="audio_ducker")
            return BrainResult(response=text, tool_used="audio_ducker")

        if quick and quick["tool"] == "_ambient_status":
            try:
                from bantz.agent.ambient import ambient_analyzer
                snap = ambient_analyzer.latest()
                if snap:
                    text = (
                        f"🎤 Ambient: **{snap.label.value.upper()}** "
                        f"(RMS={snap.rms:.0f}, ZCR={snap.zcr:.3f})\n"
                        f"{ambient_analyzer.day_summary()}"
                    )
                else:
                    text = "No ambient data yet — analyzer is waiting for samples."
            except Exception:
                text = "Ambient analyzer is not available."
            data_layer.conversations.add("assistant", text, tool_used="ambient")
            return BrainResult(response=text, tool_used="ambient")

        if quick and quick["tool"] == "_proactive_status":
            try:
                from bantz.agent.proactive import (
                    proactive_engine, _get_daily_count, _compute_adaptive_max,
                )
                from bantz.agent.rl_engine import rl_engine
                kv = data_layer.kv
                if kv:
                    count, date = _get_daily_count(kv)
                    avg_r = rl_engine.episodes.avg_reward(7) if rl_engine.initialized else 0.0
                    max_d = _compute_adaptive_max(config.proactive_max_daily, avg_r)
                    text = (
                        f"💬 Proactive Engagement Status\n"
                        f"  Enabled: {'✅' if config.proactive_enabled else '❌'}\n"
                        f"  Today: {count}/{max_d} messages\n"
                        f"  RL avg reward (7d): {avg_r:.2f}\n"
                        f"  Interval: {config.proactive_interval_hours}h ±{config.proactive_jitter_minutes}m"
                    )
                else:
                    text = "Proactive engine: KV store not available."
            except Exception:
                text = "Proactive engagement module is not available."
            data_layer.conversations.add("assistant", text, tool_used="proactive")
            return BrainResult(response=text, tool_used="proactive")

        if quick and quick["tool"] == "_health_status":
            try:
                from bantz.agent.health import health_engine
                s = health_engine.status()
                cooldown_lines = "\n".join(
                    f"    {rid}: {mins:.0f}m left" for rid, mins in s["cooldowns"].items() if mins > 0
                )
                text = (
                    f"🏥 Health & Break Status\n"
                    f"  Enabled: {'✅' if config.health_enabled else '❌'}\n"
                    f"  Active session: {s['active_hours']:.1f}h\n"
                    f"  Break taken: {'✅' if s['had_break'] else '❌'}\n"
                    f"  Since last break: {s['minutes_since_break']:.0f}m\n"
                    f"  Thermal streak: CPU={s['thermal_cpu_streak']} GPU={s['thermal_gpu_streak']}\n"
                    f"  Check interval: {config.health_check_interval}s"
                )
                if cooldown_lines:
                    text += f"\n  Active cooldowns:\n{cooldown_lines}"
            except Exception:
                text = "Health & break module is not available."
            data_layer.conversations.add("assistant", text, tool_used="health")
            return BrainResult(response=text, tool_used="health")

        if quick and quick["tool"] == "_briefing":
            from bantz.core.briefing import briefing as _briefing
            text = await _briefing.generate()
            data_layer.conversations.add("assistant", text, tool_used="briefing")
            # Speak via TTS if available (#131)
            try:
                from bantz.agent.tts import tts_engine
                if tts_engine.available():
                    await tts_engine.speak_background(text)
            except Exception:
                pass
            return BrainResult(response=text, tool_used="briefing")

        if quick and quick["tool"] == "_maintenance":
            text = await self._handle_maintenance(quick["args"].get("dry_run", False))
            data_layer.conversations.add("assistant", text, tool_used="maintenance")
            return BrainResult(response=text, tool_used="maintenance")

        if quick and quick["tool"] == "_list_reflections":
            text = self._handle_list_reflections()
            data_layer.conversations.add("assistant", text, tool_used="reflection")
            return BrainResult(response=text, tool_used="reflection")

        if quick and quick["tool"] == "_run_reflection":
            text = await self._handle_run_reflection(quick["args"].get("dry_run", False))
            data_layer.conversations.add("assistant", text, tool_used="reflection")
            return BrainResult(response=text, tool_used="reflection")

        if quick and quick["tool"] == "_location":
            text = await self._handle_location()
            data_layer.conversations.add("assistant", text, tool_used="location")
            return BrainResult(response=text, tool_used="location")

        if quick and quick["tool"] == "_save_place":
            text = await self._handle_save_place(quick["args"]["name"])
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_list_places":
            text = await self._handle_list_places()
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_delete_place":
            text = await self._handle_delete_place(quick["args"]["name"])
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_schedule_today":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_today()
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_next":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_next()
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_date":
            from bantz.core.schedule import schedule as _sched
            from datetime import datetime as _dt
            target = _dt.fromisoformat(quick["args"]["date_iso"])
            text = _sched.format_for_date(target)
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_week":
            from bantz.core.schedule import schedule as _sched
            resolved = resolve_date(user_input)
            text = _sched.format_week(resolved)
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_generate":
            cmd = await self._generate_command(user_input, en_input)
            plan = {"route": "tool", "tool_name": "shell",
                    "tool_args": {"command": cmd}, "risk_level": "moderate"}

        elif quick:
            # Resolve contextual email reads (#56)
            if quick.get("_context_read") and quick["tool"] == "gmail":
                msg_id = self._resolve_message_ref(user_input)
                if msg_id:
                    quick["args"]["message_id"] = msg_id

            # Resolve "send that mail/draft" from last draft context
            if quick.get("_send_draft") and quick["tool"] == "gmail":
                if self._last_draft:
                    quick["args"] = {
                        "action": "send",
                        "to": self._last_draft["to"],
                        "subject": self._last_draft.get("subject", ""),
                        "body": self._last_draft["body"],
                    }
                else:
                    text = "No draft to send. Compose a mail first."
                    data_layer.conversations.add("assistant", text)
                    return BrainResult(response=text, tool_used=None)

            plan = {"route": "tool", "tool_name": quick["tool"],
                    "tool_args": quick["args"], "risk_level": "safe"}

        else:
            # Short ambiguous input with recent email context?
            # e.g. "medium" after listing emails → probably "emails from medium"
            # BUT: "yes", "eh?", "aa", "ok" should NOT be routed to Gmail (#155)
            _words = en_input.strip().split()
            _short_input = len(_words) <= 2
            _email_followup = False

            if _short_input and self._last_messages:
                _input_lower = en_input.strip().lower()
                # Only route if input looks like a sender reference
                # (matches a sender name/domain from last messages, or is ordinal)
                _ORDINALS = {"first", "1st", "second", "2nd", "third", "3rd",
                             "fourth", "4th", "fifth", "5th", "last", "next",
                             "previous", "that one", "this one"}
                if _input_lower in _ORDINALS or any(w in _input_lower for w in _ORDINALS):
                    _email_followup = True
                else:
                    # Check if input matches any sender name/domain in recent messages
                    for msg in self._last_messages:
                        sender = msg.get("from", "").lower()
                        if _input_lower in sender or sender.split("@")[0] in _input_lower:
                            _email_followup = True
                            break

            if _email_followup:
                plan = {"route": "tool", "tool_name": "gmail",
                        "tool_args": {"action": "search", "from_sender": en_input.strip()},
                        "risk_level": "safe"}
            else:
                plan = await cot_route(en_input, registry.all_schemas())
                if plan is None:
                    # Stream chat responses for lower perceived latency (#67)
                    stream = self._chat_stream(en_input, tc)
                    return BrainResult(
                        response="", tool_used=None, stream=stream,
                    )

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        if route != "tool" or not tool_name:
            stream = self._chat_stream(en_input, tc)
            return BrainResult(response="", tool_used=None, stream=stream)

        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd_str = tool_args.get("command", tool_name)
            warn = (
                f"⚠️  Destructive operation: [{tool_name}] `{cmd_str}`\n"
                f"Confirm? (yes/no)"
            )
            data_layer.conversations.add("assistant", warn)
            return BrainResult(
                response=warn,
                tool_used=tool_name,
                needs_confirm=True,
                pending_command=cmd_str,
                pending_tool=tool_name,
                pending_args=tool_args,
            )

        tool = registry.get(tool_name)
        if not tool:
            err = f"Tool not found: {tool_name}"
            data_layer.conversations.add("assistant", err)
            return BrainResult(response=err, tool_used=None)

        result = await tool.execute(**tool_args)

        # ── RL reward: positive signal on successful tool use (#125) ──
        self._rl_reward_hook(tool_name, result)

        # ── Store tool results for contextual follow-ups (#56) ──
        if result.success and result.data:
            if result.data.get("messages"):
                self._last_messages = result.data["messages"]
            if result.data.get("events"):
                self._last_events = result.data["events"]

        # ── Compose/reply draft → confirmation flow ──
        if result.success and result.data and result.data.get("draft"):
            d = result.data
            self._last_draft = {
                "to": d["to"],
                "subject": d.get("subject", ""),
                "body": d["body"],
            }
            data_layer.conversations.add("assistant", result.output, tool_used=tool_name)
            return BrainResult(
                response=result.output,
                tool_used=tool_name,
                tool_result=result,
                needs_confirm=True,
                pending_tool="gmail",
                pending_args={
                    "action": "send",
                    "to": d["to"],
                    "subject": d.get("subject", ""),
                    "body": d["body"],
                },
            )

        # Try streaming finalize for long tool output (#67)
        fin_stream = await self._finalize_stream(en_input, result, tc)
        if fin_stream is not None:
            return BrainResult(
                response="", tool_used=tool_name,
                tool_result=result, stream=fin_stream,
            )

        # Short output — non-streaming finalize
        resp = await self._finalize(en_input, result, tc)
        data_layer.conversations.add("assistant", resp, tool_used=tool_name)
        await self._graph_store(user_input, resp, tool_name,
                                result.data if result else None)
        self._fire_embeddings()
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    async def _chat(self, en_input: str, tc: dict) -> str:
        """
        Chat mode with conversation history.
        history[-1] = the user message we just saved → exclude to avoid duplication.
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history
        graph_hint = await self._graph_context(en_input)
        vector_hint = await self._vector_context(en_input)
        desktop_hint = self._desktop_context()
        persona_state = _persona_hint()

        messages = [
            {"role": "system", "content": CHAT_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint(),
                style_hint=_style_hint(), graph_hint=graph_hint,
                vector_hint=vector_hint, desktop_hint=desktop_hint,
                persona_state=persona_state)},
            *prior,
            {"role": "user", "content": en_input},
        ]

        # Prefer Gemini for chat if available (#58)
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages)
                if not _is_refusal(raw):
                    return strip_markdown(raw)
        except Exception:
            pass  # fall through to Ollama

        try:
            raw = await ollama.chat(messages)
            if _is_refusal(raw):
                return "Sorry, I can't help with that. Try something else."
            return strip_markdown(raw)
        except Exception as exc:
            return f"(Ollama error: {exc})"

    async def _chat_stream(self, en_input: str, tc: dict) -> AsyncIterator[str]:
        """
        Streaming chat — yields tokens as they arrive from LLM.
        Post-processing (strip_markdown) runs on accumulated text at consumer side.
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history
        graph_hint = await self._graph_context(en_input)
        vector_hint = await self._vector_context(en_input)
        desktop_hint = self._desktop_context()
        persona_state = _persona_hint()

        messages = [
            {"role": "system", "content": CHAT_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint(),
                style_hint=_style_hint(), graph_hint=graph_hint,
                vector_hint=vector_hint, desktop_hint=desktop_hint,
                persona_state=persona_state)},
            *prior,
            {"role": "user", "content": en_input},
        ]

        # Try Gemini streaming first
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                async for token in gemini.chat_stream(messages):
                    yield token
                return
        except Exception:
            pass  # fall through to Ollama

        # Ollama streaming fallback
        try:
            async for token in ollama.chat_stream(messages):
                yield token
        except Exception as exc:
            yield f"(Ollama error: {exc})"

    async def _finalize(self, en_input: str, result: ToolResult, tc: dict) -> str:
        """Delegate to core.finalizer module."""
        return await _finalize_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            graph_hint=await self._graph_context(en_input),
        )

    async def _finalize_stream(
        self, en_input: str, result: ToolResult, tc: dict,
    ) -> AsyncIterator[str] | None:
        """Delegate to core.finalizer module."""
        return await _finalize_stream_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            graph_hint=await self._graph_context(en_input),
        )

    @staticmethod
    def _hallucination_check(response: str, tool_output: str) -> tuple[str, float]:
        """Delegate to core.finalizer module."""
        return _hallucination_check_fn(response, tool_output)


def _extract_city(text: str) -> str:
    """Extract city name from weather-related user input.

    Handles English and Turkish patterns:
      "weather in Istanbul"          → "Istanbul"
      "bugün hava nasıl"             → "" (no city, use GPS)
      "ankara'da hava nasıl"         → "Ankara"
      "what's the weather in Berlin" → "Berlin"
      "izmir hava durumu"            → "Izmir"
    """
    t = text.strip()

    # Turkish: "X'de/X'da/X'te/X'ta hava" or "X hava durumu"
    m = re.search(
        r"(\b\w[\wğüşıöçĞÜŞİÖÇ]+)\s*[''`]\s*(?:de|da|te|ta|deki|daki)\b",
        t, re.IGNORECASE,
    )
    if m:
        return m.group(1).title()

    # Turkish: "X hava durumu" or "X'nin havası"
    m = re.search(
        r"(\b\w[\wğüşıöçĞÜŞİÖÇ]+)\s+hava\b",
        t, re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).lower()
        # Skip if it's "bugün" (today), "yarın" (tomorrow), or generic words
        _SKIP_TR = {"bugün", "yarın", "şu", "bu", "o", "nasıl", "şimdi", "dışarıda"}
        if candidate not in _SKIP_TR:
            return m.group(1).title()

    # English: "weather in X", "forecast for X", "temperature in X"
    m = re.search(
        r"(?:weather|forecast|temperature|rain|degrees)\s+(?:in|at|for|of)\s+(.+?)(?:\?|$|\.|\s+today|\s+tomorrow|\s+now)",
        t, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().title()

    # English: "how's the weather in X"
    m = re.search(
        r"how(?:'s|\s+is)\s+(?:the\s+)?weather\s+in\s+(.+?)(?:\?|$|\.)",
        t, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().title()

    # Fallback: strip all known weather/time words and see if anything remains
    cleaned = re.sub(
        r"\b(weather|forecast|temperature|rain|raining|degrees|how|today|tomorrow|is|it|the|in|at|for|what|whats|what's|show|tell|me|check|get|please|now|will|does|going|to|be)\b",
        "", t, flags=re.IGNORECASE,
    )
    # Also strip Turkish filler words
    cleaned = re.sub(
        r"\b(hava|durumu|nasıl|bugün|yarın|şu|şimdi|an|ne|kadar|derece|sıcaklık|yağmur|yağacak|mı|mi|mu|mü|olacak)\b",
        "", cleaned, flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" ?!.,;:'\"")

    # Only return if it looks like a city name (1-3 words, starts with letter)
    if cleaned and len(cleaned) > 1 and len(cleaned.split()) <= 3 and cleaned[0].isalpha():
        return cleaned.title()

    return ""  # empty → WeatherTool will auto-detect via GPS/location


def _extract_mail_recipient(text: str) -> str:
    """Extract recipient from compose phrases."""
    m = re.search(r"([\w.+-]+@[\w.-]+\.\w+)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:to|email)\s+(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _extract_contact(text: str) -> tuple[str, str]:
    """Extract (alias, email) from contact-save phrases."""
    em = re.search(r"(\S+@\S+)", text)
    if not em:
        return "", ""
    email = em.group(1).rstrip("'\".,;:")
    m = re.search(r"save\s+(\S+)\s+as\s+(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(2), email
    m = re.search(r"add\s+(\S+)\s+as\s+(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(2), email
    return "", email


def _extract_event_title(text: str) -> str:
    m = re.search(
        r"(?:delete|remove|cancel|update|move|reschedule)\s+['\"]?(.+?)['\"]?"
        r"\s*(?:event|meeting|appointment|$)",
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _extract_event_create(text: str) -> tuple[str, str, str]:
    from datetime import datetime, timedelta

    time_str = ""
    # HH:MM or HH.MM format
    tm = re.search(r"(\d{1,2})[:\.](\d{2})", text)
    if tm:
        time_str = f"{int(tm.group(1)):02d}:{tm.group(2)}"
    else:
        # Bare number with am/pm attached: "5pm", "10am", "3PM" — check FIRST
        tm3 = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text, re.IGNORECASE)
        if tm3:
            h = int(tm3.group(1))
            meridiem = tm3.group(2).lower()
            if meridiem == "pm" and h < 12:
                h += 12
            elif meridiem == "am" and h == 12:
                h = 0
            time_str = f"{h:02d}:00"
        else:
            # "at 5 pm", "to 3" patterns
            tm2 = re.search(r"(?:at|to)\s+(\d{1,2})\s*(am|pm)?", text, re.IGNORECASE)
            if tm2:
                h = int(tm2.group(1))
                meridiem = (tm2.group(2) or "").lower()
                if meridiem == "pm" and h < 12:
                    h += 12
                elif meridiem == "am" and h == 12:
                    h = 0
                elif not meridiem and h < 7:
                    # Ambiguous "to 10" without am/pm — assume PM for small hours
                    h += 12
                time_str = f"{h:02d}:00"
            elif re.search(r"\bnoon\b", text, re.IGNORECASE):
                time_str = "12:00"
            elif re.search(r"\bmidnight\b", text, re.IGNORECASE):
                time_str = "00:00"

    date_str = ""
    dm = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if dm:
        date_str = dm.group(1)
    elif re.search(r"\btomorrow\b", text, re.IGNORECASE):
        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif re.search(r"\btoday\b", text, re.IGNORECASE):
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Strip routing/filler words to isolate the actual event title
    noise = (
        r"\b(calendar|event|meeting|appointment|add|create|new|set|schedule|"
        r"tomorrow|today|at|for|the|a|an|my|this|into|in|on|to|from|with|it|"
        r"can|you|please|reminder|remind|me)\b"
    )
    title = re.sub(noise, "", text, flags=re.IGNORECASE)
    title = re.sub(r"\d{1,2}[:.]\d{2}", "", title)          # strip HH:MM
    title = re.sub(r"\d{4}-\d{2}-\d{2}", "", title)          # strip ISO dates
    title = re.sub(r"\b\d{1,2}\s*(?:am|pm)\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" .,;:!?'\"").strip()
    title = re.sub(r"^[\W_]+", "", title, flags=re.UNICODE).strip()

    return title or "New Event", date_str, time_str


def _extract_event_update(text: str) -> tuple[str, str]:
    """Return (old_title, new_title) from an update/rename request."""
    o = text.lower()
    m_new = re.search(
        r"(?:just\s+)?(?:write|call it|name it|rename\s+to|change\s+(?:the\s+)?name\s+to|"
        r"it\s+should\s+be)\s+(.+?)(?:\s*$|[?.!])",
        o, re.IGNORECASE,
    )
    new_title = m_new.group(1).strip() if m_new else ""
    m_old = re.search(
        r"(?:rename|change|update)\s+['\"]?(.+?)['\"]?\s+(?:to|name|title)",
        o, re.IGNORECASE,
    )
    old_title = m_old.group(1).strip() if m_old else ""
    return old_title, new_title


brain = Brain()