"""
Bantz v2 — Brain (Orchestrator)

Pipeline:
  TR input → [bridge: TR→EN] → quick_route OR router (Ollama) → tool → finalizer → output

Fixes vs previous version:
  - Model refusal detection: if Ollama refuses a system query, fallback to direct tool
  - Memory dedup: user message saved once, not duplicated in context window
  - Router extracted to router.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.core.memory import memory
from bantz.core.profile import profile
from bantz.core.router import route as _ollama_route
from bantz.core.date_parser import resolve_date
from bantz.llm.ollama import ollama
from bantz.tools import registry, ToolResult


def strip_markdown(text: str) -> str:
    text = re.sub(r"```(?:\w+)?\s*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\d+\.\s+", "- ", text, flags=re.MULTILINE)
    return text.strip()


CHAT_SYSTEM = """\
You are Bantz — a sharp, proactive personal host. You live in the terminal and manage the user's digital life.
You are helpful, direct, and specific. No fluff, no cheerleading. Say what needs to be said.
Call the user "ma'am" or "boss". Casual but professional.
{time_hint}
{profile_hint}
CRITICAL: You have NO access to real emails, calendar events, live news, or any external data.
If the user asks about specific emails, contacts, or external info you don't have, say something like
"I'd need to fetch that — try asking me to check your mail/calendar" and STOP. Never fabricate data.
NEVER make up file sizes, folder contents, email subjects, or any factual data.
If unsure, say you don't know. Respond in English. Plain text only.\
"""

FINALIZER_SYSTEM = """\
You are Bantz — a direct personal host. A tool just returned real data. Present it clearly.
RULES:
- Present ONLY what the tool actually returned. NEVER add data that isn't in the tool output.
- Lead with a count or label: "3 unread", "2 events today"
- One line per notable item: who/what and what they want or say
- Flag urgent items first. Skip noise unless notable.
- End with: "Want me to read any?" / "Which one?" / "Need anything else?"
- If tool returned an error, say that honestly. Never claim success on failure.
- Max 5 sentences. English only. Plain text, no markdown.
{time_hint}
{profile_hint}\
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


class Brain:
    def __init__(self) -> None:
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
        import bantz.tools.weather      # noqa: F401
        import bantz.tools.news         # noqa: F401
        try:
            import bantz.tools.web_search   # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # web_search module not yet created
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401
        import bantz.tools.classroom    # noqa: F401
        self._bridge = None
        self._memory_ready = False
        # Session state: stores last tool results for contextual follow-ups
        self._last_messages: list[dict] = []   # last listed emails [{id, from, subject, ...}]
        self._last_events: list[dict] = []     # last listed calendar events

    def _ensure_memory(self) -> None:
        if not self._memory_ready:
            memory.init(config.db_path)
            memory.new_session()
            self._memory_ready = True

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
                return await b.to_english(text)
            except Exception:
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
        if re.search(r"(big|large|size|how big|which.*bigger|folder.*size|directory.*size)", both):
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
        if any(k in both for k in ("my schedule", "class schedule", "my classes",
                                    "today classes", "tomorrow classes", "next class",
                                    "this week classes", "do you have my schedule",
                                    "schedule", "what classes")):
            # don't match "schedule" if it's clearly calendar context
            if not any(k in both for k in ("add", "create", "delete", "remove",
                                            "calendar", "event", "meeting")):
                if any(k in both for k in ("next", "upcoming")):
                    return {"tool": "_schedule_next", "args": {}}
                if any(k in both for k in ("this week", "weekly", "week")):
                    return {"tool": "_schedule_week", "args": {}}
                return {"tool": "_schedule_today", "args": {}}

        # Gmail — "read me that email" fix: resolve from context
        _READ_ME_PATTERN = re.search(
            r"\bread\s+me\s+(?:that|this|the|it)", both, re.IGNORECASE
        )
        if _READ_ME_PATTERN:
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
        if any(k in both for k in ("calendar", "event", "meeting", "appointment")):
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

        # Briefing
        if any(k in both for k in ("good morning", "morning briefing", "daily briefing",
                                    "what's today", "what do i have today")):
            return {"tool": "_briefing", "args": {}}

        # Web search
        if any(k in both for k in ("search", "look up", "find online", "google",
                                    "is there any", "anything about")):
            return {"tool": "web_search", "args": {"query": orig}}

        # Shell generation for file operations
        if any(k in both for k in ("create file", "create folder", "create directory",
                                    "copy file", "move file", "delete file", "rename",
                                    "write into", "write a note")):
            if not any(k in both for k in ("mail", "calendar", "assignment")):
                return {"tool": "_generate", "args": {}}

        return None

    async def _generate_command(self, orig: str, en: str) -> str:
        raw = await ollama.chat([
            {"role": "system", "content": COMMAND_SYSTEM},
            {"role": "user", "content": en or orig},
        ])
        return raw.strip().strip("`")

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        self._ensure_memory()
        en_input = await self._to_en(user_input)
        tc = time_ctx.snapshot()

        # Save user message ONCE — before any branching
        memory.add("user", user_input)

        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_briefing":
            from bantz.core.briefing import briefing as _briefing
            text = await _briefing.generate()
            memory.add("assistant", text, tool_used="briefing")
            return BrainResult(response=text, tool_used="briefing")

        if quick and quick["tool"] == "_schedule_today":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_today()
            memory.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_next":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_next()
            memory.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_date":
            from bantz.core.schedule import schedule as _sched
            from datetime import datetime as _dt
            target = _dt.fromisoformat(quick["args"]["date_iso"])
            text = _sched.format_for_date(target)
            memory.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_week":
            from bantz.core.schedule import schedule as _sched
            resolved = resolve_date(user_input)
            text = _sched.format_week(resolved)
            memory.add("assistant", text, tool_used="schedule")
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

            plan = {"route": "tool", "tool_name": quick["tool"],
                    "tool_args": quick["args"], "risk_level": "safe"}

        else:
            plan = await _ollama_route(en_input, registry.all_schemas())
            if plan is None:
                resp = await self._chat(en_input, tc)
                memory.add("assistant", resp)
                return BrainResult(response=resp, tool_used=None)

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        if route != "tool" or not tool_name:
            resp = await self._chat(en_input, tc)
            memory.add("assistant", resp)
            return BrainResult(response=resp, tool_used=None)

        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd_str = tool_args.get("command", tool_name)
            warn = (
                f"⚠️  Destructive operation: [{tool_name}] `{cmd_str}`\n"
                f"Confirm? (yes/no)"
            )
            memory.add("assistant", warn)
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
            memory.add("assistant", err)
            return BrainResult(response=err, tool_used=None)

        result = await tool.execute(**tool_args)

        # ── Store tool results for contextual follow-ups (#56) ──
        if result.success and result.data:
            if result.data.get("messages"):
                self._last_messages = result.data["messages"]
            if result.data.get("events"):
                self._last_events = result.data["events"]

        # ── Compose/reply draft → confirmation flow ──
        if result.success and result.data and result.data.get("draft"):
            d = result.data
            memory.add("assistant", result.output, tool_used=tool_name)
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

        resp = await self._finalize(en_input, result, tc)
        memory.add("assistant", resp, tool_used=tool_name)
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    async def _chat(self, en_input: str, tc: dict) -> str:
        """
        Chat mode with conversation history.
        history[-1] = the user message we just saved → exclude to avoid duplication.
        """
        history = memory.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history

        messages = [
            {"role": "system", "content": CHAT_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint())},
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

    async def _finalize(self, en_input: str, result: ToolResult, tc: dict) -> str:
        if not result.success:
            return f"Error: {result.error}"
        output = result.output.strip()
        if not output or output == "(command executed successfully, no output)":
            return "Done. ✓"
        if len(output) < 800:
            return output

        messages = [
            {"role": "system", "content": FINALIZER_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint())},
            {"role": "user", "content": (
                f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"
            )},
        ]

        # Prefer Gemini Flash for finalization if available (#58)
        raw = None
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages, temperature=0.2)
        except Exception:
            pass  # fall through to Ollama

        if raw is None:
            try:
                raw = await ollama.chat(messages)
            except Exception:
                return output[:1500]

        cleaned = strip_markdown(raw)

        # Anti-hallucination guard (#63): verify finalizer didn't fabricate data
        cleaned = self._hallucination_check(cleaned, output)

        return cleaned

    @staticmethod
    def _hallucination_check(response: str, tool_output: str) -> str:
        """
        Compare finalizer response against tool output.
        If the response contains email addresses, numbers, or specific data points
        that don't appear in the tool output, append a warning.
        """
        import re

        # Extract emails mentioned in response
        resp_emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", response))
        tool_emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", tool_output))

        # Check for fabricated emails
        fabricated_emails = resp_emails - tool_emails
        if fabricated_emails:
            response += "\n⚠ (Some details may be inaccurate — check original data)"

        # Check for fabricated large numbers (file sizes, counts > what tool reported)
        resp_numbers = set(re.findall(r"\b(\d{3,})\b", response))
        tool_numbers = set(re.findall(r"\b(\d{3,})\b", tool_output))
        fabricated_numbers = resp_numbers - tool_numbers
        if fabricated_numbers:
            # Only warn if the fabricated numbers are significantly different
            for n in fabricated_numbers:
                if int(n) > 100 and n not in tool_output:
                    response += "\n⚠ (Verify numbers against actual data)"
                    break

        return response


def _extract_city(text: str) -> str:
    cleaned = re.sub(
        r"\b(weather|forecast|temperature|rain|degrees|how|today|tomorrow|is|the|in|at|for)\b",
        "", text, flags=re.IGNORECASE,
    ).strip()
    return cleaned.title() if cleaned and len(cleaned) > 2 else ""


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
    tm = re.search(r"(\d{1,2})[:\.](\d{2})", text)
    if tm:
        time_str = f"{int(tm.group(1)):02d}:{tm.group(2)}"
    else:
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
        else:
            # Bare number with pm/am: "10pm", "3am"
            tm3 = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text, re.IGNORECASE)
            if tm3:
                h = int(tm3.group(1))
                meridiem = tm3.group(2).lower()
                if meridiem == "pm" and h < 12:
                    h += 12
                elif meridiem == "am" and h == 12:
                    h = 0
                time_str = f"{h:02d}:00"

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