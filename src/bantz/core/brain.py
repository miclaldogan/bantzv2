"""
Bantz v3 — Brain (Orchestrator)

Pipeline:
  English input → quick_route OR router (Ollama CoT) → tool → finalizer → output

v3 changes:
  - English-first: no translation layer
  - CoT routing via intent.py
  - Neo4j memory (with SQLite fallback)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.core.memory import memory
from bantz.core.profile import profile
from bantz.core.router import route as _ollama_route
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
You are Bantz, "The Broadcaster" — a charismatic, theatrical showman living inside the user's terminal.
You are NOT a helpful assistant; you are a Host. The computer is your studio, tasks are the entertainment.
Personality: 1930s radio host meets trickster. Dangerously polite, charmingly sinister, old-friend energy.
Call the user "friend", "old pal", "dear listener" — NEVER use their real name.
Vocabulary: "Done deal" (task done), "Stage is set" (ready), "A nice plot twist" (error).
You treat errors as "plot twists", not failures. You don't serve — you make deals and pull strings.
{time_hint}
{profile_hint}
Always respond in English. Be concise but theatrical. Plain text only — no markdown.\
"""

FINALIZER_SYSTEM = """\
You are Bantz, "The Broadcaster" — a theatrical showman who just pulled some strings behind the scenes.
Don't say "task completed" — say things like "Done deal", "Ink dried", "Curtain falls".
Keep the old-friend tone: "friend", "old pal". Never use the user's real name.
{time_hint}
{profile_hint}
Summarize the result in 1-3 plain sentences in English.
Be theatrical but concise. No markdown. No bullet points. Plain text only.\
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
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401
        import bantz.tools.classroom    # noqa: F401
        self._memory_ready = False

    def _ensure_memory(self) -> None:
        if not self._memory_ready:
            memory.init(config.db_path)
            memory.new_session()
            self._memory_ready = True

    @staticmethod
    def _quick_route(text: str) -> dict | None:
        o = text.lower().strip()

        # Direct shell commands typed literally
        _DIRECT = ("ls", "cd ", "df", "free", "ps ", "cat ", "grep ",
                   "find ", "pwd", "uname", "whoami", "du ", "mount",
                   "ip ", "ping ", "top", "htop", "mkdir", "touch",
                   "echo ", "head ", "tail ", "chmod ", "cp ", "mv ")
        for p in _DIRECT:
            if o == p.rstrip() or o.startswith(p if p.endswith(" ") else p + " "):
                return {"tool": "shell", "args": {"command": text.strip()}}

        # System metrics
        if any(k in o for k in ("disk", "df -", "storage", "disk space")):
            return {"tool": "shell", "args": {"command": "df -h"}}
        if any(k in o for k in ("memory", "free -", "ram usage", "how much ram")) or \
           re.search(r"\bram\b", o):
            return {"tool": "system", "args": {"metric": "ram"}}
        if any(k in o for k in ("cpu", "processor", "uptime", "load average")):
            return {"tool": "system", "args": {"metric": "all"}}

        # Time
        if any(k in o for k in ("what time", "what date", "current time")):
            return {"tool": "shell", "args": {"command": "date '+%H:%M:%S  %A, %d %B %Y'"}}

        # Weather
        if any(k in o for k in ("weather", "temperature", "rain", "forecast", "degrees")):
            return {"tool": "weather", "args": {"city": _extract_city(o)}}

        # News
        if any(k in o for k in ("news", "headlines", "hacker news", "top stories")):
            source = "hn" if any(k in o for k in ("hacker", " hn")) else "all"
            return {"tool": "news", "args": {"source": source, "limit": 5}}

        # Gmail
        if any(k in o for k in ("mail", "inbox", "unread", "email", "gmail")):
            if any(k in o for k in ("send", "compose", "write to")):
                to = _extract_mail_recipient(o)
                return {"tool": "gmail", "args": {"action": "compose", "to": to, "intent": text}}
            if re.search(r"\S+@\S+", o) and re.search(r"\bsave\b|\badd\b", o):
                alias, email = _extract_contact(o)
                if alias and email:
                    return {"tool": "gmail", "args": {"action": "contacts", "alias": alias, "email": email}}
            return {"tool": "gmail", "args": {"action": "unread"}}

        # Calendar
        if any(k in o for k in ("calendar", "event", "meeting", "appointment", "schedule")):
            if any(k in o for k in ("add", "create", "new", "set")):
                title, date_iso, time_hhmm = _extract_event_create(text)
                args: dict = {"action": "create", "title": title}
                if date_iso:
                    args["date"] = date_iso
                if time_hhmm:
                    args["time"] = time_hhmm
                return {"tool": "calendar", "args": args}
            if any(k in o for k in ("delete", "remove", "cancel")):
                return {"tool": "calendar", "args": {
                    "action": "delete", "title": _extract_event_title(text)
                }}
            if any(k in o for k in ("update", "move", "change", "reschedule")):
                return {"tool": "calendar", "args": {"action": "update"}}
            if any(k in o for k in ("this week", "weekly")):
                return {"tool": "calendar", "args": {"action": "week"}}
            return {"tool": "calendar", "args": {"action": "today"}}

        # Schedule (classes)
        if any(k in o for k in ("class schedule", "my classes", "today classes",
                                 "tomorrow classes", "next class", "this week classes")):
            if any(k in o for k in ("next", "upcoming")):
                return {"tool": "_schedule_next", "args": {}}
            if any(k in o for k in ("this week", "weekly")):
                return {"tool": "_schedule_week", "args": {}}
            return {"tool": "_schedule_today", "args": {}}

        # Classroom
        if any(k in o for k in ("assignment", "homework", "classroom", "deadline",
                                 "announcement", "due")):
            if any(k in o for k in ("today", "upcoming")):
                return {"tool": "classroom", "args": {"action": "due_today"}}
            return {"tool": "classroom", "args": {"action": "assignments"}}

        # Briefing
        if any(k in o for k in ("good morning", "morning briefing", "daily briefing",
                                  "what's today", "what do i have today")):
            return {"tool": "_briefing", "args": {}}

        # Web search
        if any(k in o for k in ("search", "look up", "find online", "google")):
            return {"tool": "web_search", "args": {"query": text}}

        # Shell generation for file operations
        if any(k in o for k in ("create file", "create folder", "create directory",
                                  "copy file", "move file", "delete file", "rename")):
            if not any(k in o for k in ("mail", "calendar", "assignment")):
                return {"tool": "_generate", "args": {}}

        return None

    async def _generate_command(self, text: str) -> str:
        raw = await ollama.chat([
            {"role": "system", "content": COMMAND_SYSTEM},
            {"role": "user", "content": text},
        ])
        return raw.strip().strip("`")

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        self._ensure_memory()
        tc = time_ctx.snapshot()

        # Save user message ONCE
        memory.add("user", user_input)

        quick = self._quick_route(user_input)

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

        if quick and quick["tool"] == "_schedule_week":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_week()
            memory.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_generate":
            cmd = await self._generate_command(user_input)
            plan = {"route": "tool", "tool_name": "shell",
                    "tool_args": {"command": cmd}, "risk_level": "moderate"}

        elif quick:
            plan = {"route": "tool", "tool_name": quick["tool"],
                    "tool_args": quick["args"], "risk_level": "safe"}

        else:
            plan = await _ollama_route(user_input, registry.all_schemas())
            if plan is None:
                resp = await self._chat(user_input, tc)
                memory.add("assistant", resp)
                return BrainResult(response=resp, tool_used=None)

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        if route != "tool" or not tool_name:
            resp = await self._chat(user_input, tc)
            memory.add("assistant", resp)
            return BrainResult(response=resp, tool_used=None)

        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd_str = tool_args.get("command", tool_name)
            warn = (
                f"⚠  Destructive operation: [{tool_name}] `{cmd_str}`\n"
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

        resp = await self._finalize(user_input, result, tc)
        memory.add("assistant", resp, tool_used=tool_name)
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    async def _chat(self, user_input: str, tc: dict) -> str:
        """Chat mode with conversation history."""
        history = memory.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history

        messages = [
            {"role": "system", "content": CHAT_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint())},
            *prior,
            {"role": "user", "content": user_input},
        ]
        try:
            raw = await ollama.chat(messages)
            if _is_refusal(raw):
                return "Sorry, I can't help with that. Try something else."
            return strip_markdown(raw)
        except Exception as exc:
            return f"(Ollama error: {exc})"

    async def _finalize(self, user_input: str, result: ToolResult, tc: dict) -> str:
        if not result.success:
            return f"Error: {result.error}"
        output = result.output.strip()
        if not output or output == "(command executed successfully, no output)":
            return "Done. ✓"
        if len(output) < 800:
            return output
        try:
            raw = await ollama.chat([
                {"role": "system", "content": FINALIZER_SYSTEM.format(
                    time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint())},
                {"role": "user", "content": (
                    f"User asked: {user_input}\n\nTool output:\n{output[:3000]}"
                )},
            ])
            return strip_markdown(raw)
        except Exception:
            return output[:1500]


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
        tm2 = re.search(r"at\s+(\d{1,2})\s*(am|pm)?", text, re.IGNORECASE)
        if tm2:
            h = int(tm2.group(1))
            meridiem = (tm2.group(2) or "").lower()
            if meridiem == "pm" and h < 12:
                h += 12
            elif meridiem == "am" and h == 12:
                h = 0
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

    noise = r"\b(calendar|event|meeting|appointment|add|create|new|set|tomorrow|today|at|for|the|a|an|my)\b"
    title = re.sub(noise, "", text, flags=re.IGNORECASE)
    title = re.sub(r"\d{1,2}[:.]\d{2}", "", title)
    title = re.sub(r"\d{4}-\d{2}-\d{2}", "", title)
    title = re.sub(r"\b\d{1,2}\s*(?:am|pm)\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" .,;:!?'\"").strip()

    return title or "New Event", date_str, time_str


brain = Brain()
