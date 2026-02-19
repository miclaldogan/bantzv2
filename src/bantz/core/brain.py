"""
Bantz v2 — Brain (Orchestrator)

Pipeline:
  TR input → [bridge: TR→EN] → quick_route OR router (Ollama) → tool → finalizer → output
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bantz.config import config
from bantz.core.time_context import time_ctx
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


ROUTER_SYSTEM = """\
You are Bantz's routing brain. Analyze the user request and return a JSON routing decision.

AVAILABLE TOOLS:
{tool_schemas}

ROUTING RULES:
- shell: run terminal commands, list files, disk usage
- system: CPU%, RAM%, uptime
- weather: hava, sıcaklık, yağmur, forecast
- news: haberler, gündem, hacker news
- gmail: mail, inbox, mailleri oku/özetle/gönder, X'ten mailler
- calendar: takvim, toplantı, etkinlik, randevu ekle/sil/güncelle
- classroom: ödev, duyuru, kurslar, hangi sınıflar, teslim tarihi
- filesystem: read/write file content only
- chat: ONLY for greetings or questions no tool can answer

NEVER answer questions about system, mail, calendar, assignments from memory.

Return ONLY valid JSON. No markdown.

Tool: {{"route":"tool","tool_name":"<n>","tool_args":{{<args>}},"risk_level":"safe|moderate|destructive","reasoning":"<one line>"}}
Chat: {{"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","reasoning":"<one line>"}}\
"""

CHAT_SYSTEM = """\
You are Bantz, a sharp personal terminal assistant on Linux.
{time_hint}
Always respond in Turkish. Be concise. Plain text only — no markdown.\
"""

FINALIZER_SYSTEM = """\
You are Bantz. A tool just ran successfully.
{time_hint}
Summarize the result in 1-3 plain sentences in Turkish.
Be direct and specific. No markdown. No bullet points. Plain text only.\
"""

COMMAND_SYSTEM = """\
You are a Linux bash expert. The user request is given in two forms:
- TR: original Turkish
- EN: English translation

Return ONLY one bash command. No explanation. No markdown. Single line.

RULES:
1. mkdir -p for one directory — nothing else, no subdirs
2. Writing files: mkdir -p <dir> && printf '%s\\n' '<content>' > <path>
3. ~/Desktop=masaüstü, ~/Downloads=indirilenler, ~/Documents=belgeler
4. NEVER: sudo, nano, vim, brace expansion, interactive commands
5. NEVER invent extra files or directories
6. Use TR content if EN lost it\
"""


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


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
        self._bridge = None

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

    async def _to_tr(self, text: str) -> str:
        b = self._get_bridge()
        if b and b.is_enabled():
            try:
                return await b.to_turkish(text)
            except Exception:
                pass
        return text

    @staticmethod
    def _quick_route(orig: str, en: str) -> dict | None:
        o = orig.lower().strip()
        e = en.lower().strip()
        both = o + " " + e

        # Direct shell commands
        _DIRECT = ("ls", "cd ", "df", "free", "ps ", "cat ", "grep ",
                   "find ", "pwd", "uname", "whoami", "du ", "mount",
                   "ip ", "ping ", "top", "htop", "mkdir", "touch",
                   "echo ", "head ", "tail ", "chmod ", "cp ", "mv ")
        for p in _DIRECT:
            if o.startswith(p):
                return {"tool": "shell", "args": {"command": orig.strip()}}

        # Disk
        if any(k in both for k in ("disk", "df -", "depolama", "storage", "space")):
            return {"tool": "shell", "args": {"command": "df -h"}}

        # RAM
        if any(k in both for k in ("ram", "bellek", "memory", "free -")):
            return {"tool": "shell", "args": {"command": "free -h"}}

        # CPU / uptime
        if any(k in both for k in ("cpu", "işlemci", "uptime")):
            return {"tool": "system", "args": {"metric": "all"}}

        # Time / date
        if any(k in both for k in ("saat kaç", "saat ne", "tarih", "what time", "what date")):
            return {"tool": "shell", "args": {"command": "date '+%H:%M:%S  %A, %d %B %Y'"}}

        # Processes
        if any(k in both for k in ("süreç", "process listesi", "running process")):
            return {"tool": "shell", "args": {"command": "ps aux --sort=-%mem | head -15"}}

        # Weather — check for city name after "hava" keyword
        _WEATHER = ("hava", "weather", "sıcaklık", "yağmur", "forecast", "derece", "nem")
        if any(k in both for k in _WEATHER):
            # Try to extract explicit city: "istanbul hava", "hava ankara"
            city = _extract_city(o)
            return {"tool": "weather", "args": {"city": city}}

        # News
        _NEWS = ("haber", "gündem", "news", "manşet", "son dakika", "hacker news", "teknoloji haberi")
        if any(k in both for k in _NEWS):
            source = "hn" if any(k in both for k in ("hacker", "hn")) else "all"
            return {"tool": "news", "args": {"source": source, "limit": 5}}

        # ── Gmail ─────────────────────────────────────────────────────────
        _GMAIL = ("mail", "gmail", "gelen kutu", "inbox", "e-posta", "eposta", "mesaj")
        if any(k in both for k in _GMAIL):
            # Filtered search: "X'ten mailler", "X'den mailler"
            sender_match = re.search(
                r"(?:from|gönderen|kimden)[:\s]+(\S+)|(\S+)['\s](?:ten|dan|den|tan)\s+(?:gel|mail|mesaj)",
                both,
            )
            if sender_match:
                sender = (sender_match.group(1) or sender_match.group(2) or "").strip()
                return {"tool": "gmail", "args": {"action": "search", "from_sender": sender}}
            # Read specific / most recent
            if any(k in both for k in ("oku", "read", "içerik", "content", "aç", "open")):
                return {"tool": "gmail", "args": {"action": "read"}}
            # Count only
            if any(k in both for k in ("kaç", "count", "sayı", "how many")):
                return {"tool": "gmail", "args": {"action": "count"}}
            # Default: summary
            return {"tool": "gmail", "args": {"action": "summary"}}

        # ── Schedule (before calendar/classroom — "ders" keywords) ─────
        _SCHED = ("derslerim", "schedule", "sıradaki ders", "next class",
                  "bugün ders", "ders programı", "sınıfa", "derse", "dersler")
        if any(k in both for k in _SCHED):
            # "hangi ders/sınıf" → classroom, not schedule
            if any(k in both for k in ("hangi", "which", "liste", "kayıtlı")):
                pass  # fall through to classroom
            elif any(k in both for k in ("sıradaki", "sonraki", "next", "kaç kaldı")):
                return {"tool": "_schedule_next", "args": {}}
            else:
                return {"tool": "_schedule_today", "args": {}}

        # ── Briefing ─────────────────────────────────────────────────────
        _BRIEFING = ("brifing", "briefing", "günüme bak", "güne başla",
                     "sabah özeti", "daily summary", "günün özeti")
        if any(k in both for k in _BRIEFING):
            return {"tool": "_briefing", "args": {}}

        # ── Calendar ──────────────────────────────────────────────────────
        _CAL = ("takvim", "toplantı", "calendar", "etkinlik", "randevu")
        if any(k in both for k in _CAL) or "bugün ne var" in both:
            # Delete
            if any(k in both for k in ("sil", "kaldır", "iptal", "delete", "remove", "cancel")):
                title = _extract_event_title(both)
                return {"tool": "calendar", "args": {"action": "delete", "title": title}}
            # Update / move
            if any(k in both for k in ("güncelle", "taşı", "değiştir", "update", "move", "reschedule")):
                title = _extract_event_title(both)
                return {"tool": "calendar", "args": {"action": "update", "title": title}}
            # Create — extract title/date/time inline
            if any(k in both for k in ("ekle", "oluştur", "add", "create", "yeni", "new")):
                title, date, time_str = _extract_event_create(o)
                return {"tool": "calendar", "args": {
                    "action": "create", "title": title, "date": date, "time": time_str,
                }}
            # Week view
            if any(k in both for k in ("hafta", "week", "7 gün")):
                return {"tool": "calendar", "args": {"action": "week"}}
            # Default: today
            return {"tool": "calendar", "args": {"action": "today"}}

        # ── Classroom ─────────────────────────────────────────────────────
        _CLASS = ("ödev", "assignment", "classroom", "duyuru", "teslim",
                  "kurs", "ders", "sınıf", "class", "hangi ders", "hangi sınıf")
        if any(k in both for k in _CLASS):
            # Course list
            if any(k in both for k in ("hangi", "liste", "kurslar", "dersler",
                                        "sınıflar", "which", "list", "kayıtlı")):
                return {"tool": "classroom", "args": {"action": "courses"}}
            # Announcements
            if any(k in both for k in ("duyuru", "announcement")):
                return {"tool": "classroom", "args": {"action": "announcements"}}
            # Due today
            if any(k in both for k in ("bugün", "today", "bu gün")):
                return {"tool": "classroom", "args": {"action": "due_today"}}
            # Default: assignments
            return {"tool": "classroom", "args": {"action": "assignments"}}

        # Write / create files
        _WRITE = ("oluştur", "yaz", "kaydet", "create", "write", "save",
                  "mkdir", "ekle", "add", "klasör aç", "dosya aç")
        if any(w in both for w in _WRITE):
            return {"tool": "_generate", "args": {}}

        return None

    async def _generate_command(self, orig_tr: str, en_input: str) -> str:
        try:
            raw = await ollama.chat([
                {"role": "system", "content": COMMAND_SYSTEM},
                {"role": "user", "content": f"TR: {orig_tr}\nEN: {en_input}"},
            ])
        except Exception as exc:
            return f"echo 'Command generation failed: {exc}'"
        cmd = re.sub(r"^```(?:bash|sh)?\s*", "", raw.strip())
        cmd = re.sub(r"\s*```$", "", cmd)
        return cmd.splitlines()[0].strip()

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        en_input = await self._to_en(user_input)
        tc = time_ctx.snapshot()
        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_briefing":
            from bantz.core.briefing import briefing as _briefing
            text = await _briefing.generate()
            return BrainResult(response=text, tool_used="briefing")
        elif quick and quick["tool"] == "_schedule_today":
            from bantz.core.schedule import schedule as _schedule
            return BrainResult(response=_schedule.format_today(), tool_used="schedule")
        elif quick and quick["tool"] == "_schedule_next":
            from bantz.core.schedule import schedule as _schedule
            return BrainResult(response=_schedule.format_next(), tool_used="schedule")
        elif quick and quick["tool"] == "_generate":
            cmd = await self._generate_command(user_input, en_input)
            plan = {"route": "tool", "tool_name": "shell",
                    "tool_args": {"command": cmd}, "risk_level": "moderate"}
        elif quick:
            plan = {"route": "tool", "tool_name": quick["tool"],
                    "tool_args": quick["args"], "risk_level": "safe"}
        else:
            schema_str = "\n".join(
                f"  - {t['name']}: {t['description']} [risk={t['risk_level']}]"
                for t in registry.all_schemas()
            )
            try:
                raw = await ollama.chat([
                    {"role": "system", "content": ROUTER_SYSTEM.format(tool_schemas=schema_str)},
                    {"role": "user", "content": en_input},
                ])
                plan = _extract_json(raw)
            except Exception:
                resp = await self._chat(en_input, tc)
                return BrainResult(response=resp, tool_used=None)

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        if route != "tool" or not tool_name:
            resp = await self._chat(en_input, tc)
            return BrainResult(response=resp, tool_used=None)

        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd_str = tool_args.get("command", tool_name)
            return BrainResult(
                response=f"⚠️  Tehlikeli işlem: [{tool_name}] `{cmd_str}`\nOnaylıyor musun? (evet/hayır)",
                tool_used=tool_name,
                needs_confirm=True,
                pending_command=cmd_str,
                pending_tool=tool_name,
                pending_args=tool_args,
            )

        tool = registry.get(tool_name)
        if not tool:
            return BrainResult(response=f"Tool bulunamadı: {tool_name}", tool_used=None)

        result = await tool.execute(**tool_args)
        resp = await self._finalize(en_input, result, tc)
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _chat(self, en_input: str, tc: dict) -> str:
        system = CHAT_SYSTEM.format(time_hint=tc["prompt_hint"])
        try:
            raw = await ollama.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": en_input},
            ])
            return strip_markdown(raw)
        except Exception as exc:
            return f"(Ollama hatası: {exc})"

    async def _finalize(self, en_input: str, result: ToolResult, tc: dict) -> str:
        if not result.success:
            return f"Hata: {result.error}"

        output = result.output.strip()

        if not output or output == "(command executed successfully, no output)":
            return "Tamam, işlem tamamlandı. ✓"

        # Weather/news/gmail output is rich — show directly, no LLM summarization
        if len(output) < 800:
            return output

        system = FINALIZER_SYSTEM.format(time_hint=tc["prompt_hint"])
        try:
            raw = await ollama.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"},
            ])
            return strip_markdown(raw)
        except Exception:
            return output[:1500]


def _extract_city(text: str) -> str:
    cleaned = re.sub(
        r"\b(hava|durumu|weather|forecast|sıcaklık|yağmur|derece|nasıl|bugün|yarın|var mı)\b",
        "", text, flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"'(da|de|ta|te|nda|nde|daki|deki)\b", "", cleaned).strip()
    return cleaned.title() if cleaned and len(cleaned) > 2 else ""


def _extract_event_title(text: str) -> str:
    """Try to extract event title for delete/update operations."""
    # "X toplantısını sil", "X randevusunu iptal et"
    m = re.search(
        r"(?:sil|kaldır|iptal|güncelle|taşı|değiştir)\s+['\"]?(.+?)['\"]?\s*(?:toplantı|randevu|etkinlik|$)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # "toplantıyı sil" — no title, return empty
    return ""


def _extract_event_create(text: str) -> tuple[str, str, str]:
    """Extract title, date, time from a calendar create request.

    Returns (title, date_iso, time_hhmm).
    """
    from datetime import datetime, timedelta

    # ── Time: look for HH:MM or "saat X" ──
    time_str = ""
    tm = re.search(r"(\d{1,2}:\d{2})", text)
    if tm:
        time_str = tm.group(1)
    else:
        tm2 = re.search(r"saat\s+(\d{1,2})", text, re.IGNORECASE)
        if tm2:
            time_str = f"{int(tm2.group(1)):02d}:00"

    # ── Date: look for YYYY-MM-DD, "yarın", "bugün", day names ──
    date_str = ""
    dm = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if dm:
        date_str = dm.group(1)
    elif re.search(r"\byarın\b|tomorrow", text, re.IGNORECASE):
        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif re.search(r"\bbugün\b|today", text, re.IGNORECASE):
        date_str = datetime.now().strftime("%Y-%m-%d")

    # ── Title: strip noise words, keep the meaningful part ──
    noise = (
        r"\b(takvim\w*|calendar|ekle|oluştur|add|create|yeni|new|"
        r"yarın|bugün|tomorrow|today|saat|için|benim|bir|"
        r"my|to|the|at|for|a|an)\b"
    )
    title = re.sub(noise, "", text, flags=re.IGNORECASE)
    title = re.sub(r"\d{1,2}:\d{2}", "", title)          # remove time
    title = re.sub(r"\d{4}-\d{2}-\d{2}", "", title)      # remove date
    title = re.sub(r"\s+", " ", title).strip(" .,;:!?'\"")
    title = title.strip()

    return title or "Etkinlik", date_str, time_str


brain = Brain()