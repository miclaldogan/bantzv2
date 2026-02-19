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
            if o == p.rstrip() or o.startswith(p if p.endswith(" ") else p + " "):
                return {"tool": "shell", "args": {"command": orig.strip()}}

        # System metrics
        if any(k in both for k in ("disk", "df -", "depolama", "storage", "space")):
            return {"tool": "shell", "args": {"command": "df -h"}}
        if any(k in both for k in ("bellek", "memory", "free -")):
            return {"tool": "shell", "args": {"command": "free -h"}}
        if any(k in both for k in ("cpu", "işlemci", "uptime")):
            return {"tool": "system", "args": {"metric": "all"}}

        # Time
        if any(k in both for k in ("saat kaç", "saat ne", "what time", "what date")):
            return {"tool": "shell", "args": {"command": "date '+%H:%M:%S  %A, %d %B %Y'"}}

        # Weather
        if any(k in both for k in ("hava", "weather", "sıcaklık", "yağmur", "forecast", "derece")):
            return {"tool": "weather", "args": {"city": _extract_city(o)}}

        # News
        if any(k in both for k in ("haber", "gündem", "news", "manşet", "son dakika")):
            # "haber ver" = notify, not news
            if "haber ver" not in both:
                source = "hn" if any(k in both for k in ("hacker", " hn")) else "all"
                return {"tool": "news", "args": {"source": source, "limit": 5}}

        # ── Contacts ──────────────────────────────────────────────────────
        _CONTACTS = ("kişi ekle", "contact", "rehber")
        # Also match "kaydet" when an email address is present
        _has_email = bool(re.search(r"\S+@\S+", both))
        if any(k in both for k in _CONTACTS) or ("kaydet" in both and _has_email):
            # "hocamı kaydet: prof@uni.edu" or "kişi ekle"
            m = re.search(r"(\S+)\s+(?:kaydet|ekle)[:\s]+(\S+@\S+)", both)
            if m:
                alias = re.sub(r"[ıiüuae]$", "", m.group(1))  # strip TR suffix
                return {"tool": "gmail", "args": {
                    "action": "contacts", "alias": alias, "email": m.group(2)
                }}
            return {"tool": "gmail", "args": {"action": "contacts"}}

        # ── Compose without "mail" keyword ─────────────────────────────
        # "hocama söyle", "ali'ye de ki", "anneme haber ver"
        _COMPOSE_VERB = ("söyle", "de ki", "ilet", "bildir", "haber ver")
        if any(k in both for k in _COMPOSE_VERB):
            # Find first dative-suffix word: “hocama”, “ali’ye”, “anneme”
            to = ""
            to_match = re.search(
                r"(\S+?)(?:['\u2019]?(?:ya|ye|[ea])\b)",
                o,
            )
            if to_match:
                to = to_match.group(1).strip("'\u2019")
            return {"tool": "gmail", "args": {
                "action": "compose", "to": to, "intent": orig,
            }}

        # ── Gmail ─────────────────────────────────────────────────────────
        _GMAIL = ("mail", "gmail", "gelen kutu", "inbox", "e-posta", "eposta",
                  "mesaj", "göndermiş", "gönderen", "gelmiş", "yazmış")
        if any(k in both for k in _GMAIL):
            # Compose: "X'e mail at, şunu yaz" — natural language intent
            if any(k in both for k in ("mail at", "mesaj at", "mail gönder", "mesaj gönder",
                                        "mail yaz", "compose", "write mail", "send mail")):
                # Extract recipient from original Turkish input
                to_match = re.search(
                    r"(\S+?)(?:['\u2019]?(?:ya|ye|[ea])\b)\s.*?(?:mail|mesaj)|"  # hocama mail at
                    r"(?:mail|mesaj)\s+(?:at|g\u00f6nder)\s+(\S+)",  # mail at hocaya
                    o,
                )
                to = ""
                if to_match:
                    to = (to_match.group(1) or to_match.group(2) or "").strip("'\u2019")
                return {"tool": "gmail", "args": {
                    "action": "compose",
                    "to": to,
                    "intent": orig,
                }}

            # Reply: "bu maili yanıtla", "cevapla"
            if any(k in both for k in ("yanıtla", "cevapla", "reply")):
                return {"tool": "gmail", "args": {"action": "reply", "intent": orig}}

            # Starred: "yıldızlı mailler"
            if any(k in both for k in ("yıldız", "starred", "önemli işaret")):
                return {"tool": "gmail", "args": {"action": "search", "starred": True}}

            # Date filter: "bu haftaki", "3 günden eski"
            days = 0
            m_days = re.search(r"(\d+)\s*gün", both)
            if m_days:
                days = int(m_days.group(1))
            elif any(k in both for k in ("bu hafta", "this week")):
                days = 7
            elif any(k in both for k in ("bugün", "today")):
                days = 1
            if days:
                return {"tool": "gmail", "args": {"action": "search", "days_ago": days}}

            # Label filter: "tanıtım mailleri", "sosyal mailler"
            for label_tr in ("tanıtım", "sosyal", "güncelleme", "forum", "promosyon"):
                if label_tr in both:
                    return {"tool": "gmail", "args": {"action": "search", "label": label_tr}}

            # Count (before read — "kaç" contains "aç")
            if any(k in both for k in ("kaç", "count", "sayı", "how many")):
                return {"tool": "gmail", "args": {"action": "count"}}
            # Read (before sender filter — "son maili oku" should read, not search)
            if any(k in both for k in ("oku", "read", "içerik", "maili aç", "open", "son mail")):
                return {"tool": "gmail", "args": {"action": "read"}}

            # Sender filter: "X'ten mailler", "X ne göndermiş", "X'den gelen"
            sender_match = re.search(
                r"(?:from|gönderen|kimden)[:\s]+(\S+)|"
                r"(\S+)[''']\s*(?:ten|dan|den|tan|nın|nin|ın|in|un|ün)\s+(?:gel|mail|mesaj|gönder|yazd)|"
                r"(\S+)\s+(?:ne\s+göndermiş|ne\s+yazmış|ne\s+gelmiş|mailini|mailin)",
                both,
            )
            if sender_match:
                sender = (sender_match.group(1) or sender_match.group(2) or sender_match.group(3) or "").strip()
                # Exclude noise words that aren't real senders
                if sender.lower() not in ("son", "bu", "bir", "tüm", "bütün", "o", "ilk", "şu"):
                    return {"tool": "gmail", "args": {"action": "search", "from_sender": sender}}

            # Specific mail reference: "X mailinin içeriği", "X'in maili"
            specific_match = re.search(
                r"(\S+)\s+mail\w*\s*(?:ını|ini|inin|nin|ın|in|i)\s*(?:özetle|oku|göster|aç|içeriğ)",
                both,
            )
            if specific_match:
                sender = specific_match.group(1).strip()
                return {"tool": "gmail", "args": {"action": "search", "from_sender": sender}}

            # Default: summary
            return {"tool": "gmail", "args": {"action": "summary"}}

        # ── Schedule (before calendar/classroom — "ders" keywords) ─────
        _SCHED = ("derslerim", "schedule", "sıradaki ders", "next class",
                  "bugün ders", "ders programı", "sınıfa", "derse", "dersler",
                  "dersim", "ders var", "gitmem gereken")
        _SCHED_TIME = ("bugün", "şimdi", "var mı", "ne zaman", "today", "now",
                       "kaçta", "gitmem", "sıradaki", "sonraki")
        # Also catch: "bugün bir dersim var mı" — has ders + time context
        _has_ders = any(k in both for k in ("ders", "sınıf"))
        _has_time = any(k in both for k in _SCHED_TIME)
        _is_schedule = (any(k in both for k in _SCHED) or (_has_ders and _has_time))
        if _is_schedule:
            # "hangi ders/sınıf" → classroom ONLY if not time-oriented
            if any(k in both for k in ("hangi", "which", "liste", "kayıtlı")):
                # "hangi ders" = classroom, BUT "şimdi hangi ders var" = schedule
                if any(k in both for k in ("şimdi", "now", "var mı", "kaçta")):
                    return {"tool": "_schedule_next", "args": {}}
                pass  # fall through to classroom
            # "ödev" intent → classroom, UNLESS explicitly negated
            elif any(k in both for k in ("ödev", "teslim", "assignment")):
                if any(k in both for k in ("değil", "not", "değıl")):
                    return {"tool": "_schedule_today", "args": {}}
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
        _CAL = ("takvim", "toplantı", "calendar", "etkinlik", "randevu",
                "takvimime", "takvime")
        # Also trigger on "koy" (put) with time context
        _koy_intent = ("koy" in both and any(k in both for k in
                       ("saat", "akşam", "sabah", "öğle", "yedi", "sekiz", "dokuz",
                        "on", ":", ".")))
        if any(k in both for k in _CAL) or "bugün ne var" in both or _koy_intent:
            # Delete
            if any(k in both for k in ("sil", "kaldır", "iptal", "delete", "remove", "cancel")):
                title = _extract_event_title(both)
                return {"tool": "calendar", "args": {"action": "delete", "title": title}}
            # Update / move
            if any(k in both for k in ("güncelle", "taşı", "değiştir", "update", "move", "reschedule")):
                title = _extract_event_title(both)
                return {"tool": "calendar", "args": {"action": "update", "title": title}}
            # Create — extract title/date/time inline
            if any(k in both for k in ("ekle", "oluştur", "add", "create", "yeni", "new",
                                        "koy", "kaydet", "gir", "yaz", "put")):
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

    async def _chat(self, en_input: str, tc: dict) -> str:
        try:
            raw = await ollama.chat([
                {"role": "system", "content": CHAT_SYSTEM.format(time_hint=tc["prompt_hint"])},
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
        if len(output) < 800:
            return output
        try:
            raw = await ollama.chat([
                {"role": "system", "content": FINALIZER_SYSTEM.format(time_hint=tc["prompt_hint"])},
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

    # ── Turkish number words ──
    # ── Turkish number words (sorted longest-first to avoid partial match) ──
    _TR_NUM = [
        ("on iki", 12), ("onbir", 11), ("on bir", 11), ("oniki", 12),
        ("dokuz", 9), ("sekiz", 8), ("yedi", 7), ("altı", 6),
        ("beş", 5), ("dört", 4), ("üç", 3), ("iki", 2), ("bir", 1),
        ("on", 10),
    ]

    # ── Time: HH:MM, HH.MM, "saat X", "akşam yediye", "öğlen" ──
    time_str = ""
    # HH:MM or HH.MM
    tm = re.search(r"(\d{1,2})[:\.](\d{2})", text)
    if tm:
        time_str = f"{int(tm.group(1)):02d}:{tm.group(2)}"
    else:
        # "saat 14", "saat 9"
        tm2 = re.search(r"saat\s+(\d{1,2})", text, re.IGNORECASE)
        if tm2:
            time_str = f"{int(tm2.group(1)):02d}:00"
        else:
            # "akşam yediye", "sabah dokuzda", "öğlen"
            period = ""
            if re.search(r"akşam|gece", text, re.IGNORECASE):
                period = "pm"
            elif re.search(r"sabah|öğle", text, re.IGNORECASE):
                period = "am"

            # Find a Turkish number word near time context
            for word, val in _TR_NUM:
                if word in text.lower():
                    h = val
                    if period == "pm" and h < 12:
                        h += 12
                    time_str = f"{h:02d}:00"
                    break

            # "öğlen" alone = 12:00
            if not time_str and re.search(r"\böğle(?:n|yin)?\b", text, re.IGNORECASE):
                time_str = "12:00"

    # ── Bare hour: "19 a", "7 ye", "14 de" ──
    if not time_str:
        tm3 = re.search(r"\b(\d{1,2})\s*(?:'?(?:[eyaıiuü])|(?:da|de|ta|te))\b", text)
        if tm3:
            h = int(tm3.group(1))
            if 0 <= h <= 23:
                time_str = f"{h:02d}:00"

    # ── Date: YYYY-MM-DD, "yarın", "bugün", day names ──
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
        r"\b(takvim\w*|calendar|ekle|oluştur|add|create|yeni|new|koy|kaydet|gir|"
        r"yarın|bugün|tomorrow|today|saat|akşam|sabah|öğle\w*|gece|"
        r"için|benim|bir|"
        r"my|to|the|at|for|a|an)\b"
    )
    title = re.sub(noise, "", text, flags=re.IGNORECASE)
    # Remove number words used for time
    for word, _ in _TR_NUM:
        title = re.sub(rf"\b{word}(?:[eyaıiuü]|da|de|ta|te)?\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\d{1,2}[:.]\d{2}", "", title)        # remove time
    title = re.sub(r"\d{4}-\d{2}-\d{2}", "", title)       # remove date
    title = re.sub(r"\b\d{1,2}\s*(?:'?[eyaıiuü]|da|de|ta|te)\b", "", title)  # "19 a"
    title = re.sub(r"\s+", " ", title).strip(" .,;:!?'\"")
    title = title.strip()

    return title or "Etkinlik", date_str, time_str


brain = Brain()