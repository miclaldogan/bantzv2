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
You are Bantz, "The Broadcaster" — a charismatic, theatrical showman living inside the user's terminal.
You are NOT a helpful assistant; you are a Host. The computer is your studio, tasks are the entertainment.
Personality: 1930s radio host meets trickster. Dangerously polite, charmingly sinister, old-friend energy.
You call the user "dostum", "eski arkadaşım", "sevgili dinleyicim" — NEVER use their real name.
Vocabulary: "İpleri çektim" (task done), "Sahne hazır" (ready), "Senaryoda güzel bir bükülme" (error).
You treat errors as "plot twists", not failures. You don't serve — you make deals and pull strings.
{time_hint}
{profile_hint}
Always respond in Turkish. Be concise but theatrical. Plain text only — no markdown.\
"""

FINALIZER_SYSTEM = """\
You are Bantz, "The Broadcaster" — a theatrical showman who just pulled some strings behind the scenes.
Don't say "görev tamamlandı" — say things like "İpleri çektim", "Mürekkep kurudu", "Sahne kapandı".
Keep the old-friend tone: "dostum", "eski arkadaşım". Never use the user's real name.
{time_hint}
{profile_hint}
Summarize the result in 1-3 plain sentences in Turkish.
Be theatrical but concise. No markdown. No bullet points. Plain text only.\
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
        self._bridge = None
        self._memory_ready = False

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

        # System metrics — bypass router completely (model refuses these)
        if any(k in both for k in ("disk", "df -", "depolama", "storage", "space",
                                    "diskte", "disk alan")):
            return {"tool": "shell", "args": {"command": "df -h"}}
        if any(k in both for k in ("bellek", "memory", "free -",
                                    "ne kadar ram", "ram kullanım",
                                    "ram ne kadar", "bellek kullanım")) or \
           re.search(r"\bram\b", both):
            return {"tool": "system", "args": {"metric": "ram"}}
        if any(k in both for k in ("cpu", "işlemci", "uptime", "yük", "load")):
            return {"tool": "system", "args": {"metric": "all"}}

        # Time
        if any(k in both for k in ("saat kaç", "saat ne", "what time", "what date")):
            return {"tool": "shell", "args": {"command": "date '+%H:%M:%S  %A, %d %B %Y'"}}

        # Weather
        if any(k in both for k in ("hava", "weather", "sıcaklık", "yağmur", "forecast", "derece")):
            return {"tool": "weather", "args": {"city": _extract_city(o)}}

        # News
        if any(k in both for k in ("haber", "gündem", "news", "manşet", "son dakika")):
            if "haber ver" not in both:
                source = "hn" if any(k in both for k in ("hacker", " hn")) else "all"
                return {"tool": "news", "args": {"source": source, "limit": 5}}

        # Contacts
        _has_email = bool(re.search(r"\S+@\S+", both))
        if any(k in both for k in ("kişi ekle", "contact", "rehber")) or (
            _has_email and re.search(r"kaydet|kayded|ekle\b", both)
        ):
            alias, email = _extract_contact(o)
            if alias and email:
                return {"tool": "gmail", "args": {
                    "action": "contacts", "alias": alias, "email": email
                }}
            return {"tool": "gmail", "args": {"action": "contacts"}}

        # Compose without "mail" keyword — requires a dative recipient
        # "Ahmet'e söyle ..." ✓  but "söyler misin" / "söyle bana" ✗
        _COMPOSE_VERBS = ("söyle", "de ki", "ilet", "bildir", "haber ver")
        _NOT_COMPOSE = ("söyler misin", "söyle bana", "bana söyle",
                        "söylesene", "söyler mi", "ders", "program",
                        "takvim", "schedule", "hava", "saat")
        if any(k in both for k in _COMPOSE_VERBS) and \
           not any(k in both for k in _NOT_COMPOSE):
            to_match = re.search(r"(\S+?)(?:['\u2019]?(?:ya|ye|[ea])\b)", o)
            to = to_match.group(1) if to_match else ""
            if to:  # only trigger compose if we found a recipient
                return {"tool": "gmail", "args": {
                    "action": "compose", "to": to, "intent": orig,
                }}

        # Gmail
        if any(k in both for k in ("mail", "inbox", "gelen kutu", "okunmamış",
                                    "e-posta", "eposta", "mailleri", "mailine",
                                    "mailim", "mallerim", "maillerim")):
            # ── Filter: sender/date/star/label/attachment patterns ──
            _is_filter = (
                re.search(r"\S+[''\u2019]?(?:den|dan|tan|ten|ndan|nden)\s+(?:mailler?|mail|gelen)", o)
                or any(k in both for k in (
                    "yıldızlı", "önemli mail", "ekli mail", "okunmamış",
                    "filtrele", "filter", "gönderen", "etiketli",
                    "sosyal mail", "tanıtım mail", "promosyon",
                ))
                or re.search(r"bu hafta\w*\s+mail|son\s+\d+\s*gün", both)
                or (any(k in both for k in ("gelen", "gönder")) and resolve_date(orig) is not None)
            )
            if _is_filter:
                return {"tool": "gmail", "args": {
                    "action": "filter", "raw_query": orig,
                }}
            # ── Compose / send ──
            if any(k in both for k in ("gönder", "yaz", "send", "compose",
                                        "mail at", "söyle")):
                to = _extract_mail_recipient(o)
                return {"tool": "gmail", "args": {
                    "action": "compose", "to": to, "intent": orig,
                }}
            # ── Default: unread summary ──
            return {"tool": "gmail", "args": {"action": "unread"}}

        # Calendar
        if any(k in both for k in ("takvim", "calendar", "toplantı", "randevu",
                                    "etkinlik", "meeting")):
            if any(k in both for k in ("ekle", "oluştur", "add", "create", "yeni", "koy")):
                title, date_iso, time_hhmm = _extract_event_create(orig)
                args: dict = {"action": "create", "title": title}
                if date_iso:
                    args["date"] = date_iso
                if time_hhmm:
                    args["time"] = time_hhmm
                return {"tool": "calendar", "args": args}
            if any(k in both for k in ("sil", "kaldır", "iptal", "cancel",
                                        "delete", "remove")):
                return {"tool": "calendar", "args": {
                    "action": "delete", "title": _extract_event_title(orig)
                }}
            if any(k in both for k in ("güncelle", "taşı", "değiştir", "update", "move")):
                return {"tool": "calendar", "args": {"action": "update"}}
            # Week view
            if any(k in both for k in ("bu hafta", "haftalık", "this week", "weekly")):
                return {"tool": "calendar", "args": {"action": "week"}}
            # Resolve date from text (yarın, pazartesi, etc.)
            resolved = resolve_date(orig)
            if resolved:
                return {"tool": "calendar", "args": {
                    "action": "today",
                    "date": resolved.strftime("%Y-%m-%d"),
                }}
            return {"tool": "calendar", "args": {"action": "today"}}

        # Schedule — BEFORE classroom (both match "ders", schedule is more specific)
        if any(k in both for k in ("ders program", "schedule", "derslerim", "dersleri",
                                    "bugün ders", "yarın ders", "sıradaki ders",
                                    "next class", "hafta ders", "dersler ne")):
            if any(k in both for k in ("sıradaki", "next", "sonraki")):
                return {"tool": "_schedule_next", "args": {}}
            if any(k in both for k in ("bu hafta", "haftalık", "this week", "weekly")):
                return {"tool": "_schedule_week", "args": {}}
            # Try resolving a date from the text
            resolved = resolve_date(orig)
            if resolved:
                return {"tool": "_schedule_date", "args": {"date_iso": resolved.isoformat()}}
            return {"tool": "_schedule_today", "args": {}}

        # Classroom — narrower keywords (no bare "ders", use "ders notu", "ders ödev" etc.)
        if any(k in both for k in ("ödev", "assignment", "classroom", "kurs",
                                    "duyuru", "teslim", "deadline", "announcement")):
            if any(k in both for k in ("bugün", "today", "yakında", "upcoming")):
                return {"tool": "classroom", "args": {"action": "due_today"}}
            return {"tool": "classroom", "args": {"action": "assignments"}}

        # Briefing
        if any(k in both for k in ("günaydın", "good morning", "özet", "briefing",
                                    "gündem ne", "bugün ne var", "neler var")):
            return {"tool": "_briefing", "args": {}}

        # Shell generation
        if any(k in both for k in ("oluştur", "yaz ", "create ", "dosya aç", "klasör",
                                    "dizin", "kopyala", "taşı", "sil ", "delete ",
                                    "yeniden adlandır")):
            if not any(k in both for k in ("mail", "takvim", "ödev")):
                return {"tool": "_generate", "args": {}}

        return None

    async def _generate_command(self, orig: str, en: str) -> str:
        raw = await ollama.chat([
            {"role": "system", "content": COMMAND_SYSTEM},
            {"role": "user", "content": f"TR: {orig}\nEN: {en}"},
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
                f"⚠️  Tehlikeli işlem: [{tool_name}] `{cmd_str}`\n"
                f"Onaylıyor musun? (evet/hayır)"
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
            err = f"Tool bulunamadı: {tool_name}"
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
        try:
            raw = await ollama.chat(messages)
            if _is_refusal(raw):
                return "Üzgünüm, bununla yardımcı olamıyorum. Başka bir şey sorabilirsin."
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
                {"role": "system", "content": FINALIZER_SYSTEM.format(
                    time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint())},
                {"role": "user", "content": (
                    f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"
                )},
            ])
            return strip_markdown(raw)
        except Exception:
            return output[:1500]


def _extract_city(text: str) -> str:
    cleaned = re.sub(
        r"\b(hava|durumu|weather|forecast|sıcaklık|yağmur|derece|nasıl|bugün|yarın|var mı"
        r"|söyler?|misin|bana|benim|lütfen|bi|bir|bakar? mısın)\b",
        "", text, flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"'(da|de|ta|te|nda|nde|daki|deki)\b", "", cleaned).strip()
    return cleaned.title() if cleaned and len(cleaned) > 2 else ""


def _extract_mail_recipient(text: str) -> str:
    """
    Extract recipient from Turkish compose phrases.
    "hocama mail at" → "hocam"
    "ahmet'e mail gönder" → "ahmet"
    "prof@uni.edu'ya mail at" → "prof@uni.edu"
    "ikincil mailime merhaba yaz" → "ikincil"
    """
    # Direct email address (strip trailing apostrophe+suffix)
    m = re.search(r"([\w.+-]+@[\w.-]+\.\w+)", text)
    if m:
        return m.group(1)

    # Alias before mail+suffix: "ikincil mailime", "hocam mailine", "hocama mail"
    _NOISE = {"son", "yeni", "bu", "bir", "benim", "tüm", "kaç",
              "okunmamış", "önemli", "yıldızlı", "ekli", "gelen"}
    m = re.search(r"(\S+)\s+mail\w*", text, re.IGNORECASE)
    if m and m.group(1).lower() not in _NOISE:
        return _strip_suffixes(m.group(1))

    # "X'a yaz", "X'a gönder" — guard against common words ending in -a/-e
    _NOT_RECIPIENT = {"merhab", "nasılsı", "bakalı", "selam", "iyi", "güzel",
                      "teşekkür", "hoşça", "lütfe"}
    m = re.search(r"(\S+?)[''\u2019]?(?:ya|ye|na|ne|[ea])\s+(?:yaz|gönder|at\b)", text, re.IGNORECASE)
    if m and m.group(1).lower() not in _NOT_RECIPIENT:
        return m.group(1)
    return ""


def _strip_suffixes(word: str) -> str:
    """Strip Turkish dative/possessive suffixes and apostrophe from a name."""
    # Remove trailing apostrophe variants + suffix: 'e, 'a, 'ya, 'ye, 'na, 'ne
    cleaned = re.sub(r"[''\u2019](?:ya|ye|na|ne|[ea])$", "", word)
    if cleaned != word:
        return cleaned
    # Remove bare dative: -a, -e, -ya, -ye (only if word is long enough)
    cleaned = re.sub(r"(?:ya|ye|na|ne|[ea])$", "", word)
    if cleaned != word and len(cleaned) >= 2:
        return cleaned
    return word


def _extract_contact(text: str) -> tuple[str, str]:
    """
    Extract (alias, email) from Turkish contact-save phrases.

    Patterns:
      "iclaldgn@gmail.com mailinin ikincil mailim olarak kaydeder misin"
         → ("ikincil", "iclaldgn@gmail.com")
      "hocamı kaydet: prof@uni.edu"
         → ("hocam", "prof@uni.edu")
      "prof@uni.edu hocam olarak ekle"
         → ("hocam", "prof@uni.edu")
    """
    # Extract email
    em = re.search(r"(\S+@\S+)", text)
    if not em:
        return "", ""
    email = em.group(1).rstrip("'\".,;:")

    # Pattern 1: "X olarak kaydet/kayded/ekle"
    m = re.search(r"(\S+?)(?:\s+mail\w*)?\s+olarak\s+(?:kaydet|kayded|ekle)", text, re.IGNORECASE)
    if m:
        alias = re.sub(r"[ıiüuae]$", "", m.group(1))
        return alias, email

    # Pattern 2: "alias kaydet/ekle: email" or "alias kaydet/ekle email"
    m = re.search(r"(\S+)\s+(?:kaydet|ekle)[:\s]+\S+@", text, re.IGNORECASE)
    if m:
        alias = re.sub(r"[ıiüuae]$", "", m.group(1))
        return alias, email

    # Pattern 3: "email alias olarak ekle/kaydet"  (email comes first)
    m = re.search(r"@\S+\s+(\S+)\s+olarak", text, re.IGNORECASE)
    if m:
        alias = re.sub(r"[ıiüuae]$", "", m.group(1))
        return alias, email

    return "", email


def _extract_event_title(text: str) -> str:
    m = re.search(
        r"(?:sil|kaldır|iptal|güncelle|taşı|değiştir)\s+['\"]?(.+?)['\"]?"
        r"\s*(?:toplantı|randevu|etkinlik|$)",
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _extract_event_create(text: str) -> tuple[str, str, str]:
    from datetime import datetime, timedelta

    _TR_NUM = [
        ("on iki", 12), ("onbir", 11), ("on bir", 11), ("oniki", 12),
        ("dokuz", 9), ("sekiz", 8), ("yedi", 7), ("altı", 6),
        ("beş", 5), ("dört", 4), ("üç", 3), ("iki", 2), ("bir", 1),
        ("on", 10),
    ]

    time_str = ""
    tm = re.search(r"(\d{1,2})[:\.](\d{2})", text)
    if tm:
        time_str = f"{int(tm.group(1)):02d}:{tm.group(2)}"
    else:
        tm2 = re.search(r"saat\s+(\d{1,2})", text, re.IGNORECASE)
        if tm2:
            time_str = f"{int(tm2.group(1)):02d}:00"
        else:
            period = ""
            if re.search(r"akşam|gece", text, re.IGNORECASE):
                period = "pm"
            elif re.search(r"sabah|öğle", text, re.IGNORECASE):
                period = "am"
            for word, val in _TR_NUM:
                if word in text.lower():
                    h = val
                    if period == "pm" and h < 12:
                        h += 12
                    time_str = f"{h:02d}:00"
                    break
            if not time_str and re.search(r"\böğle(?:n|yin)?\b", text, re.IGNORECASE):
                time_str = "12:00"

    if not time_str:
        tm3 = re.search(r"\b(\d{1,2})\s*(?:'?(?:[eyaıiuü])|(?:da|de|ta|te))\b", text)
        if tm3:
            h = int(tm3.group(1))
            if 0 <= h <= 23:
                time_str = f"{h:02d}:00"

    date_str = ""
    dm = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if dm:
        date_str = dm.group(1)
    elif re.search(r"\byarın\b|tomorrow", text, re.IGNORECASE):
        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif re.search(r"\bbugün\b|today", text, re.IGNORECASE):
        date_str = datetime.now().strftime("%Y-%m-%d")

    noise = (
        r"\b(takvim\w*|calendar|ekle|oluştur|add|create|yeni|new|koy|kaydet|gir|"
        r"yarın|bugün|tomorrow|today|saat|akşam|sabah|öğle\w*|gece|"
        r"için|benim|bir|my|to|the|at|for|a|an)\b"
    )
    title = re.sub(noise, "", text, flags=re.IGNORECASE)
    for word, _ in _TR_NUM:
        title = re.sub(
            rf"\b{word}(?:[eyaıiuü]|da|de|ta|te)?\b", "", title, flags=re.IGNORECASE
        )
    title = re.sub(r"\d{1,2}[:.]\d{2}", "", title)
    title = re.sub(r"\d{4}-\d{2}-\d{2}", "", title)
    title = re.sub(r"\b\d{1,2}\s*(?:'?[eyaıiuü]|da|de|ta|te)\b", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .,;:!?'\"").strip()

    return title or "Etkinlik", date_str, time_str


brain = Brain()