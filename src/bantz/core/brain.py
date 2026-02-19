"""
Bantz v2 — Brain (Orchestrator)

Pipeline:
  TR input → [bridge: TR→EN] → quick_route OR router (Ollama) → tool → finalizer → output

Phase 2 additions:
- TimeContext injected into all LLM prompts
- WeatherTool + NewsTool registered and quick-routed
- LocationService used by weather/news automatically
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.llm.ollama import ollama
from bantz.tools import registry, ToolResult


# ── Markdown stripper ─────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    text = re.sub(r"```(?:\w+)?\s*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\d+\.\s+", "- ", text, flags=re.MULTILINE)
    return text.strip()


# ── Prompts ───────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """\
You are Bantz's routing brain. Analyze the user request and return a JSON routing decision.

AVAILABLE TOOLS:
{tool_schemas}

ROUTING RULES — follow strictly:

shell tool:
  - Use when user wants to RUN any terminal command (ls, df, ps, grep, find, cat, etc.)
  - Use when user asks to LIST files/dirs in a directory
  - Use when user wants disk usage → shell: "df -h"
  - Put the exact command string in tool_args.command

system tool:
  - Use ONLY for: CPU %, RAM %, disk %, uptime from psutil

weather tool:
  - Use for: weather, hava, forecast, sıcaklık, yağmur
  - Optional: tool_args.city = specific city name (empty = auto-detect)

news tool:
  - Use for: haberler, gündem, news, hacker news, teknoloji haberleri
  - Optional: tool_args.source = "hn" | "google" | "all" (default: "all")

gmail tool:
  - Use for: mail, gmail, inbox, mailleri oku/özetle/gönder, X'ten mailler
  - tool_args.action = "summary" | "count" | "read" | "search" | "send"
  - Optional: tool_args.from_sender, tool_args.message_id, tool_args.limit

calendar tool:
  - Use for: takvim, toplantı, etkinlik, randevu, bugün ne var, bu hafta
  - tool_args.action = "today" | "week" | "create" | "delete" | "update"
  - Optional: tool_args.title, tool_args.date, tool_args.time, tool_args.event_id

filesystem tool:
  - Use ONLY for: reading or writing file content

chat:
  - ONLY for greetings or questions that truly need no tool
  - NEVER answer questions about system state from memory

Return ONLY valid JSON. No markdown. No explanation.

Tool:
{{"route":"tool","tool_name":"<n>","tool_args":{{<args>}},"risk_level":"safe|moderate|destructive","reasoning":"<one line>"}}

Chat:
{{"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","reasoning":"<one line>"}}\
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
- TR: original Turkish (may contain details lost in translation)
- EN: English translation (use for intent)

Return ONLY one bash command. No explanation. No markdown fences. No comments.

STRICT RULES:
1. mkdir -p for creating a single directory — NOTHING ELSE, no subdirs
2. For writing files with content: mkdir -p <dir> && printf '%s\\n' '<content>' > <path>
3. Locations: ~ = home, ~/Desktop = masaüstü, ~/Downloads = indirilenler/downloads, ~/Documents = belgeler
4. Create EXACTLY what was asked — one folder means ONE folder, not a tree
5. NEVER use: sudo, nano, vim, vi, brace expansion {{a,b,c}}, &&cd, interactive commands
6. NEVER invent subdirectories the user did not ask for
7. If file content is in TR request but missing from EN, use the TR version
8. Output ONLY the command — single line, nothing else\
"""


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class BrainResult:
    response: str
    tool_used: str | None
    tool_result: ToolResult | None = None
    needs_confirm: bool = False
    pending_command: str = ""
    pending_tool: str = ""
    pending_args: dict = field(default_factory=dict)


# ── Brain ─────────────────────────────────────────────────────────────────────

class Brain:
    def __init__(self) -> None:
        # Phase 1 tools
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
        # Phase 2 tools
        import bantz.tools.weather      # noqa: F401
        import bantz.tools.news         # noqa: F401
        # Phase 3 tools
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401

        self._bridge = None

    # ── Bridge ────────────────────────────────────────────────────────────

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

    # ── Quick route ───────────────────────────────────────────────────────

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

        # ── Calendar ────────────────────────────────────────────────────────
        _CAL = ("takvim", "toplantı", "calendar", "etkinlik", "randevu", "program")
        if any(k in both for k in _CAL) or any(k in both for k in ("bugün ne var", "today", "bugünkü")):
            # Delete
            if any(k in both for k in ("sil", "kaldır", "iptal", "delete", "remove", "cancel")):
                title = _extract_event_title(both)
                return {"tool": "calendar", "args": {"action": "delete", "title": title}}
            # Update / move
            if any(k in both for k in ("güncelle", "taşı", "değiştir", "update", "move", "reschedule")):
                title = _extract_event_title(both)
                return {"tool": "calendar", "args": {"action": "update", "title": title}}
            # Create
            if any(k in both for k in ("ekle", "oluştur", "add", "create", "yeni", "new")):
                return {"tool": "calendar", "args": {"action": "create"}}
            # Week view
            if any(k in both for k in ("hafta", "week", "7 gün")):
                return {"tool": "calendar", "args": {"action": "week"}}
            # Default: today
            return {"tool": "calendar", "args": {"action": "today"}}

        # Write / create
        _WRITE = ("oluştur", "yaz", "kaydet", "create", "write", "save",
                  "make dir", "mkdir", "ekle", "add", "klasör aç", "dosya aç")
        # Don't trigger _generate for calendar creates already handled above
        if any(w in both for w in _WRITE):
            return {"tool": "_generate", "args": {}}

        return None

    # ── Command generator ─────────────────────────────────────────────────

    async def _generate_command(self, orig_tr: str, en_input: str) -> str:
        user_msg = f"TR: {orig_tr}\nEN: {en_input}"
        try:
            raw = await ollama.chat([
                {"role": "system", "content": COMMAND_SYSTEM},
                {"role": "user", "content": user_msg},
            ])
        except Exception as exc:
            return f"echo 'Command generation failed: {exc}'"

        cmd = raw.strip()
        cmd = re.sub(r"^```(?:bash|sh)?\s*", "", cmd)
        cmd = re.sub(r"\s*```$", "", cmd)
        cmd = cmd.splitlines()[0].strip()
        return cmd

    # ── Main ──────────────────────────────────────────────────────────────

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        # 1. TR → EN
        en_input = await self._to_en(user_input)

        # 2. Time context snapshot (used in prompts)
        tc = time_ctx.snapshot()

        # 3. Quick keyword route
        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_generate":
            cmd = await self._generate_command(user_input, en_input)
            plan = {
                "route": "tool",
                "tool_name": "shell",
                "tool_args": {"command": cmd},
                "risk_level": "moderate",
            }
        elif quick:
            plan = {
                "route": "tool",
                "tool_name": quick["tool"],
                "tool_args": quick["args"],
                "risk_level": "safe",
            }
        else:
            # 4. LLM router fallback
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


# ── City extractor helper ─────────────────────────────────────────────────────

def _extract_city(text: str) -> str:
    """Try to extract a city name from a weather query. Returns '' for auto-detect."""
    # Patterns: "istanbul hava", "hava ankara", "izmir'de hava"
    import re
    # Remove common weather words
    cleaned = re.sub(
        r"\b(hava|durumu|weather|forecast|sıcaklık|yağmur|derece|nasıl|bugün|yarın|var mı)\b",
        "", text, flags=re.IGNORECASE
    ).strip()
    # Remove Turkish location suffixes: 'da, 'de, 'ta, 'te, 'nda
    cleaned = re.sub(r"'(da|de|ta|te|nda|nde|daki|deki)\b", "", cleaned).strip()
    # What's left might be a city
    if cleaned and len(cleaned) > 2 and not cleaned.isspace():
        return cleaned.title()
    return ""


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


brain = Brain()