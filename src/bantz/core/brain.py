"""
Bantz v2 — Brain (Orchestrator)

Pipeline:
  TR input → [bridge: TR→EN] → quick_route OR router (Ollama) → tool → finalizer → output

Key design decisions:
- LLM router works in English only
- _generate_command receives BOTH TR original and EN translation to avoid bridge content loss
- No regex command parsers — LLM generates bash with strict constraints
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bantz.config import config
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
  - Use when user asks to LIST files/dirs in a directory (→ shell: "ls -la <path>")
  - Use when user wants disk usage → shell: "df -h"
  - Use when user wants to see processes → shell: "ps aux"
  - Put the exact command string in tool_args.command

system tool:
  - Use ONLY for: CPU %, RAM %, disk %, uptime from psutil
  - Use when user asks "how much RAM", "CPU usage", "memory", "uptime"
  - Do NOT use system tool for listing files

filesystem tool:
  - Use ONLY for: reading file content, writing file content
  - Do NOT use for directory listing (use shell: ls instead)

chat:
  - ONLY for greetings or questions that truly need no tool
  - NEVER answer questions about system state from memory — use a tool

CRITICAL: Never make up system information. Always use a tool to get real data.

Return ONLY valid JSON. No markdown. No explanation.

Tool:
{{"route":"tool","tool_name":"<n>","tool_args":{{<args>}},"risk_level":"safe|moderate|destructive","reasoning":"<one line>"}}

Chat:
{{"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","reasoning":"<one line>"}}\
"""

CHAT_SYSTEM = """\
You are Bantz, a sharp personal terminal assistant on Linux.
Always respond in Turkish. Be concise. Plain text only — no markdown.\
"""

FINALIZER_SYSTEM = """\
You are Bantz. A tool just ran successfully.
Summarize the result in 1-3 plain sentences in Turkish.
Be direct and specific with numbers. No markdown. No bullet points. Plain text only.\
"""

# Strict command generator — receives both TR and EN to prevent content loss
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
5. NEVER use: sudo, nano, vim, vi, brace expansion {a,b,c}, &&cd, interactive commands
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
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
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

    # ── Quick route (keyword bypass) ──────────────────────────────────────

    @staticmethod
    def _quick_route(orig: str, en: str) -> dict | None:
        """
        Fast path for obvious patterns — no LLM call needed.
        Returns {"tool": ..., "args": ...} or None (fall through to LLM router).
        Special value "_generate" = LLM generates a bash command.
        """
        o = orig.lower().strip()
        e = en.lower().strip()
        both = o + " " + e

        # User typed a real shell command directly
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

        # RAM / memory
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

        # Write / create / mkdir — LLM generates the exact bash command
        _WRITE = ("oluştur", "yaz", "kaydet", "create", "write", "save",
                  "make dir", "mkdir", "ekle", "add", "klasör aç", "dosya aç")
        if any(w in both for w in _WRITE):
            return {"tool": "_generate", "args": {}}

        return None  # fall through to LLM router

    # ── LLM command generator ─────────────────────────────────────────────

    async def _generate_command(self, orig_tr: str, en_input: str) -> str:
        """
        Ask LLM to produce a single bash command.
        Passes BOTH TR original and EN translation so bridge content loss doesn't matter.
        """
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
        # Take only first line — drop any leaked explanation
        cmd = cmd.splitlines()[0].strip()
        return cmd

    # ── Main ──────────────────────────────────────────────────────────────

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        # 1. TR → EN
        en_input = await self._to_en(user_input)

        # 2. Quick keyword route
        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_generate":
            # Write/create: LLM generates bash — gets both TR and EN
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
            # 3. LLM router fallback
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
                resp = await self._chat(en_input)
                return BrainResult(response=resp, tool_used=None)

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        # Chat only
        if route != "tool" or not tool_name:
            resp = await self._chat(en_input)
            return BrainResult(response=resp, tool_used=None)

        # Destructive confirm
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

        # Execute
        tool = registry.get(tool_name)
        if not tool:
            return BrainResult(response=f"Tool bulunamadı: {tool_name}", tool_used=None)

        result = await tool.execute(**tool_args)

        # Finalize
        resp = await self._finalize(en_input, result)
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _chat(self, en_input: str) -> str:
        try:
            raw = await ollama.chat([
                {"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": en_input},
            ])
            return strip_markdown(raw)
        except Exception as exc:
            return f"(Ollama hatası: {exc})"

    async def _finalize(self, en_input: str, result: ToolResult) -> str:
        if not result.success:
            return f"Hata: {result.error}"

        output = result.output.strip()

        # Silent commands (mkdir, touch, echo >) → confirm done
        if not output or output == "(command executed successfully, no output)":
            return "Tamam, işlem tamamlandı. ✓"

        # Short single-line → show directly
        if len(output) < 200 and "\n" not in output:
            return output

        # Long output → LLM summarize in Turkish
        try:
            raw = await ollama.chat([
                {"role": "system", "content": FINALIZER_SYSTEM},
                {"role": "user", "content": f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"},
            ])
            return strip_markdown(raw)
        except Exception:
            return output[:1500]


brain = Brain()