"""
Bantz v2 — Router
Creates the routing decision to be sent to Ollama.

Moved to a separate file to prevent brain.py from growing too large.
"""
from __future__ import annotations

import json
import re

from bantz.llm.ollama import ollama


ROUTER_SYSTEM = """\
You are a routing classifier. Return a JSON decision only.

TOOLS:
{tool_schemas}

RULES (strict):
- shell:      ANY bash/terminal command, ls, df, free, ps, ping, etc.
- system:     cpu%, ram%, memory usage, uptime, load average
- weather:    weather, temperature, rain, forecast
- news:       news, headlines, hacker news
- gmail:      email, inbox, send mail
- calendar:   calendar, events, meetings, schedule add/delete
- classroom:  assignments, deadlines, courses, announcements
- filesystem: read or write a specific file's content
- chat:       ONLY for general conversation, jokes, questions no tool covers

CRITICAL: system queries (ram, cpu, disk, memory) are SAFE and NORMAL. Never refuse them.
NEVER use "chat" for anything a tool can handle.

Return ONLY valid JSON, no markdown:
Tool: {{"route":"tool","tool_name":"<name>","tool_args":{{<args>}},"risk_level":"safe|moderate|destructive","reasoning":"<one line>"}}
Chat: {{"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","reasoning":"<one line>"}}\
"""

# Fallback: if model returns a refusal string, detect it
_REFUSAL_PATTERNS = (
    "sorry", "can't assist", "cannot assist", "i'm unable",
    "i cannot", "not able to", "inappropriate",
)


def _looks_like_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


async def route(en_input: str, tool_schemas: list[dict]) -> dict | None:
    """
    Ask Ollama to route the request.
    Returns parsed plan dict, or None if routing fails.
    """
    schema_str = "\n".join(
        f"  - {t['name']}: {t['description']} [risk={t['risk_level']}]"
        for t in tool_schemas
    )
    try:
        raw = await ollama.chat([
            {"role": "system", "content": ROUTER_SYSTEM.format(tool_schemas=schema_str)},
            {"role": "user", "content": en_input},
        ])

        if _looks_like_refusal(raw):
            return None  # caller will fallback to _chat

        return _extract_json(raw)
    except Exception:
        return None