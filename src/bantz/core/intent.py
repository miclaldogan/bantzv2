"""
Bantz v2 — Chain-of-Thought Intent Parser (#78)

Replaces the one-shot router with structured multi-step reasoning:

  Step 1 [Intent]:  What does the user want?
  Step 2 [Tool]:    Which tool handles this?
  Step 3 [Params]:  Extract parameters from the input

The reasoning chain is logged for debugging and returned alongside the
routing decision so downstream code can inspect *why* a route was chosen.

Usage:
    from bantz.core.intent import cot_route
    plan = await cot_route(en_input, registry.all_schemas())
"""
from __future__ import annotations

import json
import logging
import re

from bantz.llm.ollama import ollama

log = logging.getLogger("bantz.intent")

# ── CoT system prompt ─────────────────────────────────────────────────────────

COT_SYSTEM = """\
You are Bantz's intent classifier. Analyze the user's request step by step.

AVAILABLE TOOLS:
{tool_schemas}

TOOL PARAMETER REFERENCE (extract these from the user message):
- shell: {{"command": "<full bash command>"}}
- system: {{"metric": "all|cpu|ram|disk"}}
- weather: {{"city": "<city name or empty>"}}
- news: {{"source": "all|hn", "limit": 5}}
- web_search: {{"query": "<search terms>"}}
- gmail: {{"action": "unread|compose|read|search|filter|send|contacts", ...}}
- calendar: {{"action": "today|week|create|delete|update", "title": "...", "date": "YYYY-MM-DD", "time": "HH:MM"}}
- classroom: {{"action": "assignments|due_today"}}
- filesystem: {{"path": "<file path>", "action": "read|write", ...}}
- document: {{"path": "<file path>", "action": "summarize|read|ask", "question": "..."}}

CHAIN OF THOUGHT — follow ALL three steps before deciding:

Step 1 [INTENT]: State what the user wants in one clear sentence.
Step 2 [TOOL]:   Pick the best tool. Consider ALL options before deciding.
                 If the user typed a literal bash command, the tool is "shell".
Step 3 [PARAMS]: Extract every relevant parameter from the input text.

ROUTING RULES:
- shell:      ANY bash/terminal command, file listing, process checks
- system:     cpu%, ram%, memory, disk, uptime, load average
- weather:    weather, temperature, rain, forecast, degrees
- news:       news, headlines, hacker news, top stories
- web_search: search online, look up, find on internet, google
- gmail:      email, inbox, compose, send, read mail
- calendar:   events, meetings, appointments, schedule
- classroom:  assignments, homework, deadlines, courses
- filesystem: read/write a specific file
- document:   summarize or analyze PDF/TXT/MD/DOCX
- chat:       ONLY when absolutely no tool can handle the request

CRITICAL:
- System queries (CPU, RAM, disk) are SAFE. Never refuse them.
- NEVER route to "chat" if a tool can handle it.
- Literal bash commands (ls, df -h, etc.) → shell with the command as-is.

Return ONLY valid JSON — no markdown fences, no explanation outside the JSON:

{{"chain":[{{"step":"intent","thought":"..."}},{{"step":"tool","thought":"..."}},{{"step":"params","thought":"..."}}],"route":"tool","tool_name":"<name>","tool_args":{{...}},"risk_level":"safe|moderate|destructive","confidence":0.95}}

For chat: {{"chain":[...],"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","confidence":0.9}}\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

_REFUSAL_PATTERNS = (
    "sorry", "can't assist", "cannot assist", "i'm unable",
    "i cannot", "not able to", "inappropriate",
)


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from *text*, ignoring markdown fences."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


def _log_chain(chain: list[dict], tag: str = "") -> None:
    """Pretty-log the reasoning chain at DEBUG level."""
    if not chain:
        return
    parts = " → ".join(
        f"[{s.get('step', '?')}] {s.get('thought', '')}" for s in chain
    )
    log.debug("CoT%s: %s", f" ({tag})" if tag else "", parts)


# ── Public API ─────────────────────────────────────────────────────────────────

async def cot_route(
    en_input: str,
    tool_schemas: list[dict],
    *,
    confidence_threshold: float = 0.4,
) -> dict | None:
    """
    Chain-of-Thought routing via Ollama.

    Returns a plan dict compatible with ``brain.process()``:

        {"route", "tool_name", "tool_args", "risk_level", "chain", "confidence"}

    Returns *None* when routing fails (model refusal, parse error, or very
    low confidence) — the caller should fall back to chat.
    """
    schema_str = "\n".join(
        f"  - {t['name']}: {t['description']} [risk={t['risk_level']}]"
        for t in tool_schemas
    )

    messages: list[dict] = [
        {"role": "system", "content": COT_SYSTEM.format(tool_schemas=schema_str)},
        {"role": "user", "content": en_input},
    ]

    raw: str = ""

    # ── Attempt 1 ──────────────────────────────────────────────────────────
    try:
        raw = await ollama.chat(messages)

        if _is_refusal(raw):
            log.warning("CoT refused (attempt 1): %.100s", raw)
            return None

        plan = _extract_json(raw)
        _log_chain(plan.get("chain", []))

        if plan.get("confidence", 0.8) < confidence_threshold:
            log.info("CoT low confidence (%.2f) — falling back", plan["confidence"])
            return None

        return plan

    except (json.JSONDecodeError, AttributeError) as exc:
        log.debug("CoT JSON parse failed (attempt 1): %s — raw: %.200s", exc, raw)
    except Exception as exc:
        log.warning("CoT error (attempt 1): %s", exc)
        return None

    # ── Attempt 2: retry with correction ───────────────────────────────────
    try:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "That was not valid JSON. Return ONLY a single JSON object "
                "with keys: chain, route, tool_name, tool_args, risk_level, confidence. "
                "No markdown. No explanation."
            ),
        })

        raw2 = await ollama.chat(messages)
        plan = _extract_json(raw2)
        _log_chain(plan.get("chain", []), tag="retry")
        return plan

    except Exception as exc:
        log.debug("CoT parse failed (attempt 2): %s", exc)
        return None
