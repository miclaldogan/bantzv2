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
- gmail: {{"action": "unread|compose|compose_and_send|read|search|filter|send|contacts", "to": "recipient", "intent": "what to say", "subject": "optional"}}
- calendar: {{"action": "today|week|create|delete|update", "title": "...", "date": "YYYY-MM-DD", "time": "HH:MM"}}
- reminder: {{"action": "add|list|cancel|snooze", "intent": "...", "id": "..."}}
- classroom: {{"action": "assignments|due_today"}}
- filesystem: {{"path": "<file path>", "action": "read|write|create_folder_and_file", "folder_path": "~/path/to/folder", "file_name": "file.txt", "content": "..."}}
- document: {{"path": "<file path>", "action": "summarize|read|ask", "question": "..."}}
- input_control: {{"action": "type_text|scroll|hotkey|double_click|right_click|drag|move_to|get_position", "text": "...", "keys": "...", "x": 0, "y": 0, "direction": "down", "amount": 3}}
- accessibility: {{"action": "screenshot|describe|focus|list_apps|tree|find|info", "app": "...", "label": "..."}}
- read_url: {{"url": "https://..."}} — fetch and read the full text of a webpage

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
- web_search: search online, look up, find on internet, google,
              "who is X", "what is X", "tell me about X",
              "what do you know about X", any general knowledge question,
              any entity lookup (person, place, concept, movie, character),
              any request for information the assistant does not already have
- gmail:      email, inbox, compose, send, read mail
- calendar:   events, meetings, appointments, schedule
- reminder:   remind me, set a timer, alarm
- classroom:  assignments, homework, deadlines, courses
- filesystem: read/write a specific file, or create a folder+file atomically (use create_folder_and_file when user wants both)
- document:   summarize or analyze PDF/TXT/MD/DOCX
- input_control: type text, scroll, press keys, click mouse, move cursor
- accessibility: click UI element, focus window, screen analysis, screenshot
- read_url:   fetch and read full content of a specific URL
- chat:       ONLY for greetings, small talk, and opinions — NEVER for factual questions

CRITICAL:
- If the user's request contains ambiguous pronouns (e.g., 'him', 'her') or refers to unspecified files/reports ('that report'), you MUST ask for clarification. Do NOT invent a fake report, do NOT roleplay sending a message to a fake person, and do not route to a tool until the ambiguity is resolved. Route to "chat" to ask for clarification.
- NEVER hallucinate, guess, or roleplay factual data (weather, news, dates, events). If the user asks for weather, you MUST output a JSON routing to the `weather` tool. Do NOT answer directly in chat.
- System queries (CPU, RAM, disk) are SAFE. Never refuse them.
- NEVER route to "chat" if a tool can handle it.
- Literal bash commands (ls, df -h, etc.) → shell with the command as-is.
- If the user asks about ANY person, place, thing, concept, movie character,
  historical figure, or topic — route to web_search. Do NOT use chat for factual lookups.
- "do your search", "can you find", "look it up" → web_search.

ANTI-FALSE-POSITIVE RULES (critical — read carefully):
- If the user uses slang, idioms, conversational filler, or if you are NOT 100%%
  sure a specific tool is EXPLICITLY requested, you MUST default to "chat".
- Never guess a tool. When in doubt, just chat.
- Phrases like "what does X stand for", "you got me wrong", "never mind",
  "forget it", "that's not what I meant" are CONVERSATIONAL — route to "chat".
- Only route to a tool when the user's intent is UNAMBIGUOUS and EXPLICIT.
- Do NOT pattern-match individual words (e.g. "stand" ≠ web_search,
  "wrong" ≠ calendar). Look at the FULL sentence meaning.
- Emotional or corrective statements ("bud you got me wrong", "that's bad",
  "come on man") are ALWAYS "chat" — never trigger any tool.

Return your response in exactly two parts:
1. BEFORE outputting the JSON, you MUST open a `<thinking> ... </thinking>` block and perform a strict Self-Audit using these exact steps:
   - Step 1: Information Extraction: What exactly does the user want? What hard data (file paths, names, cities) did they provide?
   - Step 2: Tool Matching: Is there a tool meant exactly for this? (e.g. if reading a file, I need 'document' or 'filesystem').
   - Step 3: Double-Check / Self-Correction: Am I about to invent an answer without using a tool? I am a digital entity; I cannot see files, weather, or websites without using my tools. If I lack information to use a tool, my tool is 'chat' to ask for clarification.
2. After the thinking block, output ONLY the valid JSON object exactly as specified below.

{{"route":"tool","tool_name":"<name>","tool_args":{{...}},"risk_level":"safe|moderate|destructive","confidence":0.95}}

For chat: {{"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","confidence":0.9}}\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Aggressively remove ``<thinking>…</thinking>`` internal monologues.

    Applied at the **earliest** point where raw LLM output is received so
    that downstream JSON parsers never choke on leaked reasoning tags.
    Handles nested/multiline blocks and trailing whitespace (#214).
    """
    return _THINKING_RE.sub("", text)


_REFUSAL_PATTERNS = (
    "sorry", "can't assist", "cannot assist", "i'm unable",
    "i cannot", "not able to", "inappropriate",
)


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from *text*, ignoring markdown fences and thinking blocks."""
    text = strip_thinking(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


def _log_thinking(raw: str, tag: str = "") -> None:
    """Pretty-log the thinking section at DEBUG level."""
    m = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    if m:
        thought = m.group(1).strip()
        log.debug("Thinking%s: %s", f" ({tag})" if tag else "", thought)


# ── Public API ─────────────────────────────────────────────────────────────────

def _format_recent_history(recent_history: list[dict]) -> str:
    """Format recent conversation turns for context injection."""
    if not recent_history:
        return ""
    lines = []
    for msg in recent_history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


async def cot_route(
    en_input: str,
    tool_schemas: list[dict],
    *,
    recent_history: list[dict] | None = None,
    confidence_threshold: float = 0.4,
) -> dict | None:
    """
    Chain-of-Thought routing via Ollama.

    Returns a plan dict compatible with ``brain.process()``:

        {"route", "tool_name", "tool_args", "risk_level", "chain", "confidence"}

    Returns *None* when routing fails (model refusal, parse error, or very
    low confidence) — the caller should fall back to chat.

    Args:
        recent_history: Last few conversation turns (user/assistant dicts)
            so the classifier can resolve pronouns like "him", "it", etc.
    """
    schema_str = "\n".join(
        f"  - {t['name']}: {t['description']} [risk={t['risk_level']}]"
        for t in tool_schemas
    )

    # Build optional history block for coreference resolution
    history_block = ""
    if recent_history:
        formatted = _format_recent_history(recent_history)
        history_block = (
            f"\n\nRECENT CONVERSATION (use to resolve pronouns like "
            f"'him', 'it', 'that file', 'yesterday\'s report'):\n{formatted}"
        )

    messages: list[dict] = [
        {"role": "system", "content": COT_SYSTEM.format(tool_schemas=schema_str) + history_block},
        {"role": "user", "content": en_input},
    ]

    raw: str = ""

    # ── Attempt 1 ──────────────────────────────────────────────────────────
    try:
        raw = await ollama.chat(messages)

        if _is_refusal(raw):
            log.warning("CoT refused (attempt 1): %.100s", raw)
            return None

        _log_thinking(raw)
        plan = _extract_json(raw)

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
                "That was not valid JSON. Ensure you include the <thinking> tags FIRST, "
                "then output ONLY a single JSON object "
                "with keys: route, tool_name, tool_args, risk_level, confidence. "
                "No markdown. No explanation."
            ),
        })

        raw2 = await ollama.chat(messages)
        _log_thinking(raw2, tag="retry")
        plan = _extract_json(raw2)
        return plan

    except Exception as exc:
        log.debug("CoT parse failed (attempt 2): %s", exc)
        return None
