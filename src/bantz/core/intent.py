"""
Bantz v2 — Chain-of-Thought Intent Parser (#78, #273)

Replaces the one-shot router with structured multi-step reasoning:

  Step 1 [Intent]:  What does the user want?
  Step 2 [Tool]:    Which tool handles this?
  Step 3 [Params]:  Extract parameters from the input

The reasoning chain is logged for debugging and returned alongside the
routing decision so downstream code can inspect *why* a route was chosen.

#273: cot_route now uses ``ollama.chat_stream()`` to stream the LLM's
``<thinking>`` process in real-time to the TUI via EventBus events.

Usage:
    from bantz.core.intent import cot_route
    plan = await cot_route(en_input, registry.all_schemas())
"""
from __future__ import annotations

import json
import logging
import re

from bantz.llm.ollama import ollama
from bantz.core.event_bus import bus

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
- visual_click: {{"target": "<UI element description>", "app": "<optional app name>"}} — click or interact with visible UI elements on screen
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
- input_control: type text, scroll, press keys, move cursor, drag
- visual_click: click a button, menu, link, icon, tab, or any visible UI element on screen.
              "click X", "open X menu", "press the Y button" → visual_click
- accessibility: focus window, screen analysis, screenshot, describe UI, list apps
- read_url:   fetch and read full content of a specific URL
- chat:       ONLY for greetings, small talk, and opinions — NEVER for factual questions
- planner:    When the request requires TWO OR MORE different tools in sequence
              (e.g. "find articles, summarize them, and save to a file")

CRITICAL:
- If the user's request contains ambiguous pronouns (e.g., 'him', 'her') or refers to unspecified files/reports ('that report'), you MUST ask for clarification. Do NOT invent a fake report, do NOT roleplay sending a message to a fake person, and do not route to a tool until the ambiguity is resolved. Route to "chat" to ask for clarification.
- NEVER hallucinate, guess, or roleplay factual data (weather, news, dates, events). If the user asks for weather, you MUST output a JSON routing to the `weather` tool. Do NOT answer directly in chat.
- System queries (CPU, RAM, disk) are SAFE. Never refuse them.
- NEVER route to "chat" if a tool can handle it.
- Literal bash commands (ls, df -h, etc.) → shell with the command as-is.
- If the user asks about ANY person, place, thing, concept, movie character,
  historical figure, or topic — route to web_search. Do NOT use chat for factual lookups.
- "do your search", "can you find", "look it up", "research it on the internet" → web_search.
- "click X", "open X menu", "press the Y button" → visual_click (NOT shell, NOT accessibility).
- If the request clearly needs multiple DIFFERENT tools in sequence, use route "planner".
- IMPORTANT: tool_name must be the exact registered name (e.g. "web_search" not "Web Search", "visual_click" not "Visual Click"). Always use snake_case.

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
   - Step 3: Multi-Step Check: Does this need 2+ DIFFERENT tools in sequence? If yes → route "planner".
   - Step 4: Double-Check / Self-Correction: Am I about to invent an answer without using a tool? I am a digital entity; I cannot see files, weather, or websites without using my tools. If I lack information to use a tool, my tool is 'chat' to ask for clarification.
2. After the thinking block, output ONLY the valid JSON object exactly as specified below.

For single tool: {{"route":"tool","tool_name":"<name>","tool_args":{{...}},"risk_level":"safe|moderate|destructive","confidence":0.95,"reasoning":"Brief explanation"}}

For multi-step: {{"route":"planner","tool_name":null,"tool_args":{{}},"risk_level":"safe","confidence":0.9,"reasoning":"Needs X then Y then Z"}}

For chat: {{"route":"chat","tool_name":null,"tool_args":{{}},"risk_level":"safe","confidence":0.9,"reasoning":"Greeting/small talk"}}\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)

# Matches a single <thinking> open tag (with optional whitespace / nesting)
_THINKING_OPEN = re.compile(r"<thinking\s*>", re.IGNORECASE)
# Matches a single </thinking> close tag
_THINKING_CLOSE = re.compile(r"</thinking\s*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Aggressively remove ``<thinking>…</thinking>`` internal monologues.

    Applied at the **earliest** point where raw LLM output is received so
    that downstream JSON parsers never choke on leaked reasoning tags.
    Handles nested/multiline blocks and trailing whitespace (#214).
    Also strips orphaned/unclosed ``<thinking>`` blocks (#273).
    """
    result = _THINKING_RE.sub("", text)
    # Handle unclosed tags — remove from <thinking> to end of string
    result = re.sub(r"<thinking\s*>.*", "", result, flags=re.DOTALL | re.IGNORECASE)
    return result


_REFUSAL_PATTERNS = (
    "can't assist", "cannot assist", "i'm unable",
    "i cannot help", "i cannot provide", "not able to",
    "inappropriate", "i'm not able", "i refuse",
    "sorry, i can't", "sorry, i cannot",
)


def _is_refusal(text: str) -> bool:
    """Detect model safety-refusal.

    Strips ``<thinking>`` blocks first so that CoT reasoning containing
    stray words like 'sorry' doesn't falsely abort tool routing (#282).
    """
    t = strip_thinking(text).lower().strip()
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


# Maximum tokens to wait for a </thinking> close tag before force-closing.
# Prevents unbounded buffering from models that never emit the close tag.
_THINKING_MAX_TOKENS = 2000


async def _stream_and_collect(
    messages: list[dict],
    *,
    emit_thinking: bool = True,
    source: str = "cot_route",
) -> str:
    """Stream an Ollama chat call, emitting ``thinking_*`` events.

    Accumulates the full response and returns it.  While inside a
    ``<thinking>`` block, each token's inner text (tags stripped) is
    emitted as ``thinking_token`` on the EventBus.  When the block
    closes (or is force-closed after ``_THINKING_MAX_TOKENS``), a
    ``thinking_done`` event is emitted.

    Args:
        messages: Chat messages to send to Ollama.
        emit_thinking: Whether to emit thinking events to the bus.
        source: Label for the ``thinking_done`` event data.

    Returns:
        The complete raw response string (with ``<thinking>`` tags intact
        for downstream JSON extraction).
    """
    buf = ""
    in_thinking = False
    thinking_tokens = 0
    thinking_started = False

    if emit_thinking:
        await bus.emit("thinking_start", source=source)

    try:
        async for token in ollama.chat_stream(messages):
            buf += token

            if not emit_thinking:
                continue

            # ── Detect <thinking> open ────────────────────────────────
            if not in_thinking:
                if _THINKING_OPEN.search(buf):
                    in_thinking = True
                    thinking_tokens = 0
                    thinking_started = True
                    # Emit any text *after* the opening tag so far
                    m = _THINKING_OPEN.search(buf)
                    if m:
                        after = buf[m.end():]
                        inner = _clean_thinking_text(after)
                        if inner:
                            await bus.emit("thinking_token", token=inner, source=source)
                    continue

            # ── Inside <thinking> block ───────────────────────────────
            if in_thinking:
                thinking_tokens += 1
                # Check for close tag
                if _THINKING_CLOSE.search(buf):
                    in_thinking = False
                    await bus.emit("thinking_done", source=source)
                    continue

                # Force-close if model never emits </thinking> (#273 Critique 2)
                if thinking_tokens > _THINKING_MAX_TOKENS:
                    in_thinking = False
                    log.warning("Force-closing <thinking> after %d tokens (no close tag)", thinking_tokens)
                    await bus.emit("thinking_done", source=source)
                    continue

                # Emit the clean inner token
                inner = _clean_thinking_text(token)
                if inner:
                    await bus.emit("thinking_token", token=inner, source=source)

    except Exception as exc:
        log.warning("Stream error during %s: %s", source, exc)
        # If we were mid-thinking, close it
        if in_thinking and emit_thinking:
            await bus.emit("thinking_done", source=source)
        raise

    # If thinking never closed, emit done
    if thinking_started and in_thinking and emit_thinking:
        await bus.emit("thinking_done", source=source)

    return buf


def _clean_thinking_text(text: str) -> str:
    """Strip literal ``<thinking>``/``</thinking>`` tags from token text.

    Only the *inner content* should reach the TUI — raw XML tags must
    never appear on screen (#273 Critique 2).
    """
    text = _THINKING_OPEN.sub("", text)
    text = _THINKING_CLOSE.sub("", text)
    return text.strip()


async def cot_route(
    en_input: str,
    tool_schemas: list[dict],
    *,
    recent_history: list[dict] | None = None,
    tool_context: str = "",
    confidence_threshold: float = 0.4,
) -> tuple[dict | None, str | None]:
    """
    Chain-of-Thought routing via Ollama **streaming** (#273).

    Returns ``(plan, routing_error)``:

    - ``(plan_dict, None)``     — successful routing (tool or chat).
    - ``(None, None)``          — pure chat / refusal / low confidence
                                  (no tool was attempted).
    - ``(None, error_string)``  — a tool route was *attempted* but JSON
                                  parsing or validation failed.  The
                                  caller **must not** silently fall back
                                  to chat — it should inform the user
                                  (#253 People-Pleaser fix).

    Now streams via ``ollama.chat_stream()`` and emits ``thinking_token``
    events on the EventBus so the TUI can display the chain-of-thought
    in real-time.

    Parameters
    ----------
    tool_context : str
        Optional dynamic context block (e.g. recent email IDs, calendar
        events) injected only when relevant (#275 — avoids bloating the
        prompt when unrelated queries are asked).
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

    # Build optional dynamic tool context (#275)
    tool_ctx_block = ""
    if tool_context:
        tool_ctx_block = f"\n\n{tool_context}"

    messages: list[dict] = [
        {"role": "system", "content": COT_SYSTEM.format(tool_schemas=schema_str) + history_block + tool_ctx_block},
        {"role": "user", "content": en_input},
    ]

    raw: str = ""

    # ── Attempt 1 (streaming) ──────────────────────────────────────────
    try:
        raw = await _stream_and_collect(messages, emit_thinking=True, source="cot_route")

        if _is_refusal(raw):
            log.warning("CoT refused (attempt 1): %.100s", raw)
            return None, None

        _log_thinking(raw)
        plan = _extract_json(raw)

        log.info("CoT parsed: route=%s tool=%s conf=%.2f",
                 plan.get("route"), plan.get("tool_name"), plan.get("confidence", 0))

        if plan.get("confidence", 0.8) < confidence_threshold:
            log.info("CoT low confidence (%.2f) — falling back", plan["confidence"])
            return None, None

        return plan, None

    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning("CoT JSON parse failed (attempt 1): %s — raw: %.200s", exc, raw)
        # Fall through to attempt 2
    except Exception as exc:
        log.warning("CoT error (attempt 1): %s", exc)
        return None, f"CoT routing error: {exc}"

    # ── Attempt 2: retry with correction (streaming, no thinking events) ──
    try:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "That was not valid JSON. Ensure you include the <thinking> tags FIRST, "
                "then output ONLY a single JSON object "
                "with keys: route, tool_name, tool_args, risk_level, confidence, reasoning. "
                "route must be one of: tool, planner, chat. "
                "No markdown. No explanation."
            ),
        })

        raw2 = await _stream_and_collect(
            messages, emit_thinking=False, source="cot_route_retry",
        )
        _log_thinking(raw2, tag="retry")
        plan = _extract_json(raw2)
        return plan, None

    except Exception as exc:
        log.warning("CoT parse failed (attempt 2): %s", exc)
        return None, f"Intent routing failed after 2 attempts: {exc}"
