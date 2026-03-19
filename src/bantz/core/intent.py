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
- browser_control: {{"action": "open|screenshot|navigate|new_tab|find_and_click|type_in_element|hotkey|type|scroll", "app": "firefox", "url": "https://...", "target": "element name", "text": "...", "keys": "ctrl+t", "wait": 2.0}}
  • action=open  → launch the app (add url= to also navigate there immediately)
  • action=navigate → go to url= in already-open browser
  • action=new_tab → open a new tab (Ctrl+T)
  • action=find_and_click → click a UI element by name (target=)
  • action=type_in_element → find element, click it, type text= into it, press Enter (target= text= required)
  • action=screenshot → capture screen
  • action=hotkey → send key combo
  • action=type → type text at current cursor position
  SINGLE-STEP PATTERN: "open firefox and go to X" → browser_control action=open app=firefox url=https://X
  FOLLOW-UP PATTERN: "open it"/"play it"/"click it"/"aç onu" after a browser action → browser_control action=find_and_click target="first result" app=firefox
  IN-PAGE INTERACTION (use planner for multi-step): navigate → screenshot → type_in_element

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
              CRITICAL: "click files", "click chrome", "click terminal", "click firefox" →
              visual_click target="[app name]". NEVER route "click [app]" to filesystem.
- screenshot: DELIVER a screenshot image to the user — use this when the user wants to SEE/RECEIVE the photo.
              "give me a screenshot", "send me a screenshot", "show me the screen", "take a screenshot",
              "ekran görüntüsü al/gönder/ver", "can I see the screen", "what does my screen look like",
              "ss al", "/ekran", "screenshot please" → screenshot tool (NOT browser_control).
              Use browser_control action=screenshot ONLY when analyzing the screen for a click/type task.
- accessibility: focus window, screen analysis, describe UI, list apps
- read_url:   fetch and read full content of a specific URL
- browser_control: For a SINGLE browser action:
              • "open firefox/chrome" → action=open app=firefox/google-chrome
              • "go to X" / "navigate to X" / "open X.com" → action=navigate url=https://X
              • "write X there" / "type X there" / "ora X yaz" — type X into address bar and press Enter:
                action=type_in_element target="address bar" text="X" press_enter=true
              • "new tab" → action=new_tab
              • "click [element]" when browser context is clear → action=find_and_click target="[element]"
              • FOLLOW-UP after browser action: "open it", "play it", "click it",
                "aç onu", "aç", "tıkla" → action=find_and_click target="first result"
                (ONLY when PREVIOUS TOOL RESULT or RECENT CONVERSATION shows a browser/search was just performed)
              DO NOT use for any request that involves navigating AND doing something else.
- chat:       ONLY for greetings, small talk, and opinions — NEVER for factual questions
- planner:    ONLY when the request contains EXPLICIT multi-step language like "and then", "then", "and search", "and click", "and type", "after that".
              DO NOT use planner for single-action requests that happen to mention a browser.
              USE PLANNER ONLY for these patterns:
              • "go to [site] and [do something else]" — TWO distinct actions
              • "open [video/page] on youtube" → search + find + click
              • "find/play/watch [X] on youtube" → search + click
              • "search for X on [site]" → navigate + search
              • "and then", "then", "and click", "and type" — explicit chaining keywords present
              NEVER use planner for: "click X", "open X", "go to X", "write X there", "give me a screenshot".
              NEVER use planner when all actions use the same browser. "open chrome and go to wikipedia.org" =
              browser_control action=open app=chrome url=https://en.wikipedia.org — ONE step, not planner.

SCREEN CONTEXT (when CURRENT SCREEN STATE is provided in tool_context):
- Use the screen description to resolve "there", "here", "it", "that", "the page", "the bar", "the box"
- Example: screen shows Wikipedia → "write iron man there" = browser_control action=type_in_element target="Wikipedia search input" text="iron man" press_enter=true
- Example: screen shows YouTube search results → "click the first one" = browser_control action=find_and_click target="first video result"
- Example: screen shows Chrome with address bar → "write X" = browser_control action=type_in_element target="address bar" text="X" press_enter=true
- NEVER invent YouTube/Firefox steps when screen context shows you're already on a different page.

CRITICAL:
- If the user's request contains ambiguous pronouns (e.g., 'him', 'her') or refers to unspecified files/reports ('that report'), you MUST ask for clarification. Do NOT invent a fake report, do NOT roleplay sending a message to a fake person, and do not route to a tool until the ambiguity is resolved. Route to "chat" to ask for clarification.
- NEVER hallucinate, guess, or roleplay factual data (weather, news, dates, events). If the user asks for weather, you MUST output a JSON routing to the `weather` tool. Do NOT answer directly in chat.
- System queries (CPU, RAM, disk) are SAFE. Never refuse them.
- NEVER route to "chat" if a tool can handle it.
- Literal bash commands (ls, df -h, etc.) → shell with the command as-is.
- If the user asks about ANY person, place, thing, concept, movie character,
  historical figure, or topic — route to web_search. Do NOT use chat for factual lookups.
- "do your search", "can you find", "look it up", "research it on the internet" → web_search.
  EXCEPTION: "open chrome/firefox/files/terminal", "launch [app]", "click chrome/files/terminal"
  are ALWAYS browser_control or visual_click — NEVER web_search. App names are not search queries.
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
# Detects start of JSON output — our CoT always returns {"route": ...}
# When this pattern appears in the buffer, the thinking phase is done.
_JSON_MARKER = re.compile(r'\{\s*"route"\s*:', re.IGNORECASE)


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
    # Once the thinking block closes, never re-detect it.
    # Without this flag, old <thinking> tags accumulate in `buf` and every
    # subsequent token falsely re-triggers detection → ThinkingPanel spam. (#273)
    thinking_complete = False

    if emit_thinking:
        await bus.emit("thinking_start", source=source)

    try:
        async for token in ollama.chat_stream(messages):
            buf += token

            if not emit_thinking or thinking_complete:
                continue

            # ── Detect JSON start (CoT always ends with {"route": ...) ──
            # When the JSON marker appears, the thinking phase is done.
            if not in_thinking and _JSON_MARKER.search(buf):
                thinking_complete = True
                await bus.emit("thinking_done", source=source)
                continue

            # ── Detect <thinking> open (only before we've seen one block) ─
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
                else:
                    # No <thinking> tag yet — emit token directly as thinking content.
                    # This shows the model's reasoning even for models that don't use tags.
                    inner = _clean_thinking_text(token)
                    if inner:
                        if not thinking_started:
                            thinking_started = True
                        await bus.emit("thinking_token", token=inner, source=source)
                    continue

            # ── Inside <thinking> block ───────────────────────────────
            if in_thinking:
                thinking_tokens += 1
                # Check for close tag only in the current token (not full buf)
                # to avoid re-matching the tag after the block has closed.
                if _THINKING_CLOSE.search(token) or _THINKING_CLOSE.search(buf[-80:]):
                    in_thinking = False
                    thinking_complete = True
                    await bus.emit("thinking_done", source=source)
                    continue

                # Force-close if model never emits </thinking> (#273 Critique 2)
                if thinking_tokens > _THINKING_MAX_TOKENS:
                    in_thinking = False
                    thinking_complete = True
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
        if (in_thinking or thinking_started) and not thinking_complete and emit_thinking:
            await bus.emit("thinking_done", source=source)
        raise

    # If thinking phase never completed (e.g. model returned no JSON), emit done
    if not thinking_complete and emit_thinking:
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

    # ── Attempt 0: Gemini 2.0 Flash (fast, smart, no streaming needed) ──
    # Gemini is much smarter than llama3.1:8b for routing.
    # 5s timeout so we don't block if Gemini is slow.
    import asyncio as _asyncio
    try:
        from bantz.llm.gemini import gemini as _gemini
        if _gemini.is_enabled():
            _raw_g = await _asyncio.wait_for(_gemini.chat(messages, temperature=0.1), timeout=5.0)
            if _raw_g and not _is_refusal(_raw_g):
                try:
                    _plan_g = _extract_json(_raw_g)
                    log.info(
                        "Gemini route: route=%s tool=%s conf=%.2f",
                        _plan_g.get("route"), _plan_g.get("tool_name"),
                        _plan_g.get("confidence", 0),
                    )
                    if _plan_g.get("confidence", 0.8) >= confidence_threshold:
                        return _plan_g, None
                except (json.JSONDecodeError, AttributeError):
                    log.warning("Gemini returned non-JSON, falling back to Ollama: %.150s", _raw_g)
    except _asyncio.TimeoutError:
        log.warning("Gemini routing timed out — falling back to Ollama")
    except Exception as _gem_exc:
        log.warning("Gemini routing unavailable (%s) — falling back to Ollama", _gem_exc)

    # ── Attempt 1 (Ollama streaming, fallback) ────────────────────────
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
