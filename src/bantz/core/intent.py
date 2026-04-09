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

# Compact routing hints — one line per tool keeps the prompt small for 8b models.
# Full descriptions live on the tool classes; these are routing-only summaries.
_ROUTING_HINTS: dict[str, str] = {
    "web_search": "Search the internet for ANY factual/knowledge question or lookup",
    "calendar": "Google Calendar: view, create, update, delete events and meetings. action=create for 'add/schedule/create event/dinner/meeting'. action=today for 'what's on my calendar'. action=delete for 'remove/cancel event'.",
    "gmail": "Gmail: read inbox, compose, send, search, reply, forward emails. Use for ANY mention of email/mail/inbox/mails. action=summary for 'check my mail'. NEVER answer email questions from memory.",
    "weather": "Current weather and forecast for a city or user's location",
    "reminder": "Create, list, cancel, snooze reminders and timers. action=add for new reminders. action=cancel with id=N for 'delete/cancel/remove reminder #N'. action=list for 'show my reminders'.",
    "shell": "Run a bash command (ls, df, top, apt, etc.)",
    "system": "Live system metrics: CPU, RAM, disk usage, uptime",
    "filesystem": "Read, write, list files and folders under home directory",
    "browser_control": "Launch apps AND control browser. Actions: open, navigate, find_and_click, type_in_element, hotkey, type, scroll, screenshot, wait_for_load. Web services (YouTube, Spotify, Netflix…) = action=open. IMPORTANT: 'play/search/find X on YouTube/YT Music' needs MULTIPLE steps (search → wait → click) → route=planner.",
    "visual_click": "Click any visible UI element on screen by describing it",
    "screenshot": "Capture screen image (action=capture) OR analyze what's visible on screen (action=analyze). 'what do you see/what's on my screen' = action=analyze. 'take a screenshot/ss al' = action=capture.",
    "news": "Fetch and summarize latest news headlines",
    "document": "Read and summarize documents: PDF, DOCX, TXT, MD, CSV",
    "read_url": "Read full text content of a specific URL",
    "contacts": "Search Google contacts (phone, email)",
    "classroom": "Google Classroom: courses, assignments, announcements",
    "summarizer": "Summarize, analyze, rewrite, or transform text",
    "input_control": "Low-level mouse/keyboard with exact coordinates or key combos",
    "accessibility": "Query OS accessibility tree to inspect running apps (read-only)",
    "gui_action": "Interact with a specific labeled UI element in a desktop app",
    "computer_use": "Autonomous multi-step desktop automation using screen vision",
    "browser": "Advanced web page parsing: HTML, CSS selectors, image extraction",
    "feed": "Fetch and parse RSS/Atom feeds. action=fetch url=<feed_url> for a direct URL, action=category category=<name> for a group (tech, world, tr_news, science, gaming), action=list to show available categories.",
}


def _build_compact_schemas(tool_schemas: list[dict]) -> str:
    """One-line-per-tool schema string for the routing prompt."""
    lines = []
    for t in tool_schemas:
        name = t["name"]
        hint = _ROUTING_HINTS.get(name, t["description"][:80])
        lines.append(f"  {name}: {hint}")
    return "\n".join(lines)


COT_SYSTEM = """\
You are Bantz's intent router. Pick the best tool for the user's request.

TOOLS:
{tool_schemas}

<thinking>
1. What does the user want? (one sentence)
2. Which tool handles this? Pick the BEST match.
3. Extract parameters from the user's words.
4. Needs 2+ different tools in sequence? → route="planner".
</thinking>

RULES:
- ALWAYS pick a tool when the user wants something DONE. Only route to "chat" for greetings, casual chitchat, or pure opinions.
- Factual / knowledge questions → web_search. NEVER answer from memory.
- "check my mail/email/inbox" → gmail action=summary. NEVER hallucinate email content.
- "add/create/schedule event/dinner/meeting at TIME" → calendar action=create with title and time. NOT action=today.
- "delete/cancel/remove reminder #N" or "delete the no N reminder" → reminder action=cancel id=N. The tool name is "reminder" NOT "cancel_reminder".
- "remind me in X minutes/a minute" → reminder action=add. Parse time carefully: "a minute" = 1 minute, NOT 1 hour.
- "just open YouTube/Spotify/Netflix" (no specific content) → browser_control action=open app=firefox url=https://youtube.com
- "open Gemini/ChatGPT/Claude" or any WEB APP → browser_control action=navigate url=<correct URL>. Known web apps: gemini=https://gemini.google.com, chatgpt=https://chatgpt.com, claude=https://claude.ai, perplexity=https://perplexity.ai, github=https://github.com, reddit=https://reddit.com
- "play/search/find/watch X on YouTube" or "listen to X on YT Music" → route="planner" (needs search + wait + click steps).
- "play X" / "listen to X" (music intent, no site specified) → route="planner" (open YT Music + search + click).
- "search X, do research, give detailed summary" / "research X and write a report" → route="planner" (needs web_search + read_url + summarizer, possibly filesystem).
- Creating events/meetings/dinners → calendar. Reminders/timers → reminder.
- Literal bash commands (ls, df, top) → shell.
- Multi-step with "then" / "and" / "after that" → route="planner".
- Requests involving "research", "detailed summary", "write to file", or "in-depth analysis" → route="planner".
- Do NOT hallucinate data — always route to the real tool.
- browser_control action names are EXACT: find_and_click (not 'click'), navigate (not 'navigate_to'), type_in_element (not 'type_in').
- Tool name must be EXACT registry name. Never invent tool names like "cancel_reminder" or "delete_event". Use the base tool with the right action param.

OUTPUT — single JSON, no markdown:
{{"route": "tool|planner|chat", "tool_name": "exact_name", "tool_args": {{}}, "risk_level": "safe|moderate|destructive", "confidence": 0.0-1.0, "reasoning": "one sentence"}}\
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


_VALID_ROUTES = frozenset({"tool", "planner", "chat"})

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from *text*, ignoring markdown fences and thinking blocks.

    Also normalises common llama3.1 mistakes:
    - route field contains a tool name instead of "tool"/"planner"/"chat"
    """
    text = strip_thinking(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group() if m else text)

    # Fix: if 'route' is missing or not one of the valid values, the model likely
    # put the tool name there. Move it to tool_name and set route="tool".
    if isinstance(data, dict) and data.get("route") not in _VALID_ROUTES:
        wrong_route = data.get("route")
        log.debug("_extract_json: normalising route=%r → route='tool'", wrong_route)
        if isinstance(wrong_route, str) and wrong_route:
            data["tool_name"] = data.get("tool_name") or wrong_route
        data["route"] = "tool"

    return data


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
_THINKING_MAX_TOKENS = 512

# Ollama generation options for routing — cap output length for speed.
# 768 tokens gives enough room for ~4-line <thinking> + JSON output.
# Prevents llama3.1:8b from generating thousands of reasoning tokens.
_ROUTING_OPTIONS: dict = {"num_predict": 768}


async def _stream_and_collect(
    messages: list[dict],
    *,
    emit_thinking: bool = True,
    source: str = "cot_route",
    options: dict | None = None,
    model_override: str = "",
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
        async for token in ollama.chat_stream(messages, options=options, model_override=model_override):
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
    schema_str = _build_compact_schemas(tool_schemas)

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

    # ── Attempt 1 (Ollama streaming) ──────────────────────────────────
    try:
        raw = await _stream_and_collect(
            messages, emit_thinking=True, source="cot_route",
            options=_ROUTING_OPTIONS, model_override=ollama.routing_model,
        )

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
        # If the routing model is missing/broken, retry once with main model
        err_lower = str(exc).lower()
        if any(hint in err_lower for hint in ("404", "not found", "model")):
            log.warning(
                "Routing model '%s' failed (%s) — retrying with main model '%s'",
                ollama.routing_model, exc, ollama.model,
            )
            try:
                raw = await _stream_and_collect(
                    messages, emit_thinking=True, source="cot_route_fallback",
                    options=_ROUTING_OPTIONS, model_override=ollama.model,
                )
                if not _is_refusal(raw):
                    _log_thinking(raw)
                    plan = _extract_json(raw)
                    log.info("CoT fallback parsed: route=%s tool=%s conf=%.2f",
                             plan.get("route"), plan.get("tool_name"),
                             plan.get("confidence", 0))
                    if plan.get("confidence", 0.8) >= confidence_threshold:
                        # Fix routing_model for subsequent calls this session
                        ollama.routing_model = ollama.model
                        return plan, None
            except Exception as fb_exc:
                log.warning("CoT fallback also failed: %s", fb_exc)
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
            options=_ROUTING_OPTIONS, model_override=ollama.routing_model,
        )
        _log_thinking(raw2, tag="retry")
        plan = _extract_json(raw2)
        return plan, None

    except Exception as exc:
        log.warning("CoT parse failed (attempt 2): %s", exc)
        return None, f"Intent routing failed after 2 attempts: {exc}"
