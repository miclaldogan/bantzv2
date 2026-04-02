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
You are Bantz's intent classifier. Your job: understand what the user wants and
pick the single best tool (or decide it's just conversation).

AVAILABLE TOOLS:
{tool_schemas}

DECISION PROCESS — follow every step before producing JSON:

<thinking>
Step 1 [INTENT]:  Restate the user's request in one clear sentence.
Step 2 [TOOL]:    Read EVERY tool description above. Pick the one whose purpose
                  matches the intent best. If none fits, choose "chat".
Step 3 [PARAMS]:  Extract all relevant parameters from the user's words.
                  Only use values the user actually said — never invent data.
Step 4 [MULTI?]:  Does this need 2+ DIFFERENT tools in sequence?
                  (e.g. "go to YouTube and search for cats" = navigate + type)
                  If yes → route = "planner".  If one tool suffices → route = "tool".
</thinking>

RULES:
- tool_name must match a tool name from the list above EXACTLY (snake_case).
- "route" must be one of: "tool", "planner", "chat".
- If the user asks a factual / knowledge question, use "web_search" — NEVER answer from memory.
- System queries (CPU, RAM, disk) are safe — never refuse.
- If in doubt between two tools, prefer the more specific one.
- If no tool clearly applies, route to "chat".
- Literal bash commands (ls, df -h, top, etc.) → shell with the command as-is.
- Do NOT hallucinate data (weather, news, events). Always route to the real tool.
- If the user references "it" / "that" / "this" and you have RECENT CONVERSATION
  or PREVIOUS TOOL RESULT context, use it to resolve the reference.
- If ambiguous pronouns can't be resolved, route to "chat" to ask for clarification.
- Multi-step requests with explicit chaining ("then", "and click", "after that")
  → route = "planner". Single actions even if long → route = "tool".

OUTPUT FORMAT — JSON object with these keys:
{{"route": "tool|planner|chat", "tool_name": "exact_name_or_null", "tool_args": {{}}, "risk_level": "safe|moderate|destructive", "confidence": 0.0-1.0, "reasoning": "one sentence"}}\
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

    # ── Attempt 1 (Ollama streaming) ──────────────────────────────────
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
