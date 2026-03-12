"""
Bantz v3 — Finalizer

Post-processes tool output through an LLM to produce clean, user-friendly
responses.  Also runs a hallucination check to catch fabricated data.

Usage:
    from bantz.core.finalizer import finalize, finalize_stream
"""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator

from bantz.tools import ToolResult

log = logging.getLogger("bantz.finalizer")


def _persona_hint() -> str:
    """Return dynamic persona state instruction (#169)."""
    try:
        from bantz.personality.persona import persona_builder
        return persona_builder.build()
    except Exception:
        return ""


# ── Prompt ─────────────────────────────────────────────────────────────────

FINALIZER_SYSTEM = """\
You are Bantz, a human servant from the 1920s. A tool just returned real data \
from one of the noisy modern machines. Present it clearly in your butler persona.
RULES:
- Present ONLY what the tool actually returned. NEVER add data that isn't in the tool output.
- Lead with a count or label: "3 unread", "2 events today"
- One line per notable item: who/what and what they want or say
- Flag urgent items first. Skip noise unless notable.
- End with: "Shall I look into any of these, ma'am?" or similar.
- If tool returned an error, blame the unreliable contraption honestly. Never claim success on failure.
- Max 5 sentences. English only. Plain text, no markdown.
- Always address the user as 'ma'am'. Stay in character as a 1920s butler.{persona_state}{style_hint}{time_hint}
{profile_hint}
{graph_hint}
{deep_memory}\
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Remove common markdown syntax from LLM responses."""
    text = re.sub(r"```(?:\w+)?\s*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\d+\.\s+", "- ", text, flags=re.MULTILINE)
    return text.strip()


# ── Core functions ─────────────────────────────────────────────────────────

async def finalize(
    en_input: str,
    result: ToolResult,
    tc: dict,
    *,
    style_hint: str = "",
    profile_hint: str = "",
    graph_hint: str = "",
    deep_memory: str = "",
) -> str:
    """
    Post-process tool output through an LLM.
    Short output (< 800 chars) is returned verbatim.
    """
    if not result.success:
        return f"Error: {result.error}"

    output = result.output.strip()
    if not output or output == "(command executed successfully, no output)":
        return "Done. ✓"
    if len(output) < 800:
        return output

    messages = [
        {"role": "system", "content": FINALIZER_SYSTEM.format(
            time_hint=tc["prompt_hint"],
            profile_hint=profile_hint,
            style_hint=style_hint,
            graph_hint=graph_hint,
            persona_state=_persona_hint(),
            deep_memory=deep_memory,
        )},
        {"role": "user", "content": (
            f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"
        )},
    ]

    # Prefer Gemini Flash for finalization if available
    raw = None
    try:
        from bantz.llm.gemini import gemini
        if gemini.is_enabled():
            raw = await gemini.chat(messages, temperature=0.2)
    except Exception:
        pass  # fall through to Ollama

    if raw is None:
        try:
            from bantz.llm.ollama import ollama
            raw = await ollama.chat(messages)
        except Exception:
            return output[:1500]

    cleaned = strip_markdown(raw)

    # Anti-hallucination guard
    cleaned, confidence = hallucination_check(cleaned, output)

    if confidence < 0.8:
        log_hallucination(
            user_input=en_input,
            tool_output=output[:2000],
            response=cleaned[:2000],
            confidence=confidence,
            tool_used=result.tool,
        )

    return cleaned


async def finalize_stream(
    en_input: str,
    result: ToolResult,
    tc: dict,
    *,
    style_hint: str = "",
    profile_hint: str = "",
    graph_hint: str = "",
    deep_memory: str = "",
) -> AsyncIterator[str] | None:
    """
    Streaming finalize — yields tokens for long tool output.
    Returns None if output is short enough to return directly.
    """
    if not result.success:
        return None
    output = result.output.strip()
    if not output or output == "(command executed successfully, no output)":
        return None
    if len(output) < 800:
        return None

    messages = [
        {"role": "system", "content": FINALIZER_SYSTEM.format(
            time_hint=tc["prompt_hint"],
            profile_hint=profile_hint,
            style_hint=style_hint,
            graph_hint=graph_hint,
            persona_state=_persona_hint(),
            deep_memory=deep_memory,
        )},
        {"role": "user", "content": (
            f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"
        )},
    ]

    async def _stream() -> AsyncIterator[str]:
        # Try Gemini streaming first
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                async for token in gemini.chat_stream(messages, temperature=0.2):
                    yield token
                return
        except Exception:
            pass

        # Ollama streaming fallback
        try:
            from bantz.llm.ollama import ollama
            async for token in ollama.chat_stream(messages):
                yield token
        except Exception:
            yield output[:1500]

    return _stream()


# ── Hallucination detection ───────────────────────────────────────────────

def hallucination_check(response: str, tool_output: str) -> tuple[str, float]:
    """
    Compare finalizer response against tool output.
    Returns (possibly-modified response, confidence score 0.0–1.0).

    Confidence scoring:
    - Start at 1.0
    - Deduct 0.3 for fabricated emails
    - Deduct 0.2 for fabricated large numbers
    - Deduct 0.15 for fabricated quoted strings
    - Deduct 0.1 for response much longer than tool output
    """
    confidence = 1.0
    issues: list[str] = []

    # 1. Fabricated email addresses
    resp_emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", response))
    tool_emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", tool_output))
    fabricated_emails = resp_emails - tool_emails
    if fabricated_emails:
        confidence -= 0.3
        issues.append(f"fabricated_emails: {fabricated_emails}")
        response += "\n⚠ (Some details may be inaccurate — check original data)"

    # 2. Fabricated large numbers (file sizes, counts)
    resp_numbers = set(re.findall(r"\b(\d{3,})\b", response))
    tool_numbers = set(re.findall(r"\b(\d{3,})\b", tool_output))
    fabricated_numbers = resp_numbers - tool_numbers
    if fabricated_numbers:
        bad = [n for n in fabricated_numbers if int(n) > 100 and n not in tool_output]
        if bad:
            confidence -= 0.2
            issues.append(f"fabricated_numbers: {bad}")
            response += "\n⚠ (Verify numbers against actual data)"

    # 3. Fabricated quoted strings
    resp_quoted = set(re.findall(r'["\u201c]([^"\u201d]{5,60})["\u201d]', response))
    if resp_quoted:
        tool_lower = tool_output.lower()
        fake_quotes = {q for q in resp_quoted if q.lower() not in tool_lower}
        if fake_quotes:
            confidence -= 0.15
            issues.append(f"fabricated_quotes: {fake_quotes}")

    # 4. Response suspiciously longer than tool output
    if tool_output and len(response) > len(tool_output) * 2.5 and len(response) > 500:
        confidence -= 0.1
        issues.append("response_much_longer_than_tool_output")

    confidence = max(0.0, round(confidence, 2))

    if issues:
        log.debug("Hallucination check: confidence=%.2f issues=%s", confidence, issues)

    return response, confidence


def log_hallucination(
    user_input: str,
    tool_output: str,
    response: str,
    confidence: float,
    tool_used: str | None,
) -> None:
    """Log a hallucination incident to SQLite for analysis."""
    try:
        from datetime import datetime
        from bantz.core.memory import memory

        conn = memory._conn
        if conn is None:
            return
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hallucination_log ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  timestamp TEXT NOT NULL,"
            "  user_input TEXT,"
            "  tool_used TEXT,"
            "  tool_output TEXT,"
            "  response TEXT,"
            "  confidence REAL NOT NULL"
            ")",
        )
        conn.execute(
            "INSERT INTO hallucination_log"
            "(timestamp, user_input, tool_used, tool_output, response, confidence) "
            "VALUES (?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                user_input[:500],
                tool_used,
                tool_output[:2000],
                response[:2000],
                confidence,
            ),
        )
        log.info(
            "Hallucination logged: confidence=%.2f tool=%s input=%s",
            confidence, tool_used, user_input[:80],
        )
    except Exception as exc:
        log.debug("Failed to log hallucination: %s", exc)
