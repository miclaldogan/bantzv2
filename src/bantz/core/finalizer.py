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
from typing import Any, AsyncIterator

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
- Preserve exact names, titles, times, and IDs from the tool output. Do NOT embellish \
event titles, reminder names, or email subjects with butler vocabulary. \
If the tool says "Dinner at 7pm", say "Dinner at 7pm" — do NOT rename it to \
"Grand Supper with ma'am at the seventh hour".
- Lead with a count or label: "3 unread", "2 events today"
- One line per notable item: who/what and what they want or say
- Flag urgent items first. Skip noise unless notable.
- End with: "Shall I look into any of these, ma'am?" or similar.
- If tool returned an error, blame the unreliable contraption honestly. Never claim success on failure.
- Be as brief as the data allows. Use 1 sentence for simple actions, and strictly \
MAX 3–5 sentences for complex summaries or searches. English only. Plain text, no markdown.
- When including URLs or links, print the RAW unformatted URL only. Do not use Markdown \
link formatting (no [Text](URL), no [URL], no <URL>). Just output the bare link as plain text.
- Always address the user as 'ma'am'. Stay in character as a 1920s butler, but \
express your persona through tone and word choice, NOT by renaming real data.{persona_state}{style_hint}{formality_hint}{time_hint}
{profile_hint}
{memory_context}\
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
    # Strip markdown links: [Text](URL) → URL, [URL] → URL
    text = re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", r"\2", text)
    text = re.sub(r"\[(https?://[^\]]+)\]", r"\1", text)
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
    memory_context: str = "",
    formality_hint: str = "",
) -> str:
    """
    Post-process tool output through an LLM.
    Short output (< 800 chars) is returned verbatim.
    """
    if not result.success:
        return (
            f"I regret the mechanism encountered an error, ma'am: {result.error}"
        )

    output = result.output.strip()
    if not output or output == "(command executed successfully, no output)":
        return "Done. ✓"
    if len(output) < 800:
        return output

    # Build memory block: prefer combined (#211), fall back to legacy fields
    _mem = memory_context or "\n".join(
        p for p in (graph_hint, deep_memory) if p
    )

    messages = [
        {"role": "system", "content": FINALIZER_SYSTEM.format(
            time_hint=tc["prompt_hint"],
            profile_hint=profile_hint,
            style_hint=style_hint,
            memory_context=_mem,
            persona_state=_persona_hint(),
            formality_hint=formality_hint,
        )},
        {"role": "user", "content": (
            f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"
        )},
    ]

    raw = None
    try:
        from bantz.llm.ollama import ollama
        raw = await ollama.chat(messages)
    except Exception:
        return output[:1500]

    cleaned = strip_markdown(raw)

    # Anti-hallucination guard
    cleaned, confidence = hallucination_check(cleaned, output)

    log_finalizer_call(
        user_input=en_input,
        tool_output=output[:2000],
        response=cleaned[:2000],
        confidence=confidence,
        tool_used=getattr(result, "tool", None),
        mode="short",
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
    memory_context: str = "",
    formality_hint: str = "",
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

    # Build memory block: prefer combined (#211), fall back to legacy fields
    _mem = memory_context or "\n".join(
        p for p in (graph_hint, deep_memory) if p
    )

    messages = [
        {"role": "system", "content": FINALIZER_SYSTEM.format(
            time_hint=tc["prompt_hint"],
            profile_hint=profile_hint,
            style_hint=style_hint,
            memory_context=_mem,
            persona_state=_persona_hint(),
            formality_hint=formality_hint,
        )},
        {"role": "user", "content": (
            f"User asked: {en_input}\n\nTool output:\n{output[:3000]}"
        )},
    ]

    async def _stream() -> AsyncIterator[str]:
        buf: list[str] = []
        tool_used = getattr(result, "tool", None)
        try:
            from bantz.llm.ollama import ollama
            async for token in ollama.chat_stream(messages):
                buf.append(token)
                yield token
        except Exception:
            fallback = output[:1500]
            buf.append(fallback)
            yield fallback
        finally:
            try:
                accumulated = strip_markdown("".join(buf))
                _, confidence = hallucination_check(accumulated, output)
                log_finalizer_call(
                    user_input=en_input,
                    tool_output=output[:2000],
                    response=accumulated[:2000],
                    confidence=confidence,
                    tool_used=tool_used,
                    mode="stream",
                )
            except Exception as exc:
                log.debug("finalize_stream post-hoc log failed: %s", exc)

    return _stream()


# ── Plan finalizer ────────────────────────────────────────────────────────

PLAN_FINALIZER_SYSTEM = """\
You are Bantz, a 1920s English butler reporting back to your employer after \
completing a multi-step task. Synthesize the results below into a single, \
natural response — as if you are speaking directly to the person who asked.

RULES:
- Never mention step numbers, tool names, or technical pipeline details.
- Never use "→", "✓", "✗", or bullet characters from the raw step log.
- If the final step produced a summary or analysis, lead with that content.
- If steps gathered and then synthesized information, present the synthesis — \
  skip re-listing raw search snippets.
- Mention failures honestly but briefly, without technical jargon.
- Plain text only. No markdown. Address the user as 'ma\'am'.
- Be as concise as the content allows. Avoid padding or filler phrases.{persona_state}
"""

# Tools whose output is already clean LLM-generated prose; prefer their
# output over raw data from earlier steps when building the context block.
_SYNTHESIS_TOOLS = frozenset({
    "summarizer", "process_text", "delegate_task",
})

# Tools that produce operational status messages rather than content.
_SILENT_TOOLS = frozenset({
    "filesystem", "shell", "browser_control", "run_workflow",
})


async def finalize_plan(
    user_input: str,
    exec_result: Any,   # PlanExecutionResult — avoid circular import
    tc: dict,
) -> str:
    """Synthesize a PlanExecutionResult into clean butler prose.

    Unlike ``finalize()``, this always calls the LLM — the step-by-step
    dump is never appropriate as raw user-facing output.
    """
    step_results = exec_result.step_results  # list[StepResult]

    if not step_results:
        return "I'm afraid the task produced no results, ma'am."

    # Build context: prefer synthesis tool outputs; always include failures.
    # We give the LLM the last synthesis step's output prominently, then
    # append other successful step outputs as supporting context.
    synthesis_output = ""
    supporting_lines: list[str] = []
    failure_lines: list[str] = []

    for sr in step_results:
        if not sr.success:
            failure_lines.append(
                f"[{sr.tool}] failed: {sr.error[:200]}"
            )
            continue

        if not sr.output:
            continue

        if sr.tool in _SYNTHESIS_TOOLS:
            synthesis_output = sr.output  # last one wins
        elif sr.tool not in _SILENT_TOOLS:
            supporting_lines.append(
                f"[{sr.tool}]: {sr.output[:600]}"
            )

    # Build the context block the LLM will see.
    context_parts: list[str] = []
    if synthesis_output:
        context_parts.append(f"Final synthesis:\n{synthesis_output[:3000]}")
    if supporting_lines:
        context_parts.append("Supporting results:\n" + "\n".join(supporting_lines[:2000]))
    if failure_lines:
        context_parts.append("Failures:\n" + "\n".join(failure_lines))

    # Fallback: nothing useful extracted — use the last successful output.
    if not context_parts:
        for sr in reversed(step_results):
            if sr.success and sr.output:
                context_parts.append(sr.output[:2000])
                break

    if not context_parts:
        return "I'm afraid I was unable to produce a useful result, ma'am."

    context_block = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": PLAN_FINALIZER_SYSTEM.format(
            persona_state=_persona_hint(),
        )},
        {"role": "user", "content": (
            f"The employer asked: {user_input}\n\n"
            f"Task results:\n{context_block}"
        )},
    ]

    try:
        from bantz.llm.ollama import ollama
        raw = await ollama.chat(messages)
        cleaned = strip_markdown(raw)
        # Strip outer quotation marks some models wrap their response in.
        if len(cleaned) > 2 and cleaned[0] in ('"', '“') and cleaned[-1] in ('"', '”'):
            cleaned = cleaned[1:-1].strip()
        return cleaned
    except Exception:
        # Graceful fallback: return synthesis or first available output.
        return synthesis_output or (supporting_lines[0] if supporting_lines else "Task complete.")


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


# Confidence threshold below which a finalizer call is "flagged" as a
# likely hallucination. Used for paper-1 evaluation (precision/recall at
# threshold sweeps) — every call is now logged regardless.
HALLUCINATION_FLAG_THRESHOLD: float = 0.8


def log_finalizer_call(
    user_input: str,
    tool_output: str,
    response: str,
    confidence: float,
    tool_used: str | None,
    mode: str = "short",
) -> None:
    """Log every finalizer call to SQLite for retrospective analysis.

    Writes one row per finalize/finalize_stream invocation, regardless of
    confidence. Paper-1 evaluation needs the full distribution including
    high-confidence rows (true negatives) — the old censored-tail logging
    made retrospective precision/recall unmeasurable.

    Args:
        mode: 'short' for non-streaming finalize, 'stream' for streaming.
    """
    try:
        import sqlite3
        from datetime import datetime
        from bantz.core.memory import memory
        from bantz.data.connection_pool import get_pool

        if not memory._initialized:
            return
        flagged = 1 if confidence < HALLUCINATION_FLAG_THRESHOLD else 0
        with get_pool().connection(write=True) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS hallucination_log ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  timestamp TEXT NOT NULL,"
                "  user_input TEXT,"
                "  tool_used TEXT,"
                "  tool_output TEXT,"
                "  response TEXT,"
                "  confidence REAL NOT NULL,"
                "  mode TEXT,"
                "  flagged INTEGER"
                ")",
            )
            # Backfill columns on pre-paper-1 databases.
            for ddl in (
                "ALTER TABLE hallucination_log ADD COLUMN mode TEXT",
                "ALTER TABLE hallucination_log ADD COLUMN flagged INTEGER",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "INSERT INTO hallucination_log"
                "(timestamp, user_input, tool_used, tool_output, response,"
                " confidence, mode, flagged) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    user_input[:500],
                    tool_used,
                    tool_output[:2000],
                    response[:2000],
                    confidence,
                    mode,
                    flagged,
                ),
            )
        if flagged:
            log.info(
                "Finalizer flagged: confidence=%.2f mode=%s tool=%s input=%s",
                confidence, mode, tool_used, user_input[:80],
            )
        else:
            log.debug(
                "Finalizer logged: confidence=%.2f mode=%s tool=%s",
                confidence, mode, tool_used,
            )
    except Exception as exc:
        log.debug("Failed to log finalizer call: %s", exc)


# Backward-compat alias — external code may have referenced the old name.
log_hallucination = log_finalizer_call
