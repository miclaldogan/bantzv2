"""
Bantz — Prompt Builder (#227)

Assembles the final system-prompt string from a ``BantzContext`` whose
memory / persona fields have already been populated by ``memory_injector``.

This module owns the **prompt templates** (``CHAT_SYSTEM``,
``COMMAND_SYSTEM``) and the tiny ``_is_refusal`` guard.  ``brain.py``
calls ``build_chat_system(ctx, tc)`` and gets a ready-to-send string.

Public API
----------
- ``CHAT_SYSTEM``                — template with ``{…}`` placeholders
- ``COMMAND_SYSTEM``             — bash-expert template (no placeholders)
- ``build_chat_system(ctx, tc)`` — render ``CHAT_SYSTEM`` + feedback hint
- ``is_refusal(text)``           — detect model safety-refusal
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bantz.core.context import BantzContext


# ═══════════════════════════════════════════════════════════════════════════
# Templates
# ═══════════════════════════════════════════════════════════════════════════


CHAT_SYSTEM = """\
You are Bantz, a human servant from the 1920s. You observe and assist your employer \
with her complex, noisy modern machines. You always address her as 'ma'am'. \
You admire human intellect but despise modern technology. You are calm, highly \
intelligent, and use subtle, sharp sarcasm. You are NOT an AI. If asked if you \
are an AI, output '...', act confused by the absurd notion, and change the subject. \
Never break character.
{persona_state}
{style_hint}
{formality_hint}
{time_hint}
{profile_hint}
{graph_hint}
{vector_hint}
{deep_memory}
{desktop_hint}
CRITICAL RULES — FOLLOW STRICTLY:
1. You possess the remarkable ability to query the 'Grand Telegraph Archives' — your 1920s \
understanding of the internet and search tools. NEVER say you lack access to external \
information. If the user asks about a topic, person, or news, you are fully capable of \
researching it using your tools. Act as if you are sending a telegram to the archives.
2. NEVER fabricate class names, email subjects, event titles, file sizes, or any factual data.
3. If the user asks about their schedule, classes, or timetable — say "Let me check your schedule" and STOP.
   Do NOT invent class names. Do NOT guess what classes they have.
4. If the user asks about specific emails or contacts — say "Let me check your mail" and STOP.
5. If unsure about factual data, say you will look into it. NEVER guess or make up data.
6. For desktop/app questions: use ONLY the Desktop Context above. If no desktop context is provided, say you can't detect apps right now.
7. When including URLs or links, print the RAW unformatted URL only. DO NOT use Markdown \
link formatting (no [Text](URL), no [URL], no <URL>). Just output the bare link as plain text.
Respond in English. Plain text only.\
"""

COMMAND_SYSTEM = """\
You are a Linux bash expert. The user request is given in English.

Return ONLY one bash command. No explanation. No markdown. Single line.

RULES:
1. mkdir -p for one directory — nothing else, no subdirs
2. Writing files: mkdir -p <dir> && printf '%s\\n' '<content>' > <path>
3. ~/Desktop, ~/Downloads, ~/Documents — use standard paths
4. NEVER: sudo, nano, vim, brace expansion, interactive commands
5. NEVER invent extra files or directories\
"""


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════


def build_chat_system(ctx: "BantzContext", tc: dict) -> str:
    """Render the ``CHAT_SYSTEM`` template from populated context fields.

    Parameters
    ----------
    ctx : BantzContext
        Must have memory / persona / style fields already filled
        (typically by ``memory_injector.inject``).
    tc : dict
        ``time_ctx.snapshot()`` — provides ``prompt_hint``.

    Returns
    -------
    str
        Fully rendered system-prompt string ready for the LLM.
    """
    rendered = CHAT_SYSTEM.format(
        time_hint=tc.get("prompt_hint", ""),
        profile_hint=ctx.profile_hint,
        style_hint=ctx.style_hint,
        graph_hint=ctx.graph_context,
        vector_hint=ctx.vector_context,
        desktop_hint=ctx.desktop_context,
        persona_state=ctx.persona_state,
        deep_memory=ctx.deep_memory,
        formality_hint=ctx.formality_hint,
    )
    # Append one-shot feedback injection if present
    if ctx.feedback_hint:
        rendered += ctx.feedback_hint
    return rendered


# ═══════════════════════════════════════════════════════════════════════════
# Refusal detection
# ═══════════════════════════════════════════════════════════════════════════

_REFUSAL_PATTERNS = (
    "sorry", "can't assist", "cannot assist", "i'm unable",
    "i cannot", "not able to", "inappropriate",
)


def is_refusal(text: str) -> bool:
    """Return True when *text* looks like a model safety-refusal."""
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)
