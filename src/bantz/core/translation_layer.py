"""
Bantz — Translation Layer (#226)

Extracted from ``brain.py``: language-bridge helpers that run
*before* the intent router sees the user's message.

Public API
----------
- ``get_bridge()``         → cached i18n Bridge instance (or ``None``)
- ``to_en(text)``          → async translate-to-English (no-op when bridge disabled)
- ``resolve_message_ref(text, messages)`` → resolve "the first one" to a message-id
- ``detect_feedback(raw)`` → ``'positive'`` | ``'negative'`` | ``None``
- ``POSITIVE_FEEDBACK_KWS`` / ``NEGATIVE_FEEDBACK_KWS`` — keyword tuples
"""
from __future__ import annotations

import asyncio
import re
from typing import Sequence

import logging

log = logging.getLogger("bantz.translation")

# ── Bridge singleton cache ───────────────────────────────────────────

_bridge_cache: object | None = None  # None = not probed, False = unavailable


def get_bridge():
    """Return the i18n Bridge instance, or *None* if unavailable.

    Lazily imports ``bantz.i18n.bridge`` on first call and caches the
    result so subsequent calls are free.
    """
    global _bridge_cache
    if _bridge_cache is None:
        try:
            from bantz.i18n.bridge import bridge
            _bridge_cache = bridge
        except Exception:
            _bridge_cache = False
    return _bridge_cache or None


async def to_en(text: str) -> str:
    """Translate *text* to English via the i18n bridge.

    Returns the original *text* unchanged when the bridge is disabled
    or unavailable.
    """
    b = get_bridge()
    if b and b.is_enabled():
        try:
            return await asyncio.wait_for(b.to_english(text), timeout=10)
        except (asyncio.TimeoutError, Exception):
            pass
    return text


# ── Message-reference resolution ─────────────────────────────────────

_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1,
    "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4, "last": -1,
}

_SKIP_WORDS = frozenset({
    "read", "that", "this", "the", "one", "email", "mail", "from",
    "about", "please", "can", "you", "want", "open", "show", "check",
})


def resolve_message_ref(
    text: str,
    messages: Sequence[dict],
) -> str | None:
    """Resolve contextual email references like *'the first one'*.

    Parameters
    ----------
    text : str
        The user's (English) input.
    messages : sequence of dict
        Previously fetched messages, each with ``id``, ``from``, ``subject``.
    """
    if not messages:
        return None

    t = text.lower().strip()

    # Ordinals
    for word, idx in _ORDINALS.items():
        if word in t:
            try:
                return messages[idx]["id"]
            except (IndexError, KeyError):
                return None

    # Keyword match against sender / subject
    for msg in messages:
        sender = (msg.get("from") or "").lower()
        subject = (msg.get("subject") or "").lower()
        words = re.findall(r"[a-zA-Z0-9]{3,}", t)
        keywords = [w for w in words if w not in _SKIP_WORDS]
        for kw in keywords:
            if kw in sender or kw in subject:
                return msg["id"]

    # No match — return first message as fallback
    return messages[0]["id"] if messages else None


# ── Sentiment / RLHF feedback detection (#180) ──────────────────────

POSITIVE_FEEDBACK_KWS: tuple[str, ...] = (
    # English — phrases & safe single words
    "good job", "well done", "nice work", "thank you", "thanks a lot",
    "perfect", "great job", "excellent", "brilliant", "love it",
    "amazing", "awesome", "impressed", "bravo", "spot on",
    "nicely done", "good work", "great work", "much appreciated",
    # Turkish
    "aferin", "helal", "harikasın", "süpersin", "çok iyi",
    "teşekkürler", "sağ ol", "güzel iş", "mükemmel", "muhteşem",
    "bravo", "tebrikler", "eyvallah", "eline sağlık", "harika",
)

NEGATIVE_FEEDBACK_KWS: tuple[str, ...] = (
    # English — intent-bearing phrases (not bare "stop" or "bad")
    "you are wrong", "that's wrong", "that is wrong", "not right",
    "bad answer", "terrible answer", "useless answer",
    "you messed up", "not helpful", "completely wrong",
    "wrong answer", "you failed", "this is wrong",
    "are you stupid", "are you dumb", "what a mess",
    # Turkish — intent-bearing phrases
    "hata yapıyorsun", "saçmalama", "yanlış yaptın", "yanlış cevap",
    "berbat cevap", "işe yaramaz", "hiç yardımcı olmadın",
    "hayır yanlış", "olmamış", "ne saçmalıyorsun", "düzelt şunu",
)


def detect_feedback(raw_input: str) -> str | None:
    """Check if the *raw* (untranslated) user input contains explicit feedback.

    Uses regex word boundaries (``\\b``) to prevent false positives like
    'bus stop' triggering the 'stop' keyword.  Takes the RAW input so
    Turkish keywords are matched before the translation layer converts
    them to English.

    Returns ``'positive'``, ``'negative'``, or ``None``.
    Negative is checked first (higher priority).
    """
    lower = raw_input.lower()

    # Check negative FIRST — scolding outranks praise
    for kw in NEGATIVE_FEEDBACK_KWS:
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            return "negative"

    for kw in POSITIVE_FEEDBACK_KWS:
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            return "positive"

    return None
