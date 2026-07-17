"""Sentence streaming for voice replies (#voice-latency)."""
from __future__ import annotations

import pytest

from bantz.interface.ws_server import _stream_sentences


async def _gen(tokens):
    for t in tokens:
        yield t


async def collect(tokens):
    parts: list = []
    sents = [s async for s in _stream_sentences(_gen(tokens), parts)]
    return sents, "".join(parts)


async def test_sentences_emitted_incrementally():
    sents, full = await collect(
        ["Hello", " there. ", "How are", " you? ", "Good."])
    assert sents == ["Hello there.", "How are you?", "Good."]
    assert full == "Hello there. How are you? Good."


async def test_thinking_blocks_are_skipped_but_kept_in_full_text():
    sents, full = await collect(
        ["<thinking>internal ", "reasoning. steps.</thinking>",
         "The answer ", "is ready. ", "Enjoy."])
    assert sents == ["The answer is ready.", "Enjoy."]
    assert "<thinking>" in full  # UI/history still get the raw stream


async def test_unclosed_thinking_never_spoken():
    sents, _ = await collect(["Sure. ", "<thinking>never ends", " here"])
    assert sents == ["Sure."]


async def test_decimals_do_not_split():
    sents, _ = await collect(["Pi is 3.", "14 roughly. ", "Yes."])
    assert sents == ["Pi is 3.14 roughly.", "Yes."]
