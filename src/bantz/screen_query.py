"""
Bantz v2 — Natural-Language Screen Query

Answers "what do you see on the screen?" ("ekranda ne görüyorsun?") and
executes "click that icon" ("şu ikona tıkla") — one screenshot, one
vision call, optionally one verified click.

Three modes, chosen by lightweight regex (no LLM); Turkish trigger forms
are kept in the regexes because they match real user speech:
  describe — "what's on screen" / "ekranda ne var" / "ne görüyorsun"
  click    — "click the …" / "şu ikona tıkla" / "sağdaki butona bas"
  read     — "what does this say" / "bu ne yazıyor"

Routing reaches this through the ``screen_query`` tool registered in
``bantz.tools.screen_query_tool``.
"""
from __future__ import annotations

import logging
import re

from bantz.screen_control import ScreenControl
from bantz.vision_executor import VisionModel

log = logging.getLogger("bantz.screen_query")

_CLICK_RE = re.compile(
    r"\b(click|tıkla|tikla|bas|press)\b", re.IGNORECASE,
)
_READ_RE = re.compile(
    r"\b(what does .{1,40} say|ne yazıyor|ne yaziyor|read (this|that|the))\b",
    re.IGNORECASE,
)

_DESCRIBE_PROMPT = """\
Describe what you see on this screen. Be specific about: open windows and
applications, visible text and titles, and what the user appears to be
doing. Answer in the same language as the user's question.
User asked: "{utterance}"
"""

_READ_PROMPT = """\
The user is asking about something visible on screen: "{utterance}"
Read the relevant text from the screenshot and answer their question
directly, in the same language as the question.
"""

_CLICK_PROMPT = """\
The user wants to: "{utterance}"
The screenshot is {width}x{height} pixels.
Find the UI element they are referring to. Respond ONLY with JSON:
{{"x": <center x>, "y": <center y>, "label": "<what you found>"}}
or {{"x": null, "y": null, "label": null}} if it is not visible.
"""


class ScreenQueryHandler:
    """One-shot screen interactions: describe, read, or click by description."""

    def __init__(
        self,
        screen: ScreenControl | None = None,
        vision: VisionModel | None = None,
    ) -> None:
        self.vision = vision or VisionModel()
        self.screen = screen or ScreenControl()

    @staticmethod
    def classify(utterance: str) -> str:
        """describe | read | click — regex only, no LLM."""
        if _CLICK_RE.search(utterance):
            return "click"
        if _READ_RE.search(utterance):
            return "read"
        return "describe"

    async def handle(self, utterance: str) -> tuple[bool, str]:
        """Run the query; returns (success, user-facing text)."""
        shot = await self.screen.screenshot()
        if shot is None:
            return False, "I couldn't capture the screen."

        mode = self.classify(utterance)
        log.info("screen_query mode=%s: %.60s", mode, utterance)

        if mode == "click":
            return await self._click(utterance, shot)

        prompt_tpl = _READ_PROMPT if mode == "read" else _DESCRIBE_PROMPT
        try:
            answer = await self.vision.ask(
                shot, prompt_tpl.format(utterance=utterance),
            )
        except Exception as exc:
            log.warning("screen_query %s failed: %s", mode, exc)
            return False, f"My vision model is unavailable: {exc}"
        return True, answer.strip()

    async def _click(self, utterance: str, shot: bytes) -> tuple[bool, str]:
        import io
        from PIL import Image
        w, h = Image.open(io.BytesIO(shot)).size
        try:
            coords = await self.vision.ask_json(
                shot, _CLICK_PROMPT.format(utterance=utterance, width=w, height=h),
            )
        except Exception as exc:
            log.warning("screen_query click vision failed: %s", exc)
            return False, f"My vision model is unavailable: {exc}"

        x, y = coords.get("x"), coords.get("y")
        label = coords.get("label") or "the element"
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False, "I couldn't find that element on the screen."

        ok = await self.screen.click(int(x), int(y))
        if ok:
            return True, f"Clicked {label} at ({int(x)}, {int(y)})."
        return False, (f"I found {label} at ({int(x)}, {int(y)}) but the "
                       f"click did not change the screen.")
