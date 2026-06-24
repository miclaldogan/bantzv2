"""
Bantz v2 — Vision-Guided Execution Pipeline

Executes long multi-step desktop tasks the way a human would: look at the
screen, decide the single next action, act, wait for the screen to settle,
repeat.

  goal: "Open Firefox, go to music.youtube.com, play Thunderstruck by AC/DC"
  loop: screenshot → vision model JSON verdict → ScreenControl action → …

Model ladder (local-first):
  1. llama3.2-vision:11b via Ollama  (BANTZ_VISION_MODEL overrides)
  2. gemma4:e4b-it-qat via Ollama    (multimodal QAT — installed locally)
  3. moondream via Ollama            (tiny, weaker reasoning, fast)
  4. Gemini                          (cloud fallback — only if key configured)
  5. Claude                          (cloud fallback — only if key configured)

The vision verdict reuses the routing stack's hard-won defenses: the JSON
is parsed with ``bantz.core.intent._extract_json``-style extraction
(fences/thinking stripped, first balanced object), and a stuck detector
aborts after the same observation repeats.

Usage:
    from bantz.vision_executor import VisionExecutor
    ex = VisionExecutor()
    result = await ex.execute("open YT Music and play Thunderstruck")
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from bantz.config import config
from bantz.llm.ollama import ollama
from bantz.screen_control import ScreenControl

log = logging.getLogger("bantz.vision_executor")

# Local-first priority; entries tried in order, failures skipped. Local
# Ollama models come first — the cloud rungs ("gemini", "claude") only fire
# when every local model fails AND the corresponding API key is configured.
# With no cloud keys set, the system stays fully local and simply errors
# upward if no local vision model is pulled.
VISION_MODEL_PRIORITY: list[str] = [
    "llama3.2-vision:11b",
    "gemma4:e4b-it-qat",   # multimodal QAT build — installed locally
    "moondream",
    "gemini",
    "claude",
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

VISION_STEP_PROMPT = """\
You are controlling a Linux desktop to complete a task.

GOAL: {goal}
STEP: {step} / {max_steps}
FOCUSED WINDOW (compositor ground truth — trust this over the pixels):
{active_window}
COMPLETED ACTIONS SO FAR:
{history}

Look at the screenshot carefully. Identify:
1. What application/window is currently focused?
2. What is the current state relevant to the goal?
3. What is the ONE next action needed?

HARD RULES:
- If FOCUSED WINDOW is not the goal's application, your FIRST action is
  to fix that: click squarely inside the goal application's window (its
  center, not its edge). NEVER type or press shortcuts while the wrong
  window is focused — the keystrokes land in the wrong application.
- Allowed "key" values are ONLY: ctrl+l (browser address bar), Return,
  Escape, space (play/pause on media sites), tab, up/down, page_up/
  page_down. Every other shortcut (ctrl+f, ctrl+w, ctrl+t, alt+…) is
  BLOCKED by the harness — do not propose them.
- NEVER type conversational or goal text into a terminal, chat box, code
  editor, or this assistant's own window. Typing text is ONLY for search
  bars, address bars, and form fields of the goal's target application.
- "done" is allowed ONLY when the goal's end state is VISIBLE in the
  screenshot (e.g. the song actually playing in the player UI). Text in a
  terminal that merely TALKS about the goal is NOT completion.
- If the same screen appears twice in a row, do something DIFFERENT from
  the previous action.

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "observation": "brief description of what you see",
  "next_action": "click" | "type" | "scroll" | "key" | "wait" | "done" | "failed",
  "target_description": "UI element to interact with, or null",
  "target_coords": [x, y] or null,
  "value": "text to type or key combo to press, or null",
  "confidence": 0.0-1.0,
  "reason": "why this action"
}}

If the goal is fully achieved, use next_action="done".
If you are stuck or the goal is impossible, use next_action="failed".
"""

_LOCATE_PROMPT = """\
Find this UI element on the screenshot: "{description}"
The screenshot is {width}x{height} pixels.
Respond ONLY with JSON: {{"x": <center x>, "y": <center y>}} — or
{{"x": null, "y": null}} if the element is not visible.
"""


@dataclass
class VisionStep:
    step: int
    observation: str
    action: str
    target: str | None
    coords: tuple[int, int] | None
    value: str | None
    ok: bool
    elapsed_s: float


@dataclass
class ExecutionResult:
    success: bool
    goal: str
    steps: list[VisionStep] = field(default_factory=list)
    error: str = ""
    model_used: str = ""

    def summary(self) -> str:
        outcome = "completed" if self.success else f"failed ({self.error})"
        acted = ", ".join(f"{s.step}:{s.action}" for s in self.steps)
        return f"Goal {outcome} in {len(self.steps)} steps [{acted}]"


class VisionModel:
    """Vision-capable model with the local-first fallback ladder.

    ``ask`` sends one PNG + prompt and returns raw text; ``ask_json``
    additionally extracts the first JSON object (fence/prose tolerant —
    the same F1 lesson as the routing stack).
    """

    def __init__(self, priority: list[str] | None = None) -> None:
        env_model = getattr(config, "vision_model", "") or ""
        self.priority = ([env_model] if env_model else []) + (
            priority or VISION_MODEL_PRIORITY
        )
        self._active: str | None = None  # pinned after first success
        self._installed: set[str] | None = None  # ollama tags, probed once

    async def _installed_ollama(self) -> set[str]:
        """Names of locally-pulled Ollama models (probed once, cached).

        The default ladder lists ``llama3.2-vision:11b`` first, but most
        machines have not pulled it — without this filter every fresh
        process burns a 404 round-trip (and a scary warning) before it
        falls through to an installed model. Stores both the full tag and
        the bare name so ``"moondream"`` matches ``"moondream:latest"``.
        Empty set on probe failure → caller does not filter (try them all).
        """
        if self._installed is not None:
            return self._installed
        names: set[str] = set()
        try:
            resp = await ollama.client.get(
                f"{ollama.base_url}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    n = m.get("name", "")
                    if n:
                        names.add(n)
                        names.add(n.split(":")[0])
        except Exception as exc:  # pragma: no cover - network dependent
            log.debug("ollama tags probe failed: %s", exc)
        self._installed = names
        return names

    async def ask(self, image_png: bytes, prompt: str) -> str:
        img_b64 = base64.b64encode(image_png).decode()
        candidates = [self._active] if self._active else self.priority
        installed = await self._installed_ollama()
        last_exc: Exception | None = None
        tried = False
        for model in candidates:
            # Skip Ollama models that were never pulled (cloud rungs always
            # tried — their availability is gated by an API key, not tags).
            if (model not in ("gemini", "claude") and installed
                    and model not in installed):
                log.debug("vision model %s not installed — skipping", model)
                continue
            tried = True
            try:
                if model == "gemini":
                    raw = await self._ask_gemini(img_b64, prompt)
                elif model == "claude":
                    raw = await self._ask_claude(img_b64, prompt)
                else:
                    raw = await self._ask_ollama(model, img_b64, prompt)
                self._active = model
                return raw
            except Exception as exc:
                last_exc = exc
                log.warning("vision model %s failed: %s — trying next", model, exc)
                self._active = None
        if not tried:
            raise RuntimeError(
                "no vision model available: none of "
                f"{self.priority} is installed in Ollama and no cloud "
                "vision key (Gemini/Claude) is configured. Pull one, e.g. "
                "`ollama pull moondream`.")
        raise RuntimeError(f"all vision models failed: {last_exc}")

    async def ask_json(self, image_png: bytes, prompt: str) -> dict:
        raw = await self.ask(image_png, prompt)
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)
        m = _JSON_RE.search(text)
        return json.loads(m.group() if m else text)

    async def _ask_ollama(self, model: str, img_b64: str, prompt: str) -> str:
        # Ollama vision: base64 images ride on the user message.
        # Native thinking OFF for vision verdicts — on gemma4 the thinking
        # field otherwise eats the whole num_predict budget and content
        # comes back empty (only sent to models that support the flag).
        from bantz.core.intent import _supports_native_think
        think = False if _supports_native_think(model) else None
        return await ollama.chat(
            [{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"temperature": 0, "num_predict": 512},
            model_override=model,
            think=think,
        )

    async def _ask_gemini(self, img_b64: str, prompt: str) -> str:
        from bantz.llm.gemini import gemini
        if not gemini.is_enabled():
            raise RuntimeError("Gemini fallback not enabled")
        return await gemini.chat(
            [{"role": "user", "content": prompt, "images": [img_b64]}],
            temperature=0.0,
        )

    async def _ask_claude(self, img_b64: str, prompt: str) -> str:
        from bantz.llm.anthropic_client import claude
        if not claude.is_enabled():
            raise RuntimeError("Claude fallback not enabled (BANTZ_ANTHROPIC_API_KEY empty)")
        return await claude.chat(
            [{"role": "user", "content": prompt, "images": [img_b64]}],
        )

    @property
    def active_model(self) -> str:
        return self._active or "none"


class VisionExecutor:
    """Screenshot → decide → act loop for long multi-step desktop tasks."""

    STUCK_LIMIT = 3          # identical observations before aborting
    MIN_CONFIDENCE = 0.3     # below this, treat the verdict as "failed"

    def __init__(
        self,
        screen: ScreenControl | None = None,
        vision: VisionModel | None = None,
    ) -> None:
        self.vision = vision or VisionModel()
        self.screen = screen or ScreenControl(vision_locator=self._locate)

    # ── vision locator injected into ScreenControl.find_element ─────────

    async def _locate(self, description: str, shot: bytes) -> tuple[int, int] | None:
        from PIL import Image
        import io as _io
        w, h = Image.open(_io.BytesIO(shot)).size
        try:
            coords = await self.vision.ask_json(
                shot, _LOCATE_PROMPT.format(description=description,
                                            width=w, height=h),
            )
        except Exception as exc:
            log.warning("locate %r failed: %s", description, exc)
            return None
        x, y = coords.get("x"), coords.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return int(x), int(y)
        return None

    # ── main loop ────────────────────────────────────────────────────────

    # Keys the vision loop may press. A 4-8B VLM WILL eventually reach for
    # ctrl+f / ctrl+w / random shortcuts on the user's live session —
    # whitelist, don't trust. space = play/pause on media sites.
    ALLOWED_KEYS = {
        "ctrl+l", "return", "enter", "escape", "esc", "space",
        "tab", "down", "up", "page_down", "page_up", "pagedown", "pageup",
    }

    async def execute(
        self, goal: str, context: dict[str, Any] | None = None,
        max_steps: int = 15,
        verifier: Any = None,
    ) -> ExecutionResult:
        """Run the step loop until done/failed/max_steps.

        *context* (e.g. memory preferences) is folded into the goal text so
        the vision model sees resolved values, never placeholders.

        *verifier*: optional async callable ``() -> bool | str``. Called
        when the model claims "done". True accepts; False or a string
        REJECTS the claim — the string is fed back into the loop history as
        corrective context and the loop continues. VLM self-reports of
        completion hallucinate (measured: "player bar visible" on a search
        results page); a deterministic check beats the model's own word.
        """
        if context:
            ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
            goal = f"{goal}\n\nKNOWN CONTEXT (use these exact values):\n{ctx_lines}"

        result = ExecutionResult(success=False, goal=goal)
        history: list[str] = []
        recent_observations: list[str] = []

        # Per-step screenshot archive — the user can replay exactly what
        # the model saw at each decision (/tmp/bantz_vision/<run>/).
        import os
        shots_dir = os.path.join(
            "/tmp", "bantz_vision", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(shots_dir, exist_ok=True)
        log.info("vision run screenshots → %s", shots_dir)

        for step_no in range(1, max_steps + 1):
            t0 = time.monotonic()
            shot = await self.screen.screenshot()
            if shot is None:
                result.error = "screenshot unavailable"
                return result
            try:
                with open(os.path.join(shots_dir, f"step_{step_no:02d}.png"),
                          "wb") as f:
                    f.write(shot)
            except OSError:
                pass

            active = await self.screen.active_window()
            prompt = VISION_STEP_PROMPT.format(
                goal=goal, step=step_no, max_steps=max_steps,
                active_window=f"  {active}" if active else "  (unknown)",
                history="\n".join(history) or "  (none yet)",
            )
            try:
                verdict = await self.vision.ask_json(shot, prompt)
            except Exception as exc:
                result.error = f"vision verdict failed: {exc}"
                result.model_used = self.vision.active_model
                return result

            action = str(verdict.get("next_action") or "").lower()
            observation = str(verdict.get("observation") or "")[:200]
            confidence = float(verdict.get("confidence") or 0.0)
            target = verdict.get("target_description")
            value = verdict.get("value")
            coords = verdict.get("target_coords")
            if isinstance(coords, list) and len(coords) == 2:
                coords = (int(coords[0]), int(coords[1]))
            else:
                coords = None

            log.info("vision step %d: %s conf=%.2f obs=%.80s",
                     step_no, action, confidence, observation)

            # Terminal verdicts
            if action == "done":
                if verifier is not None:
                    try:
                        v = await verifier()
                    except Exception as exc:
                        log.warning("done-verifier errored (accepting): %s", exc)
                        v = True
                    if v is not True:
                        msg = v if isinstance(v, str) else (
                            "the goal's end state is not actually reached")
                        log.info("vision step %d: done REJECTED — %s",
                                 step_no, msg)
                        history.append(
                            f"  {step_no}. done claim REJECTED by verifier: "
                            f"{msg}")
                        recent_observations.clear()
                        continue
                result.steps.append(VisionStep(
                    step_no, observation, action, None, None, None, True,
                    round(time.monotonic() - t0, 2)))
                result.success = True
                result.model_used = self.vision.active_model
                return result
            if action == "failed" or confidence < self.MIN_CONFIDENCE:
                result.error = (verdict.get("reason") or "model reported failure"
                                if action == "failed"
                                else f"confidence {confidence:.2f} below floor")
                result.model_used = self.vision.active_model
                return result

            # Stuck detection: same observation N times in a row
            recent_observations.append(observation.lower())
            if len(recent_observations) >= self.STUCK_LIMIT and len(set(
                    recent_observations[-self.STUCK_LIMIT:])) == 1:
                result.error = f"stuck: screen unchanged for {self.STUCK_LIMIT} steps"
                result.model_used = self.vision.active_model
                return result

            ok = await self._act(action, coords, target, value, shot)
            result.steps.append(VisionStep(
                step_no, observation, action, target, coords,
                str(value) if value is not None else None, ok,
                round(time.monotonic() - t0, 2)))
            history.append(
                f"  {step_no}. {action}"
                + (f" → {target}" if target else "")
                + (f" [{value}]" if value else "")
                + ("" if ok else " (FAILED)"))

            # Settle: longer after actions that open windows/apps
            launchy = action == "key" and value and "super" in str(value).lower()
            await self.screen.wait_for_stable(
                timeout=3.0 if not launchy else 5.0)

        result.error = f"max_steps ({max_steps}) reached"
        result.model_used = self.vision.active_model
        return result

    # ── single action dispatch ───────────────────────────────────────────

    async def _act(
        self, action: str, coords: tuple[int, int] | None,
        target: str | None, value: Any, shot: bytes,
    ) -> bool:
        if action == "click":
            if coords is None and target:
                coords = await self.screen.find_element(str(target), shot)
            if coords is None:
                log.warning("click without resolvable target (%r)", target)
                return False
            return await self.screen.click(*coords)
        if action == "type":
            return await self.screen.type_text(str(value or ""))
        if action == "key":
            combo = str(value or "Return").strip().lower()
            if combo not in self.ALLOWED_KEYS:
                log.warning("key %r blocked (not in ALLOWED_KEYS)", combo)
                return False
            return await self.screen.key(str(value or "Return"))
        if action == "scroll":
            x, y = coords or (960, 540)
            direction = str(value or "down")
            return await self.screen.scroll(x, y, direction)
        if action == "wait":
            return await self.screen.wait_for_stable(timeout=3.0)
        log.warning("unknown vision action %r", action)
        return False
