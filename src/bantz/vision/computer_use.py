"""
Bantz — Autonomous VLM Web/Desktop Navigation Loop (#188)

Provides a closed-loop Observe → Analyze → Act → Verify cycle that chains
VLM perception with input control to complete multi-step visual tasks.

Classes exported:
    VisionGoal          — task specification with safety limits
    VisionStep          — record of one Observe-Analyze-Act cycle
    VisionLoopResult    — final result of the whole loop
    ActionResult        — outcome of executing one action
    AutonomousVisionLoop — the main loop engine
    WebNavigationMacros — pre-built shortcut sequences

Usage:
    from bantz.vision.computer_use import AutonomousVisionLoop, VisionGoal

    loop = AutonomousVisionLoop()
    goal = VisionGoal(
        description="Open Firefox and navigate to wikipedia.org",
        success_criteria="Wikipedia homepage is visible",
    )
    result = await loop.execute(goal)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("bantz.computer_use")

# ── Safety constants ──────────────────────────────────────────────────────────

_STUCK_THRESHOLD = 3          # identical screenshots in a row → abort
_DESTRUCTIVE_HOTKEYS = frozenset({
    "ctrl+w", "alt+f4", "ctrl+alt+delete",
    "super+shift+q", "ctrl+shift+w",
})

# Butler–Gemini rivalry lines
_VLM_INVOKED_LINES = [
    "I shall… reluctantly… consult the Loquacious American Calculating Machine. "
    "One does hope it will be brief, though hope springs eternal.",
    "Very well, sir. Engaging the Transatlantic Oracle. I make no promises regarding verbosity.",
    "The Machine's optical faculties are adequate — if one overlooks the verbose tendencies.",
]
_VLM_SUCCESS_LINES = [
    "The Machine performed adequately, I suppose. Even a broken clock.",
    "I shan't say I expected competence, but competence was delivered. Adequate.",
]
_VLM_FAILURE_LINES = [
    "I shan't say 'I told you so,' sir, but I did harbour reservations regarding the Machine's optical faculties.",
    "The Transatlantic Oracle has, regrettably, proven itself less than oracular.",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VisionGoal:
    """Specification of an autonomous visual navigation task."""
    description: str
    success_criteria: str
    max_steps: int = 15
    timeout_s: float = 120.0
    allowed_domains: list[str] = field(default_factory=list)


@dataclass
class VisionStep:
    """Record of one Observe → Analyze → Act cycle."""
    step_num: int
    observation: str
    reasoning: str
    action: str           # click | type | scroll | hotkey | wait | read | done
    target: str
    args: dict
    screenshot_b64: str
    success: bool = False
    error: str = ""


@dataclass
class VisionLoopResult:
    """Final outcome of AutonomousVisionLoop.execute()."""
    success: bool
    steps: list[VisionStep]
    final_observation: str
    extracted_text: str
    total_time_s: float
    error: str = ""


@dataclass
class ActionResult:
    """Outcome of executing a single action."""
    success: bool
    message: str = ""
    error: str = ""


# ── Autonomous Vision Loop ────────────────────────────────────────────────────

class AutonomousVisionLoop:
    """Closed-loop Observe → Analyze → Act → Verify visual navigation engine.

    Dependency injection via constructor: pass mock vlm_fn / input_ctrl / llm_fn
    in tests; in production the module-level ``vision_loop`` singleton uses real
    implementations (lazy-imported so no hard runtime dependency).

    Args:
        vlm_fn:     async (image_b64, prompt) -> str  — describes the screen
        llm_fn:     async (prompt) -> str             — decides next action
        input_ctrl: object with async click/type_text/scroll/hotkey methods
        navigator:  object with async execute_action method
    """

    def __init__(
        self,
        vlm_fn: Callable[..., Awaitable[str]] | None = None,
        llm_fn: Callable[..., Awaitable[str]] | None = None,
        input_ctrl: Any = None,
        navigator: Any = None,
    ) -> None:
        self._vlm_fn = vlm_fn
        self._llm_fn = llm_fn
        self._input_ctrl = input_ctrl
        self._navigator = navigator

    # ── Public API ─────────────────────────────────────────────────────────

    async def execute(
        self,
        goal: VisionGoal,
        on_step: Callable[[VisionStep], Awaitable[None]] | None = None,
    ) -> VisionLoopResult:
        """Run the Observe → Analyze → Act → Verify loop until goal met or abort.

        Args:
            goal:    What to accomplish and when to stop.
            on_step: Optional async callback called after each VisionStep.

        Returns:
            VisionLoopResult with success flag, step history, and extracted text.
        """
        start = time.monotonic()
        steps: list[VisionStep] = []
        history_summaries: list[str] = []
        extracted_text = ""
        _last_hashes: list[str] = []
        _finished_by_done = False

        log.info("AutonomousVisionLoop: starting — %s", goal.description)
        _butler_line = _VLM_INVOKED_LINES[len(goal.description) % len(_VLM_INVOKED_LINES)]
        log.info("Butler: %s", _butler_line)

        for step_num in range(1, goal.max_steps + 1):
            elapsed = time.monotonic() - start
            if elapsed >= goal.timeout_s:
                log.warning("Vision loop timeout after %.1fs", elapsed)
                return VisionLoopResult(
                    success=False,
                    steps=steps,
                    final_observation="Timeout reached.",
                    extracted_text=extracted_text,
                    total_time_s=elapsed,
                    error="timeout",
                )

            # 1. OBSERVE
            screenshot_b64 = await self._observe()
            if not screenshot_b64:
                log.warning("Vision loop: screenshot failed at step %d", step_num)
                break

            # Stuck detection
            img_hash = hashlib.md5(screenshot_b64.encode()).hexdigest()
            _last_hashes.append(img_hash)
            if len(_last_hashes) > _STUCK_THRESHOLD:
                _last_hashes.pop(0)
            if len(_last_hashes) == _STUCK_THRESHOLD and len(set(_last_hashes)) == 1:
                log.warning("Vision loop: stuck — %d identical screenshots", _STUCK_THRESHOLD)
                return VisionLoopResult(
                    success=False,
                    steps=steps,
                    final_observation="Loop stuck — screen not changing.",
                    extracted_text=extracted_text,
                    total_time_s=time.monotonic() - start,
                    error="stuck",
                )

            # 2. ANALYZE — VLM describes screen, LLM decides action
            try:
                action_dict = await self._decide_action(goal, screenshot_b64, history_summaries)
            except Exception as exc:
                log.debug("_decide_action error: %s", exc)
                action_dict = {"action": "wait", "target": "", "args": {},
                               "reasoning": "decision error", "done": False}

            observation = action_dict.get("observation", "")
            reasoning = action_dict.get("reasoning", "")
            action_name = action_dict.get("action", "wait")
            target = action_dict.get("target", "")
            args = action_dict.get("args", {})
            done = action_dict.get("done", False)

            # 3. ACT
            vs = VisionStep(
                step_num=step_num,
                observation=observation,
                reasoning=reasoning,
                action=action_name,
                target=target,
                args=args,
                screenshot_b64=screenshot_b64,
            )

            if done or action_name == "done":
                vs.success = True
                extracted_text = action_dict.get("extracted_text", "")
                steps.append(vs)
                if on_step:
                    try:
                        await on_step(vs)
                    except Exception:
                        pass
                _finished_by_done = True
                break

            if action_name == "read":
                # Extract text from current screen
                extracted_text += await self._read_screen(screenshot_b64) + "\n"
                vs.success = True
            else:
                ar = await self._execute_action(action_dict, goal)
                vs.success = ar.success
                vs.error = ar.error

            steps.append(vs)
            history_summaries.append(
                f"Step {step_num}: {action_name} {target!r} — {'ok' if vs.success else vs.error}"
            )

            if on_step:
                try:
                    await on_step(vs)
                except Exception:
                    pass

            # 4. VERIFY — ask VLM if goal is met
            goal_met, final_obs = await self._verify_goal(goal, screenshot_b64)
            if goal_met:
                log.info("Vision loop: goal met at step %d", step_num)
                log.info("Butler: %s", _VLM_SUCCESS_LINES[step_num % len(_VLM_SUCCESS_LINES)])
                return VisionLoopResult(
                    success=True,
                    steps=steps,
                    final_observation=final_obs,
                    extracted_text=extracted_text,
                    total_time_s=time.monotonic() - start,
                )

        # Done action broke the loop → success
        if _finished_by_done:
            log.info("Butler: %s", _VLM_SUCCESS_LINES[len(steps) % len(_VLM_SUCCESS_LINES)])
            return VisionLoopResult(
                success=True,
                steps=steps,
                final_observation=steps[-1].observation if steps else "",
                extracted_text=extracted_text,
                total_time_s=time.monotonic() - start,
            )

        # Max steps reached
        final_obs = "Max steps reached without achieving goal."
        log.info("Butler: %s", _VLM_FAILURE_LINES[0])
        return VisionLoopResult(
            success=False,
            steps=steps,
            final_observation=final_obs,
            extracted_text=extracted_text,
            total_time_s=time.monotonic() - start,
            error="max_steps_reached",
        )

    # ── Internal ───────────────────────────────────────────────────────────

    async def _observe(self) -> str | None:
        """Capture a screenshot and return base64 string."""
        try:
            from bantz.vision import screenshot as _ss
            b64 = await _ss.capture_base64()
            return b64
        except Exception as exc:
            log.debug("_observe error: %s", exc)
            return None

    async def _decide_action(
        self,
        goal: VisionGoal,
        screenshot_b64: str,
        history: list[str],
    ) -> dict:
        """Ask VLM/LLM to decide the next action.

        Returns a dict:
            {"action": str, "target": str, "args": dict,
             "reasoning": str, "observation": str, "done": bool}
        """
        history_text = "\n".join(f"{i+1}. {h}" for i, h in enumerate(history)) or "None"
        prompt = (
            f"You are a desktop automation agent.\n"
            f"Goal: {goal.description}\n"
            f"Success criteria: {goal.success_criteria}\n\n"
            f"Previous actions:\n{history_text}\n\n"
            f"Look at the current screenshot and decide the next action.\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"observation": "...", "reasoning": "...", '
            f'"action": "click|type|scroll|hotkey|wait|read|done", '
            f'"target": "element label or empty", '
            f'"args": {{}}, "done": false, "extracted_text": ""}}\n\n'
            f"Available actions: click, type, scroll, hotkey, wait, read, done\n"
            f"NEVER type passwords or sensitive data."
        )

        raw = await self._call_llm_with_vision(screenshot_b64, prompt)
        return self._parse_action_json(raw)

    async def _verify_goal(
        self,
        goal: VisionGoal,
        screenshot_b64: str,
    ) -> tuple[bool, str]:
        """Ask VLM if the success criteria are met.

        Returns (goal_met: bool, observation: str).
        """
        prompt = (
            f"Success criteria: {goal.success_criteria}\n\n"
            f"Is the success criteria met in the screenshot? "
            f"Reply ONLY with JSON: {{\"done\": true/false, \"observation\": \"...\"}}"
        )
        try:
            raw = await self._call_llm_with_vision(screenshot_b64, prompt)
            data = self._parse_action_json(raw)
            done = bool(data.get("done", False))
            obs = data.get("observation", "")
            return done, obs
        except Exception as exc:
            log.debug("_verify_goal error: %s", exc)
            return False, ""

    async def _execute_action(self, action_dict: dict, goal: VisionGoal) -> ActionResult:
        """Dispatch action to input_control or navigator.

        Applies safety gate for destructive hotkeys.
        """
        action = action_dict.get("action", "wait")
        target = action_dict.get("target", "")
        args = action_dict.get("args", {})

        # Destructive action gate
        if action == "hotkey":
            combo = args.get("keys", "").lower().replace(" ", "")
            if combo in _DESTRUCTIVE_HOTKEYS:
                log.warning("Destructive hotkey blocked: %s", combo)
                return ActionResult(success=False, error=f"blocked_destructive_hotkey:{combo}")

        # Domain allowlist check for URL navigation
        if action == "type" and goal.allowed_domains:
            text = args.get("text", "")
            if text.startswith("http") or "://" in text:
                if not any(d in text for d in goal.allowed_domains):
                    log.warning("URL not in allowlist: %s", text)
                    return ActionResult(success=False, error="url_not_in_allowlist")

        try:
            ctrl = self._get_input_ctrl()
            if action == "click":
                x, y = args.get("x", 0), args.get("y", 0)
                if x or y:
                    await ctrl.click(x, y)
                elif target and self._navigator:
                    await self._navigator.navigate_to(target)
            elif action == "type":
                text = args.get("text", "")
                await ctrl.type_text(text)
            elif action == "scroll":
                direction = args.get("direction", "down")
                amount = args.get("amount", 3)
                await ctrl.scroll(direction, amount)
            elif action == "hotkey":
                keys_str = args.get("keys", "")
                if keys_str:
                    await ctrl.hotkey(*keys_str.split("+"))
            elif action == "wait":
                await asyncio.sleep(args.get("seconds", 1))
            return ActionResult(success=True)
        except Exception as exc:
            log.debug("_execute_action error: %s", exc)
            return ActionResult(success=False, error=str(exc))

    async def _read_screen(self, screenshot_b64: str) -> str:
        """Extract visible text from screenshot via VLM."""
        prompt = "List all visible text on the screen, line by line."
        try:
            return await self._call_llm_with_vision(screenshot_b64, prompt)
        except Exception:
            return ""

    async def _call_llm_with_vision(self, screenshot_b64: str, prompt: str) -> str:
        """Route to injected VLM/LLM or fall back to built-in remote_vlm."""
        if self._vlm_fn is not None:
            return await self._vlm_fn(screenshot_b64, prompt)
        # Built-in fallback
        try:
            from bantz.vision.remote_vlm import describe_screen
            result = await describe_screen(screenshot_b64)
            return result.raw if hasattr(result, "raw") else str(result)
        except Exception as exc:
            log.debug("VLM call failed: %s", exc)
            return "{}"

    def _get_input_ctrl(self) -> Any:
        if self._input_ctrl is not None:
            return self._input_ctrl
        from bantz.tools import input_control
        return input_control

    @staticmethod
    def _parse_action_json(raw: str) -> dict:
        """Parse JSON from potentially noisy LLM output."""
        if not raw:
            return {}
        # Try to find JSON block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return {}


# ── Web Navigation Macros ─────────────────────────────────────────────────────

class WebNavigationMacros:
    """Pre-built shortcut sequences for common web navigation patterns.

    These reduce loop iterations by batching common UI sequences into
    single macro calls that the loop can invoke instead of individual
    click → screenshot → type steps.
    """

    def __init__(self, input_ctrl: Any = None, loop: AutonomousVisionLoop | None = None) -> None:
        self._input_ctrl = input_ctrl
        self._loop = loop

    def _ctrl(self) -> Any:
        if self._input_ctrl is not None:
            return self._input_ctrl
        from bantz.tools import input_control
        return input_control

    async def open_browser(self) -> bool:
        """Launch or focus the default browser via system tool."""
        try:
            from bantz.tools.system_tool import system_tool
            result = system_tool.run("xdg-open about:blank", timeout=5, safe_mode=False)
            return result.success
        except Exception as exc:
            log.debug("open_browser: %s", exc)
            return False

    async def navigate_to_url(self, url: str) -> bool:
        """Focus URL bar (Ctrl+L) → type URL → Enter."""
        ctrl = self._ctrl()
        try:
            await ctrl.hotkey("ctrl", "l")
            await asyncio.sleep(0.2)
            await ctrl.type_text(url)
            await asyncio.sleep(0.1)
            await ctrl.hotkey("Return")
            return True
        except Exception as exc:
            log.debug("navigate_to_url: %s", exc)
            return False

    async def search_in_page(self, query: str) -> bool:
        """Ctrl+F → type query → check via screenshot."""
        ctrl = self._ctrl()
        try:
            await ctrl.hotkey("ctrl", "f")
            await asyncio.sleep(0.2)
            await ctrl.type_text(query)
            return True
        except Exception as exc:
            log.debug("search_in_page: %s", exc)
            return False

    async def scroll_and_read(self, direction: str = "down", amount: int = 3) -> str:
        """Scroll the page and return VLM-extracted text."""
        ctrl = self._ctrl()
        try:
            await ctrl.scroll(direction, amount)
            await asyncio.sleep(0.3)
        except Exception as exc:
            log.debug("scroll_and_read scroll: %s", exc)

        if self._loop:
            b64 = await self._loop._observe()
            if b64:
                return await self._loop._read_screen(b64)
        return ""

    async def wait_for_element(self, label: str, timeout: float = 10.0) -> bool:
        """Repeatedly screenshot + VLM until element appears or timeout."""
        if self._loop is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            b64 = await self._loop._observe()
            if b64:
                prompt = (
                    f"Is there a UI element labelled '{label}' visible on screen? "
                    f"Reply ONLY: {{\"found\": true}} or {{\"found\": false}}"
                )
                raw = await self._loop._call_llm_with_vision(b64, prompt)
                data = AutonomousVisionLoop._parse_action_json(raw)
                if data.get("found"):
                    return True
            await asyncio.sleep(1.0)
        return False


# ── Module singleton ──────────────────────────────────────────────────────────

vision_loop = AutonomousVisionLoop()
web_macros = WebNavigationMacros(loop=vision_loop)
