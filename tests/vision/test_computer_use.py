"""Tests for AutonomousVisionLoop, WebNavigationMacros, ComputerUseTool (#188).

All VLM/input_control interactions are mocked — no real display or LLM needed.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from bantz.vision.computer_use import (
    AutonomousVisionLoop,
    VisionGoal,
    VisionStep,
    VisionLoopResult,
    WebNavigationMacros,
    _DESTRUCTIVE_HOTKEYS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _goal(description="open Firefox", criteria="Firefox is open", max_steps=5, timeout=10.0):
    return VisionGoal(
        description=description,
        success_criteria=criteria,
        max_steps=max_steps,
        timeout_s=timeout,
    )


def _make_loop(vlm_responses=None, screenshots=None, input_ctrl=None):
    """Build a loop with injected fakes."""
    _screenshots = iter(screenshots or ["aGVsbG8=" * 5])

    async def _fake_vlm(b64, prompt):
        resp = vlm_responses.pop(0) if vlm_responses else '{"action":"done","done":true}'
        return resp if isinstance(resp, str) else json.dumps(resp)

    loop = AutonomousVisionLoop(
        vlm_fn=_fake_vlm,
        llm_fn=None,
        input_ctrl=input_ctrl or _mock_input(),
    )

    # Patch _observe to return controlled screenshots
    async def _fake_observe():
        try:
            return next(_screenshots)
        except StopIteration:
            return "aGVsbG8=" * 5

    loop._observe = _fake_observe
    return loop


def _mock_input():
    ctrl = MagicMock()
    ctrl.click = AsyncMock(return_value={"success": True})
    ctrl.type_text = AsyncMock(return_value={"success": True})
    ctrl.scroll = AsyncMock(return_value={"success": True})
    ctrl.hotkey = AsyncMock(return_value={"success": True})
    return ctrl


# ── VisionGoal ────────────────────────────────────────────────────────────────

class TestVisionGoal:

    def test_defaults(self):
        g = VisionGoal(description="do stuff", success_criteria="stuff done")
        assert g.max_steps == 15
        assert g.timeout_s == 120.0
        assert g.allowed_domains == []

    def test_custom_fields(self):
        g = VisionGoal("desc", "criteria", max_steps=5, timeout_s=30.0,
                        allowed_domains=["wikipedia.org"])
        assert g.max_steps == 5
        assert g.allowed_domains == ["wikipedia.org"]


# ── VisionLoopResult ──────────────────────────────────────────────────────────

class TestVisionLoopResult:

    def test_fields(self):
        r = VisionLoopResult(
            success=True, steps=[], final_observation="done",
            extracted_text="text", total_time_s=1.5,
        )
        assert r.success
        assert r.error == ""


# ── _parse_action_json ────────────────────────────────────────────────────────

class TestParseActionJson:

    def test_valid_json(self):
        raw = '{"action": "click", "target": "button", "done": false}'
        d = AutonomousVisionLoop._parse_action_json(raw)
        assert d["action"] == "click"

    def test_json_with_prefix_noise(self):
        raw = 'Sure! {"action": "type", "args": {"text": "hello"}}'
        d = AutonomousVisionLoop._parse_action_json(raw)
        assert d["action"] == "type"

    def test_empty_string_returns_empty(self):
        assert AutonomousVisionLoop._parse_action_json("") == {}

    def test_no_json_returns_empty(self):
        assert AutonomousVisionLoop._parse_action_json("no json here") == {}

    def test_malformed_json_returns_empty(self):
        assert AutonomousVisionLoop._parse_action_json("{broken json") == {}


# ── execute() — basic flow ────────────────────────────────────────────────────

class TestExecuteBasic:

    @pytest.mark.asyncio
    async def test_goal_met_on_first_verify(self):
        """VLM immediately says goal met → success in 1 step."""
        loop = _make_loop(
            vlm_responses=[
                '{"action":"click","target":"button","done":false,"reasoning":"r","observation":"o"}',
                '{"done":true,"observation":"goal met"}',  # verify response
            ]
        )
        result = await loop.execute(_goal(max_steps=5))
        assert result.success is True

    @pytest.mark.asyncio
    async def test_done_action_exits_immediately(self):
        """If LLM returns action=done, loop exits without verify."""
        loop = _make_loop(
            vlm_responses=[
                '{"action":"done","done":true,"observation":"ok","reasoning":"done"}'
            ]
        )
        result = await loop.execute(_goal())
        assert result.success is True
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_max_steps_abort(self):
        """If goal never met, loop aborts after max_steps."""
        responses = []
        for i in range(10):
            responses.append('{"action":"wait","done":false,"reasoning":"r","observation":"o"}')
            responses.append('{"done":false,"observation":"not done yet"}')

        # Use unique screenshots to avoid stuck detection
        _step = 0

        async def _fake_vlm(b64, prompt):
            return responses.pop(0) if responses else '{"action":"wait","done":false}'

        loop = AutonomousVisionLoop(vlm_fn=_fake_vlm, input_ctrl=_mock_input())

        _counter = [0]

        async def _unique_screenshots():
            _counter[0] += 1
            return f"screenshot_{_counter[0]}" * 3

        loop._observe = _unique_screenshots
        result = await loop.execute(_goal(max_steps=3))
        assert result.success is False
        assert result.error == "max_steps_reached"

    @pytest.mark.asyncio
    async def test_on_step_callback_fires(self):
        """on_step callback is called once per step."""
        fired: list[VisionStep] = []

        async def _cb(vs: VisionStep):
            fired.append(vs)

        loop = _make_loop(
            vlm_responses=[
                '{"action":"done","done":true,"observation":"ok","reasoning":"r"}'
            ]
        )
        await loop.execute(_goal(), on_step=_cb)
        assert len(fired) == 1

    @pytest.mark.asyncio
    async def test_returns_extracted_text(self):
        """extracted_text from done action is propagated to result."""
        loop = _make_loop(
            vlm_responses=[
                '{"action":"done","done":true,"observation":"ok",'
                '"reasoning":"r","extracted_text":"some page content"}'
            ]
        )
        result = await loop.execute(_goal())
        assert "some page content" in result.extracted_text

    @pytest.mark.asyncio
    async def test_read_action_accumulates_text(self):
        """action=read calls _read_screen, text appended to extracted_text."""
        loop = _make_loop(
            vlm_responses=[
                '{"action":"read","done":false,"reasoning":"r","observation":"o"}',
                '{"done":true,"observation":"done"}',
            ]
        )
        loop._read_screen = AsyncMock(return_value="page text here")
        result = await loop.execute(_goal(max_steps=3))
        assert "page text here" in result.extracted_text


# ── Stuck detection ───────────────────────────────────────────────────────────

class TestStuckDetection:

    @pytest.mark.asyncio
    async def test_identical_screenshots_abort(self):
        """3 identical screenshots → stuck error."""
        same = "AAAA"
        responses = []
        for _ in range(5):
            responses.append('{"action":"wait","done":false,"reasoning":"r","observation":"o"}')
            responses.append('{"done":false,"observation":"not done"}')

        loop = AutonomousVisionLoop(
            vlm_fn=None,
            input_ctrl=_mock_input(),
        )
        _call = 0

        async def _fake_vlm(b64, prompt):
            nonlocal _call
            _call += 1
            return responses.pop(0) if responses else '{"action":"wait","done":false}'

        loop._vlm_fn = _fake_vlm

        async def _same_screenshot():
            return same

        loop._observe = _same_screenshot

        result = await loop.execute(_goal(max_steps=10))
        assert result.error == "stuck"


# ── Timeout ───────────────────────────────────────────────────────────────────

class TestTimeout:

    @pytest.mark.asyncio
    async def test_timeout_abort(self):
        """timeout_s=0 → immediate timeout."""
        loop = _make_loop(vlm_responses=[])
        result = await loop.execute(_goal(timeout=0.0, max_steps=100))
        assert result.error == "timeout"
        assert result.success is False


# ── _execute_action ───────────────────────────────────────────────────────────

class TestExecuteAction:

    @pytest.mark.asyncio
    async def test_click_delegates_to_input_ctrl(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        ar = await loop._execute_action(
            {"action": "click", "args": {"x": 100, "y": 200}}, _goal()
        )
        ctrl.click.assert_called_once_with(100, 200)
        assert ar.success

    @pytest.mark.asyncio
    async def test_type_delegates_to_input_ctrl(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        ar = await loop._execute_action(
            {"action": "type", "args": {"text": "hello"}}, _goal()
        )
        ctrl.type_text.assert_called_once_with("hello")
        assert ar.success

    @pytest.mark.asyncio
    async def test_scroll_delegates_to_input_ctrl(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        ar = await loop._execute_action(
            {"action": "scroll", "args": {"direction": "down", "amount": 3}}, _goal()
        )
        ctrl.scroll.assert_called_once_with("down", 3)
        assert ar.success

    @pytest.mark.asyncio
    async def test_hotkey_delegates_to_input_ctrl(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        ar = await loop._execute_action(
            {"action": "hotkey", "args": {"keys": "ctrl+t"}}, _goal()
        )
        ctrl.hotkey.assert_called_once_with("ctrl", "t")
        assert ar.success

    @pytest.mark.asyncio
    async def test_destructive_hotkey_blocked(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        ar = await loop._execute_action(
            {"action": "hotkey", "args": {"keys": "alt+f4"}}, _goal()
        )
        ctrl.hotkey.assert_not_called()
        assert not ar.success
        assert "blocked_destructive_hotkey" in ar.error

    @pytest.mark.asyncio
    async def test_url_not_in_allowlist_blocked(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        goal = _goal()
        goal.allowed_domains = ["wikipedia.org"]
        ar = await loop._execute_action(
            {"action": "type", "args": {"text": "http://evil.com"}}, goal
        )
        ctrl.type_text.assert_not_called()
        assert "url_not_in_allowlist" in ar.error

    @pytest.mark.asyncio
    async def test_url_in_allowlist_allowed(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)
        goal = _goal()
        goal.allowed_domains = ["wikipedia.org"]
        ar = await loop._execute_action(
            {"action": "type", "args": {"text": "https://wikipedia.org/wiki/Python"}}, goal
        )
        ctrl.type_text.assert_called_once()
        assert ar.success

    @pytest.mark.asyncio
    async def test_wait_action_returns_success(self):
        loop = AutonomousVisionLoop(input_ctrl=_mock_input())
        ar = await loop._execute_action(
            {"action": "wait", "args": {"seconds": 0}}, _goal()
        )
        assert ar.success


# ── _verify_goal ──────────────────────────────────────────────────────────────

class TestVerifyGoal:

    @pytest.mark.asyncio
    async def test_verify_true_when_vlm_says_done(self):
        async def _vlm(b64, prompt):
            return '{"done": true, "observation": "Firefox is open"}'

        loop = AutonomousVisionLoop(vlm_fn=_vlm)
        done, obs = await loop._verify_goal(_goal(), "img_b64")
        assert done is True
        assert "Firefox" in obs

    @pytest.mark.asyncio
    async def test_verify_false_when_vlm_says_not_done(self):
        async def _vlm(b64, prompt):
            return '{"done": false, "observation": "still loading"}'

        loop = AutonomousVisionLoop(vlm_fn=_vlm)
        done, obs = await loop._verify_goal(_goal(), "img_b64")
        assert done is False

    @pytest.mark.asyncio
    async def test_verify_false_on_vlm_error(self):
        async def _vlm(b64, prompt):
            raise RuntimeError("VLM down")

        loop = AutonomousVisionLoop(vlm_fn=_vlm)
        done, obs = await loop._verify_goal(_goal(), "img_b64")
        assert done is False


# ── Destructive hotkeys set ───────────────────────────────────────────────────

class TestDestructiveHotkeys:

    def test_alt_f4_in_set(self):
        assert "alt+f4" in _DESTRUCTIVE_HOTKEYS

    def test_ctrl_w_in_set(self):
        assert "ctrl+w" in _DESTRUCTIVE_HOTKEYS

    def test_safe_hotkey_not_in_set(self):
        assert "ctrl+t" not in _DESTRUCTIVE_HOTKEYS


# ── WebNavigationMacros ───────────────────────────────────────────────────────

class TestWebNavigationMacros:

    @pytest.mark.asyncio
    async def test_navigate_to_url(self):
        ctrl = _mock_input()
        macros = WebNavigationMacros(input_ctrl=ctrl)
        result = await macros.navigate_to_url("https://wikipedia.org")
        ctrl.hotkey.assert_called()
        ctrl.type_text.assert_called_once_with("https://wikipedia.org")
        assert result is True

    @pytest.mark.asyncio
    async def test_search_in_page(self):
        ctrl = _mock_input()
        macros = WebNavigationMacros(input_ctrl=ctrl)
        result = await macros.search_in_page("Ottoman Empire")
        ctrl.hotkey.assert_called()
        ctrl.type_text.assert_called_once_with("Ottoman Empire")
        assert result is True

    @pytest.mark.asyncio
    async def test_scroll_and_read(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)

        async def _observe():
            return "aGVsbG8="

        loop._observe = _observe
        loop._read_screen = AsyncMock(return_value="article text")
        macros = WebNavigationMacros(input_ctrl=ctrl, loop=loop)
        text = await macros.scroll_and_read()
        assert "article text" in text

    @pytest.mark.asyncio
    async def test_wait_for_element_found(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)

        async def _observe():
            return "aGVsbG8="

        call_count = 0

        async def _vlm(b64, prompt):
            nonlocal call_count
            call_count += 1
            return '{"found": true}'

        loop._observe = _observe
        loop._vlm_fn = _vlm
        macros = WebNavigationMacros(input_ctrl=ctrl, loop=loop)
        result = await macros.wait_for_element("search box", timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_element_timeout(self):
        ctrl = _mock_input()
        loop = AutonomousVisionLoop(input_ctrl=ctrl)

        async def _observe():
            return "aGVsbG8="

        async def _vlm(b64, prompt):
            return '{"found": false}'

        loop._observe = _observe
        loop._vlm_fn = _vlm
        macros = WebNavigationMacros(input_ctrl=ctrl, loop=loop)
        result = await macros.wait_for_element("nonexistent button", timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_element_no_loop_returns_false(self):
        macros = WebNavigationMacros()
        result = await macros.wait_for_element("anything")
        assert result is False


# ── ComputerUseTool ───────────────────────────────────────────────────────────

class TestComputerUseTool:

    def test_tool_name(self):
        from bantz.tools.computer_use import ComputerUseTool
        assert ComputerUseTool.name == "computer_use"

    def test_tool_registered(self):
        import bantz.tools.computer_use  # ensure registration  # noqa
        from bantz.tools import registry
        assert registry.get("computer_use") is not None

    @pytest.mark.asyncio
    async def test_execute_missing_task_returns_error(self):
        from bantz.tools.computer_use import ComputerUseTool
        tool = ComputerUseTool()
        result = await tool.execute()
        assert not result.success
        assert "task" in result.error

    @pytest.mark.asyncio
    async def test_execute_success_path(self):
        from bantz.tools.computer_use import ComputerUseTool
        tool = ComputerUseTool()
        # Inject a loop that immediately succeeds
        mock_loop = MagicMock()
        from bantz.vision.computer_use import VisionLoopResult
        mock_loop.execute = AsyncMock(return_value=VisionLoopResult(
            success=True,
            steps=[],
            final_observation="Done!",
            extracted_text="page content",
            total_time_s=0.5,
        ))
        tool._loop = mock_loop
        result = await tool.execute(task="open Firefox")
        assert result.success
        assert "Done!" in result.output

    @pytest.mark.asyncio
    async def test_execute_failure_path(self):
        from bantz.tools.computer_use import ComputerUseTool
        from bantz.vision.computer_use import VisionLoopResult
        tool = ComputerUseTool()
        mock_loop = MagicMock()
        mock_loop.execute = AsyncMock(return_value=VisionLoopResult(
            success=False,
            steps=[],
            final_observation="",
            extracted_text="",
            total_time_s=1.0,
            error="max_steps_reached",
        ))
        tool._loop = mock_loop
        result = await tool.execute(task="do impossible thing")
        assert not result.success
        assert result.error == "max_steps_reached"

    @pytest.mark.asyncio
    async def test_execute_exception_returns_error(self):
        from bantz.tools.computer_use import ComputerUseTool
        tool = ComputerUseTool()
        mock_loop = MagicMock()
        mock_loop.execute = AsyncMock(side_effect=RuntimeError("crash"))
        tool._loop = mock_loop
        result = await tool.execute(task="crash test")
        assert not result.success
        assert "crash" in result.error


# ── Module singletons ─────────────────────────────────────────────────────────

class TestSingletons:

    def test_vision_loop_singleton(self):
        from bantz.vision.computer_use import vision_loop, AutonomousVisionLoop
        assert isinstance(vision_loop, AutonomousVisionLoop)

    def test_web_macros_singleton(self):
        from bantz.vision.computer_use import web_macros, WebNavigationMacros
        assert isinstance(web_macros, WebNavigationMacros)
