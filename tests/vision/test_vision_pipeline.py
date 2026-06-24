"""
Vision-guided execution pipeline — integration test for the AC/DC flow.

Mocks the vision model (scripted JSON verdicts) and the screen backend
(synthetic screenshots that change after each action), then drives:

    profile.get_relevant("play_music", artist="AC/DC")
      → goal string with resolved favorites
      → VisionExecutor.execute() step loop
      → done in N steps

No Ollama, no display server, no real input events.
"""
from __future__ import annotations

import io
import json

import pytest

from bantz.screen_control import ScreenControl, _image_diff
from bantz.vision_executor import ExecutionResult, VisionExecutor, VisionModel


# ── helpers ──────────────────────────────────────────────────────────────


def _png(color: tuple[int, int, int]) -> bytes:
    """Tiny solid-color PNG — distinct colors mean 'screen changed'."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(buf, format="PNG")
    return buf.getvalue()


class ScriptedVision(VisionModel):
    """VisionModel that replays a fixed list of JSON verdicts."""

    def __init__(self, verdicts: list[dict]) -> None:
        super().__init__()
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []
        self._active = "scripted"

    async def ask(self, image_png: bytes, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._verdicts:
            return json.dumps({"observation": "exhausted",
                               "next_action": "failed",
                               "confidence": 1.0,
                               "reason": "script exhausted"})
        return json.dumps(self._verdicts.pop(0))


class FakeScreen(ScreenControl):
    """ScreenControl with synthetic screenshots and recorded actions.

    Screenshot color advances after every action, so click verification
    and wait_for_stable behave like a live, settling screen.
    """

    def __init__(self) -> None:
        # Deliberately skip ScreenControl.__init__ backend probing.
        self.backend = "fake"
        self._vision_locator = None
        self._frame = 0
        self.actions: list[tuple] = []

    async def screenshot(self) -> bytes:
        palette = [(10, 10, 10), (200, 30, 30), (30, 200, 30),
                   (30, 30, 200), (200, 200, 30), (200, 30, 200),
                   (30, 200, 200), (240, 240, 240)]
        return _png(palette[self._frame % len(palette)])

    async def _click_raw(self, x: int, y: int, button: str) -> bool:
        self.actions.append(("click", x, y))
        self._frame += 1
        return True

    async def type_text(self, text: str, interval_ms: int = 50) -> bool:
        self.actions.append(("type", text))
        self._frame += 1
        return True

    async def key(self, keycombo: str) -> bool:
        self.actions.append(("key", keycombo))
        self._frame += 1
        return True

    async def wait_for_stable(self, timeout: float = 3.0,
                              threshold: float = 0.02,
                              poll: float = 0.3) -> bool:
        return True  # synthetic screen settles instantly


# ── unit-level checks ────────────────────────────────────────────────────


def test_image_diff_detects_change_and_stability():
    a, b = _png((10, 10, 10)), _png((200, 30, 30))
    assert _image_diff(a, a) < 0.005
    assert _image_diff(a, b) > 0.02


def test_profile_get_relevant_resolves_favorites(tmp_path, monkeypatch):
    from bantz.core import profile as profile_mod
    monkeypatch.setattr(profile_mod, "_PROFILE_PATH", tmp_path / "profile.json")
    p = profile_mod.Profile()
    p.save({
        "name": "Iclal",
        "preferences": {
            "music": {
                "favorite_artists": ["AC/DC"],
                "artist_favorites": {"AC/DC": "Thunderstruck"},
                "preferred_service": "yt_music",
            },
            "apps": {"browser": "firefox"},
        },
    })
    ctx = p.get_relevant("play_music", artist="ac/dc")
    assert ctx["song"] == "Thunderstruck"
    assert ctx["artist"] == "AC/DC"          # canonical casing restored
    assert ctx["service_url"] == "https://music.youtube.com"
    assert ctx["browser"] == "firefox"


def test_screen_query_classify_modes():
    from bantz.screen_query import ScreenQueryHandler
    assert ScreenQueryHandler.classify("ekranda ne var?") == "describe"
    assert ScreenQueryHandler.classify("şu ikona tıkla") == "click"
    assert ScreenQueryHandler.classify("bu ne yazıyor?") == "read"
    assert ScreenQueryHandler.classify("click the blue Subscribe button") == "click"


# ── the AC/DC integration flow ───────────────────────────────────────────


ACDC_SCRIPT = [
    {"observation": "empty desktop", "next_action": "key",
     "value": "super", "confidence": 0.9, "reason": "open launcher"},
    {"observation": "launcher open", "next_action": "type",
     "value": "firefox", "confidence": 0.9, "reason": "find firefox"},
    {"observation": "firefox highlighted", "next_action": "key",
     "value": "Return", "confidence": 0.9, "reason": "launch"},
    {"observation": "firefox open", "next_action": "type",
     "value": "music.youtube.com", "confidence": 0.85, "reason": "go to YT Music"},
    {"observation": "YT Music loaded", "next_action": "click",
     "target_coords": [320, 18], "target_description": "search bar",
     "confidence": 0.9, "reason": "focus search"},
    {"observation": "search focused", "next_action": "type",
     "value": "AC/DC Thunderstruck", "confidence": 0.9, "reason": "search song"},
    {"observation": "results visible", "next_action": "click",
     "target_coords": [400, 200], "target_description": "first result",
     "confidence": 0.9, "reason": "play it"},
    {"observation": "Thunderstruck playing", "next_action": "done",
     "confidence": 1.0, "reason": "goal achieved"},
]


@pytest.mark.asyncio
async def test_acdc_flow_end_to_end(tmp_path, monkeypatch):
    # 1. Memory: favorite song pre-fetched from the profile
    from bantz.core import profile as profile_mod
    monkeypatch.setattr(profile_mod, "_PROFILE_PATH", tmp_path / "profile.json")
    p = profile_mod.Profile()
    p.save({"name": "Iclal", "preferences": {
        "music": {"artist_favorites": {"AC/DC": "Thunderstruck"},
                  "preferred_service": "yt_music"},
        "apps": {"browser": "firefox"},
    }})
    ctx = p.get_relevant("play_music", artist="AC/DC")
    goal = (f"Open {ctx['browser']}, go to {ctx['service_url']}, "
            f"search for {ctx['song']} by {ctx['artist']}, and play it")

    # 2. Vision loop with scripted verdicts + fake screen
    screen = FakeScreen()
    executor = VisionExecutor(screen=screen, vision=ScriptedVision(ACDC_SCRIPT))
    result = await executor.execute(goal, ctx, max_steps=12)

    # 3. The pipeline completes and the action trace is the human flow
    assert isinstance(result, ExecutionResult)
    assert result.success, result.error
    assert len(result.steps) == len(ACDC_SCRIPT)
    assert result.steps[-1].action == "done"
    typed = [a[1] for a in screen.actions if a[0] == "type"]
    assert "AC/DC Thunderstruck" in typed
    assert "music.youtube.com" in typed
    # Goal text carried the resolved memory values into every prompt
    vision: ScriptedVision = executor.vision  # type: ignore[assignment]
    assert "Thunderstruck" in vision.prompts[0]
    assert "firefox" in vision.prompts[0]


@pytest.mark.asyncio
async def test_executor_aborts_when_stuck():
    same = {"observation": "frozen dialog", "next_action": "click",
            "target_coords": [10, 10], "confidence": 0.9, "reason": "retry"}
    screen = FakeScreen()

    # Freeze the screen: clicks succeed but never advance the frame
    async def _noop(x, y, button="left"):
        return True
    screen._click_raw = _noop  # type: ignore[assignment]

    executor = VisionExecutor(screen=screen, vision=ScriptedVision([same] * 6))
    result = await executor.execute("close the dialog", max_steps=6)
    assert not result.success
    assert "stuck" in result.error


@pytest.mark.asyncio
async def test_verifier_rejects_premature_done():
    # Model claims done twice; verifier rejects the first claim with
    # corrective text, accepts the second.
    script = [
        {"observation": "results page", "next_action": "done",
         "confidence": 0.95, "reason": "looks complete"},
        {"observation": "clicked play", "next_action": "click",
         "target_coords": [200, 300], "confidence": 0.9, "reason": "play"},
        {"observation": "player bar active", "next_action": "done",
         "confidence": 0.95, "reason": "music playing"},
    ]
    calls = {"n": 0}

    async def verifier():
        calls["n"] += 1
        return True if calls["n"] > 1 else "nothing is playing yet"

    executor = VisionExecutor(screen=FakeScreen(), vision=ScriptedVision(script))
    result = await executor.execute("play a song", max_steps=6, verifier=verifier)
    assert result.success
    assert calls["n"] == 2
    # The rejection message was fed back into the next prompt
    vision: ScriptedVision = executor.vision  # type: ignore[assignment]
    assert any("REJECTED" in p for p in vision.prompts)


@pytest.mark.asyncio
async def test_executor_respects_low_confidence_floor():
    verdicts = [{"observation": "??", "next_action": "click",
                 "target_coords": [5, 5], "confidence": 0.1, "reason": "guess"}]
    executor = VisionExecutor(screen=FakeScreen(), vision=ScriptedVision(verdicts))
    result = await executor.execute("anything", max_steps=3)
    assert not result.success
    assert "confidence" in result.error
