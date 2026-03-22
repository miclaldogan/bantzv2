"""Tests for Telegram screenshot (daguerreotype) feature (#189).

Covers: ScreenshotTool, Attachment dataclass, BrainResult.attachments,
_send_photo(), handle_message attachment dispatch, rate limiting, config.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip('telegram')


# ── Attachment dataclass ──────────────────────────────────────────────────────

class TestAttachment:

    def test_fields_accessible(self):
        from bantz.core.types import Attachment
        att = Attachment(type="image", data=b"\xff\xd8", caption="hi")
        assert att.type == "image"
        assert att.data == b"\xff\xd8"
        assert att.caption == "hi"
        assert att.mime_type == "image/jpeg"

    def test_default_filename(self):
        from bantz.core.types import Attachment
        att = Attachment(type="image", data=b"x")
        assert att.filename == "bantz_attachment"

    def test_brain_result_has_attachments(self):
        from bantz.core.types import BrainResult
        r = BrainResult(response="hi", tool_used=None)
        assert r.attachments == []

    def test_brain_result_attachments_carried(self):
        from bantz.core.types import BrainResult, Attachment
        att = Attachment(type="image", data=b"x")
        r = BrainResult(response="hi", tool_used="screenshot", attachments=[att])
        assert len(r.attachments) == 1
        assert r.attachments[0].data == b"x"


# ── ScreenshotTool ────────────────────────────────────────────────────────────

class TestScreenshotTool:

    @pytest.mark.asyncio
    async def test_returns_jpeg_bytes_in_data(self):
        from bantz.tools.screenshot_tool import ScreenshotTool

        fake_shot = MagicMock()
        fake_shot.data = b"\xff\xd8\xff\xe0JPEG"
        fake_shot.width = 1920
        fake_shot.height = 1080

        with patch("bantz.vision.screenshot.capture", AsyncMock(return_value=fake_shot)), \
             patch("bantz.tools.screenshot_tool._reencode_jpeg", side_effect=lambda d, q: d):
            result = await ScreenshotTool().execute()

        assert result.success
        assert result.data["screenshot"] == b"\xff\xd8\xff\xe0JPEG"
        assert result.data["width"] == 1920
        assert result.data["height"] == 1080

    @pytest.mark.asyncio
    async def test_failure_when_capture_returns_none(self):
        from bantz.tools.screenshot_tool import ScreenshotTool

        with patch("bantz.vision.screenshot.capture", AsyncMock(return_value=None)):
            result = await ScreenshotTool().execute()

        assert not result.success
        assert "daguerreotype" in result.error.lower()

    @pytest.mark.asyncio
    async def test_window_capture_when_app_given(self):
        from bantz.tools.screenshot_tool import ScreenshotTool

        fake_shot = MagicMock()
        fake_shot.data = b"JPEG"
        fake_shot.width = 800
        fake_shot.height = 600

        with patch("bantz.vision.screenshot.capture_window", AsyncMock(return_value=fake_shot)) as mock_cw, \
             patch("bantz.tools.screenshot_tool._reencode_jpeg", side_effect=lambda d, q: d):
            result = await ScreenshotTool().execute(app="vscode")

        mock_cw.assert_called_once_with("vscode")
        assert result.success

    @pytest.mark.asyncio
    async def test_disabled_config_returns_error(self):
        from bantz.tools.screenshot_tool import ScreenshotTool

        with patch("bantz.config.config.telegram_screenshot_enabled", False):
            result = await ScreenshotTool().execute()

        assert not result.success
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_exception_handled_gracefully(self):
        from bantz.tools.screenshot_tool import ScreenshotTool

        with patch("bantz.vision.screenshot.capture", AsyncMock(side_effect=RuntimeError("display error"))):
            result = await ScreenshotTool().execute()

        assert not result.success
        assert "display error" in result.error

    def test_tool_registered(self):
        import bantz.tools.screenshot_tool  # ensure registration  # noqa
        from bantz.tools import registry
        assert registry.get("screenshot") is not None

    def test_tool_name(self):
        from bantz.tools.screenshot_tool import ScreenshotTool
        assert ScreenshotTool.name == "screenshot"

    def test_daguerreotype_vocabulary_in_output(self):
        """The tool output must use butler vocabulary."""
        from bantz.tools.screenshot_tool import ScreenshotTool
        import asyncio

        fake_shot = MagicMock()
        fake_shot.data = b"JPEG"
        fake_shot.width = 1920
        fake_shot.height = 1080

        async def _run():
            with patch("bantz.vision.screenshot.capture", AsyncMock(return_value=fake_shot)), \
                 patch("bantz.tools.screenshot_tool._reencode_jpeg", side_effect=lambda d, q: d):
                return await ScreenshotTool().execute()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert "daguerreotype" in result.output.lower()


# ── SCREENSHOT_TRIGGERS set ───────────────────────────────────────────────────

class TestScreenshotTriggers:

    def test_screenshot_in_triggers(self):
        from bantz.tools.screenshot_tool import SCREENSHOT_TRIGGERS
        assert "screenshot" in SCREENSHOT_TRIGGERS

    def test_ekran_in_triggers(self):
        from bantz.tools.screenshot_tool import SCREENSHOT_TRIGGERS
        assert "ekran" in SCREENSHOT_TRIGGERS

    def test_show_me_in_triggers(self):
        from bantz.tools.screenshot_tool import SCREENSHOT_TRIGGERS
        assert "show me" in SCREENSHOT_TRIGGERS


# ── _send_photo() ─────────────────────────────────────────────────────────────

class TestSendPhoto:

    def _make_update(self):
        update = MagicMock()
        update.message.reply_photo = AsyncMock()
        update.message.reply_text = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_sends_jpeg_via_reply_photo(self):
        from bantz.interface.telegram_bot import _send_photo
        update = self._make_update()
        await _send_photo(update, b"\xff\xd8JPEG", caption="Here it is, ma'am.")
        update.message.reply_photo.assert_called_once()
        call_kwargs = update.message.reply_photo.call_args[1]
        assert call_kwargs["caption"] == "Here it is, ma'am."

    @pytest.mark.asyncio
    async def test_no_caption_when_empty(self):
        from bantz.interface.telegram_bot import _send_photo
        update = self._make_update()
        await _send_photo(update, b"\xff\xd8JPEG", caption="")
        update.message.reply_photo.assert_called_once()
        call_kwargs = update.message.reply_photo.call_args[1]
        assert call_kwargs.get("caption") is None

    @pytest.mark.asyncio
    async def test_long_caption_truncated_and_sent_separately(self):
        from bantz.interface.telegram_bot import _send_photo
        update = self._make_update()
        long_caption = "x" * 2000
        await _send_photo(update, b"\xff\xd8JPEG", caption=long_caption)
        # Photo was sent with truncated caption
        assert update.message.reply_photo.call_count == 1
        # Full caption sent as follow-up text
        assert update.message.reply_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_reply_photo_exception_logged_not_raised(self):
        from bantz.interface.telegram_bot import _send_photo
        update = self._make_update()
        update.message.reply_photo.side_effect = RuntimeError("network error")
        # Must not raise
        await _send_photo(update, b"x", caption="test")


# ── Screenshot rate limiting ──────────────────────────────────────────────────

class TestScreenshotRateLimit:

    def test_first_call_allowed(self):
        from bantz.interface.telegram_bot import _screenshot_rate_ok, _SCREENSHOT_RATE
        _SCREENSHOT_RATE.clear()
        assert _screenshot_rate_ok(99999) is True

    def test_rapid_second_call_blocked(self):
        from bantz.interface.telegram_bot import _screenshot_rate_ok, _SCREENSHOT_RATE
        _SCREENSHOT_RATE.clear()
        _screenshot_rate_ok(88888)   # first call
        # Immediate second call — within 5s minimum interval
        assert _screenshot_rate_ok(88888) is False

    def test_different_users_independent(self):
        from bantz.interface.telegram_bot import _screenshot_rate_ok, _SCREENSHOT_RATE
        _SCREENSHOT_RATE.clear()
        _screenshot_rate_ok(11111)
        # Different user should be allowed
        assert _screenshot_rate_ok(22222) is True


# ── Maintenance filter fix ────────────────────────────────────────────────────

class TestMaintenanceFilter:

    def _result(self, tool, response):
        r = MagicMock()
        r.tool_used = tool
        r.response = response
        return r

    def test_suppresses_empty_maintenance(self):
        from bantz.interface.telegram_bot import _is_maintenance_spam
        assert _is_maintenance_spam(self._result("maintenance", "")) is True

    def test_suppresses_all_clear_maintenance(self):
        from bantz.interface.telegram_bot import _is_maintenance_spam
        assert _is_maintenance_spam(self._result("maintenance", "All systems nominal")) is True

    def test_lets_through_actionable_maintenance(self):
        from bantz.interface.telegram_bot import _is_maintenance_spam
        assert _is_maintenance_spam(
            self._result("maintenance", "⚠️ Neo4j connection failed: timeout")
        ) is False

    def test_non_maintenance_never_suppressed(self):
        from bantz.interface.telegram_bot import _is_maintenance_spam
        assert _is_maintenance_spam(self._result("web_search", "")) is False
        assert _is_maintenance_spam(self._result(None, "")) is False


# ── handle_message attachment dispatch ───────────────────────────────────────

class TestHandleMessageAttachments:

    @pytest.mark.asyncio
    async def test_attachment_triggers_send_photo(self):
        """When BrainResult has attachments, _send_photo is called."""
        from bantz.core.types import BrainResult, Attachment

        att = Attachment(type="image", data=b"\xff\xd8JPEG", caption="daguerreotype")
        brain_result = BrainResult(
            response="Here it is, ma'am.",
            tool_used="screenshot",
            attachments=[att],
        )

        update = MagicMock()
        update.effective_user.id = 42
        update.effective_chat.id = 1
        update.message.text = "take a screenshot"
        update.message.reply_text = AsyncMock(return_value=MagicMock(
            edit_text=AsyncMock(return_value=True),
            delete=AsyncMock(),
        ))
        update.message.reply_photo = AsyncMock()
        update.message.chat.send_action = AsyncMock()
        context = MagicMock()

        with patch("bantz.config.config.telegram_llm_mode", True), \
             patch("bantz.interface.telegram_bot._ALLOWED", None), \
             patch("bantz.interface.telegram_bot._is_rate_limited", return_value=False), \
             patch("bantz.interface.telegram_bot._screenshot_rate_ok", return_value=True), \
             patch("bantz.core.brain.brain.process", AsyncMock(return_value=brain_result)), \
             patch("bantz.interface.telegram_bot._safe_edit", AsyncMock(return_value=True)):
            from bantz.interface.telegram_bot import handle_message
            await handle_message(update, context)

        update.message.reply_photo.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_attachment_text_only(self):
        """When no attachments, reply_photo is NOT called."""
        from bantz.core.types import BrainResult

        brain_result = BrainResult(
            response="The weather is fine, ma'am.",
            tool_used="weather",
        )

        update = MagicMock()
        update.effective_user.id = 42
        update.effective_chat.id = 1
        update.message.text = "what's the weather"
        placeholder = MagicMock()
        placeholder.edit_text = AsyncMock()
        placeholder.delete = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=placeholder)
        update.message.reply_photo = AsyncMock()
        update.message.chat.send_action = AsyncMock()
        context = MagicMock()

        with patch("bantz.config.config.telegram_llm_mode", True), \
             patch("bantz.interface.telegram_bot._ALLOWED", None), \
             patch("bantz.interface.telegram_bot._is_rate_limited", return_value=False), \
             patch("bantz.core.brain.brain.process", AsyncMock(return_value=brain_result)), \
             patch("bantz.interface.telegram_bot._safe_edit", AsyncMock(return_value=True)):
            from bantz.interface.telegram_bot import handle_message
            await handle_message(update, context)

        update.message.reply_photo.assert_not_called()
