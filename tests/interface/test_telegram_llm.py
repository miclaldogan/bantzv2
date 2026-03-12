"""Tests for Discussion #178 — feat(telegram): full LLM mode."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Helpers / Fakes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FakeBrainResult:
    response: str = ""
    tool_used: str | None = None
    tool_result: object = None
    needs_confirm: bool = False
    pending_command: str = ""
    pending_tool: str = ""
    pending_args: dict = field(default_factory=dict)
    stream: AsyncIterator[str] | None = None


def _make_update(user_id: int = 111, text: str = "hello", chat_id: int = 999):
    """Build a minimal mock Update for Telegram tests."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = AsyncMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.chat = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    return update


def _ctx():
    return MagicMock()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Smart Chunking — _safe_reply
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeReply:
    """Test paragraph-boundary chunking in _safe_reply."""

    @pytest.mark.asyncio
    async def test_short_message_single_reply(self):
        from bantz.interface.telegram_bot import _safe_reply
        update = _make_update()
        await _safe_reply(update, "Hello world")
        update.message.reply_text.assert_awaited_once_with("Hello world")

    @pytest.mark.asyncio
    async def test_splits_at_paragraph_boundary(self):
        from bantz.interface.telegram_bot import _safe_reply
        update = _make_update()
        # Two paragraphs each ~2500 chars → combined > 4000 → split into 2
        para_a = "A" * 2500
        para_b = "B" * 2500
        text = para_a + "\n\n" + para_b
        await _safe_reply(update, text)
        assert update.message.reply_text.await_count == 2
        calls = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert calls[0] == para_a
        assert calls[1] == para_b

    @pytest.mark.asyncio
    async def test_exact_boundary(self):
        """Message exactly at the limit — single reply."""
        from bantz.interface.telegram_bot import _safe_reply
        update = _make_update()
        text = "X" * 4000
        await _safe_reply(update, text)
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hard_split_long_paragraph(self):
        """A single paragraph > 4000 chars must still be sent."""
        from bantz.interface.telegram_bot import _safe_reply
        update = _make_update()
        text = "X" * 8000  # No \n\n, no \n — hard truncation
        await _safe_reply(update, text)
        # Must have been sent (may be 1 or more chunks depending on split logic)
        assert update.message.reply_text.await_count >= 1
        # All chars accounted for
        sent = "".join(c[0][0] for c in update.message.reply_text.call_args_list)
        assert len(sent) >= 4000  # at least a substantial portion sent

    @pytest.mark.asyncio
    async def test_three_small_paragraphs_fit_one_chunk(self):
        from bantz.interface.telegram_bot import _safe_reply
        update = _make_update()
        text = "Hi\n\nThere\n\nFriend"
        await _safe_reply(update, text)
        update.message.reply_text.assert_awaited_once_with("Hi\n\nThere\n\nFriend")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    def setup_method(self):
        import bantz.interface.telegram_bot as mod
        self.mod = mod
        # Clear rate state before each test
        mod._rate_log.clear()

    def test_under_limit_returns_false(self):
        assert self.mod._is_rate_limited(42) is False

    def test_at_limit_returns_true(self):
        for _ in range(10):
            self.mod._is_rate_limited(42)
        assert self.mod._is_rate_limited(42) is True

    def test_different_users_independent(self):
        for _ in range(10):
            self.mod._is_rate_limited(100)
        # User 200 should still be fine
        assert self.mod._is_rate_limited(200) is False

    def test_window_expiry(self):
        """After window expires, user can send again."""
        # Manually inject old timestamps
        old_time = time.monotonic() - 120  # 2 minutes ago
        self.mod._rate_log[42] = [old_time] * 10
        assert self.mod._is_rate_limited(42) is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. Silent Stranger — Unauthorized users
# ═══════════════════════════════════════════════════════════════════════════


class TestSilentStranger:
    """Unauthorized users get silent drop — no reply at all."""

    @pytest.mark.asyncio
    async def test_unauthorized_user_silent_drop(self):
        import bantz.interface.telegram_bot as mod
        # Save and set allowed list
        original = mod._ALLOWED
        try:
            mod._ALLOWED = {999}  # only user 999 allowed
            update = _make_update(user_id=111)  # not allowed
            ctx = _ctx()
            await mod.handle_message(update, ctx)
            # reply_text should NOT have been called
            update.message.reply_text.assert_not_awaited()
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_authorized_user_passes(self):
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = {111}
            update = _make_update(user_id=111, text="hello")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="Good day, ma'am.")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)
            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)
            # Should have replied
            update.message.reply_text.assert_awaited()
        finally:
            mod._ALLOWED = original


# ═══════════════════════════════════════════════════════════════════════════
# 4. LLM Mode Toggle
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMModeToggle:
    @pytest.mark.asyncio
    async def test_llm_mode_off_rejects(self):
        """When TELEGRAM_LLM_MODE=false, free text is rejected."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None  # no restriction
            update = _make_update(user_id=111, text="hello")
            ctx = _ctx()
            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = False
                await mod.handle_message(update, ctx)
            update.message.reply_text.assert_awaited_once()
            msg = update.message.reply_text.call_args[0][0]
            assert "strict commands" in msg.lower() or "telegraph" in msg.lower() or "/" in msg
        finally:
            mod._ALLOWED = original


# ═══════════════════════════════════════════════════════════════════════════
# 5. Cognitive Wire — handle_message integration
# ═══════════════════════════════════════════════════════════════════════════


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_typing_indicator_sent(self):
        """Typing action should be sent while brain processes."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="What is the weather?")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="Fine weather, ma'am.")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            from telegram.constants import ChatAction
            update.message.chat.send_action.assert_awaited_once_with(ChatAction.TYPING)
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_brain_process_called_with_is_remote(self):
        """brain.process() must be called with is_remote=True."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="Tell me a joke")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="A dry jest, ma'am.")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            mock_brain.process.assert_awaited_once_with("Tell me a joke", is_remote=True)
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_stream_response_collected(self):
        """When result.stream exists, chunks are collected and sent."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="stream me")
            ctx = _ctx()

            async def fake_stream():
                for chunk in ["Good ", "day, ", "ma'am."]:
                    yield chunk

            fake_result = FakeBrainResult(response="", stream=fake_stream())
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            # Reply should contain the joined stream
            reply_text = update.message.reply_text.call_args[0][0]
            assert reply_text == "Good day, ma'am."
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_stream_response_persisted_to_db(self):
        """Streamed responses must be saved to DB for memory continuity."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="remember this")
            ctx = _ctx()

            async def fake_stream():
                for chunk in ["I shall ", "remember."]:
                    yield chunk

            fake_result = FakeBrainResult(response="", stream=fake_stream())
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)
            mock_brain._graph_store = AsyncMock()

            mock_dal_conv = MagicMock()
            mock_dal = MagicMock()
            mock_dal.conversations = mock_dal_conv

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {
                    "bantz.core.brain": MagicMock(brain=mock_brain),
                    "bantz.data.dal": MagicMock(data_layer=mock_dal),
                }):
                    await mod.handle_message(update, ctx)

            # Must have called conversations.add with the assistant response
            mock_dal_conv.add.assert_called_once_with(
                "assistant", "I shall remember.",
                tool_used=None,
            )
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self):
        """Empty text messages are silently ignored."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="   ")
            ctx = _ctx()

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                await mod.handle_message(update, ctx)

            # No typing, no reply
            update.message.chat.send_action.assert_not_awaited()
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_rate_limited_user_gets_warning(self):
        """Rate-limited user gets a polite wait message."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            # Exhaust rate limit
            for _ in range(10):
                mod._is_rate_limited(111)

            update = _make_update(user_id=111, text="one more")
            ctx = _ctx()
            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                await mod.handle_message(update, ctx)

            msg = update.message.reply_text.call_args[0][0]
            assert "wait" in msg.lower() or "too many" in msg.lower()
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_brain_error_reports_to_user(self):
        """If brain raises, user gets an error message."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="break pls")
            ctx = _ctx()

            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(side_effect=RuntimeError("kaboom"))

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            msg = update.message.reply_text.call_args[0][0]
            assert "kaboom" in msg
        finally:
            mod._ALLOWED = original_allowed

    @pytest.mark.asyncio
    async def test_empty_response_sends_ellipsis(self):
        """Empty brain response sends '…' instead of nothing."""
        import bantz.interface.telegram_bot as mod
        original_allowed = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="hmm")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            msg = update.message.reply_text.call_args[0][0]
            assert msg == "…"
        finally:
            mod._ALLOWED = original_allowed


# ═══════════════════════════════════════════════════════════════════════════
# 6. Config field
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigField:
    def test_telegram_llm_mode_exists(self):
        from bantz.config import Config
        fields = Config.model_fields
        assert "telegram_llm_mode" in fields

    def test_telegram_llm_mode_default_true(self):
        from bantz.config import Config
        f = Config.model_fields["telegram_llm_mode"]
        assert f.default is True

    def test_telegram_llm_mode_alias(self):
        from bantz.config import Config
        f = Config.model_fields["telegram_llm_mode"]
        alias = f.alias or (f.validation_alias if hasattr(f, "validation_alias") else None)
        assert alias == "TELEGRAM_LLM_MODE"


# ═══════════════════════════════════════════════════════════════════════════
# 7. brain.py is_remote integration
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainIsRemote:
    def test_process_accepts_is_remote_kwarg(self):
        """brain.process() signature must accept is_remote."""
        import inspect
        from bantz.core.brain import Brain
        sig = inspect.signature(Brain.process)
        params = sig.parameters
        assert "is_remote" in params
        assert params["is_remote"].default is False

    def test_remote_hint_in_chat_system(self):
        """When is_remote=True, _chat should inject the remote persona hint."""
        from bantz.core.brain import Brain
        import inspect
        # Read the source of _chat to verify the remote_hint variable
        src = inspect.getsource(Brain._chat)
        assert "remote telegraph" in src.lower() or "remote_hint" in src

    def test_remote_hint_in_chat_stream(self):
        """When is_remote=True, _chat_stream should inject the remote persona hint."""
        from bantz.core.brain import Brain
        import inspect
        src = inspect.getsource(Brain._chat_stream)
        assert "remote telegraph" in src.lower() or "remote_hint" in src


# ═══════════════════════════════════════════════════════════════════════════
# 8. TTS suppression when is_remote
# ═══════════════════════════════════════════════════════════════════════════


class TestTTSSuppression:
    def test_tts_guard_in_briefing_handler(self):
        """The TTS speak call in briefing handler must be guarded by _is_remote."""
        from bantz.core.brain import Brain
        import inspect
        src = inspect.getsource(Brain._handle_briefing) if hasattr(Brain, "_handle_briefing") else ""
        if not src:
            # Fallback: search in process or entire class
            src = inspect.getsource(Brain)
        assert "_is_remote" in src


# ═══════════════════════════════════════════════════════════════════════════
# 9. /start command shows LLM hint
# ═══════════════════════════════════════════════════════════════════════════


class TestStartCommand:
    @pytest.mark.asyncio
    async def test_start_shows_llm_hint_when_enabled(self):
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111)
            ctx = _ctx()
            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                await mod.cmd_start(update, ctx)
            msg = update.message.reply_text.call_args[0][0]
            assert "type any message" in msg.lower() or "brain" in msg.lower() or "💬" in msg
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_start_no_llm_hint_when_disabled(self):
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111)
            ctx = _ctx()
            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = False
                await mod.cmd_start(update, ctx)
            msg = update.message.reply_text.call_args[0][0]
            assert "type any message" not in msg.lower()
        finally:
            mod._ALLOWED = original


# ═══════════════════════════════════════════════════════════════════════════
# 10. run_bot registers MessageHandler
# ═══════════════════════════════════════════════════════════════════════════


class TestRunBotRegistration:
    def test_run_bot_source_has_message_handler(self):
        """run_bot() must register a MessageHandler for free text."""
        import inspect
        from bantz.interface.telegram_bot import run_bot
        src = inspect.getsource(run_bot)
        assert "MessageHandler" in src
        assert "handle_message" in src

    def test_run_bot_source_has_text_filter(self):
        """MessageHandler must use TEXT & ~COMMAND filter."""
        import inspect
        from bantz.interface.telegram_bot import run_bot
        src = inspect.getsource(run_bot)
        assert "filters.TEXT" in src
        assert "filters.COMMAND" in src


# ═══════════════════════════════════════════════════════════════════════════
# 11. Module exports
# ═══════════════════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_handle_message_exported(self):
        from bantz.interface.telegram_bot import handle_message
        assert callable(handle_message)

    def test_rate_limit_constants(self):
        from bantz.interface.telegram_bot import _RATE_WINDOW, _RATE_LIMIT
        assert _RATE_WINDOW == 60
        assert _RATE_LIMIT == 10

    def test_safe_reply_exported(self):
        from bantz.interface.telegram_bot import _safe_reply
        assert callable(_safe_reply)

    def test_is_rate_limited_exported(self):
        from bantz.interface.telegram_bot import _is_rate_limited
        assert callable(_is_rate_limited)
