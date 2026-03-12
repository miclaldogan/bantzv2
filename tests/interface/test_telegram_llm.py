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
    # reply_text returns a placeholder message mock (#181)
    placeholder = AsyncMock()
    placeholder.edit_text = AsyncMock()
    placeholder.delete = AsyncMock()
    update.message.reply_text = AsyncMock(return_value=placeholder)
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

            # Response should have been edited into the placeholder (#181)
            placeholder = update.message.reply_text.return_value
            placeholder.edit_text.assert_awaited_once()
            edited_text = placeholder.edit_text.call_args[0][0]
            assert edited_text == "Good day, ma'am."
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

            # Error should be edited into the placeholder (#181)
            placeholder = update.message.reply_text.return_value
            edited = placeholder.edit_text.call_args[0][0]
            assert "kaboom" in edited
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

            # Empty response → placeholder edited to "…" (#181)
            placeholder = update.message.reply_text.return_value
            edited = placeholder.edit_text.call_args[0][0]
            assert edited == "…"
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


# ═══════════════════════════════════════════════════════════════════════════
# 12. Progress Indicators & Message Editing (#181)
# ═══════════════════════════════════════════════════════════════════════════


class TestProgressIndicators:
    """Tests for placeholder → edit_text flow (#181)."""

    @pytest.mark.asyncio
    async def test_placeholder_sent_immediately(self):
        """reply_text() called with a placeholder BEFORE brain.process()."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="What's the weather?")
            ctx = _ctx()

            call_order: list[str] = []

            original_reply = update.message.reply_text

            async def tracked_reply(*a, **kw):
                call_order.append("reply_text")
                return await original_reply(*a, **kw)

            update.message.reply_text = AsyncMock(
                side_effect=tracked_reply,
                return_value=original_reply.return_value,
            )

            fake_result = FakeBrainResult(response="Fine weather, ma'am.")
            mock_brain = MagicMock()

            async def tracked_process(*a, **kw):
                call_order.append("brain.process")
                return fake_result

            mock_brain.process = AsyncMock(side_effect=tracked_process)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            assert call_order.index("reply_text") < call_order.index("brain.process")
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_placeholder_edited_with_response(self):
        """edit_text() called with the actual LLM response on the placeholder."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="hello")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="Good day, ma'am.")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            placeholder = update.message.reply_text.return_value
            placeholder.edit_text.assert_awaited_once()
            assert placeholder.edit_text.call_args[0][0] == "Good day, ma'am."
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_long_response_chunked(self):
        """>4000 chars: first chunk edits placeholder, rest are new messages."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="tell me everything")
            ctx = _ctx()

            # Two large paragraphs → 2 chunks
            para_a = "A" * 2500
            para_b = "B" * 2500
            long_response = para_a + "\n\n" + para_b

            fake_result = FakeBrainResult(response=long_response)
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            placeholder = update.message.reply_text.return_value
            # First chunk edits the placeholder
            placeholder.edit_text.assert_awaited_once_with(para_a)
            # Second chunk sent as new message (call_args_list[0] = placeholder, [1] = extra)
            calls = update.message.reply_text.call_args_list
            assert len(calls) >= 2  # placeholder + at least one extra chunk
            assert calls[-1][0][0] == para_b
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_edit_failure_fallback_plain_text(self):
        """If edit_text fails with markdown, retry with parse_mode=None."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="markdown problem")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="*broken markdown")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            placeholder = update.message.reply_text.return_value
            call_count = 0

            async def flaky_edit(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Markdown parse error")
                # Second call (with parse_mode=None) succeeds
                return None

            placeholder.edit_text = AsyncMock(side_effect=flaky_edit)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            assert call_count == 2
            # Second call should have parse_mode=None
            second_call = placeholder.edit_text.call_args_list[1]
            assert second_call[1].get("parse_mode") is None
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_edit_failure_complete_fallback(self):
        """Both edit_text calls fail → delete placeholder + reply_text."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="total failure")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="The response.")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            placeholder = update.message.reply_text.return_value
            placeholder.edit_text = AsyncMock(side_effect=Exception("API down"))

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            # Placeholder should have been deleted
            placeholder.delete.assert_awaited_once()
            # Response sent via reply_text (call[0]=placeholder, call[1]=response)
            calls = update.message.reply_text.call_args_list
            assert any("The response." in str(c) for c in calls)
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_error_response_edits_placeholder(self):
        """Brain error → placeholder edited to show error message."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="cause error")
            ctx = _ctx()

            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(side_effect=RuntimeError("server crash"))

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            placeholder = update.message.reply_text.return_value
            edited = placeholder.edit_text.call_args[0][0]
            assert "server crash" in edited
            assert "⚠️" in edited
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_placeholder_is_in_character(self):
        """Placeholder text is from PLACEHOLDER_MESSAGES (butler persona)."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="hello butler")
            ctx = _ctx()

            fake_result = FakeBrainResult(response="At your service.")
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            # First reply_text call is the placeholder
            placeholder_text = update.message.reply_text.call_args_list[0][0][0]
            assert placeholder_text in mod.PLACEHOLDER_MESSAGES
        finally:
            mod._ALLOWED = original

    @pytest.mark.asyncio
    async def test_streaming_collected_then_edited(self):
        """Streaming tokens collected in full, then single edit_text."""
        import bantz.interface.telegram_bot as mod
        original = mod._ALLOWED
        mod._rate_log.clear()
        try:
            mod._ALLOWED = None
            update = _make_update(user_id=111, text="stream it")
            ctx = _ctx()

            async def fake_stream():
                for chunk in ["Good ", "evening, ", "ma'am."]:
                    yield chunk

            fake_result = FakeBrainResult(response="", stream=fake_stream())
            mock_brain = MagicMock()
            mock_brain.process = AsyncMock(return_value=fake_result)

            with patch.object(mod, "config") as mock_cfg:
                mock_cfg.telegram_llm_mode = True
                with patch.dict("sys.modules", {"bantz.core.brain": MagicMock(brain=mock_brain)}):
                    await mod.handle_message(update, ctx)

            placeholder = update.message.reply_text.return_value
            placeholder.edit_text.assert_awaited_once()
            edited = placeholder.edit_text.call_args[0][0]
            assert edited == "Good evening, ma'am."
        finally:
            mod._ALLOWED = original


class TestPlaceholderMessages:
    """Tests for the PLACEHOLDER_MESSAGES constant."""

    def test_placeholder_messages_exist(self):
        from bantz.interface.telegram_bot import PLACEHOLDER_MESSAGES
        assert isinstance(PLACEHOLDER_MESSAGES, list)
        assert len(PLACEHOLDER_MESSAGES) >= 5

    def test_placeholder_messages_are_strings(self):
        from bantz.interface.telegram_bot import PLACEHOLDER_MESSAGES
        for msg in PLACEHOLDER_MESSAGES:
            assert isinstance(msg, str)
            assert len(msg) > 10

    def test_placeholder_messages_have_telegraph_emoji(self):
        from bantz.interface.telegram_bot import PLACEHOLDER_MESSAGES
        for msg in PLACEHOLDER_MESSAGES:
            assert "📟" in msg

    def test_placeholder_messages_in_character(self):
        """All placeholders should sound like a 1920s butler."""
        from bantz.interface.telegram_bot import PLACEHOLDER_MESSAGES
        butler_words = {"ma'am", "moment", "please", "telegraph", "archives",
                        "enquiry", "bureau", "dispatching", "stand by"}
        for msg in PLACEHOLDER_MESSAGES:
            lower = msg.lower()
            assert any(w in lower for w in butler_words), (
                f"Placeholder not in character: {msg}"
            )


class TestChunkText:
    """Tests for the _chunk_text helper."""

    def test_short_text_single_chunk(self):
        from bantz.interface.telegram_bot import _chunk_text
        assert _chunk_text("Hello world") == ["Hello world"]

    def test_exact_boundary(self):
        from bantz.interface.telegram_bot import _chunk_text
        text = "X" * 4000
        assert _chunk_text(text) == [text]

    def test_splits_at_paragraph_boundary(self):
        from bantz.interface.telegram_bot import _chunk_text
        a = "A" * 2500
        b = "B" * 2500
        chunks = _chunk_text(a + "\n\n" + b)
        assert chunks == [a, b]

    def test_hard_split_no_newlines(self):
        from bantz.interface.telegram_bot import _chunk_text
        text = "X" * 8000
        chunks = _chunk_text(text)
        assert len(chunks) == 2
        assert "".join(chunks) == text

    def test_custom_max_len(self):
        from bantz.interface.telegram_bot import _chunk_text
        text = "short"
        assert _chunk_text(text, max_len=3) == ["sho", "rt"]


class TestSafeEdit:
    """Tests for the _safe_edit fallback chain."""

    @pytest.mark.asyncio
    async def test_normal_edit_succeeds(self):
        from bantz.interface.telegram_bot import _safe_edit
        ph = AsyncMock()
        ph.edit_text = AsyncMock()
        result = await _safe_edit(ph, "hello")
        assert result is True
        ph.edit_text.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_fallback_to_plain_text(self):
        from bantz.interface.telegram_bot import _safe_edit
        call_count = 0

        async def flaky(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Markdown error")
            return None

        ph = AsyncMock()
        ph.edit_text = AsyncMock(side_effect=flaky)
        result = await _safe_edit(ph, "test")
        assert result is True
        assert call_count == 2
        # Second call has parse_mode=None
        assert ph.edit_text.call_args_list[1][1]["parse_mode"] is None

    @pytest.mark.asyncio
    async def test_complete_failure_returns_false(self):
        from bantz.interface.telegram_bot import _safe_edit
        ph = AsyncMock()
        ph.edit_text = AsyncMock(side_effect=Exception("total failure"))
        result = await _safe_edit(ph, "test")
        assert result is False
