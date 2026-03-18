"""
Bantz v3 — Telegram Bot

Dual-path remote telegraph for phone access:
  1. Express Wire (/commands) — fast, LLM-free direct tool calls
  2. Cognitive Wire (free text) — full brain.process() with LLM (#178)

Commands:
    /briefing   → full morning summary
    /hava       → weather report
    /mail       → unread emails
    /takvim     → today's calendar events
    /odev       → upcoming assignments
    /ders       → today's schedule
    /siradaki   → next class
    /haber      → latest news

Usage:
    python -m bantz.interface.telegram_bot

Env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_ALLOWED_USERS=123456,789012   # optional whitelist
    TELEGRAM_LLM_MODE=true                 # enable free-text → LLM
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections import defaultdict
from typing import Callable, Coroutine, Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bantz.config import config

# ── Proxy support (Turkey blocks api.telegram.org) ──────────────────────
_PROXY = config.telegram_proxy.strip() or os.environ.get("HTTPS_PROXY", "").strip()
if _PROXY:
    # python-telegram-bot uses httpx under the hood;
    # setting these env vars makes httpx route through the proxy.
    os.environ.setdefault("HTTPS_PROXY", _PROXY)
    os.environ.setdefault("HTTP_PROXY", _PROXY)

logging.basicConfig(
    format="%(asctime)s [bantz-tg] %(levelname)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Security: allowed user whitelist ──────────────────────────────────────────

_ALLOWED: set[int] | None = None
if config.telegram_allowed_users.strip():
    _ALLOWED = {
        int(uid.strip())
        for uid in config.telegram_allowed_users.split(",")
        if uid.strip().isdigit()
    }

# ── Active chat IDs for proactive notifications ──────────────────────────────
_active_chats: set[int] = set()

# ── Rate limiter (max 10 messages per minute per user) ────────────────────────
_RATE_WINDOW = 60  # seconds
_RATE_LIMIT = 10   # max messages per window
_rate_log: dict[int, list[float]] = defaultdict(list)

# ── Placeholder messages for progress indication (#181) ───────────────────────
PLACEHOLDER_MESSAGES: list[str] = [
    "📟 One moment, ma'am — consulting the Grand Telegraph Archives...",
    "📟 Dispatching a query to the archives. Please hold the line, ma'am...",
    "📟 The telegraph office is processing your request. A moment, if you please...",
    "📟 Reaching out to the information bureau, ma'am. Do stand by...",
    "📟 The wires are humming with your enquiry. Just a moment...",
]

# ── Throttled streaming config (#181 follow-up) ──────────────────────────────
_STREAM_INTERVAL: float = 2.0  # seconds between edit_text updates

# ── Message processing lock — strict serial execution ─────────────────────────
# Prevents context-window corruption when user sends burst messages.
_msg_lock = asyncio.Lock()


# ── Telegram Spam Filter (#184) ───────────────────────────────────────────────

class _TelegramSpamFilter:
    """Aggregates rapid-fire system/maintenance messages into a single summary.

    User messages always pass through immediately.
    System messages are buffered: if ≥ SPAM_THRESHOLD arrive within
    WINDOW_SECONDS, they are collapsed into one butler-style summary.
    """

    WINDOW_SECONDS: float = 60.0
    SPAM_THRESHOLD: int = 5

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._window_start: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def should_send(self, message: str, *, is_system: bool = False) -> str | None:
        """Return the message to send, or None if it was buffered.

        Non-system messages always return immediately.
        System messages are rate-checked against the sliding window.
        When the threshold is reached the buffer is flushed as a summary.
        """
        if not is_system:
            return message

        async with self._lock:
            now = time.monotonic()

            if now - self._window_start > self.WINDOW_SECONDS:
                self._buffer.clear()
                self._window_start = now

            self._buffer.append(message)

            if len(self._buffer) >= self.SPAM_THRESHOLD:
                count = len(self._buffer)
                summary = (
                    f"Ma'am, the machines have been undergoing routine maintenance. "
                    f"{count} tasks completed in the past minute. All systems nominal."
                )
                self._buffer.clear()
                self._window_start = now
                log.info("_TelegramSpamFilter: bundled %d system messages into summary", count)
                return summary

            return message

    async def flush(self) -> str | None:
        """Flush any buffered messages (e.g. on shutdown). Returns summary or None."""
        async with self._lock:
            if not self._buffer:
                return None
            count = len(self._buffer)
            summary = f"Ma'am, {count} maintenance tasks completed. All quiet now."
            self._buffer.clear()
            return summary


_spam_filter = _TelegramSpamFilter()


async def send_system_notification(text: str) -> None:
    """Send a system/background notification to all active chats, via spam filter.

    Use this instead of raw bot.send_message() for proactive/background sends
    so that rapid-fire maintenance messages are automatically bundled (#184).
    Silently no-ops when no bot application is running or no chats are active.
    """
    filtered = await _spam_filter.should_send(text, is_system=True)
    if filtered is None:
        return  # buffered, not yet due for sending

    if not _active_chats:
        return

    try:
        from telegram.ext import Application as _TgApp
        app: _TgApp | None = _current_app
        if app is None:
            return
        for chat_id in _active_chats:
            try:
                await app.bot.send_message(chat_id=chat_id, text=filtered)
            except Exception as exc:
                log.debug("send_system_notification: failed for %d: %s", chat_id, exc)
    except Exception as exc:
        log.debug("send_system_notification: %s", exc)


# Module-level reference to the running Application, set in run_bot()
_current_app = None


def _authorized(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    """Decorator: silently drop messages from non-whitelisted users (#178)."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if _ALLOWED and update.effective_user and update.effective_user.id not in _ALLOWED:
            return  # Silent stranger — no reply, total stealth
        return await func(update, context)
    return wrapper


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_len: int = 4000) -> list[str]:
    """Split *text* at paragraph boundaries, respecting *max_len*.

    Returns a list of chunks, each ≤ max_len characters.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = ""
            # If single paragraph exceeds max_len, hard-split on newlines
            if len(para) > max_len:
                for line in para.split("\n"):
                    if len(line) > max_len:
                        # No newline split possible — brute-force at max_len
                        for i in range(0, len(line), max_len):
                            chunks.append(line[i:i + max_len])
                        current = ""
                    elif current and len(current) + len(line) + 1 <= max_len:
                        current = current + "\n" + line
                    else:
                        if current:
                            chunks.append(current)
                        current = line
            else:
                current = para
    if current:
        chunks.append(current)

    return chunks


async def _safe_reply(update: Update, text: str) -> None:
    """Send a reply, splitting at paragraph boundaries for Telegram's 4096 limit."""
    for chunk in _chunk_text(text):
        await update.message.reply_text(chunk)


async def _safe_edit(placeholder, text: str) -> bool:
    """Edit placeholder message with fallback chain.

    Senior Architect note — MarkdownV2 trap:
      1. edit_text(text) — may fail if Telegram parses markdown entities
      2. edit_text(text, parse_mode=None) — force plain text
      3. return False → caller should delete placeholder + reply_text
    """
    try:
        await placeholder.edit_text(text)
        return True
    except Exception:
        try:
            await placeholder.edit_text(text, parse_mode=None)
            return True
        except Exception:
            return False


async def _stream_to_placeholder(placeholder, stream, *, interval: float = _STREAM_INTERVAL) -> str:
    """Consume *stream*, updating *placeholder* every *interval* seconds.

    Returns the full accumulated response text.  The placeholder is edited
    with the latest accumulated text at each tick, so the user can watch
    the response being generated live without hitting Telegram rate limits.
    """
    parts: list[str] = []
    last_edit = 0.0
    last_text = ""

    async for chunk in stream:
        parts.append(chunk)
        now = asyncio.get_event_loop().time()
        if now - last_edit >= interval:
            current = "".join(parts).strip()
            if current and current != last_text:
                try:
                    await placeholder.edit_text(current + " ▍")
                except Exception:
                    try:
                        await placeholder.edit_text(current + " ▍", parse_mode=None)
                    except Exception:
                        pass  # skip this tick, retry next
                last_text = current
                last_edit = now

    return "".join(parts)


# ── Command Handlers ─────────────────────────────────────────────────────────

@_authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _active_chats.add(update.effective_chat.id)
    llm_hint = (
        "\n\n💬 Or just type any message — Bantz will respond "
        "with the full power of his brain."
    ) if config.telegram_llm_mode else ""
    await update.message.reply_text(
        "🦌 Bantz is live!\n\n"
        "Commands:\n"
        "/briefing — daily summary\n"
        "/hava — weather report\n"
        "/mail — unread emails\n"
        "/takvim — today's calendar\n"
        "/odev — upcoming assignments\n"
        "/ders — today's schedule\n"
        "/siradaki — next class\n"
        "/haber — latest news\n"
        "/hatirlatici — list reminders\n"
        "/digest — evening daily digest\n"
        "/weekly — weekly summary"
        + llm_hint
    )


@_authorized
async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.briefing import briefing
        text = await briefing.generate()
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Briefing error: {exc}")


@_authorized
async def cmd_hava(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.weather import WeatherTool
        result = await WeatherTool().execute(city="")
        await _safe_reply(update, result.output if result.success else f"Error: {result.error}")
    except Exception as exc:
        await update.message.reply_text(f"Weather error: {exc}")


@_authorized
async def cmd_mail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.gmail import GmailTool
        result = await GmailTool().execute(action="filter", q="is:unread", max_results=10)
        if result.success:
            text = result.output.strip() or "No unread emails ✓"
        else:
            text = f"Error: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Mail error: {exc}")


@_authorized
async def cmd_takvim(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.calendar import CalendarTool
        result = await CalendarTool().execute(action="today")
        if result.success:
            text = result.output.strip() or "No events today ✓"
        else:
            text = f"Error: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Calendar error: {exc}")


@_authorized
async def cmd_odev(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.classroom import ClassroomTool
        result = await ClassroomTool().execute(action="upcoming")
        if result.success:
            text = result.output.strip() or "No upcoming assignments ✓"
        else:
            text = f"Error: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Assignment error: {exc}")


@_authorized
async def cmd_ders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.schedule import schedule
        text = schedule.format_today()
        await _safe_reply(update, text or "No classes today ✓")
    except Exception as exc:
        await update.message.reply_text(f"Schedule error: {exc}")


@_authorized
async def cmd_siradaki(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.schedule import schedule
        text = schedule.format_next()
        await _safe_reply(update, text or "No upcoming classes ✓")
    except Exception as exc:
        await update.message.reply_text(f"Schedule error: {exc}")


@_authorized
async def cmd_haber(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.news import NewsTool
        result = await NewsTool().execute(source="all", limit=5)
        if result.success:
            text = result.output.strip() or "No news found"
        else:
            text = f"Error: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"News error: {exc}")


@_authorized
async def cmd_hatirlatici(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List upcoming reminders."""
    _active_chats.add(update.effective_chat.id)
    try:
        from bantz.core.scheduler import scheduler
        text = scheduler.format_upcoming(limit=10)
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Reminder error: {exc}")


@_authorized
async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """On-demand daily digest."""
    _active_chats.add(update.effective_chat.id)
    await update.message.reply_text("⏳ Generating digest…")
    try:
        from bantz.core.digest import digest_manager
        text = await digest_manager.daily_digest()
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Digest error: {exc}")


@_authorized
async def cmd_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """On-demand weekly digest."""
    _active_chats.add(update.effective_chat.id)
    await update.message.reply_text("⏳ Generating weekly digest…")
    try:
        from bantz.core.digest import digest_manager
        text = await digest_manager.weekly_digest()
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Weekly digest error: {exc}")


# ── Proactive digest notifications ───────────────────────────────────────────

async def _daily_digest_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduled job: send evening daily digest to all active chats."""
    if not _active_chats or not config.daily_digest_enabled:
        return

    try:
        from bantz.core.digest import digest_manager
        text = await digest_manager.daily_digest()
    except Exception as exc:
        log.warning("Daily digest job failed: %s", exc)
        return

    for chat_id in _active_chats:
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            log.debug("Failed to send daily digest to %d: %s", chat_id, exc)


async def _weekly_digest_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduled job: send weekly digest on configured day."""
    if not _active_chats or not config.weekly_digest_enabled:
        return

    from datetime import datetime
    now = datetime.now()
    _DAY_MAP = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    target_day = _DAY_MAP.get(config.weekly_digest_day.lower(), 6)
    if now.weekday() != target_day:
        return

    try:
        from bantz.core.digest import digest_manager
        text = await digest_manager.weekly_digest()
    except Exception as exc:
        log.warning("Weekly digest job failed: %s", exc)
        return

    for chat_id in _active_chats:
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            log.debug("Failed to send weekly digest to %d: %s", chat_id, exc)


# ── Proactive reminder notifications ─────────────────────────────────────────

async def _check_reminders_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic job: check for due reminders and send notifications."""
    if not _active_chats:
        return

    try:
        from bantz.core.scheduler import scheduler
        due = scheduler.check_due()
    except Exception:
        due = []

    # Also check location-triggered reminders
    try:
        from bantz.core.places import places
        place_due = places.pop_place_reminders()
        due.extend(place_due)
    except Exception:
        pass

    for item in due:
        title = item.get("title", "Reminder")
        place_label = item.get("_place_label")

        if place_label:
            text = f"📍⏰ Reminder (at {place_label}): {title}"
        else:
            text = f"⏰ Reminder: {title}"

        # Route through spam filter so burst reminders are bundled (#184)
        filtered = await _spam_filter.should_send(text, is_system=True)
        if filtered is None:
            continue

        for chat_id in _active_chats:
            try:
                await context.bot.send_message(chat_id=chat_id, text=filtered)
            except Exception as exc:
                log.debug("Failed to send reminder to %d: %s", chat_id, exc)


# ── Cognitive Wire: free-text → brain.process() (#178) ────────────────────────


def _is_maintenance_spam(result) -> bool:
    """Return True for ANY maintenance result on Telegram.

    Maintenance output is noisy and irrelevant on mobile — suppress ALL of it.
    Users can run /briefing to see system health if they want.
    """
    if getattr(result, "tool_used", None) != "maintenance":
        return False
    return True


def _is_rate_limited(user_id: int) -> bool:
    """Return True if user has exceeded 10 messages/minute."""
    now = time.monotonic()
    timestamps = _rate_log[user_id]
    # Prune old entries outside the window
    _rate_log[user_id] = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(_rate_log[user_id]) >= _RATE_LIMIT:
        return True
    _rate_log[user_id].append(now)
    return False


@_authorized
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cognitive Wire — route free text through brain.process(is_remote=True).

    Progress indicator flow (#181):
      1. Send placeholder immediately (covers Telegram's 5s typing limit)
      2. brain.process() with typing indicator
      3. Edit placeholder with actual response (chunked if >4096)

    Messages are serialised via _msg_lock so burst sends don't corrupt
    the context window.
    """
    if not config.telegram_llm_mode:
        await update.message.reply_text(
            "🔒 The telegraph only accepts strict commands (/).\n"
            "Use /start to see available commands."
        )
        return

    user_id = update.effective_user.id if update.effective_user else 0
    if _is_rate_limited(user_id):
        await update.message.reply_text(
            "⏳ Too many messages — please wait a moment."
        )
        return

    _active_chats.add(update.effective_chat.id)
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    # Step 1: Immediate placeholder — user sees instant feedback (#181)
    placeholder = await update.message.reply_text(
        random.choice(PLACEHOLDER_MESSAGES)
    )

    # Step 2: Typing indicator while brain processes
    await update.message.chat.send_action(ChatAction.TYPING)

    # Acquire lock — serialise message processing to prevent context corruption
    async with _msg_lock:
        try:
            from bantz.core.brain import brain
            result = await brain.process(user_text, is_remote=True)

            # ── Maintenance spam filter ──────────────────────────────
            if _is_maintenance_spam(result):
                # All steps green — suppress on Telegram, just remove placeholder
                try:
                    await placeholder.delete()
                except Exception:
                    pass
                return

            # Collect response: prefer stream if available
            if result.stream:
                response = await _stream_to_placeholder(placeholder, result.stream)
            else:
                response = result.response

            if response and response.strip():
                cleaned = response.strip()

                # Step 3: Edit placeholder with actual response (#181)
                chunks = _chunk_text(cleaned)
                if not await _safe_edit(placeholder, chunks[0]):
                    # Fallback: delete placeholder, send via reply_text
                    try:
                        await placeholder.delete()
                    except Exception:
                        pass
                    await _safe_reply(update, cleaned)
                else:
                    # Remaining chunks as new messages
                    for extra in chunks[1:]:
                        await update.message.reply_text(extra)

                # ── Persist streamed response to memory + graph (#178 fix) ────
                # Brain only auto-saves non-streaming responses; for streams the
                # consumer is responsible (mirrors TUI behaviour in app.py L718).
                if result.stream:
                    try:
                        from bantz.data import data_layer
                        data_layer.conversations.add(
                            "assistant", cleaned,
                            tool_used=result.tool_used,
                        )
                    except Exception:
                        log.debug("Failed to persist Telegram response to DB")
                    try:
                        await brain._graph_store(
                            user_text, cleaned, result.tool_used,
                        )
                    except Exception:
                        log.debug("Failed to persist Telegram response to graph")
            else:
                if not await _safe_edit(placeholder, "…"):
                    await update.message.reply_text("…")
        except Exception as exc:
            log.exception("Cognitive wire error for user %d", user_id)
            if not await _safe_edit(placeholder, f"⚠️ Error: {exc}"):
                await update.message.reply_text(f"⚠️ Error: {exc}")


# ── Bot runner ────────────────────────────────────────────────────────────────

def run_bot() -> None:
    token = config.telegram_bot_token
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        print("   → Add TELEGRAM_BOT_TOKEN=... to .env")
        print("   → or run: bantz --setup telegram")
        return

    global _current_app
    app = Application.builder().token(token).build()
    _current_app = app

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(CommandHandler("hava", cmd_hava))
    app.add_handler(CommandHandler("mail", cmd_mail))
    app.add_handler(CommandHandler("takvim", cmd_takvim))
    app.add_handler(CommandHandler("odev", cmd_odev))
    app.add_handler(CommandHandler("ders", cmd_ders))
    app.add_handler(CommandHandler("siradaki", cmd_siradaki))
    app.add_handler(CommandHandler("haber", cmd_haber))
    app.add_handler(CommandHandler("hatirlatici", cmd_hatirlatici))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("weekly", cmd_weekly))

    # Cognitive Wire — free text → LLM (#178)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Proactive reminder check — runs every 30 seconds
    app.job_queue.run_repeating(
        _check_reminders_job,
        interval=config.reminder_check_interval,
        first=10,
        name="reminder_check",
    )

    # Daily digest — runs every 60s, job itself checks if it's the right time
    import datetime as _dt
    _digest_time = _dt.time(
        hour=config.daily_digest_hour,
        minute=config.daily_digest_minute,
    )
    app.job_queue.run_daily(
        _daily_digest_job,
        time=_digest_time,
        name="daily_digest",
    )

    # Weekly digest — runs daily at configured time, job checks day-of-week
    _weekly_time = _dt.time(
        hour=config.weekly_digest_hour,
        minute=config.weekly_digest_minute,
    )
    app.job_queue.run_daily(
        _weekly_digest_job,
        time=_weekly_time,
        name="weekly_digest",
    )

    # ── Warm-up: pre-load the model into VRAM (mirrors TUI on_mount) ──────
    async def _warm_up_ollama(context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a throwaway prompt so the first real message hits a warm model."""
        try:
            from bantz.llm.ollama_client import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
            log.info("   Ollama warm-up complete ✓")
        except Exception:
            log.debug("Ollama warm-up skipped (not available)")

    app.job_queue.run_once(_warm_up_ollama, when=2, name="ollama_warmup")

    # ── Daily session rotation — prevent infinite session accumulation ────
    async def _rotate_session(context: ContextTypes.DEFAULT_TYPE) -> None:
        """Start a fresh memory session each day (mirrors TUI behaviour)."""
        try:
            from bantz.core.brain import brain
            brain._memory_ready = False  # force re-init → new_session()
            log.info("   Daily session rotation ✓")
        except Exception:
            log.debug("Session rotation failed")

    import datetime as _dt
    _session_rotate_time = _dt.time(hour=4, minute=0)  # 4 AM daily
    app.job_queue.run_daily(
        _rotate_session, time=_session_rotate_time, name="session_rotation",
    )

    log.info("🦌 Bantz Telegram bot starting...")
    if _PROXY:
        log.info(f"   Proxy: {_PROXY}")
    if _ALLOWED:
        log.info(f"   Allowed users: {_ALLOWED}")
    else:
        log.info("   ⚠ No user restriction — anyone can use it")
    log.info(f"   LLM mode: {'ON' if config.telegram_llm_mode else 'OFF'}")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
