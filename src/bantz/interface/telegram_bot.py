"""
Bantz v3 — Telegram Bot

Lightweight bot for phone access. No LLM needed — calls tools directly.
Each command is isolated: if one service fails, others keep working.

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
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Coroutine, Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
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


def _authorized(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    """Decorator: reject messages from non-whitelisted users."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if _ALLOWED and update.effective_user and update.effective_user.id not in _ALLOWED:
            await update.message.reply_text("⛔ Unauthorized access.")
            return
        return await func(update, context)
    return wrapper


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _safe_reply(update: Update, text: str) -> None:
    """Send a reply, splitting if too long for Telegram's 4096 char limit."""
    if len(text) <= 4000:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), 4000):
            await update.message.reply_text(text[i:i + 4000])


# ── Command Handlers ─────────────────────────────────────────────────────────

@_authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _active_chats.add(update.effective_chat.id)
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
        "/hatirlatici — list reminders"
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

        for chat_id in _active_chats:
            try:
                await context.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                log.debug("Failed to send reminder to %d: %s", chat_id, exc)


# ── Bot runner ────────────────────────────────────────────────────────────────

def run_bot() -> None:
    token = config.telegram_bot_token
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        print("   → Add TELEGRAM_BOT_TOKEN=... to .env")
        print("   → or run: bantz --setup telegram")
        return

    app = Application.builder().token(token).build()

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

    # Proactive reminder check — runs every 30 seconds
    app.job_queue.run_repeating(
        _check_reminders_job,
        interval=config.reminder_check_interval,
        first=10,
        name="reminder_check",
    )

    log.info("🦌 Bantz Telegram bot starting...")
    if _PROXY:
        log.info(f"   Proxy: {_PROXY}")
    if _ALLOWED:
        log.info(f"   Allowed users: {_ALLOWED}")
    else:
        log.info("   ⚠ No user restriction — anyone can use it")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
