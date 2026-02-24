"""
Bantz v3 — Telegram Bot

Full phone access: slash commands for quick actions + free-text chat routed
through Brain (same pipeline as the TUI).

Commands:
    /start      → help & command list
    /briefing   → full morning summary
    /weather    → current weather
    /mail       → unread emails
    /calendar   → today's events
    /assignments→ upcoming assignments
    /schedule   → today's class schedule
    /next       → next class
    /news       → latest headlines
    /graph      → Neo4j memory stats

Free text:
    Any message that isn't a command is routed through brain.process()
    — same tools, same CoT routing, same Gemini/Ollama finalizer as the TUI.

Usage:
    python -m bantz.integrations.telegram_bot
    # or via __main__.py: bantz --telegram

Env:
    BANTZ_TELEGRAM_BOT_TOKEN=...
    BANTZ_TELEGRAM_ALLOWED_USERS=123456,789012   # comma-separated user IDs
    BANTZ_TELEGRAM_PROXY=http://...              # optional, for restricted networks
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Coroutine

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from bantz.config import config

# ── Proxy support ─────────────────────────────────────────────────────────────
_PROXY = config.telegram_proxy.strip() or os.environ.get("HTTPS_PROXY", "").strip()
if _PROXY:
    os.environ.setdefault("HTTPS_PROXY", _PROXY)
    os.environ.setdefault("HTTP_PROXY",  _PROXY)

logging.basicConfig(
    format="%(asctime)s [bantz-tg] %(levelname)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Allowed user whitelist ─────────────────────────────────────────────────────
_ALLOWED: set[int] | None = None
if config.telegram_allowed_users.strip():
    _ALLOWED = {
        int(uid.strip())
        for uid in config.telegram_allowed_users.split(",")
        if uid.strip().isdigit()
    }


def _authorized(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if _ALLOWED and update.effective_user and update.effective_user.id not in _ALLOWED:
            await update.message.reply_text("⛔ Unauthorized.")
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


# ── Helper ────────────────────────────────────────────────────────────────────

async def _reply(update: Update, text: str) -> None:
    """Send reply, splitting at Telegram's 4096-char limit."""
    text = text.strip() or "(no output)"
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i + 4000])


# ── Command Handlers ──────────────────────────────────────────────────────────

@_authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(update,
        "Bantz v3 — Operations Director\n\n"
        "Commands:\n"
        "/briefing     — morning summary\n"
        "/weather      — current weather\n"
        "/mail         — unread emails\n"
        "/calendar     — today's events\n"
        "/assignments  — upcoming deadlines\n"
        "/schedule     — today's classes\n"
        "/next         — next class\n"
        "/news         — latest headlines\n"
        "/graph        — memory graph stats\n\n"
        "Or just type anything — I'll route it."
    )


@_authorized
async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Generating briefing...")
    try:
        from bantz.core.briefing import briefing
        text = await briefing.generate()
        await _reply(update, text)
    except Exception as exc:
        await _reply(update, f"Briefing error: {exc}")


@_authorized
async def cmd_weather(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.weather import WeatherTool
        result = await WeatherTool().execute(city=config.location_city or "")
        await _reply(update, result.output if result.success else f"Error: {result.error}")
    except Exception as exc:
        await _reply(update, f"Weather error: {exc}")


@_authorized
async def cmd_mail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.gmail import GmailTool
        result = await GmailTool().execute(action="unread")
        await _reply(update, result.output if result.success else f"Error: {result.error}")
    except Exception as exc:
        await _reply(update, f"Mail error: {exc}")


@_authorized
async def cmd_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.calendar import CalendarTool
        result = await CalendarTool().execute(action="today")
        await _reply(update, result.output if result.success else f"Error: {result.error}")
    except Exception as exc:
        await _reply(update, f"Calendar error: {exc}")


@_authorized
async def cmd_assignments(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.classroom import ClassroomTool
        result = await ClassroomTool().execute(action="upcoming")
        await _reply(update, result.output if result.success else f"Error: {result.error}")
    except Exception as exc:
        await _reply(update, f"Assignments error: {exc}")


@_authorized
async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.schedule import schedule
        await _reply(update, schedule.format_today() or "No classes today.")
    except Exception as exc:
        await _reply(update, f"Schedule error: {exc}")


@_authorized
async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.schedule import schedule
        await _reply(update, schedule.format_next() or "No upcoming classes.")
    except Exception as exc:
        await _reply(update, f"Schedule error: {exc}")


@_authorized
async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.news import NewsTool
        result = await NewsTool().execute(source="all", limit=5)
        await _reply(update, result.output if result.success else f"Error: {result.error}")
    except Exception as exc:
        await _reply(update, f"News error: {exc}")


@_authorized
async def cmd_graph(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.memory.graph import graph_memory
        stats = graph_memory.stats()
        if stats["available"]:
            await _reply(update,
                f"Neo4j graph memory\n"
                f"Nodes:     {stats['nodes']:,}\n"
                f"Relations: {stats['relations']:,}"
            )
        else:
            await _reply(update, "Neo4j offline — using SQLite fallback.")
    except Exception as exc:
        await _reply(update, f"Graph error: {exc}")


# ── Free-text chat handler ────────────────────────────────────────────────────

@_authorized
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route free-text messages through Brain — same pipeline as the TUI."""
    text = update.message.text.strip()
    if not text:
        return

    await update.message.reply_text("⟳ thinking...")

    try:
        from bantz.core.brain import brain
        result = await brain.process(text)
        response = result.response or "(no response)"
        if result.tool_used:
            response = f"[{result.tool_used}]\n{response}"
        await _reply(update, response)
    except Exception as exc:
        await _reply(update, f"Error: {exc}")


# ── Bot runner ────────────────────────────────────────────────────────────────

def run_bot() -> None:
    token = config.telegram_bot_token
    if not token:
        print("BANTZ_TELEGRAM_BOT_TOKEN is not set.")
        print("Add it to .env:  BANTZ_TELEGRAM_BOT_TOKEN=<your token>")
        return

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_start))
    app.add_handler(CommandHandler("briefing",    cmd_briefing))
    app.add_handler(CommandHandler("weather",     cmd_weather))
    app.add_handler(CommandHandler("mail",        cmd_mail))
    app.add_handler(CommandHandler("calendar",    cmd_calendar))
    app.add_handler(CommandHandler("assignments", cmd_assignments))
    app.add_handler(CommandHandler("schedule",    cmd_schedule))
    app.add_handler(CommandHandler("next",        cmd_next))
    app.add_handler(CommandHandler("news",        cmd_news))
    app.add_handler(CommandHandler("graph",       cmd_graph))

    # Free-text → Brain
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bantz Telegram bot starting...")
    if _PROXY:
        log.info("  Proxy: %s", _PROXY)
    if _ALLOWED:
        log.info("  Allowed users: %s", _ALLOWED)
    else:
        log.warning("  No user whitelist — anyone can message this bot")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
