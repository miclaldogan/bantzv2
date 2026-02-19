"""
Bantz v2 — Telegram Bot

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
    python -m bantz.integrations.telegram_bot

Env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_ALLOWED_USERS=123456,789012   # optional whitelist
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from bantz.config import config

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


def _authorized(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    """Decorator: reject messages from non-whitelisted users."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if _ALLOWED and update.effective_user and update.effective_user.id not in _ALLOWED:
            await update.message.reply_text("⛔ Yetkisiz erişim.")
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
    await update.message.reply_text(
        "🦌 Bantz yayında!\n\n"
        "Komutlar:\n"
        "/briefing — günlük özet\n"
        "/hava — hava durumu\n"
        "/mail — okunmamış mailler\n"
        "/takvim — bugünün takvimi\n"
        "/odev — yaklaşan ödevler\n"
        "/ders — bugünün ders programı\n"
        "/siradaki — sıradaki ders\n"
        "/haber — son haberler"
    )


@_authorized
async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.briefing import briefing
        text = await briefing.generate()
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Briefing hatası: {exc}")


@_authorized
async def cmd_hava(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.weather import WeatherTool
        result = await WeatherTool().execute(city="")
        await _safe_reply(update, result.output if result.success else f"Hata: {result.error}")
    except Exception as exc:
        await update.message.reply_text(f"Hava hatası: {exc}")


@_authorized
async def cmd_mail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.gmail import GmailTool
        result = await GmailTool().execute(action="filter", q="is:unread", max_results=10)
        if result.success:
            text = result.output.strip() or "Okunmamış mail yok ✓"
        else:
            text = f"Hata: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Mail hatası: {exc}")


@_authorized
async def cmd_takvim(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.calendar import CalendarTool
        result = await CalendarTool().execute(action="today")
        if result.success:
            text = result.output.strip() or "Bugün etkinlik yok ✓"
        else:
            text = f"Hata: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Takvim hatası: {exc}")


@_authorized
async def cmd_odev(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.classroom import ClassroomTool
        result = await ClassroomTool().execute(action="upcoming")
        if result.success:
            text = result.output.strip() or "Yaklaşan ödev yok ✓"
        else:
            text = f"Hata: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Ödev hatası: {exc}")


@_authorized
async def cmd_ders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.schedule import schedule
        text = schedule.format_today()
        await _safe_reply(update, text or "Bugün ders yok ✓")
    except Exception as exc:
        await update.message.reply_text(f"Ders hatası: {exc}")


@_authorized
async def cmd_siradaki(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.core.schedule import schedule
        text = schedule.format_next()
        await _safe_reply(update, text or "Sırada ders yok ✓")
    except Exception as exc:
        await update.message.reply_text(f"Ders hatası: {exc}")


@_authorized
async def cmd_haber(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from bantz.tools.news import NewsTool
        result = await NewsTool().execute(source="all", limit=5)
        if result.success:
            text = result.output.strip() or "Haber bulunamadı"
        else:
            text = f"Hata: {result.error}"
        await _safe_reply(update, text)
    except Exception as exc:
        await update.message.reply_text(f"Haber hatası: {exc}")


# ── Bot runner ────────────────────────────────────────────────────────────────

def run_bot() -> None:
    token = config.telegram_bot_token
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN ayarlanmamış!")
        print("   → .env dosyasına TELEGRAM_BOT_TOKEN=... ekle")
        print("   → ya da: bantz --setup telegram")
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

    log.info("🦌 Bantz Telegram bot başlatılıyor...")
    if _ALLOWED:
        log.info(f"   İzinli kullanıcılar: {_ALLOWED}")
    else:
        log.info("   ⚠ Kullanıcı kısıtlaması yok — herkes kullanabilir")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
