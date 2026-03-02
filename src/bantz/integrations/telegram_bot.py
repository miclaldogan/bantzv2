"""
Bantz — Telegram Bot (backward-compatibility shim)

The Telegram bot has moved to bantz.interface.telegram_bot.
This file re-exports for backward compatibility.
"""
from bantz.interface.telegram_bot import run_bot  # noqa: F401

if __name__ == "__main__":
    run_bot()
