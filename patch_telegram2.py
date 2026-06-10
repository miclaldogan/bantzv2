import re

with open("src/bantz/interface/telegram_bot.py", "r") as f:
    content = f.read()

# Fix send_system_notification
content = re.sub(
    r"coros = \[app\.bot\.send_message\(chat_id=chat_id, text=filtered\) for chat_id in _active_chats\]\n\s+if coros:\n\s+results = await asyncio\.gather\(\*coros, return_exceptions=True\)\n\s+for chat_id, res in zip\(_active_chats, results\):\n\s+if isinstance\(res, Exception\):\n\s+log\.debug\(\"send_system_notification: failed for %d: %s\", chat_id, res\)",
    r"""# ⚡ Bolt Optimization: Broadcast messages concurrently to eliminate N+1 latency bottlenecks
        chats = list(_active_chats)
        coros = [app.bot.send_message(chat_id=chat_id, text=filtered) for chat_id in chats]
        if coros:
            results = await asyncio.gather(*coros, return_exceptions=True)
            for chat_id, res in zip(chats, results):
                if isinstance(res, Exception):
                    log.debug("send_system_notification: failed for %d: %s", chat_id, res)""",
    content
)

# Fix _daily_digest_job
content = re.sub(
    r"coros = \[context\.bot\.send_message\(chat_id=chat_id, text=text\) for chat_id in _active_chats\]\n\s+if coros:\n\s+results = await asyncio\.gather\(\*coros, return_exceptions=True\)\n\s+for chat_id, res in zip\(_active_chats, results\):\n\s+if isinstance\(res, Exception\):\n\s+log\.debug\(\"Failed to send daily digest to %d: %s\", chat_id, res\)",
    r"""# ⚡ Bolt Optimization: Broadcast messages concurrently to eliminate N+1 latency bottlenecks
    chats = list(_active_chats)
    coros = [context.bot.send_message(chat_id=chat_id, text=text) for chat_id in chats]
    if coros:
        results = await asyncio.gather(*coros, return_exceptions=True)
        for chat_id, res in zip(chats, results):
            if isinstance(res, Exception):
                log.debug("Failed to send daily digest to %d: %s", chat_id, res)""",
    content
)

# Fix _weekly_digest_job
content = re.sub(
    r"coros = \[context\.bot\.send_message\(chat_id=chat_id, text=text\) for chat_id in _active_chats\]\n\s+if coros:\n\s+results = await asyncio\.gather\(\*coros, return_exceptions=True\)\n\s+for chat_id, res in zip\(_active_chats, results\):\n\s+if isinstance\(res, Exception\):\n\s+log\.debug\(\"Failed to send weekly digest to %d: %s\", chat_id, res\)",
    r"""# ⚡ Bolt Optimization: Broadcast messages concurrently to eliminate N+1 latency bottlenecks
    chats = list(_active_chats)
    coros = [context.bot.send_message(chat_id=chat_id, text=text) for chat_id in chats]
    if coros:
        results = await asyncio.gather(*coros, return_exceptions=True)
        for chat_id, res in zip(chats, results):
            if isinstance(res, Exception):
                log.debug("Failed to send weekly digest to %d: %s", chat_id, res)""",
    content
)

# Fix _check_reminders_job
content = re.sub(
    r"coros = \[context\.bot\.send_message\(chat_id=chat_id, text=filtered\) for chat_id in _active_chats\]\n\s+if coros:\n\s+results = await asyncio\.gather\(\*coros, return_exceptions=True\)\n\s+for chat_id, res in zip\(_active_chats, results\):\n\s+if isinstance\(res, Exception\):\n\s+log\.debug\(\"Failed to send reminder to %d: %s\", chat_id, res\)",
    r"""# ⚡ Bolt Optimization: Broadcast messages concurrently to eliminate N+1 latency bottlenecks
        chats = list(_active_chats)
        coros = [context.bot.send_message(chat_id=chat_id, text=filtered) for chat_id in chats]
        if coros:
            results = await asyncio.gather(*coros, return_exceptions=True)
            for chat_id, res in zip(chats, results):
                if isinstance(res, Exception):
                    log.debug("Failed to send reminder to %d: %s", chat_id, res)""",
    content
)

with open("src/bantz/interface/telegram_bot.py", "w") as f:
    f.write(content)
