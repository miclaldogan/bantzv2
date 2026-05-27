"""
Bantz v2 — Entry point

Commands:
  bantz                         → TUI
  bantz --once "query"          → single query, no UI
  bantz --daemon                → headless daemon (scheduler + GPS, no TUI)
  bantz --doctor                → system health check
  bantz --setup onboarding         → first-run personalization wizard
  bantz --setup profile         → user profile setup
  bantz --setup google gmail    → OAuth setup for Gmail
  bantz --setup google classroom → OAuth setup for Classroom
  bantz --setup schedule        → class schedule setup
  bantz --setup telegram        → Telegram bot token setup
  bantz --setup places          → Known locations setup
  bantz --setup gemini          → Gemini API key setup
  bantz --setup voice           → Guided voice setup wizard (STT + TTS + wake word)
  bantz --setup systemd         → install systemd user service
"""
from __future__ import annotations

import argparse
import asyncio

from bantz.cli.setup import _handle_setup, _doctor, _show_config, _cache_stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="bantz", description="Bantz v2 — your terminal host")
    parser.add_argument("--once", metavar="QUERY", help="Run single query, no UI")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as headless daemon (scheduler + GPS, no TUI)")
    parser.add_argument("--doctor", action="store_true", help="System health check")
    parser.add_argument("--cache-stats", action="store_true",
                        help="Show spatial cache statistics")
    parser.add_argument("--setup", nargs="+", metavar="SERVICE",
                        help="Setup integrations: --setup google gmail")
    parser.add_argument("--jobs", action="store_true",
                        help="List all scheduled APScheduler jobs")
    parser.add_argument("--run-job", metavar="JOB_ID",
                        help="Manually trigger a scheduled job by ID")
    parser.add_argument("--maintenance", action="store_true",
                        help="Run nightly maintenance workflow now")
    parser.add_argument("--reflect", action="store_true",
                        help="Run nightly memory reflection now")
    parser.add_argument("--reflections", action="store_true",
                        help="View past daily reflections")
    parser.add_argument("--overnight-poll", action="store_true",
                        help="Run one overnight poll cycle (Gmail/Calendar/Classroom)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate actions without changes (use with --maintenance/--reflect/--overnight-poll)")
    parser.add_argument("--mood-history", action="store_true",
                        help="Show 24h mood transition history")
    parser.add_argument("--config", action="store_true",
                        help="Show current configuration (secrets masked)")
    args = parser.parse_args()

    if args.doctor:
        asyncio.run(_doctor())
        return

    if args.cache_stats:
        _cache_stats()
        return

    if args.setup:
        _handle_setup(args.setup)
        return

    if args.jobs:
        asyncio.run(_list_jobs())
        return

    if args.run_job:
        asyncio.run(_run_job(args.run_job))
        return

    if args.maintenance:
        asyncio.run(_maintenance(args.dry_run))
        return

    if args.reflect:
        asyncio.run(_reflect(args.dry_run))
        return

    if args.reflections:
        _view_reflections()
        return

    if args.overnight_poll:
        asyncio.run(_overnight_poll(args.dry_run))
        return

    if args.mood_history:
        _mood_history()
        return

    if args.config:
        _show_config()
        return

    if args.once:
        asyncio.run(_once(args.once))
        return

    if args.daemon:
        asyncio.run(_daemon())
        return

    from bantz.interface.live_ui import run
    run()


async def _start_ws_server() -> None:
    """Start the WebSocket broadcast server (non-blocking background task)."""
    try:
        from bantz.interface.ws_server import ws_server
        await ws_server.start()
    except Exception as exc:
        import logging
        logging.getLogger("bantz.main").warning("WS server failed to start: %s", exc)


async def _list_jobs() -> None:
    """List all scheduled APScheduler jobs (bantz --jobs)."""
    from bantz.config import config
    config.ensure_dirs()

    from bantz.agent.job_scheduler import job_scheduler
    await job_scheduler.start(config.db_path, enable_night_jobs=True)
    print(job_scheduler.format_jobs())
    await job_scheduler.shutdown()


async def _run_job(job_id: str) -> None:
    """Manually trigger a job (bantz --run-job <id>)."""
    from bantz.config import config
    config.ensure_dirs()

    from bantz.core.memory import memory
    memory.init(config.db_path)
    memory.new_session()

    from bantz.core.scheduler import scheduler
    scheduler.init(config.db_path)

    from bantz.agent.job_scheduler import job_scheduler
    await job_scheduler.start(config.db_path, enable_night_jobs=True)

    ok = job_scheduler.run_job_now(job_id)
    if ok:
        print(f"✓ Triggered job: {job_id}")
        # Give async jobs a moment to run
        await asyncio.sleep(3)
    else:
        print(f"✗ Job not found: {job_id}")
        print("Available jobs:")
        for j in job_scheduler.list_jobs():
            print(f"  {j['id']}")

    await job_scheduler.shutdown()


async def _maintenance(dry_run: bool) -> None:
    """Run the 6-step nightly maintenance workflow (bantz --maintenance)."""
    from bantz.config import config
    config.ensure_dirs()

    from bantz.core.memory import memory
    memory.init(config.db_path)

    from bantz.agent.workflows.maintenance import run_maintenance

    tag = " (dry-run)" if dry_run else ""
    print(f"🔧 Running maintenance{tag}...")
    report = await run_maintenance(dry_run=dry_run)
    print(report.summary())


async def _reflect(dry_run: bool) -> None:
    """Run the nightly memory reflection workflow (bantz --reflect)."""
    from bantz.config import config
    config.ensure_dirs()

    from bantz.core.memory import memory
    memory.init(config.db_path)

    from bantz.agent.workflows.reflection import run_reflection

    tag = " (dry-run)" if dry_run else ""
    print(f"🤔 Running reflection{tag}...")
    result = await run_reflection(dry_run=dry_run)
    print(result.summary_line())


def _view_reflections() -> None:
    """View past daily reflections (bantz --reflections)."""
    from bantz.config import config
    config.ensure_dirs()

    from bantz.core.memory import memory
    memory.init(config.db_path)

    from bantz.agent.workflows.reflection import list_reflections

    reflections = list_reflections(limit=10)
    if not reflections:
        print("No reflections yet. Run 'bantz --reflect' first.")
        return

    for r in reflections:
        date = r.get("date", "?")
        summary = r.get("summary", "")
        sessions = r.get("sessions", 0)
        msgs = r.get("total_messages", 0)
        print(f"\n📅 {date} ({sessions} sessions, {msgs} msgs)")
        if summary:
            print(f"   {summary}")
        reflection = r.get("reflection", "")
        if reflection:
            print(f"   💡 {reflection}")
        decisions = r.get("decisions", [])
        if decisions:
            print(f"   Decisions: {', '.join(decisions)}")
        unresolved = r.get("unresolved", [])
        if unresolved:
            print(f"   ❓ {', '.join(unresolved)}")


async def _overnight_poll(dry_run: bool) -> None:
    """Run one overnight poll cycle (bantz --overnight-poll)."""
    from bantz.config import config
    config.ensure_dirs()

    from bantz.agent.workflows.overnight_poll import run_overnight_poll

    tag = " (dry-run)" if dry_run else ""
    print(f"📬 Running overnight poll{tag}...")
    result = await run_overnight_poll(dry_run=dry_run)
    print(result.summary_line())
    if result.errors:
        for src in (result.gmail, result.calendar, result.classroom):
            if src and src.status != "ok":
                print(f"  ⚠️ {src.source}: {src.status} — {src.error_message}")


def _mood_history() -> None:
    """Show 24h mood transition history (bantz --mood-history)."""
    print("Mood history is unavailable — the Textual TUI has been removed.")
    print("Mood data is still recorded in the database but has no display command yet.")


async def _once(query: str) -> None:
    print("Loading models…", flush=True)
    from bantz.core.brain import brain
    result = await brain.process(query)
    print(result.response)

    # Draft confirmation flow — auto-send for --once
    if result.needs_confirm and result.pending_tool and result.pending_args:
        answer = input().strip().lower()
        if answer in ("evet", "e", "yes", "y", "ok", "tamam"):
            from bantz.tools import registry as _reg
            tool = _reg.get(result.pending_tool)
            if tool:
                tr = await tool.execute(**result.pending_args)
                print(tr.output if tr.success else f"Error: {tr.error}")
        else:
            print("Cancelled.")


async def _daemon() -> None:
    """Run Bantz as a headless daemon — APScheduler-driven.

    Replaces the old manual polling loops with APScheduler for:
    - Reminder checks (30s interval)
    - Night maintenance (3 AM cron)
    - Night reflection (11 PM cron)
    - Overnight email/cal poll (every 2h 00-07)
    - Morning briefing prep (6 AM cron)

    Key features: misfire_grace_time=86400, coalesce=True,
    systemd-inhibit for night jobs, persistent SQLAlchemy job store.
    """
    import signal
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("bantz.daemon")
    log.info("Bantz daemon starting (APScheduler)...")

    from bantz.config import config
    config.ensure_dirs()

    # Init memory + legacy scheduler (for backward compat)
    from bantz.core.memory import memory
    memory.init(config.db_path)
    memory.new_session()

    from bantz.core.scheduler import scheduler
    scheduler.init(config.db_path)
    log.info("Legacy scheduler: %s", scheduler.status_line())

    # Init KV store for briefing cache
    from bantz.data.sqlite_store import SQLiteKVStore
    _ = SQLiteKVStore(config.db_path)

    # Start APScheduler
    if config.job_scheduler_enabled:
        from bantz.agent.job_scheduler import job_scheduler
        await job_scheduler.start(
            config.db_path,
            enable_night_jobs=True,
        )
        log.info("APScheduler: %s", job_scheduler.status_line())
    else:
        log.info("APScheduler disabled — falling back to legacy loops")

    # Start GPS server
    gps_ok = False
    try:
        from bantz.core.gps_server import gps_server
        gps_ok = await gps_server.start()
        if gps_ok:
            log.info("GPS server: %s (relay: %s)", gps_server.url, gps_server.relay_topic)
    except Exception as exc:
        log.warning("GPS server failed to start: %s", exc)

    # Start WebSocket server for Tauri UI
    await _start_ws_server()

    # Warm up TTS engine so the first chat response isn't delayed by model load
    if config.tts_enabled:
        try:
            from bantz.agent.tts import tts_engine
            if tts_engine._ensure_init():
                log.info(
                    "TTS engine initialized, model=%s speak_all=%s",
                    tts_engine._model_path, config.tts_speak_all_responses,
                )
            else:
                log.warning(
                    "TTS enabled but engine failed to initialize "
                    "(piper binary or model file missing?)"
                )
        except Exception as _tts_exc:
            log.warning("TTS warm-up failed: %s", _tts_exc)

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _signal_handler(sig, _frame):
        log.info("Received %s — shutting down...", signal.Signals(sig).name)
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    import os
    log.info("Daemon running — APScheduler=%s. PID: %d",
             "active" if config.job_scheduler_enabled else "off", os.getpid())

    # If APScheduler is disabled, fall back to legacy polling loops
    tasks = []
    if not config.job_scheduler_enabled:
        async def _reminder_loop():
            interval = config.reminder_check_interval
            while not stop_event.is_set():
                try:
                    due = scheduler.check_due()
                    for r in due:
                        repeat_tag = f" (repeats {r['repeat']})" if r["repeat"] != "none" else ""
                        log.info("⏰ REMINDER: %s%s", r["title"], repeat_tag)
                        memory.add("assistant", f"⏰ Reminder: {r['title']}{repeat_tag}",
                                   tool_used="reminder")
                except Exception as exc:
                    log.debug("Reminder check error: %s", exc)
                await asyncio.sleep(interval)

        async def _briefing_loop():
            while not stop_event.is_set():
                try:
                    from bantz.personality.greeting import greeting_manager
                    text = await greeting_manager.morning_briefing_if_due()
                    if text:
                        log.info("📋 Morning briefing:\n%s", text)
                        memory.add("assistant", text, tool_used="briefing")
                        # TTS: speak the briefing in daemon mode (#171)
                        try:
                            from bantz.agent.tts import tts_engine
                            if config.tts_auto_briefing and tts_engine.available():
                                await tts_engine.speak_background(text)
                        except Exception:
                            pass
                except Exception as exc:
                    log.debug("Briefing check error: %s", exc)
                await asyncio.sleep(60)

        tasks = [
            asyncio.create_task(_reminder_loop()),
            asyncio.create_task(_briefing_loop()),
        ]

    # Wait for stop signal
    await stop_event.wait()

    # Cleanup
    for t in tasks:
        t.cancel()
    if config.job_scheduler_enabled:
        try:
            from bantz.agent.job_scheduler import job_scheduler
            await job_scheduler.shutdown()
        except Exception:
            pass
    if gps_ok:
        try:
            await gps_server.stop()
        except Exception:
            pass
    log.info("Bantz daemon stopped.")


if __name__ == "__main__":
    main()
