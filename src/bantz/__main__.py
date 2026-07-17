"""
Bantz v2 — Entry point

Commands:
  bantz                         → TUI
  bantz --ui                    → desktop UI (Tauri; starts daemon if not running)
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
    parser.add_argument("--ui", action="store_true",
                        help="Launch the Bantz desktop UI (Tauri)")
    parser.add_argument("--telegram", action="store_true",
                        help="Run the Telegram bot (brain-routed, polling)")
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

    if args.ui:
        _open_ui()
        return

    if args.daemon:
        asyncio.run(_daemon())
        return

    if args.telegram:
        from bantz.interface.telegram_bot import run_bot
        run_bot()
        return

    from bantz.interface.live_ui import run
    run()


def _open_ui() -> None:
    import shutil
    import socket
    import subprocess
    import sys
    from pathlib import Path

    # bantz-ui/ sits next to the repo root (3 levels up from src/bantz/__main__.py)
    ui_dir = Path(__file__).resolve().parent.parent.parent / "bantz-ui"
    if not ui_dir.exists():
        sys.exit(
            f"[✗] bantz-ui directory not found at {ui_dir}\n"
            "    Manual fallback: cd ~/bantzv2/bantz-ui && npm run tauri:dev"
        )

    # Start backend daemon if WebSocket port 8765 is not already open
    def _ws_up() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 8765), timeout=1):
                return True
        except OSError:
            return False

    if not _ws_up():
        print("[→] Backend not running — starting bantz --daemon …")
        subprocess.Popen(
            [sys.executable, "-m", "bantz", "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Binary name is platform-specific
    is_win = sys.platform == "win32"
    bin_name = "bantz-ui.exe" if is_win else "bantz-ui"

    # Prefer compiled release binary, then AppImage (Linux), then dev server
    release_bin = ui_dir / "src-tauri" / "target" / "release" / bin_name
    appimage_dir = ui_dir / "src-tauri" / "target" / "release" / "bundle" / "appimage"
    appimages = sorted(appimage_dir.glob("*.AppImage")) if appimage_dir.exists() else []

    if release_bin.exists():
        print("[→] Launching Bantz UI …")
        subprocess.run([str(release_bin)], check=False)
        return

    if appimages:
        print("[→] Launching Bantz UI …")
        subprocess.run([str(appimages[0])], check=False)
        return

    # No compiled binary — fall back to Vite/Tauri dev server
    npm = shutil.which("npm")
    if not npm:
        sys.exit("[✗] npm not found in PATH. Install Node.js from https://nodejs.org")

    if not (ui_dir / "node_modules").exists():
        print("[→] Installing npm dependencies (first run) …")
        subprocess.run([npm, "install"], cwd=ui_dir, check=True)

    print("[→] Starting Bantz UI (dev mode) …")
    subprocess.run([npm, "run", "tauri:dev"], cwd=ui_dir, check=False)


async def _start_ws_server(voice_chat: bool = False) -> None:
    """Start the WebSocket broadcast server (non-blocking background task).

    voice_chat=True (daemon mode) makes ws_server the consumer of
    wake-word "voice_input" events; live_ui handles those itself."""
    try:
        from bantz.interface.ws_server import ws_server
        ws_server.voice_chat = voice_chat
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
    """Single-shot query — no TUI.

    Emits progress to *stderr* so scripts piping stdout still get clean
    output.  Drains streaming responses and optionally speaks them via
    TTS (#435).
    """
    import sys

    def _progress(msg: str) -> None:  # writes to stderr — stdout stays clean
        print(msg, file=sys.stderr, flush=True)

    from bantz.core.brain import brain
    result = await brain.process(query, progress_cb=_progress)

    # Drain streaming response when brain returns a generator (chat route)
    if result.stream is not None:
        _progress("Streaming response…")
        chunks: list[str] = []
        async for chunk in result.stream:
            chunks.append(chunk)
        response = "".join(chunks)
    else:
        response = result.response

    print(response)

    # Speak the response when TTS is fully configured (#435)
    from bantz.config import config
    if (
        config.tts_enabled
        and config.tts_speak_all_responses
        and response
        and result.tool_used != "tts"
    ):
        _progress("Synthesising voice…")
        try:
            from bantz.agent.tts import tts_engine
            await tts_engine.speak(response)
        except Exception:
            pass

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

    # Self-heal the Wayland/Hyprland env: a systemd user service started
    # before the compositor imports its environment has no WAYLAND_DISPLAY /
    # HYPRLAND_INSTANCE_SIGNATURE, which breaks grim + hyprctl.
    from bantz.core.desktop_env import ensure_wayland_env
    ensure_wayland_env()

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
    await _start_ws_server(voice_chat=True)

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

    # Start the voice front-end: wake word listener + ghost loop (#165/#36).
    # Both self-gate on their own config flags and no-op gracefully when a
    # dependency (Picovoice key, mic, whisper) is missing, so this is safe to
    # attempt whenever wake word is enabled.
    voice_started = False
    if config.wake_word_enabled:
        try:
            from bantz.agent.ghost_loop import ghost_loop
            from bantz.agent.wake_word import wake_listener
            ghost_loop.start()
            voice_started = wake_listener.start()
            if voice_started:
                log.info("Voice front-end: wake word listener + ghost loop running")
            else:
                log.warning(
                    "Wake word enabled but listener failed to start "
                    "(missing BANTZ_PICOVOICE_ACCESS_KEY or mic?)"
                )
        except Exception as _voice_exc:
            log.warning("Voice front-end failed to start: %s", _voice_exc)

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
