"""
Bantz v2 — Entry point

Commands:
  bantz                         → TUI
  bantz --once "query"          → single query, no UI
  bantz --daemon                → headless daemon (scheduler + GPS, no TUI)
  bantz --doctor                → system health check
  bantz --setup profile         → user profile setup
  bantz --setup google gmail    → OAuth setup for Gmail
  bantz --setup google classroom → OAuth setup for Classroom
  bantz --setup schedule        → class schedule setup
  bantz --setup telegram        → Telegram bot token setup
  bantz --setup places          → Known locations setup
  bantz --setup systemd         → install systemd user service
"""
from __future__ import annotations

import argparse
import asyncio


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

    from bantz.interface.tui.app import run
    run()


def _handle_setup(parts: list[str]) -> None:
    if len(parts) >= 1 and parts[0].lower() == "profile":
        _setup_profile()
        return
    if len(parts) >= 1 and parts[0].lower() == "schedule":
        _setup_schedule()
        return
    if len(parts) >= 1 and parts[0].lower() == "telegram":
        _setup_telegram()
        return
    if len(parts) >= 1 and parts[0].lower() == "places":
        asyncio.run(_setup_places())
        return
    if len(parts) >= 1 and parts[0].lower() == "gemini":
        _setup_gemini()
        return
    if len(parts) >= 1 and parts[0].lower() == "systemd":
        if len(parts) >= 2 and parts[1].lower() == "--check":
            _systemd_check()
        else:
            _setup_systemd()
        return
    if len(parts) >= 2 and parts[0].lower() == "google":
        service = parts[1].lower()
        from bantz.auth.google_oauth import setup_google
        setup_google(service)
    else:
        print(f"Unknown setup target: {' '.join(parts)}")
        print("Available:")
        print("  bantz --setup profile")
        print("  bantz --setup google [gmail|classroom|calendar]")
        print("  bantz --setup schedule")
        print("  bantz --setup telegram")
        print("  bantz --setup places")
        print("  bantz --setup gemini")
        print("  bantz --setup systemd")
        print("  bantz --setup systemd --check")


def _setup_telegram() -> None:
    """Interactive Telegram bot token setup."""
    from pathlib import Path

    print("\n🦌 Telegram Bot Setup")
    print("─" * 40)
    print("1. Go to @BotFather → /newbot → get token")
    print("2. Paste the token below:")
    print()

    token = input("Bot token: ").strip()
    if not token:
        print("Token required. Cancelled.")
        return

    # Optionally get allowed user IDs
    print()
    print("(Security) Restrict to specific users?")
    print("Enter Telegram user IDs, comma-separated (blank=everyone):")
    allowed = input("User IDs: ").strip()

    # Proxy (Turkey blocks api.telegram.org)
    print()
    print("(Proxy) You may need an HTTPS proxy in some regions.")
    print("Example: socks5://127.0.0.1:1080 or http://proxy:8080")
    proxy = input("Proxy URL (blank=skip): ").strip()

    # Write to .env
    env_path = Path.cwd() / ".env"
    existing = ""
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8")

    lines = existing.splitlines()
    # Remove old telegram entries
    lines = [l for l in lines if not l.startswith("TELEGRAM_BOT_TOKEN=")
             and not l.startswith("TELEGRAM_ALLOWED_USERS=")
             and not l.startswith("TELEGRAM_PROXY=")]

    lines.append(f"TELEGRAM_BOT_TOKEN={token}")
    if allowed:
        lines.append(f"TELEGRAM_ALLOWED_USERS={allowed}")
    if proxy:
        lines.append(f"TELEGRAM_PROXY={proxy}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Token saved: {env_path}")
    print("Start with: python -m bantz.interface.telegram_bot")


def _setup_gemini() -> None:
    """Interactive Gemini API key setup."""
    from pathlib import Path

    print("\n🦌 Gemini API Setup")
    print("─" * 40)
    print("1. Go to https://aistudio.google.com/apikey")
    print("2. Create an API key")
    print("3. Paste it below:")
    print()

    api_key = input("Gemini API key: ").strip()
    if not api_key:
        print("No key provided. Cancelled.")
        return

    model = input("Model (default: gemini-2.0-flash): ").strip() or "gemini-2.0-flash"

    # Write to .env
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    # Remove old gemini entries
    lines = [l for l in lines if not l.startswith("BANTZ_GEMINI_")]

    lines.append(f"BANTZ_GEMINI_ENABLED=true")
    lines.append(f"BANTZ_GEMINI_API_KEY={api_key}")
    lines.append(f"BANTZ_GEMINI_MODEL={model}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Gemini configured: {env_path}")
    print(f"   Model: {model}")
    print("   Gemini will be used as the finalizer for high-quality responses.")


async def _setup_places() -> None:
    """Interactive known-places setup — writes places.json.
    Three coordinate options: auto IP, city name search, manual entry.
    First place (or explicit choice) becomes default location in .env.
    """
    import json
    import urllib.request
    from pathlib import Path
    from bantz.core.places import places
    from bantz.core.location import location_service

    def _nominatim_search(query: str) -> list[dict]:
        """Forward geocode via Nominatim (no API key)."""
        import urllib.parse
        encoded = urllib.parse.quote(query)
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={encoded}&format=json&limit=3&accept-language=tr"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Bantz/2.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    print("\n📍 Known Locations Setup")
    print("─" * 40)

    data = dict(places.all_places())
    if data:
        print("Existing locations:")
        for k, v in data.items():
            prim = " ★" if v.get("primary") else ""
            print(f"  {k}: {v.get('label', k)}  ({v.get('lat', 0):.4f}, {v.get('lon', 0):.4f}){prim}")
        print()

    # Get IP location for option 1
    print("Detecting IP location...")
    loc = await location_service.get()
    if loc.lat != 0.0 and loc.lon != 0.0:
        print(f"  📡 {loc.display}  ({loc.lat:.4f}, {loc.lon:.4f})  via {loc.source}")
    else:
        print("  ⚠  Could not detect IP location.")
    print()

    print("Add locations (e.g.: dorm, campus, home). Leave blank to finish.\n")

    primary_key: str | None = None

    while True:
        try:
            key = input("Location code (e.g.: dorm): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not key:
            break

        label = input(f"  Display name [{key.capitalize()}]: ").strip() or key.capitalize()

        # 3 options for coordinates
        print()
        print("  Coordinate option:")
        if loc.lat != 0.0:
            print(f"  [1] Automatic (IP location: {loc.city}, {loc.lat:.4f}, {loc.lon:.4f})")
        else:
            print("  [1] Automatic (IP location unavailable)")
        print("  [2] Search by city/address name")
        print("  [3] Enter manually (lat, lon)")
        choice = input("  Choice [2]: ").strip() or "2"

        lat, lon = 0.0, 0.0

        if choice == "1" and loc.lat != 0.0:
            lat, lon = loc.lat, loc.lon
            print(f"  → {loc.city}  ({lat:.4f}, {lon:.4f})")

        elif choice == "3":
            raw = input("  Coordinates (lat, lon): ").strip()
            try:
                parts = raw.replace(" ", "").split(",")
                lat, lon = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                print("  ✗ Invalid coordinates, skipping.")
                continue

        else:  # default: option 2 — search
            query = input("  City/address: ").strip()
            if not query:
                print("  ✗ Empty query, skipping.")
                continue
            try:
                results = _nominatim_search(query)
                if not results:
                    print("  ✗ No results found.")
                    continue

                if len(results) == 1:
                    pick = results[0]
                else:
                    print("  Results found:")
                    for i, r in enumerate(results, 1):
                        print(f"    [{i}] {r['display_name'][:80]}")
                    idx = input(f"  Choice [1]: ").strip() or "1"
                    try:
                        pick = results[int(idx) - 1]
                    except (ValueError, IndexError):
                        pick = results[0]

                lat = float(pick["lat"])
                lon = float(pick["lon"])
                display = pick.get("display_name", "")[:60]
                print(f"  → Found: {lat:.4f}, {lon:.4f}  ({display})")

                confirm = input("  Correct? [Y/n]: ").strip().lower()
                if confirm in ("n", "no"):
                    print("  Skipping.")
                    continue

            except Exception as e:
                print(f"  ✗ Search error: {e}")
                continue

        data[key] = {"label": label, "lat": lat, "lon": lon}
        print(f"  ✓ {key}: {label} ({lat:.4f}, {lon:.4f})")

        # Ask if this is primary location
        if not primary_key:
            is_primary = input("  Is this your primary location? (for weather etc.) [Y/n]: ").strip().lower()
            if is_primary in ("", "y", "yes"):
                primary_key = key
                data[key]["primary"] = True
        print()

    if not data:
        print("\nNo locations added.")
        return

    places.save(data)
    print(f"\n✅ Locations saved: {places.setup_path()}")
    print(f"  {len(data)} locations defined")

    # Write primary location to .env so location_service uses it
    pkey = primary_key
    if not pkey:
        # Find any existing primary
        for k, v in data.items():
            if v.get("primary"):
                pkey = k
                break
    if not pkey and len(data) == 1:
        pkey = next(iter(data))

    if pkey and pkey in data:
        place = data[pkey]
        _write_location_to_env(
            city=place["label"],
            lat=place["lat"],
            lon=place["lon"],
        )
        print(f"  📡 Primary location written to .env: {place['label']} ({place['lat']:.4f}, {place['lon']:.4f})")
        print("  → location_service will use these coordinates")


def _write_location_to_env(city: str, lat: float, lon: float) -> None:
    """Write BANTZ_CITY / BANTZ_LAT / BANTZ_LON to .env."""
    from pathlib import Path

    env_path = Path.cwd() / ".env"
    existing = ""
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8")

    lines = existing.splitlines()
    # Remove old location entries
    lines = [
        l for l in lines
        if not l.startswith("BANTZ_CITY=")
        and not l.startswith("BANTZ_LAT=")
        and not l.startswith("BANTZ_LON=")
    ]

    lines.append(f"BANTZ_CITY={city}")
    lines.append(f"BANTZ_LAT={lat}")
    lines.append(f"BANTZ_LON={lon}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup_profile() -> None:
    """Interactive profile setup — writes profile.json."""
    from bantz.core.profile import profile, ALL_BRIEFING_SECTIONS, ALL_NEWS_SOURCES

    print("\n👤 User Profile Setup")
    print("─" * 40)
    if profile.is_configured():
        print(f"Current profile: {profile.get('name')} ({profile.response_style})")
        print()

    name = input("Name: ").strip()
    if not name:
        print("Name required. Cancelled.")
        return

    university = input("University (blank=skip): ").strip()
    department = input("Department (blank=skip): ").strip()
    year_raw = input("Year (1-6, blank=skip): ").strip()
    year = int(year_raw) if year_raw.isdigit() else 0

    print("\nResponse style:")
    print("  1) casual  — friendly, like an old friend")
    print("  2) formal  — professional, respectful")
    style_choice = input("Choice [1]: ").strip()
    response_style = "formal" if style_choice == "2" else "casual"

    # Briefing sections
    print("\nBriefing sections (comma-separated numbers, or Enter for all):")
    for i, sec in enumerate(ALL_BRIEFING_SECTIONS, 1):
        print(f"  {i}) {sec}")
    sec_input = input(f"Sections [1-{len(ALL_BRIEFING_SECTIONS)}, default=all]: ").strip()
    if sec_input:
        indices = [int(x.strip()) - 1 for x in sec_input.split(",") if x.strip().isdigit()]
        briefing_sections = [
            ALL_BRIEFING_SECTIONS[i] for i in indices
            if 0 <= i < len(ALL_BRIEFING_SECTIONS)
        ]
        if not briefing_sections:
            briefing_sections = list(ALL_BRIEFING_SECTIONS)
    else:
        briefing_sections = list(ALL_BRIEFING_SECTIONS)

    # News sources
    print("\nNews sources (comma-separated numbers, or Enter for all):")
    for i, src in enumerate(ALL_NEWS_SOURCES, 1):
        print(f"  {i}) {src}")
    src_input = input(f"Sources [1-{len(ALL_NEWS_SOURCES)}, default=all]: ").strip()
    if src_input:
        indices = [int(x.strip()) - 1 for x in src_input.split(",") if x.strip().isdigit()]
        news_sources = [
            ALL_NEWS_SOURCES[i] for i in indices
            if 0 <= i < len(ALL_NEWS_SOURCES)
        ]
        if not news_sources:
            news_sources = list(ALL_NEWS_SOURCES)
    else:
        news_sources = list(ALL_NEWS_SOURCES)

    profile.save({
        "name": name,
        "university": university,
        "department": department,
        "year": year,
        "pronoun": response_style,
        "tone": response_style,
        "preferences": {
            "briefing_sections": briefing_sections,
            "news_sources": news_sources,
            "response_style": response_style,
        },
    })
    print(f"\n✅ Profile saved: {profile.path}")
    print(f"  → {profile.prompt_hint()}")


def _setup_schedule() -> None:
    """Interactive schedule setup — writes schedule.json."""
    import json
    from bantz.core.schedule import Schedule, DAYS_EN, DAYS_TR

    path = Schedule.setup_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if present
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"Existing schedule loaded: {path}")
        except Exception:
            pass

    print("\n📅 Class Schedule Setup")
    print("─" * 40)
    print("Enter classes day by day. Leave blank to finish.")
    print("Format: HH:MM  Class-Name  Duration(min)  Location")
    print()

    for day_en in DAYS_EN:
        day_tr = DAYS_TR[day_en]
        print(f"\n{day_tr}:")
        existing = data.get(day_en, [])
        if existing:
            for c in existing:
                print(f"  (existing) {c.get('time','')} {c.get('name','')} {c.get('location','')}")

        classes = list(existing)  # keep existing
        while True:
            try:
                raw = input(f"  New class (blank=skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                break
            parts = raw.split(None, 3)
            if len(parts) < 2:
                print("  Enter at least time and class name.")
                continue
            time_str = parts[0]
            name = parts[1]
            duration = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 90
            location = parts[3] if len(parts) > 3 else ""

            cls: dict = {"name": name, "time": time_str, "duration": duration}
            if location:
                cls["location"] = location
            classes.append(cls)
            print(f"  ✓ Added: {time_str} {name}")

        if classes:
            data[day_en] = sorted(classes, key=lambda c: c.get("time", ""))

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Schedule saved: {path}")
    print("Test: bantz --once 'my classes today'")


def _cache_stats() -> None:
    """Display spatial cache statistics (#121)."""
    from bantz.config import config
    from bantz.vision.spatial_cache import spatial_db

    spatial_db.init(config.db_path)
    stats = spatial_db.stats()

    print("\n🗺  Spatial Cache Statistics")
    print("─" * 50)
    print(f"  Total entries : {stats['total_entries']} / {stats.get('max_entries', 1000)}")
    print(f"  Total hits    : {stats['total_hits']}")
    print(f"  Expired       : {stats['expired']}")
    print(f"  TTL           : {stats.get('ttl_hours', 24)}h")

    if stats.get("apps"):
        print("\n  Applications:")
        for app, cnt in stats["apps"].items():
            print(f"    {app}: {cnt} elements")

    if stats.get("sources"):
        print("\n  Sources:")
        for src, cnt in stats["sources"].items():
            print(f"    {src}: {cnt} entries")

    if stats.get("top_elements"):
        print("\n  Top elements (by hits):")
        for elem in stats["top_elements"]:
            print(
                f"    [{elem['source']}] {elem['app']}/{elem['label']} "
                f"— {elem['hits']} hits (conf={elem['confidence']:.2f})"
            )

    print()
    spatial_db.close()


# ── Config display ────────────────────────────────────────────────────────

_SECRET_FIELDS = frozenset({
    "gemini_api_key", "neo4j_password", "telegram_bot_token",
    "gps_relay_token",
})


def _mask(value: str) -> str:
    """Mask a secret value: show first 4 chars + ****."""
    if not value:
        return "(empty)"
    if len(value) <= 6:
        return "****"
    return value[:4] + "****"


def _show_config() -> None:
    """Print all configuration values with secrets masked."""
    from bantz.config import Config, config

    print("Bantz v2 — Current Configuration")
    print("─" * 52)

    section = ""
    for name, field in Config.model_fields.items():
        # Derive section from field comment / alias prefix
        alias = field.alias or name
        val = getattr(config, name)

        # Section headers (group by config.py sections)
        new_section = _section_for(name)
        if new_section != section:
            section = new_section
            print(f"\n  ── {section} ──")

        # Mask secrets
        display = _mask(str(val)) if name in _SECRET_FIELDS else str(val)

        # Pad for alignment
        env_hint = f"  ({alias})" if alias != name else ""
        print(f"  {name:<35s} = {display}{env_hint}")

    print()


def _section_for(field_name: str) -> str:
    """Map a config field name to a human-readable section label."""
    _MAP = [
        (("ollama_",), "Ollama"),
        (("embedding_", "vector_search_"), "Embeddings"),
        (("distillation_",), "Distillation"),
        (("vlm_", "screenshot_"), "Vision / VLM"),
        (("input_control_", "input_confirm_", "input_type_"), "Input Control"),
        (("gemini_",), "Gemini"),
        (("language", "translation_"), "Language"),
        (("shell_",), "Shell Security"),
        (("location_",), "Location"),
        (("gps_relay_",), "GPS Relay"),
        (("neo4j_",), "Neo4j"),
        (("data_dir",), "Storage"),
        (("morning_briefing_",), "Morning Briefing"),
        (("daily_digest_", "weekly_digest_"), "Digests"),
        (("reminder_",), "Scheduler / Reminders"),
        (("job_scheduler_", "night_", "briefing_prep_", "overnight_poll_hours",), "Job Scheduler"),
        (("urgent_keywords",), "Overnight Poll"),
        (("telegram_",), "Telegram"),
        (("observer_",), "Observer"),
        (("rl_",), "RL Engine"),
        (("intervention_",), "Interventions"),
        (("app_detector_",), "App Detector"),
        (("desktop_notifications", "notification_",), "Notifications"),
        (("tts_",), "TTS / Audio"),
        (("audio_duck_",), "Audio Ducking"),
        (("wake_word_", "picovoice_",), "Wake Word"),
        (("ambient_",), "Ambient Sound"),
    ]
    for prefixes, label in _MAP:
        if any(field_name.startswith(p) or field_name == p for p in prefixes):
            return label
    return "General"


async def _doctor() -> None:
    from bantz.llm.ollama import ollama
    from bantz.config import config
    from bantz.tools import registry
    from bantz.auth.token_store import token_store
    import bantz.tools.shell
    import bantz.tools.system
    import bantz.tools.filesystem
    import bantz.tools.weather
    import bantz.tools.news
    import bantz.tools.web_reader
    import bantz.tools.gmail
    import bantz.tools.calendar
    import bantz.tools.classroom
    import bantz.tools.reminder
    import bantz.tools.contacts
    import bantz.tools.input_control

    print("Bantz v2 — System Check")
    print("─" * 52)

    # Ollama
    ok = await ollama.is_available()
    if ok:
        print(f"✅ Ollama: connected ({config.ollama_model})")
    else:
        print(f"❌ Ollama: UNREACHABLE ({config.ollama_base_url})")

    # Gemini
    from bantz.llm.gemini import gemini as _gem
    if _gem.is_enabled():
        gem_ok = await _gem.is_available()
        if gem_ok:
            print(f"✅ Gemini: connected ({config.gemini_model})")
        else:
            print(f"❌ Gemini: UNREACHABLE")
    else:
        print(f"⚪ Gemini: disabled  → bantz --setup gemini")

    # Neo4j
    if config.neo4j_enabled:
        try:
            from bantz.data.layer import layer as _lay
            _lay.init(config.db_path)
            print(f"✅ Neo4j: connected ({config.neo4j_uri})")
        except Exception:
            print(f"❌ Neo4j: enabled but connection failed")
    else:
        print(f"⚪ Neo4j: disabled  → BANTZ_NEO4J_ENABLED=true")

    # Vision / VLM
    if config.vlm_enabled:
        print(f"✅ Vision/VLM: enabled ({config.vlm_endpoint})")
    else:
        print(f"⚪ Vision/VLM: disabled  → BANTZ_VLM_ENABLED=true")

    # psutil
    import psutil
    print(f"✅ psutil: CPU {psutil.cpu_percent(interval=0.3):.0f}%")

    # Tools
    names = [t["name"] for t in registry.all_schemas()]
    print(f"✅ Tools ({len(names)}): {', '.join(names)}")

    # Translation / Bridge
    if config.translation_enabled and config.language == "tr":
        try:
            from transformers import AutoTokenizer  # noqa: F401
            print("✅ MarianMT: available")
        except ImportError:
            print("❌ MarianMT: NOT installed  → pip install 'bantz[translation]'")
    else:
        print(f"⚪ Translation: {'disabled' if not config.translation_enabled else f'not needed (lang={config.language})'}")

    # Location
    from bantz.core.location import location_service
    loc = await location_service.get()
    if loc.is_live:
        print(f"✅ Location: {loc.display}  (via {loc.source})")
    else:
        print(f"⚪ Location: unknown  → enable phone GPS relay")

    # Google integrations
    g_status = token_store.status()
    for svc, st in g_status.items():
        icon = "✅" if st == "ok" else "⚪"
        print(f"  {icon} Google {svc}: {st}")
    if any(st != "ok" for st in g_status.values()):
        print("     → bantz --setup google gmail  /  bantz --setup google classroom")

    # Memory DB
    config.ensure_dirs()
    from bantz.core.memory import memory as _mem
    _mem.init(config.db_path)
    s = _mem.stats()
    print(f"✅ Memory DB: {s['total_conversations']} conversations, {s['total_messages']} messages")

    # Embeddings
    if config.embedding_enabled:
        print(f"✅ Embeddings: {config.embedding_model}  (weight={config.vector_search_weight})")
    else:
        print(f"⚪ Embeddings: disabled  → BANTZ_EMBEDDING_ENABLED=true")

    # Profile
    from bantz.core.profile import profile as _prof
    if _prof.is_configured():
        print(f"✅ Profile: {_prof.status_line()}")
    else:
        print(f"⚪ Profile: not configured  → bantz --setup profile")

    # Input Control (#122)
    if config.input_control_enabled:
        try:
            from bantz.tools.input_control import _detect_backend
            ic_backend = _detect_backend()
            print(f"✅ Input Control: {ic_backend}  (confirm_destructive={config.input_confirm_destructive})")
        except Exception:
            print(f"❌ Input Control: enabled but detection failed")
    else:
        print(f"⚪ Input Control: disabled  → BANTZ_INPUT_CONTROL_ENABLED=true")

    # Spatial Cache (#121)
    try:
        from bantz.vision.spatial_cache import spatial_db as _sc
        _sc.init(config.db_path)
        sc_stats = _sc.stats()
        print(f"✅ Spatial Cache: {sc_stats['total_entries']} entries, {sc_stats['total_hits']} hits")
    except Exception:
        print("⚪ Spatial Cache: not initialized")

    # Navigator Pipeline (#123)
    try:
        from bantz.vision.navigator import navigator as _nav
        _nav.init(config.db_path)
        nav_stats = _nav.stats()
        total = nav_stats.get('total_attempts', 0)
        if total > 0:
            methods = nav_stats.get('methods', [])
            summary = ', '.join(f"{m['method']}={m['successes']}/{m['attempts']}" for m in methods)
            print(f"✅ Navigator: {total} attempts  ({summary})")
        else:
            print(f"✅ Navigator: ready (no attempts yet)")
    except Exception:
        print("⚪ Navigator: not initialized")

    # Background Observer (#124)
    if config.observer_enabled:
        print(f"✅ Observer: threshold={config.observer_severity_threshold}, model={config.observer_analysis_model}")
    else:
        print(f"⚪ Observer: disabled  → BANTZ_OBSERVER_ENABLED=true")

    # RL Engine (#125)
    if config.rl_enabled:
        try:
            from bantz.agent.rl_engine import rl_engine as _rl
            _rl.init(config.db_path)
            print(f"✅ RL Engine: {_rl.status_line()}")
        except Exception:
            print(f"❌ RL Engine: enabled but init failed")
    else:
        print(f"⚪ RL Engine: disabled  → BANTZ_RL_ENABLED=true")

    # Intervention Queue (#126)
    if config.rl_enabled:
        try:
            from bantz.agent.interventions import intervention_queue as _ivq
            _ivq.init(config.db_path, rate_limit=config.intervention_rate_limit, default_ttl=config.intervention_toast_ttl)
            print(f"✅ Interventions: {_ivq.status_line()}")
        except Exception:
            print(f"❌ Interventions: enabled but init failed")
    else:
        print(f"⚪ Interventions: disabled (requires RL)")

    # App Detector (#127)
    if config.app_detector_enabled:
        try:
            from bantz.agent.app_detector import app_detector as _ad
            _ad.init(
                cache_ttl=config.app_detector_cache_ttl,
                polling_interval=config.app_detector_polling_interval,
            )
            print(f"✅ App Detector: {_ad.status_line()}")
        except Exception:
            print(f"❌ App Detector: enabled but init failed")
    else:
        print(f"⚪ App Detector: disabled  → BANTZ_APP_DETECTOR_ENABLED=true")

    # Desktop Notifications (#153)
    if config.desktop_notifications:
        try:
            from bantz.agent.notifier import notifier as _dn
            _dn.init(
                enabled=config.desktop_notifications,
                icon=config.notification_icon,
                sound=config.notification_sound,
            )
            print(f"✅ Notifications: {_dn.status_line()}")
        except Exception:
            print(f"❌ Notifications: enabled but init failed")
    else:
        print(f"⚪ Notifications: disabled  → BANTZ_DESKTOP_NOTIFICATIONS=true")

    # TTS / Audio Briefing (#131)
    if config.tts_enabled:
        try:
            from bantz.agent.tts import tts_engine as _tts
            if _tts.available():
                print(f"✅ TTS: ready (model={config.tts_model}, auto_briefing={config.tts_auto_briefing})")
            else:
                print(f"❌ TTS: enabled but piper/aplay not found")
        except Exception:
            print(f"❌ TTS: enabled but init failed")
    else:
        print(f"⚪ TTS: disabled  → BANTZ_TTS_ENABLED=true")

    # Audio Ducking (#171)
    if config.audio_duck_enabled:
        try:
            from bantz.agent.audio_ducker import audio_ducker as _ducker
            diag = _ducker.diagnose()
            if diag["pactl_available"]:
                print(f"✅ Audio Ducking: ready (duck to {config.audio_duck_pct}%)")
            else:
                print(f"❌ Audio Ducking: enabled but pactl not found")
        except Exception as exc:
            print(f"❌ Audio Ducking: init failed — {exc}")
    else:
        print(f"⚪ Audio Ducking: disabled  → BANTZ_AUDIO_DUCK_ENABLED=true")

    # Wake Word Detection (#165)
    if config.wake_word_enabled:
        if not config.picovoice_access_key:
            print(f"❌ Wake Word: enabled but BANTZ_PICOVOICE_ACCESS_KEY not set")
        else:
            try:
                from bantz.agent.wake_word import wake_listener
                diag = wake_listener.diagnose()
                if diag["porcupine_available"]:
                    if diag["pyaudio_available"]:
                        print(f"✅ Wake Word: ready (sensitivity={config.wake_word_sensitivity})")
                    else:
                        print(f"❌ Wake Word: pvporcupine OK but pyaudio not found")
                else:
                    print(f"❌ Wake Word: pvporcupine not installed  → pip install pvporcupine")
            except Exception as exc:
                print(f"❌ Wake Word: init failed — {exc}")
    else:
        print(f"⚪ Wake Word: disabled  → BANTZ_WAKE_WORD_ENABLED=true")

    # Ambient Sound Analysis (#166)
    if config.ambient_enabled:
        if not config.wake_word_enabled:
            print(f"❌ Ambient: enabled but wake_word disabled (ambient piggybacks on wake word mic)")
        else:
            try:
                from bantz.agent.ambient import ambient_analyzer as _amb
                diag = _amb.diagnose()
                print(f"✅ Ambient: ready (interval={config.ambient_interval}s, window={config.ambient_window}s)")
            except Exception as exc:
                print(f"❌ Ambient: init failed — {exc}")
    else:
        print(f"⚪ Ambient: disabled  → BANTZ_AMBIENT_ENABLED=true")

    # Telegram
    if config.telegram_bot_token:
        print(f"✅ Telegram: token set")
    else:
        print(f"⚪ Telegram: not configured  → bantz --setup telegram")

    # Habits
    from bantz.core.habits import habits as _hab
    print(f"✅ Habits: {_hab.status_line()}")

    # Places
    from bantz.core.places import places as _plc
    if _plc.is_configured():
        print(f"✅ Places: {_plc.status_line()}")
    else:
        print(f"⚪ Places: not configured  → bantz --setup places")

    # GPS
    from bantz.core.gps_server import gps_server
    gps_line = gps_server.status_line()
    gps_icon = "✅" if config.gps_relay_token else "⚪"
    print(f"{gps_icon} {gps_line}")

    # Scheduler
    from bantz.core.scheduler import scheduler as _sched
    _sched.init(config.db_path)
    print(f"✅ {_sched.status_line()}")

    # Job Scheduler — APScheduler (#128)
    if config.job_scheduler_enabled:
        try:
            from bantz.agent.job_scheduler import job_scheduler as _js
            await _js.start(config.db_path, enable_night_jobs=True)
            print(f"✅ Job Scheduler: {_js.status_line()}")
            await _js.shutdown()
        except ImportError:
            print(f"❌ Job Scheduler: apscheduler not installed  → pip install apscheduler")
        except Exception as exc:
            print(f"❌ Job Scheduler: enabled but init failed: {exc}")
    else:
        print(f"⚪ Job Scheduler: disabled  → BANTZ_JOB_SCHEDULER_ENABLED=true")

    # systemd service (#173) — quick summary
    import os as _os
    from pathlib import Path as _Path
    _user = _os.environ.get("USER", "")
    _svc_path = _Path.home() / ".config" / "systemd" / "user" / "bantz.service"
    if _svc_path.exists():
        import subprocess as _sp
        _active = _sp.run(
            ["systemctl", "--user", "is-active", "bantz.service"],
            capture_output=True, text=True,
        )
        _state = _active.stdout.strip()
        if _state == "active":
            _linger = "linger=yes" if _check_linger(_user) else "linger=no"
            print(f"✅ systemd: active ({_linger})  → bantz --setup systemd --check")
        else:
            print(f"⚪ systemd: {_state}  → systemctl --user start bantz.service")
    else:
        print(f"⚪ systemd: not installed  → bantz --setup systemd")

    print("─" * 52)


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
    from bantz.config import config
    config.ensure_dirs()

    from bantz.interface.tui.mood import MoodHistory, MOOD_FACES

    history = MoodHistory()
    history.init(config.db_path)

    entries = history.recent(hours=24)
    if not entries:
        print("No mood transitions recorded yet. Start the TUI first.")
        return

    print("\n🎭 Bantz — Mood History (last 24h)")
    print("─" * 50)
    for e in entries:
        ts = e["timestamp"][:19].replace("T", " ")
        mood = e["mood"]
        prev = e["prev_mood"]
        face = MOOD_FACES.get(mood, "(?)")  # type: ignore[arg-type]
        reason = e.get("reason", "")
        cpu = e.get("cpu_pct", 0)
        print(f"  {ts}  {prev:>8} → {mood:<8} {face}  CPU:{cpu:.0f}%  {reason}")

    # Summary
    summary = history.summary_24h()
    if summary:
        print(f"\n⏱ Time in mood (minutes):")
        for mood, mins in sorted(summary.items(), key=lambda x: -x[1]):
            face = MOOD_FACES.get(mood, "(?)")  # type: ignore[arg-type]
            print(f"  {mood:<10} {face}  {mins:.0f} min")
    print()


async def _once(query: str) -> None:
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
    kv = SQLiteKVStore(config.db_path)

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


def _setup_systemd() -> None:
    """Install Bantz systemd user service with linger + verification."""
    import os
    import subprocess
    from pathlib import Path

    print("\n🦌 Bantz — systemd Service Setup")
    print("─" * 52)

    user = os.environ.get("USER", "")
    if not user:
        print("✗ Cannot determine current user.")
        return

    project_dir = Path(__file__).resolve().parent.parent.parent  # src/bantz → repo root
    venv_python = project_dir / ".venv" / "bin" / "python"
    env_file = project_dir / ".env"

    # Build user-mode service (no User=, no ProtectSystem=strict)
    content = f"""[Unit]
Description=Bantz v2 — Personal AI Assistant (Daemon)
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={project_dir}
EnvironmentFile={env_file}
ExecStart={venv_python} -m bantz --daemon
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bantz

[Install]
WantedBy=default.target
"""

    # Install to user systemd directory
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    target = systemd_dir / "bantz.service"
    target.write_text(content)

    print(f"✅ Service file installed: {target}")
    print()

    # Ensure linger for 24/7 operation
    _ensure_linger(user)
    print()

    # Ask if user wants to enable + start now
    answer = input("Enable and start now? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        ok = True
        ok = _systemctl("daemon-reload") and ok
        ok = _systemctl("enable", "bantz.service") and ok
        ok = _systemctl("start", "bantz.service") and ok
        if ok:
            print()
            _verify_service()
        else:
            print("\n⚠  Some steps failed. Check errors above.")
    else:
        print("Setup complete — enable manually when ready:")
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable bantz.service")
        print("  systemctl --user start bantz.service")


# ── systemd helpers ────────────────────────────────────────────────────────


def _check_linger(user: str) -> bool:
    """Check if loginctl linger is enabled for user.

    Uses ``loginctl show-user`` (official systemd API) instead of
    probing the filesystem directly.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["loginctl", "show-user", user, "--property=Linger"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            # Output: "Linger=yes" or "Linger=no"
            return "Linger=yes" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _ensure_linger(user: str) -> bool:
    """Check and enable loginctl linger for persistent user services."""
    if _check_linger(user):
        print(f"✅ Linger: already enabled for {user}")
        return True

    print("⚠  Linger is NOT enabled.")
    print("   Without linger, Bantz stops when you log out.")
    print(f"   This will run: loginctl enable-linger {user}")
    answer = input("   Enable linger for 24/7 operation? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        print("   Skipped. Service will stop on logout.")
        return False

    import subprocess
    # NOTE: capture_output=False intentionally — if polkit prompts for a
    # password, the user must see the prompt and be able to type it.
    # With capture_output=True the terminal would freeze silently.
    result = subprocess.run(
        ["loginctl", "enable-linger", user],
        capture_output=False, text=True,
    )
    if result.returncode == 0:
        # Double-check via the official API
        if _check_linger(user):
            print(f"✅ Linger enabled for {user}")
            return True
        # Command succeeded but check fails (rare edge case)
        print(f"✅ Linger command succeeded for {user}")
        return True
    else:
        print(f"❌ Failed to enable linger (exit code {result.returncode})")
        print(f"   Try manually: sudo loginctl enable-linger {user}")
        return False


def _systemctl(*args: str) -> bool:
    """Run systemctl --user <args> with proper error handling.

    Uses subprocess.run for safe execution with error capture.
    Returns True on success, False on failure with stderr printed.
    """
    import subprocess
    cmd = ["systemctl", "--user", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"❌ systemctl --user {' '.join(args)}: {stderr or f'exit code {result.returncode}'}")
        return False
    return True


def _verify_service() -> None:
    """Check that bantz.service is active and healthy."""
    import subprocess
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "bantz.service"],
        capture_output=True, text=True,
    )
    state = result.stdout.strip()
    if state == "active":
        print(f"✅ bantz.service is running")
        # Show compact status (let output go to terminal)
        subprocess.run(
            ["systemctl", "--user", "status", "bantz.service", "--no-pager", "-l"],
            capture_output=False,
        )
    else:
        print(f"⚠  Service state: {state}")
        print("   Check logs: journalctl --user -u bantz.service -n 20")


def _systemd_check() -> None:
    """Full systemd health diagnostic (bantz --setup systemd --check)."""
    import os
    import subprocess
    from pathlib import Path

    print("\nBantz — systemd Health Check")
    print("─" * 52)

    user = os.environ.get("USER", "")

    # 1. Service file
    service_path = Path.home() / ".config" / "systemd" / "user" / "bantz.service"
    if service_path.exists():
        print(f"✅ Service file: {service_path}")
    else:
        print(f"❌ Service file: NOT FOUND — run: bantz --setup systemd")
        return

    # 2. Linger
    if _check_linger(user):
        print(f"✅ Linger: enabled for {user}")
    else:
        print(f"⚪ Linger: disabled — run: loginctl enable-linger {user}")

    # 3. Service state
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "bantz.service"],
        capture_output=True, text=True,
    )
    state = result.stdout.strip()
    if state == "active":
        print(f"✅ Service: active (running)")
    elif state == "inactive":
        print(f"⚪ Service: inactive — run: systemctl --user start bantz.service")
    else:
        print(f"❌ Service: {state}")

    # 4. Detailed properties (PID, Memory, uptime)
    if state == "active":
        props = subprocess.run(
            ["systemctl", "--user", "show", "bantz.service",
             "--property=MainPID,MemoryCurrent,ActiveEnterTimestamp"],
            capture_output=True, text=True,
        )
        if props.returncode == 0:
            prop_map: dict[str, str] = {}
            for line in props.stdout.strip().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    prop_map[k] = v

            pid = prop_map.get("MainPID", "?")
            print(f"   PID: {pid}")

            # Memory (bytes → human-readable)
            mem_raw = prop_map.get("MemoryCurrent", "")
            if mem_raw and mem_raw.isdigit() and int(mem_raw) > 0:
                mem_mb = int(mem_raw) / (1024 * 1024)
                print(f"   Memory: {mem_mb:.1f}M")
            elif mem_raw == "[not set]" or not mem_raw:
                print("   Memory: (cgroup accounting not available)")

            # Uptime
            ts_raw = prop_map.get("ActiveEnterTimestamp", "")
            if ts_raw:
                try:
                    from datetime import datetime
                    # systemd outputs local time like "Thu 2025-01-09 14:30:00 TRT"
                    # Parse with dateutil for flexibility, fall back to showing raw
                    _uptime_str = _format_uptime(ts_raw)
                    print(f"   Uptime: {_uptime_str}")
                except Exception:
                    print(f"   Started: {ts_raw}")

    # 5. Journal errors (last 24h)
    journal = subprocess.run(
        ["journalctl", "--user", "-u", "bantz.service",
         "--since", "24 hours ago", "-p", "err", "--no-pager", "-q"],
        capture_output=True, text=True,
    )
    if journal.returncode == 0:
        error_lines = [l for l in journal.stdout.strip().splitlines() if l.strip()]
        print(f"   Journal errors (24h): {len(error_lines)}")
        if error_lines:
            for line in error_lines[-3:]:  # Show last 3
                print(f"     {line}")
    else:
        print("   Journal: could not read (journalctl error)")

    # 6. Enabled on boot?
    enabled_result = subprocess.run(
        ["systemctl", "--user", "is-enabled", "bantz.service"],
        capture_output=True, text=True,
    )
    en_state = enabled_result.stdout.strip()
    if en_state == "enabled":
        print(f"✅ Boot: enabled (starts on login)")
    else:
        print(f"⚪ Boot: {en_state} — run: systemctl --user enable bantz.service")

    print("─" * 52)


def _format_uptime(timestamp_str: str) -> str:
    """Parse systemd ActiveEnterTimestamp and return human-readable uptime."""
    from datetime import datetime, timezone
    import re

    # Strip timezone abbreviation (e.g. "TRT", "UTC", "CET")
    # systemd format: "Thu 2025-01-09 14:30:00 TRT"
    cleaned = re.sub(r"\s+[A-Z]{2,5}\s*$", "", timestamp_str.strip())
    # Try common systemd datetime formats
    for fmt in (
        "%a %Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            started = datetime.strptime(cleaned, fmt)
            delta = datetime.now() - started
            days = delta.days
            hours, rem = divmod(delta.seconds, 3600)
            minutes = rem // 60
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")
            return " ".join(parts)
        except ValueError:
            continue
    return timestamp_str  # fallback: return raw


if __name__ == "__main__":
    main()