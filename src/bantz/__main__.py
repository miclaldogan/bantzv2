"""
Bantz v2 — Entry point

Commands:
  bantz                         → TUI
  bantz --once "query"          → single query, no UI
  bantz --doctor                → system health check
  bantz --setup profile         → user profile setup
  bantz --setup google gmail    → OAuth setup for Gmail
  bantz --setup google classroom → OAuth setup for Classroom
  bantz --setup schedule        → class schedule setup
  bantz --setup telegram        → Telegram bot token setup
  bantz --setup places          → Known locations setup
"""
from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(prog="bantz", description="Bantz v2 — your terminal host")
    parser.add_argument("--once", metavar="QUERY", help="Run single query, no UI")
    parser.add_argument("--doctor", action="store_true", help="System health check")
    parser.add_argument("--setup", nargs="+", metavar="SERVICE",
                        help="Setup integrations: --setup google gmail")
    args = parser.parse_args()

    if args.doctor:
        asyncio.run(_doctor())
        return

    if args.setup:
        _handle_setup(args.setup)
        return

    if args.once:
        asyncio.run(_once(args.once))
        return

    from bantz.app import run
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
    print("Start with: python -m bantz.integrations.telegram_bot")


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
                if confirm in ("h", "hayır", "n", "no"):
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
            if is_primary in ("", "e", "evet", "y", "yes"):
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
    import bantz.tools.gmail
    import bantz.tools.calendar
    import bantz.tools.classroom

    print("Bantz v2 — System Check")
    print("─" * 44)

    # Ollama
    ok = await ollama.is_available()
    status = "connected" if ok else "UNREACHABLE"
    print(f"{'✓' if ok else '✗'} Ollama ({config.ollama_base_url}): {status}")
    print(f"  model: {config.ollama_model}")

    # Gemini
    from bantz.llm.gemini import gemini as _gem
    if _gem.is_enabled():
        gem_ok = await _gem.is_available()
        gem_status = "connected" if gem_ok else "UNREACHABLE"
        print(f"{'✓' if gem_ok else '✗'} Gemini ({config.gemini_model}): {gem_status}")
    else:
        print(f"○ Gemini: disabled  → bantz --setup gemini")

    # psutil
    import psutil
    print(f"✓ psutil: CPU {psutil.cpu_percent(interval=0.3):.0f}%")

    # Tools
    names = [t["name"] for t in registry.all_schemas()]
    print(f"✓ Tools ({len(names)}): {', '.join(names)}")

    # Translation / Bridge
    print(f"  translation_enabled: {config.translation_enabled}")
    if config.translation_enabled and config.language == "tr":
        try:
            from transformers import AutoTokenizer  # noqa: F401
            print("✓ MarianMT: available")
        except ImportError:
            print("✗ MarianMT: NOT installed  → pip install 'bantz[translation]'")

    # Location
    from bantz.core.location import location_service
    loc = await location_service.get()
    print(f"✓ Location: {loc.display}  (via {loc.source})")

    # Google integrations
    print("  Google integrations:")
    g_status = token_store.status()
    for svc, st in g_status.items():
        icon = "✓" if st == "ok" else "○"
        print(f"  {icon} {svc}: {st}")
    if any(st != "ok" for st in g_status.values()):
        print("  → Run: bantz --setup google gmail  /  bantz --setup google classroom")

    # Memory DB
    config.ensure_dirs()
    from bantz.core.memory import memory as _mem
    _mem.init(config.db_path)
    s = _mem.stats()
    print(f"✓ Memory DB: {s['db_path']}")
    print(f"  {s['total_conversations']} conversations  |  {s['total_messages']} total messages")

    # Profile
    from bantz.core.profile import profile as _prof
    icon = "✓" if _prof.is_configured() else "○"
    print(f"{icon} Profile: {_prof.status_line()}")

    # Telegram
    tg_ok = bool(config.telegram_bot_token)
    tg_icon = "✓" if tg_ok else "○"
    tg_status = "token set" if tg_ok else "not configured  → bantz --setup telegram"
    print(f"{tg_icon} Telegram: {tg_status}")

    # Habits
    from bantz.core.habits import habits as _hab
    print(f"✓ Habits: {_hab.status_line()}")

    # Places
    from bantz.core.places import places as _plc
    plc_icon = "✓" if _plc.is_configured() else "○"
    print(f"{plc_icon} Places: {_plc.status_line()}")

    # GPS
    from bantz.core.gps_server import gps_server
    print(f"○ {gps_server.status_line()}")

    print("─" * 44)


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


if __name__ == "__main__":
    main()