"""
Bantz v2 — CLI setup wizards, diagnostics, and configuration display.

All interactive --setup commands, --doctor, --config, and --cache-stats
live here so __main__.py stays focused on running Bantz.
"""
from __future__ import annotations

import asyncio


def _handle_setup(parts: list[str]) -> None:
    if len(parts) >= 1 and parts[0].lower() == "onboarding":
        _setup_onboarding()
        return
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
    if len(parts) >= 1 and parts[0].lower() == "claude":
        _setup_claude()
        return
    if len(parts) >= 1 and parts[0].lower() == "openai":
        _setup_openai()
        return
    if len(parts) >= 1 and parts[0].lower() == "voice":
        _setup_voice()
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
        print("  bantz --setup onboarding")
        print("  bantz --setup profile")
        print("  bantz --setup google [gmail|classroom|calendar]")
        print("  bantz --setup schedule")
        print("  bantz --setup telegram")
        print("  bantz --setup places")
        print("  bantz --setup claude        ← Anthropic Claude API")
        print("  bantz --setup openai        ← OpenAI / compatible API")
        print("  bantz --setup gemini        ← Google Gemini API")
        print("  bantz --setup voice")
        print("  bantz --setup systemd")
        print("  bantz --setup systemd --check")


def _setup_onboarding() -> None:
    """Run the first-run onboarding wizard (bantz --setup onboarding)."""
    from pathlib import Path
    from bantz.config import config

    print("\n🧠 Bantz — First-Run Onboarding")
    print("─" * 40)

    config.ensure_dirs()

    try:
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.entity_registry import EntityRegistry
    except ImportError:
        print("⚠  MemPalace is not installed.")
        print("   Run: pip install 'bantz[memory]'")
        return

    from bantz.memory.onboarding import is_onboarding_done, run_onboarding

    palace_path = config.resolved_palace_path
    kg_path = config.resolved_kg_path
    identity_path = config.resolved_identity_path
    palace_parent = str(Path(palace_path).parent)

    if is_onboarding_done(palace_parent):
        print("Onboarding was already completed.")
        redo = input("Run again? [y/N] ").strip().lower()
        if redo not in ("y", "yes", "evet", "e"):
            print("Skipped.")
            return

    Path(palace_path).mkdir(parents=True, exist_ok=True)
    Path(kg_path).parent.mkdir(parents=True, exist_ok=True)

    kg = KnowledgeGraph(db_path=kg_path)
    registry = EntityRegistry.load()

    run_onboarding(
        identity_path=identity_path,
        kg=kg,
        registry=registry,
        palace_parent=palace_parent,
    )
    print("\n✅ Onboarding complete. Bantz now knows who you are.")


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
    lines = [line for line in lines if not line.startswith("TELEGRAM_BOT_TOKEN=")
             and not line.startswith("TELEGRAM_ALLOWED_USERS=")
             and not line.startswith("TELEGRAM_PROXY=")]

    lines.append(f"TELEGRAM_BOT_TOKEN={token}")
    if allowed:
        lines.append(f"TELEGRAM_ALLOWED_USERS={allowed}")
    if proxy:
        lines.append(f"TELEGRAM_PROXY={proxy}")

    import os
    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ Token saved: {env_path}")
    print("Start with: python -m bantz.interface.telegram_bot")


def _env_path():
    """Return the .env file path — prefers ~/.local/share/bantz/.env, falls back to CWD."""
    from pathlib import Path
    data_env = Path.home() / ".local" / "share" / "bantz" / ".env"
    return data_env if data_env.exists() else Path.cwd() / ".env"


def _write_env(updates: dict[str, str], *, prefix_strip: list[str] | None = None) -> None:
    """Write key=value pairs into the .env file, replacing existing entries."""
    import os
    env_path = _env_path()
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for pfx in (prefix_strip or []):
        lines = [line for line in lines if not line.startswith(pfx)]
    for k, v in updates.items():
        lines = [line for line in lines if not line.startswith(f"{k}=")]
        lines.append(f"{k}={v}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n   Saved to: {env_path}")


def _setup_claude() -> None:
    """Interactive Claude (Anthropic) API key setup."""
    print("\nBantz — Claude Setup")
    print("─" * 40)
    print("1. Go to https://console.anthropic.com/settings/keys")
    print("2. Create an API key")
    print("3. Paste it below (input is hidden):\n")

    import getpass
    api_key = getpass.getpass("Anthropic API key: ").strip()
    if not api_key:
        print("No key provided. Cancelled.")
        return

    print("\nAvailable models:")
    print("  1  claude-sonnet-4-6        (recommended — fast + smart)")
    print("  2  claude-opus-4-8          (most capable, slower)")
    print("  3  claude-haiku-4-5-20251001 (fastest, cheapest)")
    choice = input("\nModel choice [1]: ").strip() or "1"
    model = {
        "1": "claude-sonnet-4-6",
        "2": "claude-opus-4-8",
        "3": "claude-haiku-4-5-20251001",
    }.get(choice, "claude-sonnet-4-6")

    _write_env({
        "BANTZ_LLM_PROVIDER":    "claude",
        "BANTZ_ANTHROPIC_API_KEY": api_key,
        "BANTZ_ANTHROPIC_MODEL": model,
    })
    print("   Provider: Claude / Anthropic")
    print(f"   Model:    {model}")
    print("\nRun 'bantz --doctor' to verify the connection.")


def _setup_openai() -> None:
    """Interactive OpenAI API key setup."""
    print("\nBantz — OpenAI Setup")
    print("─" * 40)
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Create an API key")
    print("3. Paste it below (input is hidden):\n")

    import getpass
    api_key = getpass.getpass("OpenAI API key: ").strip()
    if not api_key:
        print("No key provided. Cancelled.")
        return

    print("\nAvailable models:")
    print("  1  gpt-4o-mini   (recommended — fast + cheap)")
    print("  2  gpt-4o        (most capable)")
    print("  3  o3-mini       (reasoning model)")
    choice = input("\nModel choice [1]: ").strip() or "1"
    model = {
        "1": "gpt-4o-mini",
        "2": "gpt-4o",
        "3": "o3-mini",
    }.get(choice, "gpt-4o-mini")

    base_url = input("API base URL (Enter for default openai.com): ").strip()
    updates = {
        "BANTZ_LLM_PROVIDER":  "openai",
        "BANTZ_OPENAI_API_KEY": api_key,
        "BANTZ_OPENAI_MODEL":   model,
    }
    if base_url:
        updates["BANTZ_OPENAI_BASE_URL"] = base_url

    _write_env(updates)
    print("   Provider: OpenAI")
    print(f"   Model:    {model}")
    print("\nRun 'bantz --doctor' to verify the connection.")


def _setup_gemini() -> None:
    """Interactive Gemini API key setup."""
    print("\nBantz — Gemini Setup")
    print("─" * 40)
    print("1. Go to https://aistudio.google.com/apikey")
    print("2. Create an API key")
    print("3. Paste it below:\n")

    import getpass
    api_key = getpass.getpass("Gemini API key: ").strip()
    if not api_key:
        print("No key provided. Cancelled.")
        return

    model = input("Model (default: gemini-2.0-flash): ").strip() or "gemini-2.0-flash"

    _write_env({
        "BANTZ_LLM_PROVIDER":  "gemini",
        "BANTZ_GEMINI_ENABLED": "true",
        "BANTZ_GEMINI_API_KEY": api_key,
        "BANTZ_GEMINI_MODEL":   model,
    })
    print("   Provider: Gemini")
    print(f"   Model:    {model}")
    print("\nRun 'bantz --doctor' to verify the connection.")


# ── Voice packages checked by _setup_voice ──────────────────────────────────
# Maps pip install name → importable name
_VOICE_PACKAGES: dict[str, str] = {
    "faster-whisper": "faster_whisper",
    "pyaudio": "pyaudio",
    "webrtcvad": "webrtcvad",
    "pvporcupine": "pvporcupine",
}

# .env keys removed / re-added by _setup_voice
_VOICE_ENV_PREFIXES: tuple[str, ...] = (
    "BANTZ_VOICE_ENABLED=",
    "BANTZ_TTS_ENABLED=",
    "BANTZ_STT_ENABLED=",
    "BANTZ_GHOST_LOOP_ENABLED=",
    "BANTZ_WAKE_WORD_ENABLED=",
    "BANTZ_PICOVOICE_ACCESS_KEY=",
)


def _setup_voice() -> None:
    """Guided voice setup wizard (`bantz --setup voice`).

    1. Checks which of faster-whisper, pyaudio, webrtcvad, pvporcupine
       are installed.
    2. Offers to run ``pip install`` for missing packages.
    3. Prompts for ``BANTZ_PICOVOICE_ACCESS_KEY`` (wake-word detection).
    4. Writes ``BANTZ_VOICE_ENABLED``, ``BANTZ_TTS_ENABLED``,
       ``BANTZ_STT_ENABLED``, ``BANTZ_GHOST_LOOP_ENABLED`` (and
       ``BANTZ_WAKE_WORD_ENABLED`` when a Picovoice key is supplied) to
       ``.env``.
    5. Runs a mic test, STT availability check, and TTS binary check.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    print("\n🎤 Bantz — Voice Setup Wizard")
    print("─" * 40)
    print("Sets up voice input (STT / wake-word) and TTS output.")
    print()

    # ── Step 1: check installed packages ────────────────────────────────
    print("Step 1/4 — Checking voice dependencies…")
    print()
    missing: list[str] = []
    for pkg_name, import_name in _VOICE_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name}")
        except ImportError:
            missing.append(pkg_name)
            print(f"  ❌ {pkg_name} (not installed)")
    print()

    # ── Step 2: offer pip install for missing packages ───────────────────
    if missing:
        print(f"Step 2/4 — Install missing packages: {', '.join(missing)}")
        answer = input("  Install now? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes", "evet", "e"):
            for pkg in missing:
                print(f"  pip install {pkg}…")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg],
                )
                if result.returncode == 0:
                    print(f"  ✅ {pkg} installed")
                else:
                    print(f"  ❌ {pkg} install failed (exit {result.returncode})")
        else:
            print("  Skipped. Install manually:")
            print(f"    pip install {' '.join(missing)}")
        print()
    else:
        print("Step 2/4 — All packages present. ✅")
        print()

    # ── Step 3: Picovoice access key ─────────────────────────────────────
    print("Step 3/4 — Picovoice Access Key (wake-word detection)")
    print("  Get a free key at: https://console.picovoice.ai/")
    print("  Leave blank to skip wake-word detection.")
    picovoice_key = input("  Picovoice Access Key: ").strip()
    print()

    # ── Step 4: write .env ───────────────────────────────────────────────
    print("Step 4/4 — Saving settings to .env…")
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    # Strip any existing voice entries so we write a clean set
    lines = [
        line for line in lines
        if not any(line.startswith(p) for p in _VOICE_ENV_PREFIXES)
    ]

    lines.append("BANTZ_VOICE_ENABLED=true")
    lines.append("BANTZ_TTS_ENABLED=true")
    lines.append("BANTZ_STT_ENABLED=true")
    lines.append("BANTZ_GHOST_LOOP_ENABLED=true")
    if picovoice_key:
        lines.append(f"BANTZ_PICOVOICE_ACCESS_KEY={picovoice_key}")
        lines.append("BANTZ_WAKE_WORD_ENABLED=true")

    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✅ Saved: {env_path}")
    print()

    # ── Step 5: component tests ──────────────────────────────────────────
    print("─" * 40)
    print("Running voice component tests…")
    print()
    mic_ok = _voice_test_mic()
    stt_ok = _voice_test_stt()
    tts_ok = _voice_test_tts()
    print()

    if all([mic_ok, stt_ok, tts_ok]):
        print("✅ All voice components ready.")
    else:
        print("⚠  Some components failed — see above for details.")
    print("Restart Bantz for settings to take effect: bantz")
    print()


def _voice_test_mic() -> bool:
    """Test microphone access (reads one PyAudio frame; no recording kept).

    Returns True on success, False on failure.
    """
    print("  Mic test…", end=" ", flush=True)
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=480,
        )
        data = stream.read(480, exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        if data:
            print("PASS ✅")
            return True
        print("FAIL ❌ (no audio data)")
        return False
    except ImportError:
        print("SKIP ⚠  (pyaudio not installed)")
        return False
    except Exception as exc:
        print(f"FAIL ❌ ({exc})")
        return False


def _voice_test_stt() -> bool:
    """Test STT availability — import check only (no model loaded here).

    Returns True if faster-whisper is importable, False otherwise.
    """
    print("  STT (faster-whisper) test…", end=" ", flush=True)
    try:
        import faster_whisper  # noqa: F401
        print("PASS ✅")
        return True
    except ImportError:
        print("FAIL ❌ (faster-whisper not installed)")
        return False


def _voice_test_tts() -> bool:
    """Test TTS availability — checks for the Piper binary.

    Returns True if a Piper executable is found, False otherwise.
    """
    import shutil
    from pathlib import Path

    print("  TTS (Piper) test…", end=" ", flush=True)
    candidates = [
        shutil.which("piper"),
        shutil.which("piper-tts"),
        str(Path.home() / ".local" / "bin" / "piper"),
        str(Path.home() / "miniforge3" / "bin" / "piper"),
    ]
    if any(c and Path(c).exists() for c in candidates):
        print("PASS ✅")
        return True
    print("FAIL ❌ (piper binary not found — see https://github.com/rhasspy/piper)")
    return False


async def _setup_places() -> None:
    """Interactive known-places setup — writes places.json.
    Three coordinate options: auto IP, city name search, manual entry.
    First place (or explicit choice) becomes default location in .env.
    """
    import json
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
                    idx = input("  Choice [1]: ").strip() or "1"
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
        line for line in lines
        if not line.startswith("BANTZ_CITY=")
        and not line.startswith("BANTZ_LAT=")
        and not line.startswith("BANTZ_LON=")
    ]

    lines.append(f"BANTZ_CITY={city}")
    lines.append(f"BANTZ_LAT={lat}")
    lines.append(f"BANTZ_LON={lon}")

    import os
    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
    import os
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
                raw = input("  New class (blank=skip): ").strip()
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

    # Write securely
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
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
    "gemini_api_key", "telegram_bot_token",
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
        (("vector_search_",), "Vector Search"),
        (("distillation_",), "Distillation"),
        (("vlm_", "screenshot_"), "Vision / VLM"),
        (("input_control_", "input_confirm_", "input_type_"), "Input Control"),
        (("gemini_",), "Gemini"),
        (("language", "translation_"), "Language"),
        (("shell_",), "Shell Security"),
        (("location_",), "Location"),
        (("gps_relay_",), "GPS Relay"),
        (("mempalace_", "palace_"), "MemPalace"),
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
        (("voice_enabled",), "Voice"),
        (("tts_",), "TTS / Audio"),
        (("audio_duck_",), "Audio Ducking"),
        (("wake_word_", "picovoice_",), "Wake Word"),
        (("stt_", "vad_", "ghost_loop_",), "Ghost Loop / STT"),
        (("ambient_",), "Ambient Sound"),
    ]
    for prefixes, label in _MAP:
        if any(field_name.startswith(p) or field_name == p for p in prefixes):
            return label
    return "General"


def _check_whisper_model_cached(model_name: str) -> bool:
    """Check if a faster-whisper / CTranslate2 model is already on disk."""
    try:
        from pathlib import Path
        # HuggingFace hub cache: ~/.cache/huggingface/hub/models--*
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            # Model repos follow pattern: models--Systran--faster-whisper-<name>
            for d in hf_cache.iterdir():
                if d.is_dir() and model_name in d.name.lower() and "whisper" in d.name.lower():
                    return True
        # Also check CTranslate2 default cache
        # faster-whisper may also download to a direct path
        alt_cache = Path.home() / ".cache" / "faster_whisper"
        if alt_cache.exists():
            for d in alt_cache.iterdir():
                if d.is_dir() and model_name in d.name.lower():
                    return True
    except Exception:
        pass
    return False


async def _doctor() -> None:
    from bantz.llm.ollama import ollama
    from bantz.config import config
    from bantz.tools import registry
    from bantz.auth.token_store import token_store

    print("Bantz v2 — System Check")
    print("─" * 52)

    # ── Active LLM provider ──────────────────────────────────────────────────
    _provider = (config.llm_provider or "ollama").lower()

    if _provider == "claude":
        from bantz.llm.anthropic_client import claude as _claude
        if _claude.is_enabled():
            _ok = await _claude.is_available()
            if _ok:
                print(f"✅ Claude: connected — {config.anthropic_model}")
            else:
                print("❌ Claude: UNREACHABLE — check BANTZ_ANTHROPIC_API_KEY")
        else:
            print("❌ Claude: API key not set  → bantz --setup claude")
    elif _provider == "openai":
        from bantz.llm.openai_client import openai_client as _oai
        if _oai.is_enabled():
            _ok = await _oai.is_available()
            if _ok:
                print(f"✅ OpenAI: connected — {config.openai_model}")
            else:
                print("❌ OpenAI: UNREACHABLE — check BANTZ_OPENAI_API_KEY")
        else:
            print("❌ OpenAI: API key not set  → bantz --setup openai")
    elif _provider == "gemini":
        from bantz.llm.gemini import gemini as _gem
        _ok = await _gem.is_available()
        if _ok:
            print(f"✅ Gemini: connected — {config.gemini_model}")
        else:
            print("❌ Gemini: UNREACHABLE — check BANTZ_GEMINI_API_KEY")
    else:
        # Ollama (default)
        is_remote = "localhost" not in config.ollama_base_url and "127.0.0.1" not in config.ollama_base_url
        mode_label = "remote (GPU VPS)" if is_remote else "local"
        try:
            await ollama.verify_connection()
            print(f"✅ Ollama [{mode_label}]: connected — {config.ollama_model} @ {config.ollama_base_url}")
        except RuntimeError as _e:
            print(f"❌ Ollama [{mode_label}]: {_e}")

    # Ollama always shown as secondary if it's not the active provider
    if _provider != "ollama":
        is_remote = "localhost" not in config.ollama_base_url and "127.0.0.1" not in config.ollama_base_url
        mode_label = "remote" if is_remote else "local"
        try:
            await ollama.verify_connection()
            print(f"   Ollama [{mode_label}]: available — {config.ollama_model} (used for routing/tools)")
        except RuntimeError:
            print(f"   Ollama [{mode_label}]: offline (routing/tools will be degraded)")

    # MemPalace
    if config.mempalace_enabled:
        try:
            from bantz.memory.bridge import palace_bridge
            await palace_bridge.init()  # #432: .enabled is always False until init() is called
            if palace_bridge and palace_bridge.enabled:
                print(f"✅ MemPalace: enabled (wing={config.mempalace_wing})")
            else:
                print("❌ MemPalace: enabled but bridge not initialized")
        except Exception:
            print("❌ MemPalace: enabled but import failed")
    else:
        print("⚪ MemPalace: disabled  → BANTZ_MEMPALACE_ENABLED=true")

    # Vision / VLM
    if config.vlm_enabled:
        print(f"✅ Vision/VLM: enabled ({config.vlm_endpoint})")
    else:
        print("⚪ Vision/VLM: disabled  → BANTZ_VLM_ENABLED=true")

    # psutil
    import psutil
    print(f"✅ psutil: CPU {psutil.cpu_percent(interval=0.3):.0f}%")

    # Tools — import modules so they register themselves (#432: count was 0 at import time)
    import importlib as _importlib
    for _mod in (
        "bantz.tools.shell", "bantz.tools.system", "bantz.tools.filesystem",
        "bantz.tools.weather", "bantz.tools.web_search", "bantz.tools.web_reader",
        "bantz.tools.gmail", "bantz.tools.calendar", "bantz.tools.classroom",
        "bantz.tools.reminder",
    ):
        _importlib.import_module(_mod)
    for _opt in (
        "bantz.tools.news", "bantz.tools.document", "bantz.tools.accessibility",
        "bantz.tools.visual_click", "bantz.tools.browser_control",
        "bantz.tools.screenshot_tool", "bantz.tools.desktop",
        "bantz.tools.delegate_task",
    ):
        try:
            _importlib.import_module(_opt)
        except (ImportError, Exception):
            pass
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
        print("⚪ Location: unknown  → enable phone GPS relay")

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

    # MemPalace (ChromaDB embeddings are handled internally)
    from bantz.memory.bridge import palace_bridge
    if not palace_bridge.enabled:
        # init() may not have been called yet (e.g. mempalace_enabled was False above)
        await palace_bridge.init()  # #432: safe to call again — re-checks config flag
    if palace_bridge.enabled:
        print(f"✅ MemPalace: active  (vector_weight={config.vector_search_weight})")
    else:
        print("⚪ MemPalace: not initialised")

    # Profile
    from bantz.core.profile import profile as _prof
    if _prof.is_configured():
        print(f"✅ Profile: {_prof.status_line()}")
    else:
        print("⚪ Profile: not configured  → bantz --setup profile")

    # AT-SPI Accessibility
    try:
        from bantz.tools.accessibility import _init_atspi, list_applications, detect_display_server
        atspi_ok = _init_atspi()
        if atspi_ok:
            _atspi_apps = list_applications()
            _display = detect_display_server()
            print(f"✅ AT-SPI2: available — {len(_atspi_apps)} accessible app(s), display={_display}")
            if not _atspi_apps:
                print("   ⚠ No accessible apps found — ensure AT-SPI bus is active and apps support accessibility")
        else:
            print("❌ AT-SPI2: unavailable — sudo apt install python3-gi gir1.2-atspi-2.0")
    except Exception as _exc:
        print(f"❌ AT-SPI2: import failed ({_exc})")

    # Input Control (#122)
    if config.input_control_enabled:
        try:
            from bantz.tools.input_control import _detect_backend
            ic_backend = _detect_backend()
            print(f"✅ Input Control: {ic_backend}  (confirm_destructive={config.input_confirm_destructive})")
        except Exception:
            print("❌ Input Control: enabled but detection failed")
    else:
        print("⚪ Input Control: disabled  → BANTZ_INPUT_CONTROL_ENABLED=true")

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
            print("✅ Navigator: ready (no attempts yet)")
    except Exception:
        print("⚪ Navigator: not initialized")

    # Background Observer (#124)
    if config.observer_enabled:
        print(f"✅ Observer: threshold={config.observer_severity_threshold}, model={config.observer_analysis_model}")
    else:
        print("⚪ Observer: disabled  → BANTZ_OBSERVER_ENABLED=true")

    # Affinity Engine (#221)
    if config.rl_enabled:
        try:
            from bantz.agent.affinity_engine import affinity_engine as _ae
            _ae.init(config.db_path)
            print(f"✅ Affinity Engine: {_ae.status_line()}")
        except Exception:
            print("❌ Affinity Engine: enabled but init failed")
    else:
        print("⚪ Affinity Engine: disabled  → BANTZ_RL_ENABLED=true")

    # Intervention Queue (#126)
    if config.rl_enabled:
        try:
            from bantz.agent.interventions import intervention_queue as _ivq
            _ivq.init(config.db_path, rate_limit=config.intervention_rate_limit, default_ttl=config.intervention_toast_ttl)
            print(f"✅ Interventions: {_ivq.status_line()}")
        except Exception:
            print("❌ Interventions: enabled but init failed")
    else:
        print("⚪ Interventions: disabled (requires RL)")

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
            print("❌ App Detector: enabled but init failed")
    else:
        print("⚪ App Detector: disabled  → BANTZ_APP_DETECTOR_ENABLED=true")

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
            print("❌ Notifications: enabled but init failed")
    else:
        print("⚪ Notifications: disabled  → BANTZ_DESKTOP_NOTIFICATIONS=true")

    # ── Voice Pipeline (#277 — consolidated diagnostics) ───────────────
    print()
    _voice_on = config.voice_enabled
    if _voice_on:
        print("🎙️  Voice Master Switch: ON  (BANTZ_VOICE_ENABLED=true)")
    else:
        _any_voice = any([
            config.tts_enabled, config.wake_word_enabled,
            config.stt_enabled, config.ghost_loop_enabled,
            config.audio_duck_enabled, config.ambient_enabled,
        ])
        if _any_voice:
            print("🎙️  Voice (individual flags active)")
        else:
            print("⚪ Voice: disabled  → BANTZ_VOICE_ENABLED=true")

    # ── OS-level audio dependency (PortAudio) ────────────────────────
    _voice_pip_missing: list[str] = []  # #432: collect pip deps for consolidated fix command
    _voice_apt_missing: list[str] = []
    _portaudio_ok = False
    try:
        import ctypes.util
        _pa_lib = ctypes.util.find_library("portaudio")
        _portaudio_ok = _pa_lib is not None
    except Exception:
        pass
    if config.wake_word_enabled or config.stt_enabled or config.ghost_loop_enabled:
        if _portaudio_ok:
            print(f"  ✅ PortAudio: found ({_pa_lib})")
        else:
            print("  ❌ PortAudio: NOT found  → sudo apt install portaudio19-dev")
            _voice_apt_missing.append("portaudio19-dev")

    # ── PyAudio stream test ──────────────────────────────────────────
    _pyaudio_ok = False
    if config.wake_word_enabled or config.stt_enabled:
        try:
            import pyaudio  # noqa: F401
            _pyaudio_ok = True
            print("  ✅ PyAudio: importable")
        except ImportError:
            print("  ❌ PyAudio: NOT installed  → pip install pyaudio  (requires portaudio19-dev)")
            _voice_pip_missing.append("pyaudio")

    # TTS / Audio Briefing (#131)
    if config.tts_enabled:
        try:
            from bantz.agent.tts import tts_engine as _tts
            if _tts.available():
                print(f"  ✅ TTS: ready (model={config.tts_model}, auto_briefing={config.tts_auto_briefing})")
            else:
                print("  ❌ TTS: enabled but piper/aplay not found")
        except Exception:
            print("  ❌ TTS: enabled but init failed")
    elif _voice_on:
        print("  ⚪ TTS: master=ON but tts_enabled overridden to false")
    else:
        print("  ⚪ TTS: disabled  → BANTZ_TTS_ENABLED=true")

    # Audio Ducking (#171)
    if config.audio_duck_enabled:
        try:
            from bantz.agent.audio_ducker import audio_ducker as _ducker
            diag = _ducker.diagnose()
            if diag["pactl_available"]:
                print(f"  ✅ Audio Ducking: ready (duck to {config.audio_duck_pct}%)")
            else:
                print("  ❌ Audio Ducking: enabled but pactl not found")
        except Exception as exc:
            print(f"  ❌ Audio Ducking: init failed — {exc}")
    else:
        print("  ⚪ Audio Ducking: disabled  → BANTZ_AUDIO_DUCK_ENABLED=true")

    # Wake Word Detection (#165)
    if config.wake_word_enabled:
        if not config.picovoice_access_key:
            print("  ❌ Wake Word: enabled but BANTZ_PICOVOICE_ACCESS_KEY not set")
            print("       → Get free key: https://console.picovoice.ai/")
        else:
            try:
                from bantz.agent.wake_word import wake_listener
                diag = wake_listener.diagnose()
                if diag["porcupine_available"]:
                    if diag.get("pyaudio_available", _pyaudio_ok):
                        print(f"  ✅ Wake Word: ready (sensitivity={config.wake_word_sensitivity})")
                    else:
                        print("  ❌ Wake Word: pvporcupine OK but pyaudio not found")
                else:
                    print("  ❌ Wake Word: pvporcupine not installed  → pip install pvporcupine")
                    _voice_pip_missing.append("pvporcupine")
            except Exception as exc:
                print(f"  ❌ Wake Word: init failed — {exc}")
    else:
        print("  ⚪ Wake Word: disabled  → BANTZ_WAKE_WORD_ENABLED=true")

    # Ghost Loop / STT (#36) — Whisper model check
    if config.ghost_loop_enabled and config.stt_enabled:
        try:
            from bantz.agent.ghost_loop import ghost_loop as _gl
            diag = _gl.diagnose()
            stt_ok = diag["stt"].get("faster_whisper_available", False)
            vad_ok = diag["voice_capture"].get("webrtcvad_available", False)
            if stt_ok and vad_ok:
                # Check if Whisper model is already cached
                _model_cached = _check_whisper_model_cached(config.stt_model)
                _cache_note = "" if _model_cached else "  ⚠️  model not cached — will download on first run"
                print(f"  ✅ Ghost Loop: ready (model={config.stt_model}, lang={config.stt_language}, vad_silence={config.vad_silence_ms}ms)")
                if _cache_note:
                    print(f"     {_cache_note}")
            else:
                missing = []
                if not stt_ok:
                    missing.append("faster-whisper")
                    _voice_pip_missing.append("faster-whisper")
                if not vad_ok:
                    missing.append("webrtcvad")
                    _voice_pip_missing.append("webrtcvad")
                print(f"  ❌ Ghost Loop: missing deps → pip install {' '.join(missing)}")
        except Exception as exc:
            print(f"  ❌ Ghost Loop: init failed — {exc}")
    elif config.ghost_loop_enabled:
        print("  ❌ Ghost Loop: enabled but BANTZ_STT_ENABLED=false")
    else:
        print("  ⚪ Ghost Loop: disabled  → BANTZ_GHOST_LOOP_ENABLED=true + BANTZ_STT_ENABLED=true")

    # Ambient Sound Analysis (#166, #441)
    if config.ambient_enabled:
        if not config.wake_word_enabled:
            try:
                from bantz.agent.ambient import ambient_analyzer as _amb
                diag = _amb.diagnose()
                print(f"  ✅ Ambient: ready via standalone sampler (interval={config.ambient_interval}s, window={config.ambient_window}s)")
                print("       → wake word disabled; StandaloneAmbientSampler will open its own mic stream")
            except Exception as exc:
                print(f"  ❌ Ambient: init failed — {exc}")
        else:
            try:
                from bantz.agent.ambient import ambient_analyzer as _amb
                diag = _amb.diagnose()
                print(f"  ✅ Ambient: ready (interval={config.ambient_interval}s, window={config.ambient_window}s)")
            except Exception as exc:
                print(f"  ❌ Ambient: init failed — {exc}")
    else:
        print("  ⚪ Ambient: disabled  → BANTZ_AMBIENT_ENABLED=true")

    # ── Consolidated voice fix commands (#432) ───────────────────────
    if _voice_pip_missing or _voice_apt_missing:
        print()
        print("  ── Voice setup fix commands:")
        if _voice_apt_missing:
            print(f"     sudo apt install {' '.join(_voice_apt_missing)}")
        if _voice_pip_missing:
            print(f"     pip install {' '.join(_voice_pip_missing)}")

    print()  # end voice section

    # Telegram
    if config.telegram_bot_token:
        print("✅ Telegram: token set")
    else:
        print("⚪ Telegram: not configured  → bantz --setup telegram")

    # Habits
    from bantz.core.habits import habits as _hab
    print(f"✅ Habits: {_hab.status_line()}")

    # Places
    from bantz.core.places import places as _plc
    if _plc.is_configured():
        print(f"✅ Places: {_plc.status_line()}")
    else:
        print("⚪ Places: not configured  → bantz --setup places")

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
            print("❌ Job Scheduler: apscheduler not installed  → pip install apscheduler")
        except Exception as exc:
            print(f"❌ Job Scheduler: enabled but init failed: {exc}")
    else:
        print("⚪ Job Scheduler: disabled  → BANTZ_JOB_SCHEDULER_ENABLED=true")

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
        print("⚪ systemd: not installed  → bantz --setup systemd")

    print("─" * 52)


def _setup_systemd() -> None:
    """Install Bantz systemd user service with linger + verification."""
    import os
    from pathlib import Path

    print("\n🦌 Bantz — systemd Service Setup")
    print("─" * 52)

    user = os.environ.get("USER", "")
    if not user:
        print("✗ Cannot determine current user.")
        return

    # src/bantz/cli/setup.py → repo root is 4 levels up
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
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
        print("✅ bantz.service is running")
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
        print("❌ Service file: NOT FOUND — run: bantz --setup systemd")
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
        print("✅ Service: active (running)")
    elif state == "inactive":
        print("⚪ Service: inactive — run: systemctl --user start bantz.service")
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
        error_lines = [line for line in journal.stdout.strip().splitlines() if line.strip()]
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
        print("✅ Boot: enabled (starts on login)")
    else:
        print(f"⚪ Boot: {en_state} — run: systemctl --user enable bantz.service")

    print("─" * 52)


def _format_uptime(timestamp_str: str) -> str:
    """Parse systemd ActiveEnterTimestamp and return human-readable uptime."""
    from datetime import datetime
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
