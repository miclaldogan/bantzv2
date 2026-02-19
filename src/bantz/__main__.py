"""
Bantz v2 — Entry point

Commands:
  bantz                         → TUI
  bantz --once "query"          → single query, no UI
  bantz --doctor                → system health check
  bantz --setup google gmail    → OAuth setup for Gmail
  bantz --setup google classroom → OAuth setup for Classroom
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
    if len(parts) >= 1 and parts[0].lower() == "schedule":
        _setup_schedule()
        return
    if len(parts) >= 2 and parts[0].lower() == "google":
        service = parts[1].lower()
        from bantz.auth.google_oauth import setup_google
        setup_google(service)
    else:
        print(f"Unknown setup target: {' '.join(parts)}")
        print("Available:")
        print("  bantz --setup google [gmail|classroom|calendar]")
        print("  bantz --setup schedule")


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
            print(f"Mevcut program yüklendi: {path}")
        except Exception:
            pass

    print("\n📅 Ders Programı Kurulumu")
    print("─" * 40)
    print("Dersleri gün gün gir. Bitirmek için boş bırak.")
    print("Format: HH:MM  Ders Adı  Süre(dk)  Konum")
    print()

    for day_en in DAYS_EN:
        day_tr = DAYS_TR[day_en]
        print(f"\n{day_tr}:")
        existing = data.get(day_en, [])
        if existing:
            for c in existing:
                print(f"  (mevcut) {c.get('time','')} {c.get('name','')} {c.get('location','')}")

        classes = list(existing)  # keep existing
        while True:
            try:
                raw = input(f"  Yeni ders (boş=geç): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                break
            parts = raw.split(None, 3)
            if len(parts) < 2:
                print("  En az saat ve ders adı gir.")
                continue
            time_str = parts[0]
            name = parts[1]
            duration = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 90
            location = parts[3] if len(parts) > 3 else ""

            cls: dict = {"name": name, "time": time_str, "duration": duration}
            if location:
                cls["location"] = location
            classes.append(cls)
            print(f"  ✓ Eklendi: {time_str} {name}")

        if classes:
            data[day_en] = sorted(classes, key=lambda c: c.get("time", ""))

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Ders programı kaydedildi: {path}")
    print("Test: bantz --once 'bugün derslerim'")


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

    # DB
    config.ensure_dirs()
    print(f"✓ DB: {config.db_path}")
    print("─" * 44)


async def _once(query: str) -> None:
    from bantz.core.brain import brain
    result = await brain.process(query)
    print(result.response)


if __name__ == "__main__":
    main()