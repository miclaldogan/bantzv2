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
    if len(parts) >= 2 and parts[0].lower() == "google":
        service = parts[1].lower()
        from bantz.auth.google_oauth import setup_google
        setup_google(service)
    else:
        print(f"Unknown setup target: {' '.join(parts)}")
        print("Available: bantz --setup google [gmail|classroom|calendar]")


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