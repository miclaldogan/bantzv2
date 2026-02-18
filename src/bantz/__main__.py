"""
Bantz v2 — Entry point
`python -m bantz` veya `bantz` komutuyla çalışır.
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bantz",
        description="Bantz v2 — your terminal host",
    )
    parser.add_argument(
        "--once",
        metavar="QUERY",
        help="Tek sorgu çalıştır, UI açma",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Sistem kontrolü",
    )
    args = parser.parse_args()

    if args.doctor:
        asyncio.run(_doctor())
        return

    if args.once:
        asyncio.run(_once(args.once))
        return

    # Varsayılan: TUI aç
    from bantz.app import run
    run()


async def _doctor() -> None:
    """Bağlantı ve sistem kontrolü."""
    from bantz.llm.ollama import ollama
    from bantz.config import config

    print("Bantz v2 — Sistem Kontrolü")
    print("─" * 40)

    ok = await ollama.is_available()
    status = "✓" if ok else "✗"
    print(f"{status} Ollama ({config.ollama_base_url}): {'bağlı' if ok else 'BAĞLANAMADI'}")
    print(f"  Model: {config.ollama_model}")

    import psutil
    print(f"✓ psutil: CPU %{psutil.cpu_percent(interval=0.3):.0f}")

    from bantz.tools import registry
    import bantz.tools.shell       # noqa: F401
    import bantz.tools.system      # noqa: F401
    import bantz.tools.filesystem  # noqa: F401
    print(f"✓ Tools: {', '.join(registry.names())}")

    config.ensure_dirs()
    print(f"✓ DB yolu: {config.db_path}")
    print("─" * 40)


async def _once(query: str) -> None:
    """Tek sorgu modu — stdout'a yaz."""
    from bantz.core.brain import brain
    result = await brain.process(query)
    print(result.response)


if __name__ == "__main__":
    main()