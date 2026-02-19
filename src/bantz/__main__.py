"""
Bantz v2 — Entry point
`python -m bantz` or `bantz` command.
"""
from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(prog="bantz", description="Bantz v2 — your terminal host")
    parser.add_argument("--once", metavar="QUERY", help="Run single query, no UI")
    parser.add_argument("--doctor", action="store_true", help="System health check")
    args = parser.parse_args()

    if args.doctor:
        asyncio.run(_doctor())
        return
    if args.once:
        asyncio.run(_once(args.once))
        return

    from bantz.app import run
    run()


async def _doctor() -> None:
    from bantz.llm.ollama import ollama
    from bantz.config import config
    from bantz.tools import registry
    import bantz.tools.shell        # noqa: F401
    import bantz.tools.system       # noqa: F401
    import bantz.tools.filesystem   # noqa: F401

    print("Bantz v2 — System Check")
    print("─" * 42)

    # Ollama
    ok = await ollama.is_available()
    print(f"{'✓' if ok else '✗'} Ollama ({config.ollama_base_url}): {'connected' if ok else 'UNREACHABLE'}")
    print(f"  model: {config.ollama_model}")

    # psutil
    import psutil
    print(f"✓ psutil: CPU {psutil.cpu_percent(interval=0.3):.0f}%")

    # Tools
    print(f"✓ Tools: {', '.join(registry.names())}")

    # Bridge
    print(f"  translation_enabled: {config.translation_enabled}")
    if config.translation_enabled and config.language == "tr":
        try:
            from transformers import pipeline  # noqa: F401
            print("✓ MarianMT (transformers): available")
        except ImportError:
            print("✗ MarianMT: NOT installed")
            print("  → pip install 'bantz[translation]'")
    else:
        print("  Bridge: disabled (set BANTZ_TRANSLATION_ENABLED=true to enable)")

    # DB
    config.ensure_dirs()
    print(f"✓ DB: {config.db_path}")
    print("─" * 42)


async def _once(query: str) -> None:
    from bantz.core.brain import brain
    result = await brain.process(query)
    print(result.response)


if __name__ == "__main__":
    main()