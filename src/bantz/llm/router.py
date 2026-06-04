"""
Bantz LLM router — returns the active provider based on BANTZ_LLM_PROVIDER.

Provider       Config key              Required env var
─────────────────────────────────────────────────────
ollama         BANTZ_LLM_PROVIDER=ollama   BANTZ_OLLAMA_BASE_URL (default localhost)
claude         BANTZ_LLM_PROVIDER=claude   BANTZ_ANTHROPIC_API_KEY
openai         BANTZ_LLM_PROVIDER=openai   BANTZ_OPENAI_API_KEY
gemini         BANTZ_LLM_PROVIDER=gemini   BANTZ_GEMINI_API_KEY

Backward compat: if BANTZ_LLM_PROVIDER=ollama but BANTZ_GEMINI_ENABLED=true
and a key is set, Gemini is used (preserves pre-router behaviour).
"""
from __future__ import annotations

import logging

from bantz.config import config

log = logging.getLogger("bantz.llm.router")


def get_provider():
    """Return the active LLM client for conversation.

    All providers expose the same interface:
        await provider.chat(messages) -> str
        provider.chat_stream(messages) -> AsyncIterator[str]
        await provider.is_available() -> bool
    """
    provider = (config.llm_provider or "ollama").lower().strip()

    if provider == "claude":
        from bantz.llm.anthropic_client import claude
        if not claude.is_enabled():
            raise RuntimeError(
                "BANTZ_LLM_PROVIDER=claude but BANTZ_ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file."
            )
        return claude

    if provider == "openai":
        from bantz.llm.openai_client import openai_client
        if not openai_client.is_enabled():
            raise RuntimeError(
                "BANTZ_LLM_PROVIDER=openai but BANTZ_OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )
        return openai_client

    if provider == "gemini":
        from bantz.llm.gemini import gemini
        if not gemini.is_enabled():
            raise RuntimeError(
                "BANTZ_LLM_PROVIDER=gemini but BANTZ_GEMINI_API_KEY is not set "
                "or BANTZ_GEMINI_ENABLED=false. Add a key to your .env file."
            )
        return gemini

    # ollama (default) — backward compat: prefer Gemini if explicitly enabled
    if config.gemini_enabled and config.gemini_api_key:
        from bantz.llm.gemini import gemini
        log.debug("router: using Gemini (backward compat, BANTZ_GEMINI_ENABLED=true)")
        return gemini

    from bantz.llm.ollama import ollama
    return ollama


# Alias used by finalizer and other call-sites
get_llm = get_provider
