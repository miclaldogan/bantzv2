"""
Bantz v3 — Gemini Flash 2.0 Client

Used as the finalizer for long-context reasoning: email threads, PDF summaries,
calendar briefings, and any tool output > 800 chars where deeper synthesis helps.

Requires:
  BANTZ_GEMINI_ENABLED=true
  BANTZ_GEMINI_API_KEY=<your key>

Falls back silently to Ollama when not configured.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GeminiClient:
    """Thin wrapper around google-generativeai SDK."""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import google.generativeai as genai
            from bantz.config import config

            if not config.gemini_enabled or not config.gemini_api_key:
                return None

            genai.configure(api_key=config.gemini_api_key)
            self._model = genai.GenerativeModel(config.gemini_model)
            return self._model
        except ImportError:
            logger.warning("google-generativeai not installed — Gemini disabled.")
            return None
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)
            return None

    @property
    def is_available(self) -> bool:
        try:
            from bantz.config import config
            return bool(config.gemini_enabled and config.gemini_api_key)
        except Exception:
            return False

    async def chat(self, system: str, user: str) -> str:
        """
        Single-turn chat with system + user messages.
        Returns the text response or raises on failure.
        """
        model = self._get_model()
        if model is None:
            raise RuntimeError("Gemini not configured")

        import asyncio

        # google-generativeai is synchronous — run in executor
        loop = asyncio.get_event_loop()
        prompt = f"{system}\n\n{user}" if system else user

        def _call() -> str:
            response = model.generate_content(prompt)
            return response.text

        return await loop.run_in_executor(None, _call)


gemini = GeminiClient()
