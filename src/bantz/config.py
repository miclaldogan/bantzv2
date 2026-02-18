"""
Bantz v2 — Configuration
All settings are read from here. They can be overridden by a .env file or environment variables.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BANTZ_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"

    gemini_enabled: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Language / Translation ─────────────────────────────────────────────────────
    language: str = "tr"           # "tr" → MarianMT bridge active, "en" → direct
    translation_enabled: bool = True

    # ── Database ────────────────────────────────────────────────────────
    db_path: Path = Path.home() / ".local" / "share" / "bantz" / "store.db"

    # ── Shell Security ────────────────────────────────────────────────────
    shell_confirm_destructive: bool = True   # rm, sudo etc. require confirmation
    shell_timeout_seconds: int = 30          # command timeout

    # ── UI ────────────────────────────────────────────────────────────────
    ui_theme: str = "dark"                   # textual theme

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# Singleton — can be used from anywhere with `from bantz.config import config`
config = Config()