"""
Bantz v2 — Configuration
Reads from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_model: str = Field("qwen2.5-coder:7b", alias="BANTZ_OLLAMA_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="BANTZ_OLLAMA_BASE_URL")

    # ── Gemini (optional) ─────────────────────────────────────────────────
    gemini_enabled: bool = Field(False, alias="BANTZ_GEMINI_ENABLED")
    gemini_api_key: str = Field("", alias="BANTZ_GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", alias="BANTZ_GEMINI_MODEL")

    # ── Language / Translation ────────────────────────────────────────────
    language: str = Field("tr", alias="BANTZ_LANGUAGE")
    translation_enabled: bool = Field(True, alias="BANTZ_TRANSLATION_ENABLED")

    # ── Shell Security ────────────────────────────────────────────────────
    shell_confirm_destructive: bool = Field(True, alias="BANTZ_SHELL_CONFIRM_DESTRUCTIVE")
    shell_timeout_seconds: int = Field(30, alias="BANTZ_SHELL_TIMEOUT_SECONDS")

    # ── Location (manual override — optional) ─────────────────────────────
    # Priority: .env manual > GeoClue2 > ipinfo.io > fallback
    location_city: str = Field("", alias="BANTZ_CITY")
    location_country: str = Field("TR", alias="BANTZ_COUNTRY")
    location_timezone: str = Field("Europe/Istanbul", alias="BANTZ_TIMEZONE")
    location_region: str = Field("", alias="BANTZ_REGION")
    location_lat: float = Field(0.0, alias="BANTZ_LAT")
    location_lon: float = Field(0.0, alias="BANTZ_LON")

    # ── Storage ───────────────────────────────────────────────────────────
    data_dir: str = Field("", alias="BANTZ_DATA_DIR")

    @property
    def db_path(self) -> Path:
        base = (
            Path(self.data_dir)
            if self.data_dir
            else Path.home() / ".local" / "share" / "bantz"
        )
        new = base / "bantz.db"
        # One-time migration: store.db → bantz.db
        if not new.exists():
            old = base / "store.db"
            if old.exists():
                old.rename(new)
        return new

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = {"env_file": ".env", "extra": "ignore"}


config = Config()