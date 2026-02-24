"""
Bantz v3 — Configuration
Reads from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_model: str = Field("qwen2.5:7b", alias="BANTZ_OLLAMA_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="BANTZ_OLLAMA_BASE_URL")

    # ── Gemini (finalizer) ────────────────────────────────────────────────
    gemini_enabled: bool = Field(False, alias="BANTZ_GEMINI_ENABLED")
    gemini_api_key: str = Field("", alias="BANTZ_GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", alias="BANTZ_GEMINI_MODEL")

    # ── Neo4j (graph memory) ──────────────────────────────────────────────
    neo4j_enabled: bool = Field(False, alias="BANTZ_NEO4J_ENABLED")
    neo4j_uri: str = Field("bolt://localhost:7687", alias="BANTZ_NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="BANTZ_NEO4J_USER")
    neo4j_password: str = Field("bantzpass", alias="BANTZ_NEO4J_PASSWORD")

    # ── Shell Security ────────────────────────────────────────────────────
    shell_confirm_destructive: bool = Field(True, alias="BANTZ_SHELL_CONFIRM_DESTRUCTIVE")
    shell_timeout_seconds: int = Field(30, alias="BANTZ_SHELL_TIMEOUT_SECONDS")

    # ── Location (manual override — optional) ─────────────────────────────
    location_city: str = Field("", alias="BANTZ_CITY")
    location_country: str = Field("US", alias="BANTZ_COUNTRY")
    location_timezone: str = Field("UTC", alias="BANTZ_TIMEZONE")
    location_region: str = Field("", alias="BANTZ_REGION")
    location_lat: float = Field(0.0, alias="BANTZ_LAT")
    location_lon: float = Field(0.0, alias="BANTZ_LON")

    # ── Storage ───────────────────────────────────────────────────────────
    data_dir: str = Field("", alias="BANTZ_DATA_DIR")

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_bot_token: str = Field("", alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_users: str = Field("", alias="TELEGRAM_ALLOWED_USERS")
    telegram_proxy: str = Field("", alias="TELEGRAM_PROXY")

    @property
    def db_path(self) -> Path:
        base = (
            Path(self.data_dir)
            if self.data_dir
            else Path.home() / ".local" / "share" / "bantz"
        )
        new = base / "bantz.db"
        if not new.exists():
            old = base / "store.db"
            if old.exists():
                old.rename(new)
        return new

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = {"env_file": ".env", "extra": "ignore"}


config = Config()
