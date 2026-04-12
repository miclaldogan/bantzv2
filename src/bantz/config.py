"""
Bantz v2 — Configuration
Reads from environment variables / .env file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

log = logging.getLogger("bantz.config")


class Config(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_model: str = Field("llama3.1:8b", alias="BANTZ_OLLAMA_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="BANTZ_OLLAMA_BASE_URL")
    # Optional fast/quant model for intent routing only.
    # When set, cot_route uses this smaller model for speed while the main
    # model handles conversation.  Examples: qwen2.5:3b, gemma3:4b-it-qat,
    # llama3.2:3b, phi4-mini.
    # Leave empty (default) to use the main ollama_model for everything.
    ollama_routing_model: str = Field("", alias="BANTZ_OLLAMA_ROUTING_MODEL")

    # ── Vector Search ─────────────────────────────────────────────────────
    vector_search_weight: float = Field(0.5, alias="BANTZ_VECTOR_SEARCH_WEIGHT")

    # ── Session Distillation ──────────────────────────────────────────────
    distillation_enabled: bool = Field(True, alias="BANTZ_DISTILLATION_ENABLED")

    # ── Vision / Remote VLM ───────────────────────────────────────────────
    vlm_enabled: bool = Field(False, alias="BANTZ_VLM_ENABLED")
    vlm_endpoint: str = Field("http://localhost:8090", alias="BANTZ_VLM_ENDPOINT")
    vlm_timeout: int = Field(5, alias="BANTZ_VLM_TIMEOUT")
    screenshot_quality: int = Field(70, alias="BANTZ_SCREENSHOT_QUALITY")

    # ── Input Control (#122) ──────────────────────────────────────────────
    input_control_enabled: bool = Field(False, alias="BANTZ_INPUT_CONTROL_ENABLED")
    input_confirm_destructive: bool = Field(True, alias="BANTZ_INPUT_CONFIRM_DESTRUCTIVE")
    input_type_interval_ms: int = Field(50, alias="BANTZ_INPUT_TYPE_INTERVAL_MS")
    desktop_automation_enabled: bool = Field(False, alias="BANTZ_DESKTOP_AUTOMATION_ENABLED")

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

    # ── GPS Relay ─────────────────────────────────────────────────────────
    gps_relay_token: str = Field("", alias="BANTZ_GPS_RELAY_TOKEN")

    # ── MemPalace Memory ──────────────────────────────────────────────
    mempalace_enabled: bool = Field(True, alias="BANTZ_MEMPALACE_ENABLED")
    palace_path: str = Field("", alias="BANTZ_PALACE_PATH")
    mempalace_kg_path: str = Field("", alias="BANTZ_MEMPALACE_KG_PATH")
    mempalace_wing: str = Field("bantz", alias="BANTZ_MEMPALACE_WING")
    mempalace_identity_path: str = Field("", alias="BANTZ_MEMPALACE_IDENTITY_PATH")

    # ── Storage ───────────────────────────────────────────────────────────
    data_dir: str = Field("", alias="BANTZ_DATA_DIR")

    # ── Morning Briefing ──────────────────────────────────────────────────
    morning_briefing_enabled: bool = Field(True, alias="BANTZ_MORNING_BRIEFING")
    morning_briefing_hour: int = Field(8, alias="BANTZ_MORNING_HOUR")
    morning_briefing_minute: int = Field(0, alias="BANTZ_MORNING_MINUTE")

    # ── Digests ───────────────────────────────────────────────────────────
    daily_digest_enabled: bool = Field(True, alias="BANTZ_DAILY_DIGEST_ENABLED")
    daily_digest_hour: int = Field(20, alias="BANTZ_DAILY_DIGEST_HOUR")
    daily_digest_minute: int = Field(0, alias="BANTZ_DAILY_DIGEST_MINUTE")
    weekly_digest_enabled: bool = Field(True, alias="BANTZ_WEEKLY_DIGEST_ENABLED")
    weekly_digest_day: str = Field("sunday", alias="BANTZ_WEEKLY_DIGEST_DAY")
    weekly_digest_hour: int = Field(20, alias="BANTZ_WEEKLY_DIGEST_HOUR")
    weekly_digest_minute: int = Field(0, alias="BANTZ_WEEKLY_DIGEST_MINUTE")

    # ── Scheduler / Reminders ─────────────────────────────────────────────
    reminder_check_interval: int = Field(30, alias="BANTZ_REMINDER_CHECK_INTERVAL")

    # ── Job Scheduler / APScheduler (#128) ────────────────────────────────
    job_scheduler_enabled: bool = Field(True, alias="BANTZ_JOB_SCHEDULER_ENABLED")
    night_maintenance_hour: int = Field(3, alias="BANTZ_MAINTENANCE_HOUR")
    night_reflection_hour: int = Field(23, alias="BANTZ_REFLECTION_HOUR")
    briefing_prep_hour: int = Field(6, alias="BANTZ_BRIEFING_PREP_HOUR")
    overnight_poll_hours: str = Field("0,2,4,6", alias="BANTZ_OVERNIGHT_POLL_HOURS")

    # ── Overnight Poll / Urgent Keywords (#132) ───────────────────────────
    # Comma-separated list of keywords that mark an email as urgent.
    # Matched case-insensitively against sender + subject.
    # Example: "final,deadline,acil,erasmus,gargantua,jetson"
    urgent_keywords: str = Field(
        "urgent,acil,deadline,final,emergency,important",
        alias="BANTZ_URGENT_KEYWORDS",
    )

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_bot_token: str = Field("", alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_users: str = Field("", alias="TELEGRAM_ALLOWED_USERS")
    telegram_proxy: str = Field("", alias="TELEGRAM_PROXY")
    telegram_llm_mode: bool = Field(True, alias="TELEGRAM_LLM_MODE")
    telegram_screenshot_enabled: bool = Field(True, alias="TELEGRAM_SCREENSHOT_ENABLED")
    telegram_screenshot_quality: int = Field(80, alias="TELEGRAM_SCREENSHOT_QUALITY")
    telegram_screenshot_max_dimension: int = Field(1920, alias="TELEGRAM_SCREENSHOT_MAX_DIM")
    screenshot_auto_after_action: bool = Field(True, alias="SCREENSHOT_AUTO_AFTER_ACTION")

    # ── Background Observer (#124) ────────────────────────────────────────
    observer_enabled: bool = Field(False, alias="BANTZ_OBSERVER_ENABLED")
    observer_severity_threshold: str = Field("warning", alias="BANTZ_OBSERVER_SEVERITY_THRESHOLD")
    observer_batch_seconds: float = Field(5.0, alias="BANTZ_OBSERVER_BATCH_SECONDS")
    observer_dedup_window: float = Field(60.0, alias="BANTZ_OBSERVER_DEDUP_WINDOW")
    observer_analysis_model: str = Field("qwen2.5:0.5b", alias="BANTZ_OBSERVER_ANALYSIS_MODEL")
    observer_enable_llm: bool = Field(True, alias="BANTZ_OBSERVER_ENABLE_LLM")

    # ── RL Engine (#125) ───────────────────────────────────────────────────
    rl_enabled: bool = Field(False, alias="BANTZ_RL_ENABLED")
    rl_learning_rate: float = Field(0.3, alias="BANTZ_RL_LEARNING_RATE")
    rl_discount_factor: float = Field(0.9, alias="BANTZ_RL_DISCOUNT_FACTOR")
    rl_exploration_rate: float = Field(0.15, alias="BANTZ_RL_EXPLORATION_RATE")
    rl_exploration_min: float = Field(0.02, alias="BANTZ_RL_EXPLORATION_MIN")
    rl_confidence_threshold: float = Field(0.7, alias="BANTZ_RL_CONFIDENCE_THRESHOLD")
    rl_suggestion_interval: int = Field(1800, alias="BANTZ_RL_SUGGESTION_INTERVAL")

    # ── Bonding Meter (#172) ────────────────────────────────────────────────
    bonding_enabled: bool = Field(True, alias="BANTZ_BONDING_ENABLED")
    bonding_sigmoid_rate: float = Field(0.04, alias="BANTZ_BONDING_SIGMOID_RATE")
    bonding_sigmoid_midpoint: float = Field(25.0, alias="BANTZ_BONDING_SIGMOID_MIDPOINT")

    # ── Proactive Interventions (#126) ────────────────────────────────────
    intervention_rate_limit: int = Field(3, alias="BANTZ_INTERVENTION_RATE_LIMIT")
    intervention_toast_ttl: float = Field(20.0, alias="BANTZ_INTERVENTION_TOAST_TTL")
    intervention_quiet_mode: bool = Field(False, alias="BANTZ_INTERVENTION_QUIET_MODE")
    intervention_focus_mode: bool = Field(False, alias="BANTZ_INTERVENTION_FOCUS_MODE")

    # ── App Detector (#127) ────────────────────────────────────────────────
    app_detector_enabled: bool = Field(False, alias="BANTZ_APP_DETECTOR_ENABLED")
    app_detector_cache_ttl: float = Field(5.0, alias="BANTZ_APP_DETECTOR_CACHE_TTL")
    app_detector_polling_interval: int = Field(5, alias="BANTZ_APP_DETECTOR_POLLING_INTERVAL")
    app_detector_auto_focus: bool = Field(True, alias="BANTZ_APP_DETECTOR_AUTO_FOCUS")

    # ── Desktop Notifications (#153) ─────────────────────────────────────
    desktop_notifications: bool = Field(True, alias="BANTZ_DESKTOP_NOTIFICATIONS")
    notification_icon: str = Field("", alias="BANTZ_NOTIFICATION_ICON")
    notification_sound: bool = Field(False, alias="BANTZ_NOTIFICATION_SOUND")

    # ── Voice Master Switch (#277) ────────────────────────────────────────
    # Setting BANTZ_VOICE_ENABLED=true cascades into tts, wake word, stt,
    # ghost loop, audio ducking, and ambient.  Individual flags still
    # override when set explicitly.
    voice_enabled: bool = Field(False, alias="BANTZ_VOICE_ENABLED")

    # ── TTS / Audio Briefing (#131) ──────────────────────────────────────
    tts_enabled: bool = Field(False, alias="BANTZ_TTS_ENABLED")
    tts_model: str = Field("en_US-danny-low", alias="BANTZ_TTS_MODEL")
    tts_model_path: str = Field("", alias="BANTZ_TTS_MODEL_PATH")
    tts_speaker: int = Field(0, alias="BANTZ_TTS_SPEAKER")
    tts_rate: float = Field(1.0, alias="BANTZ_TTS_RATE")
    tts_auto_briefing: bool = Field(True, alias="BANTZ_TTS_AUTO_BRIEFING")
    tts_speak_all_responses: bool = Field(False, alias="BANTZ_TTS_SPEAK_ALL_RESPONSES")
    tts_animatronic_filter: bool = Field(False, alias="BANTZ_TTS_ANIMATRONIC_FILTER")
    tts_gain: float = Field(12.0, alias="BANTZ_TTS_GAIN")  # dB boost via sox gain (both paths)

    # ── Audio Ducking (#171) ──────────────────────────────────────────────
    audio_duck_enabled: bool = Field(False, alias="BANTZ_AUDIO_DUCK_ENABLED")
    audio_duck_pct: int = Field(30, alias="BANTZ_AUDIO_DUCK_PCT")

    # ── Wake Word Detection (#165) ───────────────────────────────────────
    wake_word_enabled: bool = Field(False, alias="BANTZ_WAKE_WORD_ENABLED")
    picovoice_access_key: str = Field("", alias="BANTZ_PICOVOICE_ACCESS_KEY")
    wake_word_sensitivity: float = Field(0.5, alias="BANTZ_WAKE_WORD_SENSITIVITY")

    # ── Ghost Loop / STT (#36) ───────────────────────────────────────────
    stt_enabled: bool = Field(False, alias="BANTZ_STT_ENABLED")
    stt_model: str = Field("tiny", alias="BANTZ_STT_MODEL")
    stt_language: str = Field("en", alias="BANTZ_STT_LANGUAGE")
    stt_device: str = Field("cpu", alias="BANTZ_STT_DEVICE")
    vad_silence_ms: int = Field(800, alias="BANTZ_VAD_SILENCE_MS")
    vad_aggressiveness: int = Field(2, alias="BANTZ_VAD_AGGRESSIVENESS")
    ghost_loop_enabled: bool = Field(False, alias="BANTZ_GHOST_LOOP_ENABLED")

    # ── Ambient Sound Analysis (#166) ────────────────────────────────────
    ambient_enabled: bool = Field(False, alias="BANTZ_AMBIENT_ENABLED")
    ambient_interval: int = Field(600, alias="BANTZ_AMBIENT_INTERVAL")
    ambient_window: float = Field(3.0, alias="BANTZ_AMBIENT_WINDOW")

    # ── Proactive Engagement (#167) ──────────────────────────────────────
    proactive_enabled: bool = Field(False, alias="BANTZ_PROACTIVE_ENABLED")
    proactive_interval_hours: float = Field(3.0, alias="BANTZ_PROACTIVE_INTERVAL_HOURS")
    proactive_jitter_minutes: int = Field(30, alias="BANTZ_PROACTIVE_JITTER_MINUTES")
    proactive_max_daily: int = Field(1, alias="BANTZ_PROACTIVE_MAX_DAILY")
    proactive_away_timeout: int = Field(1800, alias="BANTZ_PROACTIVE_AWAY_TIMEOUT")

    # ── Health & Break Interventions (#168) ──────────────────────────────
    health_enabled: bool = Field(False, alias="BANTZ_HEALTH_ENABLED")
    health_check_interval: int = Field(300, alias="BANTZ_HEALTH_CHECK_INTERVAL")
    health_late_hour: int = Field(2, alias="BANTZ_HEALTH_LATE_HOUR")
    health_session_max_hours: float = Field(4.0, alias="BANTZ_HEALTH_SESSION_MAX_HOURS")
    health_thermal_cpu: float = Field(85.0, alias="BANTZ_HEALTH_THERMAL_CPU")
    health_thermal_gpu: float = Field(80.0, alias="BANTZ_HEALTH_THERMAL_GPU")
    health_eye_strain_hours: float = Field(2.0, alias="BANTZ_HEALTH_EYE_STRAIN_HOURS")

    # ── Dynamic Persona (#169) ───────────────────────────────────────────
    persona_enabled: bool = Field(True, alias="BANTZ_PERSONA_ENABLED")

    # ── Deep Memory / Spontaneous Recall (#170) ──────────────────────────
    deep_memory_enabled: bool = Field(True, alias="BANTZ_DEEP_MEMORY_ENABLED")
    deep_memory_threshold: float = Field(0.72, alias="BANTZ_DEEP_MEMORY_THRESHOLD")
    deep_memory_max_results: int = Field(3, alias="BANTZ_DEEP_MEMORY_MAX_RESULTS")

    # ── Continuous Awareness (#325) ───────────────────────────────────────
    awareness_enabled: bool = Field(False, alias="BANTZ_AWARENESS_ENABLED")
    awareness_interval_s: float = Field(15.0, alias="BANTZ_AWARENESS_INTERVAL_S")

    # ── Validators ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _voice_master_switch(self) -> Self:
        """Cascade BANTZ_VOICE_ENABLED into sub-flags.

        When the master switch is *on*, every voice sub-system that was
        left at its default (False) gets promoted to True.  If a user
        explicitly set a sub-flag to False via .env the raw env value
        will still be False — but since Pydantic parses *before* the
        validator runs and all defaults are already False, we treat
        "still False" as "not explicitly configured" and flip it.

        This means: BANTZ_VOICE_ENABLED=true alone is enough to turn
        on TTS + Wake Word + STT + Ghost Loop + Audio Ducking.
        """
        if not self.voice_enabled:
            return self

        _VOICE_SUBS = [
            "tts_enabled",
            "wake_word_enabled",
            "stt_enabled",
            "ghost_loop_enabled",
            "audio_duck_enabled",
            "ambient_enabled",
        ]
        flipped: list[str] = []
        for attr in _VOICE_SUBS:
            if not getattr(self, attr):
                object.__setattr__(self, attr, True)
                flipped.append(attr)

        if flipped:
            log.debug(
                "Voice master switch ON → enabled: %s",
                ", ".join(flipped),
            )
        return self

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

    @property
    def resolved_palace_path(self) -> str:
        """Resolved path for MemPalace ChromaDB storage."""
        if self.palace_path:
            return str(Path(self.palace_path).expanduser())
        return str(Path.home() / ".mempalace" / "palace")

    @property
    def resolved_kg_path(self) -> str:
        """Resolved path for MemPalace knowledge graph SQLite DB."""
        if self.mempalace_kg_path:
            return str(Path(self.mempalace_kg_path).expanduser())
        return str(Path.home() / ".mempalace" / "knowledge_graph.sqlite3")

    @property
    def resolved_identity_path(self) -> str:
        """Resolved path for MemPalace identity file."""
        if self.mempalace_identity_path:
            return str(Path(self.mempalace_identity_path).expanduser())
        return str(Path.home() / ".mempalace" / "identity.txt")

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = {"env_file": ".env", "extra": "ignore"}


config = Config()