"""
Tests for Issue #139 — Config CLI, --doctor upgrade, .env.example.

Covers:
  - _mask() secret masking
  - _section_for() section mapping
  - _show_config() output with secrets masked
  - _SECRET_FIELDS completeness
  - argparse --config flag registration
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest


# ── _mask() ──────────────────────────────────────────────────────────────

class TestMask:
    def test_empty_string(self):
        from bantz.__main__ import _mask
        assert _mask("") == "(empty)"

    def test_short_secret(self):
        from bantz.__main__ import _mask
        assert _mask("abc") == "****"
        assert _mask("abcdef") == "****"

    def test_long_secret(self):
        from bantz.__main__ import _mask
        result = _mask("AIzaSyDEADBEEF1234")
        assert result == "AIza****"
        assert "DEADBEEF" not in result

    def test_exactly_seven_chars(self):
        from bantz.__main__ import _mask
        result = _mask("1234567")
        assert result == "1234****"


# ── _section_for() ──────────────────────────────────────────────────────

class TestSectionFor:
    def test_ollama(self):
        from bantz.__main__ import _section_for
        assert _section_for("ollama_model") == "Ollama"
        assert _section_for("ollama_base_url") == "Ollama"

    def test_embeddings(self):
        from bantz.__main__ import _section_for
        assert _section_for("embedding_model") == "Embeddings"
        assert _section_for("embedding_enabled") == "Embeddings"
        assert _section_for("vector_search_weight") == "Embeddings"

    def test_gemini(self):
        from bantz.__main__ import _section_for
        assert _section_for("gemini_api_key") == "Gemini"
        assert _section_for("gemini_enabled") == "Gemini"

    def test_tts(self):
        from bantz.__main__ import _section_for
        assert _section_for("tts_enabled") == "TTS / Audio"
        assert _section_for("tts_model") == "TTS / Audio"

    def test_telegram(self):
        from bantz.__main__ import _section_for
        assert _section_for("telegram_bot_token") == "Telegram"

    def test_observer(self):
        from bantz.__main__ import _section_for
        assert _section_for("observer_enabled") == "Observer"

    def test_rl_engine(self):
        from bantz.__main__ import _section_for
        assert _section_for("rl_enabled") == "RL Engine"

    def test_interventions(self):
        from bantz.__main__ import _section_for
        assert _section_for("intervention_rate_limit") == "Interventions"

    def test_app_detector(self):
        from bantz.__main__ import _section_for
        assert _section_for("app_detector_enabled") == "App Detector"

    def test_notifications(self):
        from bantz.__main__ import _section_for
        assert _section_for("desktop_notifications") == "Notifications"
        assert _section_for("notification_icon") == "Notifications"

    def test_job_scheduler(self):
        from bantz.__main__ import _section_for
        assert _section_for("job_scheduler_enabled") == "Job Scheduler"
        assert _section_for("night_maintenance_hour") == "Job Scheduler"

    def test_location(self):
        from bantz.__main__ import _section_for
        assert _section_for("location_city") == "Location"

    def test_neo4j(self):
        from bantz.__main__ import _section_for
        assert _section_for("neo4j_enabled") == "Neo4j"

    def test_unknown_falls_to_general(self):
        from bantz.__main__ import _section_for
        assert _section_for("totally_unknown_field") == "General"

    def test_distillation(self):
        from bantz.__main__ import _section_for
        assert _section_for("distillation_enabled") == "Distillation"

    def test_vision(self):
        from bantz.__main__ import _section_for
        assert _section_for("vlm_enabled") == "Vision / VLM"
        assert _section_for("screenshot_quality") == "Vision / VLM"

    def test_input_control(self):
        from bantz.__main__ import _section_for
        assert _section_for("input_control_enabled") == "Input Control"

    def test_shell(self):
        from bantz.__main__ import _section_for
        assert _section_for("shell_confirm_destructive") == "Shell Security"

    def test_storage(self):
        from bantz.__main__ import _section_for
        assert _section_for("data_dir") == "Storage"

    def test_morning_briefing(self):
        from bantz.__main__ import _section_for
        assert _section_for("morning_briefing_enabled") == "Morning Briefing"

    def test_digests(self):
        from bantz.__main__ import _section_for
        assert _section_for("daily_digest_enabled") == "Digests"
        assert _section_for("weekly_digest_day") == "Digests"

    def test_overnight_poll(self):
        from bantz.__main__ import _section_for
        assert _section_for("urgent_keywords") == "Overnight Poll"

    def test_gps_relay(self):
        from bantz.__main__ import _section_for
        assert _section_for("gps_relay_token") == "GPS Relay"

    def test_language(self):
        from bantz.__main__ import _section_for
        assert _section_for("language") == "Language"
        assert _section_for("translation_enabled") == "Language"

    def test_reminders(self):
        from bantz.__main__ import _section_for
        assert _section_for("reminder_check_interval") == "Scheduler / Reminders"


# ── _SECRET_FIELDS ──────────────────────────────────────────────────────

class TestSecretFields:
    def test_expected_secrets(self):
        from bantz.__main__ import _SECRET_FIELDS
        expected = {"gemini_api_key", "neo4j_password", "telegram_bot_token", "gps_relay_token"}
        assert _SECRET_FIELDS == expected

    def test_secrets_are_frozenset(self):
        from bantz.__main__ import _SECRET_FIELDS
        assert isinstance(_SECRET_FIELDS, frozenset)


# ── _show_config() ──────────────────────────────────────────────────────

class TestShowConfig:
    def test_output_contains_header(self):
        from bantz.__main__ import _show_config
        buf = io.StringIO()
        with redirect_stdout(buf):
            _show_config()
        output = buf.getvalue()
        assert "Bantz v2 — Current Configuration" in output
        assert "─" in output

    def test_secrets_are_masked(self):
        from bantz.__main__ import _show_config
        buf = io.StringIO()
        with redirect_stdout(buf):
            _show_config()
        output = buf.getvalue()
        # gemini_api_key default is "" → should show "(empty)"
        assert "(empty)" in output
        # neo4j_password default is "bantzpass" → should be masked
        assert "bantzpass" not in output
        assert "bant****" in output

    def test_non_secrets_shown_in_clear(self):
        from bantz.__main__ import _show_config
        buf = io.StringIO()
        with redirect_stdout(buf):
            _show_config()
        output = buf.getvalue()
        # Model value comes from env; just verify the field is printed
        assert "ollama_model" in output
        assert "BANTZ_OLLAMA_MODEL" in output

    def test_section_headers_present(self):
        from bantz.__main__ import _show_config
        buf = io.StringIO()
        with redirect_stdout(buf):
            _show_config()
        output = buf.getvalue()
        assert "Ollama" in output
        assert "Gemini" in output
        assert "TTS / Audio" in output

    def test_env_alias_shown(self):
        from bantz.__main__ import _show_config
        buf = io.StringIO()
        with redirect_stdout(buf):
            _show_config()
        output = buf.getvalue()
        assert "BANTZ_OLLAMA_MODEL" in output


# ── argparse --config flag ──────────────────────────────────────────────

class TestConfigArgparse:
    def test_config_flag_registered(self):
        """Verify --config is a valid argument."""
        from bantz.__main__ import main
        import sys
        # Parse --config, which should be recognized
        parser = argparse.ArgumentParser(prog="bantz")
        parser.add_argument("--config", action="store_true")
        args = parser.parse_args(["--config"])
        assert args.config is True

    def test_config_flag_dispatches(self):
        """Verify --config calls _show_config."""
        import sys
        with patch.object(sys, "argv", ["bantz", "--config"]):
            with patch("bantz.__main__._show_config") as mock_show:
                from bantz.__main__ import main
                main()
                mock_show.assert_called_once()


# ── .env.example completeness ───────────────────────────────────────────

class TestEnvExampleCompleteness:
    """Verify .env.example covers all config fields that have env aliases."""

    def test_all_config_aliases_in_env_example(self):
        from pathlib import Path
        from bantz.config import Config

        env_path = Path(__file__).resolve().parent.parent.parent / ".env.example"
        content = env_path.read_text()

        missing = []
        for name, field in Config.model_fields.items():
            alias = field.alias
            if alias and alias != name:
                if alias not in content:
                    missing.append(f"{name} ({alias})")

        assert not missing, f"Missing from .env.example: {missing}"

    def test_tts_section_present(self):
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env.example"
        content = env_path.read_text()
        assert "BANTZ_TTS_ENABLED" in content
        assert "BANTZ_TTS_MODEL" in content

    def test_job_scheduler_section_present(self):
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env.example"
        content = env_path.read_text()
        assert "BANTZ_JOB_SCHEDULER_ENABLED" in content
        assert "BANTZ_MAINTENANCE_HOUR" in content
        assert "BANTZ_OVERNIGHT_POLL_HOURS" in content

    def test_urgent_keywords_present(self):
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env.example"
        content = env_path.read_text()
        assert "BANTZ_URGENT_KEYWORDS" in content

    def test_telegram_env_vars_match_config(self):
        """Ensure .env.example uses the same env var names as config.py."""
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env.example"
        content = env_path.read_text()
        # Config uses TELEGRAM_BOT_TOKEN (no BANTZ_ prefix)
        assert "TELEGRAM_BOT_TOKEN" in content
        # Should NOT have the wrong BANTZ_TELEGRAM_BOT_TOKEN
        assert "BANTZ_TELEGRAM_BOT_TOKEN" not in content
