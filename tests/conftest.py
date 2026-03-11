"""
Bantz — Shared test fixtures.

Provides common helpers for all test suites:
  - In-memory SQLite DAL
  - Isolated temp directories
  - Common mock helpers
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is importable regardless of how pytest is invoked
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Return a temporary SQLite database path."""
    return str(tmp_path / "test_bantz.db")


@pytest.fixture
def mock_config():
    """Return a MagicMock pretending to be bantz.config.config."""
    cfg = MagicMock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_model = "test-model"
    cfg.rl_enabled = True
    cfg.intervention_toast_ttl = 20.0
    cfg.intervention_rate_limit = 3
    cfg.observer_enabled = False
    cfg.reminder_check_interval = 30
    cfg.intervention_quiet_mode = False
    cfg.intervention_focus_mode = False
    cfg.rl_suggestion_interval = 300
    cfg.app_detector_enabled = False
    cfg.app_detector_auto_focus = False
    return cfg
