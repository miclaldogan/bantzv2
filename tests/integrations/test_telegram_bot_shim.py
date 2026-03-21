"""Tests for bantz.integrations.telegram_bot shim."""
from __future__ import annotations

import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_tg_bot_module():
    """Fixture to mock bantz.interface.telegram_bot and clean up sys.modules."""
    mock_mod = MagicMock()
    # We only need to mock the interface module itself,
    # which avoids any imports of its dependencies (telegram, pydantic, etc.)
    with patch.dict(sys.modules, {"bantz.interface.telegram_bot": mock_mod}):
        yield mock_mod


def test_telegram_bot_shim_exports_run_bot(mock_tg_bot_module):
    """Verify the shim re-exports run_bot correctly."""
    from bantz.integrations.telegram_bot import run_bot
    assert run_bot is mock_tg_bot_module.run_bot


def test_telegram_bot_shim_executes_run_bot(mock_tg_bot_module):
    """Verify that the shim calls run_bot() when run as a script."""
    # run_path executes the file as if it were the main script
    runpy.run_path("src/bantz/integrations/telegram_bot.py", run_name="__main__")
    mock_tg_bot_module.run_bot.assert_called_once()
