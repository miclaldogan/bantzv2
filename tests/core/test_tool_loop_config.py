"""Tool-loop budget config (audit S3, issue #501).

The C1 recovery loop is bounded by config, not constants, so eval
conditions (steps=1 vs 3) are pure env flips. max_steps counts tool
EXECUTIONS: 1 = single-shot = exact current behaviour.
"""
import pytest
from pydantic import ValidationError

from bantz.config import Config


class TestToolLoopConfig:
    def test_defaults_are_single_shot(self, monkeypatch):
        monkeypatch.delenv("BANTZ_TOOL_LOOP_MAX_STEPS", raising=False)
        monkeypatch.delenv("BANTZ_TOOL_LOOP_TOKEN_BUDGET", raising=False)
        cfg = Config()
        assert cfg.tool_loop_max_steps == 1
        assert cfg.tool_loop_token_budget == 4096

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("BANTZ_TOOL_LOOP_MAX_STEPS", "3")
        monkeypatch.setenv("BANTZ_TOOL_LOOP_TOKEN_BUDGET", "2048")
        cfg = Config()
        assert cfg.tool_loop_max_steps == 3
        assert cfg.tool_loop_token_budget == 2048

    def test_zero_steps_rejected(self, monkeypatch):
        # 0 would mean "never execute the tool" — must be a config error,
        # not a silently-dead pipeline.
        monkeypatch.setenv("BANTZ_TOOL_LOOP_MAX_STEPS", "0")
        with pytest.raises(ValidationError):
            Config()

    def test_negative_budget_rejected(self, monkeypatch):
        monkeypatch.setenv("BANTZ_TOOL_LOOP_TOKEN_BUDGET", "-1")
        with pytest.raises(ValidationError):
            Config()

    def test_mode_default_and_env_override(self, monkeypatch):
        monkeypatch.delenv("BANTZ_TOOL_LOOP_MODE", raising=False)
        assert Config().tool_loop_mode == "redecide"
        monkeypatch.setenv("BANTZ_TOOL_LOOP_MODE", "retry")
        assert Config().tool_loop_mode == "retry"

    def test_mode_invalid_rejected(self, monkeypatch):
        monkeypatch.setenv("BANTZ_TOOL_LOOP_MODE", "yolo")
        with pytest.raises(ValidationError):
            Config()
