"""Tool argument schema: validation, deterministic repair, and the guarantee
that the whole feature is inert while its flags are off.

The OFF-path tests are the important ones. Everything here ships default-off
so an experiment can sweep it as a condition, and a regression that makes the
default path behave differently would silently invalidate every baseline
number measured against it.
"""
from __future__ import annotations

import pytest

from bantz.config import config
from bantz.tools import arg_schema
from bantz.tools.arg_schema import repair, validate


CAL = {
    "action": {"type": "string", "required": True,
               "enum": ["today", "create", "delete"]},
    "title": {"type": "string"},
    "date": {"type": "string", "format": "date"},
    "time": {"type": "string", "format": "time"},
    "duration": {"type": "integer"},
}


# ── validation ───────────────────────────────────────────────────────────────

class TestValidate:
    def test_undeclared_tool_is_never_policed(self):
        assert validate({}, {"anything": "goes", "n": 3}) == []

    def test_clean_args_pass(self):
        assert validate(CAL, {"action": "create", "title": "X",
                              "date": "2026-07-06", "time": "16:00"}) == []

    def test_missing_required(self):
        errs = validate(CAL, {"title": "X"})
        assert any("action" in e and "required" in e for e in errs)

    def test_empty_string_counts_as_missing_for_required(self):
        errs = validate(CAL, {"action": ""})
        assert any("required" in e for e in errs)

    def test_enum_violation_names_the_alternatives(self):
        errs = validate(CAL, {"action": "upcoming"})
        assert len(errs) == 1
        assert "upcoming" in errs[0] and "today" in errs[0]

    def test_wrong_type(self):
        errs = validate(CAL, {"action": "today", "duration": "long"})
        assert any("duration" in e and "integer" in e for e in errs)

    def test_bool_is_not_an_integer(self):
        errs = validate(CAL, {"action": "today", "duration": True})
        assert any("duration" in e for e in errs)

    def test_bad_date_and_time_formats(self):
        errs = validate(CAL, {"action": "create", "date": "July 6th",
                              "time": "8pm"})
        assert any("date" in e for e in errs)
        assert any("time" in e for e in errs)

    def test_unknown_key_reported(self):
        errs = validate(CAL, {"action": "today", "bogus": 1})
        assert any("unknown" in e and "bogus" in e for e in errs)

    def test_absent_optional_is_fine(self):
        assert validate(CAL, {"action": "today"}) == []


# ── deterministic repair ─────────────────────────────────────────────────────

class TestRepair:
    def test_undeclared_tool_is_a_noop(self):
        args = {"whatever": 1}
        out, notes = repair({}, args)
        assert out == args and notes == []

    def test_does_not_mutate_the_input(self):
        args = {"action": "today", "bogus": 1}
        repair(CAL, args)
        assert args == {"action": "today", "bogus": 1}

    @pytest.mark.parametrize("raw,want", [
        ("July 6th", "2026-07-06"),
        ("6th of July", "2026-07-06"),
        ("July 6, 2026", "2026-07-06"),
    ])
    def test_month_name_dates_normalise(self, raw, want):
        out, _ = repair(CAL, {"action": "create", "date": raw})
        assert out["date"] == want

    def test_twelve_hour_time_normalises(self):
        out, _ = repair(CAL, {"action": "create", "time": "8pm"})
        assert out["time"] == "20:00"

    def test_numeric_string_coerces(self):
        out, _ = repair(CAL, {"action": "today", "duration": "90"})
        assert out["duration"] == 90

    def test_unknown_key_dropped(self):
        out, notes = repair(CAL, {"action": "today", "bogus": 1})
        assert "bogus" not in out
        assert any("bogus" in n for n in notes)

    def test_default_fills_only_when_absent(self):
        params = {"limit": {"type": "integer", "default": 5}}
        assert repair(params, {})[0]["limit"] == 5
        assert repair(params, {"limit": 99})[0]["limit"] == 99

    def test_repair_then_validate_clears_format_errors(self):
        out, _ = repair(CAL, {"action": "create", "date": "July 6th",
                              "time": "8pm", "duration": "30"})
        assert validate(CAL, out) == []

    # The refusals matter as much as the repairs: guessing here would make the
    # agent look better than it is and hide a real decision error.
    def test_enum_mismatch_is_never_guessed(self):
        out, notes = repair(CAL, {"action": "upcoming"})
        assert out["action"] == "upcoming"
        assert not any("action" in n for n in notes)
        assert validate(CAL, out)

    def test_identifier_is_never_invented(self):
        params = {"message_id": {"type": "string"}}
        out, notes = repair(params, {"message_id": "the final exam email"})
        assert out["message_id"] == "the final exam email"
        assert notes == []

    def test_impossible_date_is_refused(self):
        out, _ = repair(CAL, {"action": "create", "date": "February 31st"})
        assert out["date"] == "February 31st"

    def test_semantic_values_untouched(self):
        params = {"path": {"type": "string"}, "city": {"type": "string"}}
        args = {"path": "notes", "city": "Elazig, Turkey"}
        out, notes = repair(params, args)
        assert out == args and notes == []


# ── the OFF guarantee ────────────────────────────────────────────────────────

class TestDefaultsAreInert:
    def test_flags_default_off(self):
        assert config.arg_validation is False
        assert config.arg_repair_mode == "none"
        assert config.arg_schema_in_prompt is False

    def test_schema_omits_parameters_unless_asked(self, monkeypatch):
        import bantz.tools.calendar  # noqa: F401  — registers the tool
        from bantz.tools import registry
        tool = registry.get("calendar")
        assert tool.parameters, "calendar should declare a schema"

        monkeypatch.setattr(config, "arg_schema_in_prompt", False)
        assert "parameters" not in tool.schema()
        monkeypatch.setattr(config, "arg_schema_in_prompt", True)
        assert tool.schema()["parameters"] == tool.parameters

    def test_check_tool_args_is_silent_for_undeclared_tools(self):
        from bantz.core.brain import Brain
        assert Brain._check_tool_args("no_such_tool", {"x": 1}) == []

    def test_check_tool_args_never_raises(self, monkeypatch):
        """A validator crash must degrade to 'valid', never break the call."""
        def boom(*a, **k):
            raise RuntimeError("validator exploded")
        monkeypatch.setattr(arg_schema, "validate", boom)
        monkeypatch.setattr(config, "arg_validation", True)
        from bantz.core.brain import Brain
        import bantz.tools.calendar  # noqa: F401
        assert Brain._check_tool_args("calendar", {"action": "today"}) == []


# ── schema/signature drift ───────────────────────────────────────────────────

class TestNoSignatureDrift:
    """A declared parameter that execute() cannot accept would be rejected at
    call time as an unexpected keyword — caught here instead of in a batch."""

    @pytest.mark.parametrize("module,tool_name", [
        ("bantz.tools.gmail", "gmail"),
        ("bantz.tools.calendar", "calendar"),
        ("bantz.tools.filesystem", "filesystem"),
        ("bantz.tools.shell", "shell"),
        ("bantz.tools.weather", "weather"),
        ("bantz.tools.reminder", "reminder"),
        ("bantz.tools.web_search", "web_search"),
    ])
    def test_declared_params_are_accepted_by_execute(self, module, tool_name):
        import importlib
        import inspect
        importlib.import_module(module)
        from bantz.tools import registry

        tool = registry.get(tool_name)
        assert tool is not None and tool.parameters

        sig = inspect.signature(tool.execute)
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in sig.parameters.values())
        if takes_kwargs:
            return  # **kwargs absorbs anything; nothing to drift against
        unknown = set(tool.parameters) - set(sig.parameters)
        assert not unknown, (
            f"{tool_name}.parameters declares {sorted(unknown)} which "
            f"execute() does not accept")
