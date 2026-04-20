"""
Tests for bantz.workflows.registry — YAML loading, parsing, lookups.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from bantz.workflows.registry import WorkflowRegistry
from bantz.workflows.errors import WorkflowNotFoundError, WorkflowValidationError
from bantz.workflows.models import WorkflowDef


VALID_YAML = """\
name: test-basic
description: "A simple test workflow"
version: "1.0"
steps:
  - name: greet
    action: set_variable
    variable: msg
    value: "hello world"
"""

INVALID_YAML = """\
name: bad
steps: "not a list"
"""

BROKEN_YAML = """\
name: broken
  indentation: bad
 yaml: here
"""

MULTI_STEP_YAML = """\
name: multi-step
steps:
  - name: a
    action: tool
    tool: weather
  - name: b
    action: tool
    tool: news
    depends_on:
      - a
"""


class TestParseYaml:
    def test_valid(self):
        wf = WorkflowRegistry.parse_yaml(VALID_YAML)
        assert wf.name == "test-basic"
        assert len(wf.steps) == 1
        assert wf.steps[0].action == "set_variable"

    def test_invalid_schema(self):
        with pytest.raises(WorkflowValidationError, match="Schema validation"):
            WorkflowRegistry.parse_yaml(INVALID_YAML)

    def test_broken_yaml(self):
        with pytest.raises(WorkflowValidationError, match="Invalid YAML"):
            WorkflowRegistry.parse_yaml(BROKEN_YAML)

    def test_not_a_mapping(self):
        with pytest.raises(WorkflowValidationError, match="Expected a mapping"):
            WorkflowRegistry.parse_yaml("- item1\n- item2")

    def test_multi_step(self):
        wf = WorkflowRegistry.parse_yaml(MULTI_STEP_YAML)
        assert len(wf.steps) == 2
        assert wf.steps[1].depends_on == ["a"]


class TestRegistryLoad:
    def test_load_from_directory(self, tmp_path):
        (tmp_path / "wf1.yaml").write_text(VALID_YAML)
        reg = WorkflowRegistry()
        count = reg.load(tmp_path)
        assert count == 1
        assert "test-basic" in reg.list_names()

    def test_load_ignores_invalid(self, tmp_path):
        (tmp_path / "good.yaml").write_text(VALID_YAML)
        (tmp_path / "bad.yaml").write_text(INVALID_YAML)
        reg = WorkflowRegistry()
        count = reg.load(tmp_path)
        assert count == 1

    def test_load_nonexistent_dir(self, tmp_path):
        reg = WorkflowRegistry()
        count = reg.load(tmp_path / "nope")
        assert count == 0

    def test_later_dirs_override(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "test.yaml").write_text(VALID_YAML)
        override = VALID_YAML.replace("hello world", "overridden")
        (d2 / "test.yaml").write_text(override)
        reg = WorkflowRegistry()
        reg.load(d1, d2)
        wf = reg.get("test-basic")
        assert wf.steps[0].value == "overridden"


class TestRegistryGet:
    def test_get_existing(self, tmp_path):
        (tmp_path / "wf.yaml").write_text(VALID_YAML)
        reg = WorkflowRegistry()
        reg.load(tmp_path)
        wf = reg.get("test-basic")
        assert wf.name == "test-basic"

    def test_get_missing_raises(self):
        reg = WorkflowRegistry()
        reg._loaded = True
        with pytest.raises(WorkflowNotFoundError, match="not found"):
            reg.get("nonexistent")

    def test_get_shows_available(self, tmp_path):
        (tmp_path / "wf.yaml").write_text(VALID_YAML)
        reg = WorkflowRegistry()
        reg.load(tmp_path)
        with pytest.raises(WorkflowNotFoundError, match="test-basic"):
            reg.get("doesnt-exist")


class TestRegistryListAll:
    def test_list_all(self, tmp_path):
        (tmp_path / "wf.yaml").write_text(VALID_YAML)
        reg = WorkflowRegistry()
        reg.load(tmp_path)
        items = reg.list_all()
        assert len(items) == 1
        assert items[0]["name"] == "test-basic"
        assert items[0]["steps"] == 1

    def test_list_empty(self):
        reg = WorkflowRegistry()
        reg._loaded = True
        assert reg.list_all() == []


class TestRegistryRegister:
    def test_manual_register(self):
        reg = WorkflowRegistry()
        reg._loaded = True
        wf = WorkflowDef(
            name="dynamic",
            steps=[{"name": "s", "action": "set_variable", "variable": "x", "value": "1"}],
        )
        reg.register(wf)
        assert reg.get("dynamic").name == "dynamic"
        assert reg.count == 1


class TestRegistryClear:
    def test_clear(self, tmp_path):
        (tmp_path / "wf.yaml").write_text(VALID_YAML)
        reg = WorkflowRegistry()
        reg.load(tmp_path)
        assert reg.count == 1
        reg.clear()
        assert reg.count == 0


class TestBuiltins:
    def test_builtins_directory_parseable(self):
        """All YAML files in builtins/ should parse without error."""
        builtin_dir = Path(__file__).parent.parent.parent / "src" / "bantz" / "workflows" / "builtins"
        if not builtin_dir.is_dir():
            pytest.skip("Builtins directory not found")
        reg = WorkflowRegistry()
        count = reg.load(builtin_dir)
        assert count >= 1, f"Expected at least 1 builtin workflow, got {count}"
        # All should have valid names and at least 1 step
        for wf in reg._workflows.values():
            assert wf.name
            assert len(wf.steps) >= 1
