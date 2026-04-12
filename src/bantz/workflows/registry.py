"""
Workflow Registry — discovers and caches WorkflowDef from YAML files.

Scans:
  1. ``<data_dir>/workflows/`` (user-defined)
  2. ``src/bantz/workflows/builtins/`` (shipped with Bantz)

If two files share the same ``name:`` field, the user-defined version wins.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from bantz.workflows.errors import (
    WorkflowNotFoundError,
    WorkflowValidationError,
)
from bantz.workflows.models import WorkflowDef

log = logging.getLogger(__name__)

_BUILTIN_DIR = Path(__file__).parent / "builtins"


class WorkflowRegistry:
    """In-memory cache of parsed ``WorkflowDef`` objects."""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowDef] = {}
        self._loaded = False

    # ── public API ────────────────────────────────────────────────────────

    def load(self, *dirs: Path) -> int:
        """Scan directories for ``*.yaml`` / ``*.yml`` and parse them.

        Returns the number of successfully loaded workflows.
        Later directories take priority (overwrite earlier names).
        """
        count = 0
        for d in dirs:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.y*ml")):
                try:
                    wf = self.parse_file(p)
                    self._workflows[wf.name] = wf
                    count += 1
                    log.info("Loaded workflow '%s' from %s", wf.name, p)
                except (WorkflowValidationError, Exception) as exc:
                    log.warning("Skipping %s: %s", p.name, exc)
        self._loaded = True
        return count

    def load_defaults(self) -> int:
        """Load builtins + user dir (from config). Idempotent."""
        if self._loaded:
            return len(self._workflows)
        dirs: list[Path] = [_BUILTIN_DIR]
        try:
            from bantz.config import config
            user_dir = Path(config.workflows_dir) if config.workflows_dir else None
            if user_dir is None:
                user_dir = config.db_path.parent / "workflows"
            dirs.append(user_dir)
        except Exception:
            pass
        return self.load(*dirs)

    def get(self, name: str) -> WorkflowDef:
        """Look up a workflow by name. Raises ``WorkflowNotFoundError``."""
        self.load_defaults()
        wf = self._workflows.get(name)
        if wf is None:
            available = ", ".join(sorted(self._workflows)) or "(none)"
            raise WorkflowNotFoundError(
                f"Workflow '{name}' not found. Available: {available}"
            )
        return wf

    def list_names(self) -> list[str]:
        """Return sorted list of available workflow names."""
        self.load_defaults()
        return sorted(self._workflows.keys())

    def list_all(self) -> list[dict[str, Any]]:
        """Return summary dicts of all workflows (name, description, inputs)."""
        self.load_defaults()
        return [
            {
                "name": wf.name,
                "description": wf.description,
                "inputs": {k: v.model_dump() for k, v in wf.inputs.items()},
                "steps": len(wf.steps),
                "version": wf.version,
            }
            for wf in self._workflows.values()
        ]

    def register(self, wf: WorkflowDef) -> None:
        """Manually register an in-memory workflow (e.g. agent-generated)."""
        self._workflows[wf.name] = wf

    @property
    def count(self) -> int:
        return len(self._workflows)

    # ── static helpers ────────────────────────────────────────────────────

    @staticmethod
    def parse_file(path: Path) -> WorkflowDef:
        """Parse a single YAML file into a ``WorkflowDef``."""
        text = path.read_text(encoding="utf-8")
        return WorkflowRegistry.parse_yaml(text, source=str(path))

    @staticmethod
    def parse_yaml(text: str, source: str = "<string>") -> WorkflowDef:
        """Parse raw YAML text into a ``WorkflowDef``."""
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise WorkflowValidationError(f"Invalid YAML in {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise WorkflowValidationError(
                f"Expected a mapping at top level in {source}, got {type(raw).__name__}"
            )
        try:
            return WorkflowDef(**raw)
        except ValidationError as exc:
            raise WorkflowValidationError(
                f"Schema validation failed for {source}: {exc}"
            ) from exc

    def clear(self) -> None:
        """Remove all loaded workflows (for testing)."""
        self._workflows.clear()
        self._loaded = False


# Global singleton
registry = WorkflowRegistry()
