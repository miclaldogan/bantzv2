"""
Bantz v2 — Tool Base & Registry
Each tool implements this interface. Router generates JSON → Registry finds the correct tool.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: str                        # text to be shown to the user
    data: dict[str, Any] = field(default_factory=dict)  # structured data (optional)
    error: str = ""


# ── Base Tool ─────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    name: str
    description: str
    risk_level: Literal["safe", "moderate", "destructive"] = "safe"

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult: ...

    def schema(self) -> dict:
        """Short description of this tool for the router."""
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

def _normalise_tool_name(name: str) -> str:
    """Normalise a tool name: lowercase, strip, spaces/hyphens → underscores.

    Small LLMs frequently return 'Web Search', 'Visual Click', 'web-search',
    etc.  This gives us a canonical form for lookup.
    """
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name — fuzzy: case-insensitive, space/hyphen tolerant."""
        # Exact match first (fast path)
        tool = self._tools.get(name)
        if tool:
            return tool
        # Normalised fuzzy lookup
        norm = _normalise_tool_name(name)
        for registered_name, t in self._tools.items():
            if _normalise_tool_name(registered_name) == norm:
                return t
        return None

    def all_schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


# Global registry — `from bantz.tools import registry`
registry = ToolRegistry()