"""
Bantz v2 — Tool Base & Registry
Each tool implements this interface. Router generates JSON → Registry finds the correct tool.
"""
from __future__ import annotations

import asyncio
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

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


# Global registry — `from bantz.tools import registry`
registry = ToolRegistry()