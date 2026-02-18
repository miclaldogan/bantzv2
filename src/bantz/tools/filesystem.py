"""
Bantz v2 — Filesystem Tool
File system operations: listing, reading, writing.
Path security check — cannot go outside the home directory (by default).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from bantz.tools import BaseTool, ToolResult, registry

# Safe root — default is home directory
SAFE_ROOT = Path.home()
MAX_READ_BYTES = 50_000   # 50KB üstü dosyaları kırparak okur


def _safe_path(raw: str) -> Path | None:
    """Resolves the given path and checks if it is under SAFE_ROOT."""
    try:
        p = Path(raw).expanduser().resolve()
        p.relative_to(SAFE_ROOT)  # ValueError if outside root
        return p
    except (ValueError, RuntimeError):
        return None


class FilesystemTool(BaseTool):
    name = "filesystem"
    description = (
        "File system operations: listing (ls), reading (read), "
        "writing (write). Only works under the home directory."
    )
    risk_level = "moderate"

    async def execute(
        self,
        action: Literal["ls", "read", "write"] = "ls",
        path: str = "~",
        content: str = "",
        **kwargs: Any,
    ) -> ToolResult:

        p = _safe_path(path)
        if p is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Security: '{path}' is outside the home directory or invalid.",
            )

        if action == "ls":
            return await self._ls(p)
        elif action == "read":
            return await self._read(p)
        elif action == "write":
            return await self._write(p, content)
        else:
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")

    # ── ls ────────────────────────────────────────────────────────────────

    async def _ls(self, p: Path) -> ToolResult:
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {p}")
        if p.is_file():
            stat = p.stat()
            return ToolResult(
                success=True,
                output=f"{p.name}  ({stat.st_size} bytes)",
                data={"type": "file", "path": str(p), "size": stat.st_size},
            )
        # directory
        try:
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
            lines = []
            for e in entries[:100]:   # max 100 entries
                if e.is_dir():
                    lines.append(f"📁 {e.name}/")
                else:
                    lines.append(f"📄 {e.name}  ({e.stat().st_size} bytes)")
            output = f"{p}\n" + "\n".join(lines)
            if len(list(p.iterdir())) > 100:
                output += "\n... (first 100 shown)"
            return ToolResult(success=True, output=output, data={"path": str(p), "count": len(lines)})
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {p}")

    # ── read ──────────────────────────────────────────────────────────────

    async def _read(self, p: Path) -> ToolResult:
        if not p.exists():
            return ToolResult(success=False, output="", error=f"File not found: {p}")
        if p.is_dir():
            return ToolResult(success=False, output="", error=f"This is a directory: {p}. Use 'ls'.")
        try:
            raw = p.read_bytes()
            truncated = len(raw) > MAX_READ_BYTES
            text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
            if truncated:
                text += f"\n\n... (file truncated, first {MAX_READ_BYTES // 1000}KB shown)"
            return ToolResult(
                success=True,
                output=text,
                data={"path": str(p), "size": len(raw), "truncated": truncated},
            )
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {p}")

    # ── write ─────────────────────────────────────────────────────────────

    async def _write(self, p: Path, content: str) -> ToolResult:
        if not content:
            return ToolResult(success=False, output="", error="Content cannot be empty.")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Written: {p}  ({len(content)} characters)",
                data={"path": str(p), "bytes_written": len(content.encode())},
            )
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {p}")


registry.register(FilesystemTool())