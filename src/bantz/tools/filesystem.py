"""Bantz v2 - Filesystem Tool (#258: PDF support + binary rejection)

File system operations: listing, reading (text + PDF), writing.
Path security check - cannot go outside the home directory (by default).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.filesystem")

# Safe root - default is home directory
SAFE_ROOT = Path.home()
MAX_READ_BYTES = 50_000   # truncates text files larger than 50KB
MAX_PDF_CHARS = 25_000    # truncates PDF text to ~10-15 pages (#258)

# Binary extensions that cannot be meaningfully read as text
_BINARY_EXTENSIONS = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".webp",
    ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma",
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".iso", ".img", ".dmg",
    ".pyc", ".pyo", ".class", ".o",
    ".db", ".sqlite", ".sqlite3",
})

# Lazy import for PyMuPDF - app must not crash if missing (#258)
_fitz = None


def _get_fitz():
    """Lazy-load fitz (PyMuPDF). Returns module or None."""
    global _fitz
    if _fitz is None:
        try:
            import fitz  # type: ignore[import-untyped]
            _fitz = fitz
        except ImportError:
            return None
    return _fitz


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
        "writing (write), or atomic folder+file creation (create_folder_and_file). "
        "Only works under the home directory. "
        "Use action='create_folder_and_file' when the user wants to create a "
        "folder AND put a file in it in a single step — avoids multi-step hallucination. "
        "Always use absolute paths (e.g. ~/Documents/file.txt). "
        "NEVER guess file names — if unsure, list the directory first."
    )
    risk_level = "moderate"

    async def execute(
        self,
        action: Literal["ls", "read", "write", "create_folder_and_file"] = "ls",
        path: str = "~",
        content: str = "",
        folder_path: str = "",
        file_name: str = "",
        **kwargs: Any,
    ) -> ToolResult:

        # ── Atomic folder+file creation shortcut ──────────────────────
        if action == "create_folder_and_file":
            return await self._create_folder_and_file(
                folder_path=folder_path,
                file_name=file_name,
                content=content,
            )

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
                try:
                    if e.is_dir():
                        lines.append(f"📁 {e.name}/")
                    else:
                        lines.append(f"📄 {e.name}  ({e.stat().st_size} bytes)")
                except (OSError, FileNotFoundError):
                    # Broken symlink or inaccessible entry
                    lines.append(f"⛓ {e.name}  (broken link)")
            output = f"{p}\n" + "\n".join(lines)
            if len(entries) > 100:
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

        suffix = p.suffix.lower()

        # ── Binary file rejection (#258) ──────────────────────────────
        if suffix in _BINARY_EXTENSIONS:
            return ToolResult(
                success=False,
                output="Error: Cannot read binary file format as text.",
                error=f"Unsupported binary format: {suffix}",
            )

        # ── PDF reading via PyMuPDF (#258) ────────────────────────────
        if suffix == ".pdf":
            return await self._read_pdf(p)

        # ── Plain text reading ────────────────────────────────────────
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

    # ── PDF reader (#258) ─────────────────────────────────────────────────

    async def _read_pdf(self, p: Path) -> ToolResult:
        """Extract text from PDF with context-bloat safeguard."""
        fitz = _get_fitz()
        if fitz is None:
            return ToolResult(
                success=False,
                output="Error: PDF reading requires PyMuPDF. Install with: pip install pymupdf",
                error="PyMuPDF (fitz) not installed.",
            )

        try:
            doc = fitz.open(str(p))
        except Exception as exc:
            return ToolResult(
                success=False, output="",
                error=f"Failed to open PDF: {exc}",
            )

        # Encrypted / password-protected
        if doc.is_encrypted:
            doc.close()
            return ToolResult(
                success=False,
                output="Error: PDF is password protected or encrypted.",
                error="Encrypted PDF.",
            )

        # Extract text from all pages
        try:
            page_count = len(doc)
            text = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

        # No readable text (scanned images only)
        if not text.strip():
            return ToolResult(
                success=False,
                output="Error: PDF contains no readable text (might be scanned images only).",
                error="Empty PDF text.",
            )

        # Context bloat safeguard — truncate long PDFs
        original_len = len(text)
        truncated = original_len > MAX_PDF_CHARS
        if truncated:
            text = (
                text[:MAX_PDF_CHARS]
                + "\n\n[SYSTEM WARNING: File truncated due to length limits. "
                "Only the beginning is shown.]"
            )
            log.warning("PDF %s truncated from %d to %d chars", p.name, original_len, MAX_PDF_CHARS)

        return ToolResult(
            success=True,
            output=text,
            data={
                "path": str(p),
                "pages": page_count,
                "length": len(text),
                "truncated": truncated,
            },
        )

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

    # ── create_folder_and_file (atomic combo — #196) ─────────────────────

    async def _create_folder_and_file(
        self, folder_path: str, file_name: str, content: str,
    ) -> ToolResult:
        """Create a directory AND write a file inside it in one atomic step.

        This avoids multi-step hallucination where the LLM claims success
        without actually calling the tools sequentially.
        """
        if not folder_path:
            return ToolResult(
                success=False, output="",
                error="Missing folder_path — where should I create the folder?",
            )
        if not file_name:
            return ToolResult(
                success=False, output="",
                error="Missing file_name — what should the file be called?",
            )

        folder = _safe_path(folder_path)
        if folder is None:
            return ToolResult(
                success=False, output="",
                error=f"Security: '{folder_path}' is outside the home directory or invalid.",
            )

        file_path = folder / file_name
        # Validate the final file path is also under SAFE_ROOT
        if _safe_path(str(file_path)) is None:
            return ToolResult(
                success=False, output="",
                error=f"Security: '{file_path}' is outside the home directory.",
            )

        try:
            folder.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return ToolResult(
                success=False, output="",
                error=f"Permission denied: cannot create '{folder}'.",
            )

        file_content = content or ""
        try:
            file_path.write_text(file_content, encoding="utf-8")
        except PermissionError:
            return ToolResult(
                success=False, output="",
                error=f"Permission denied: cannot write '{file_path}'.",
            )

        return ToolResult(
            success=True,
            output=(
                f"✓ Created folder: {folder}\n"
                f"✓ Created file: {file_path}  ({len(file_content)} characters)"
            ),
            data={
                "folder_path": str(folder),
                "file_path": str(file_path),
                "bytes_written": len(file_content.encode()),
            },
        )


registry.register(FilesystemTool())