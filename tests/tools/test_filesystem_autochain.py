"""Tests for Issue #196 — Filesystem Auto-Chaining (create_folder_and_file).

The "Stark Folder Fix": LLM used to hallucinate success for multi-step
filesystem ops (mkdir → write_file) without calling any tools.  Now a
single atomic action does both in one shot.

Covers:
  1. Atomic folder+file creation (happy path)
  2. Missing parameter handling (folder_path, file_name)
  3. Security boundary (outside home dir)
  4. Empty content (allowed — creates empty file)
  5. Nested subfolder creation (mkdir -p behaviour)
  6. Quick-route regex matching for English + Turkish patterns
  7. COT_SYSTEM filesystem schema includes create_folder_and_file
  8. Tool description advertises create_folder_and_file
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(coro):
    """Sync wrapper for async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Atomic folder+file creation
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateFolderAndFile:
    """The core atomic operation: create dir + write file in one step."""

    @pytest.mark.asyncio
    async def test_creates_folder_and_file(self, tmp_path):
        """Happy path: folder is created, file is written, paths returned."""
        from bantz.tools.filesystem import FilesystemTool, SAFE_ROOT
        tool = FilesystemTool()
        folder = str(tmp_path / "Stark")
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path=folder,
                file_name="notes.txt",
                content="Winter is coming.",
            )
        assert result.success is True
        assert "✓ Created folder" in result.output
        assert "✓ Created file" in result.output
        assert (tmp_path / "Stark" / "notes.txt").exists()
        assert (tmp_path / "Stark" / "notes.txt").read_text() == "Winter is coming."
        assert result.data["folder_path"] == folder
        assert result.data["file_path"] == str(Path(folder) / "notes.txt")

    @pytest.mark.asyncio
    async def test_creates_nested_subfolders(self, tmp_path):
        """mkdir -p behaviour: creates intermediate directories."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        deep = str(tmp_path / "level1" / "level2" / "level3")
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path=deep,
                file_name="deep.txt",
                content="Hello from the deep.",
            )
        assert result.success is True
        assert (tmp_path / "level1" / "level2" / "level3" / "deep.txt").exists()

    @pytest.mark.asyncio
    async def test_empty_content_creates_empty_file(self, tmp_path):
        """Content can be empty — creates a 0-byte file."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        folder = str(tmp_path / "EmptyTest")
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path=folder,
                file_name="blank.txt",
                content="",
            )
        assert result.success is True
        assert (tmp_path / "EmptyTest" / "blank.txt").read_text() == ""
        assert result.data["bytes_written"] == 0

    @pytest.mark.asyncio
    async def test_existing_folder_no_error(self, tmp_path):
        """If the folder already exists, it should NOT fail (exist_ok=True)."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        folder = tmp_path / "Existing"
        folder.mkdir()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path=str(folder),
                file_name="new.txt",
                content="Added to existing folder.",
            )
        assert result.success is True
        assert (folder / "new.txt").read_text() == "Added to existing folder."


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Missing parameter handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestMissingParams:
    """Graceful error messages when required params are omitted."""

    @pytest.mark.asyncio
    async def test_missing_folder_path(self, tmp_path):
        """Missing folder_path → clear error."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path="",
                file_name="test.txt",
                content="data",
            )
        assert result.success is False
        assert "folder_path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_file_name(self, tmp_path):
        """Missing file_name → clear error."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path=str(tmp_path / "Foo"),
                file_name="",
                content="data",
            )
        assert result.success is False
        assert "file_name" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Security boundary
# ═══════════════════════════════════════════════════════════════════════════════


class TestSecurityBoundary:
    """Paths outside SAFE_ROOT must be rejected."""

    @pytest.mark.asyncio
    async def test_outside_home_rejected(self, tmp_path):
        """Folder path outside SAFE_ROOT → security error."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path="/etc/evil",
                file_name="pwned.txt",
                content="bad",
            )
        assert result.success is False
        assert "security" in result.error.lower() or "outside" in result.error.lower()

    @pytest.mark.asyncio
    async def test_traversal_attack_rejected(self, tmp_path):
        """Path traversal (../../) must be caught."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(
                action="create_folder_and_file",
                folder_path=str(tmp_path / ".." / ".." / "tmp" / "evil"),
                file_name="escape.txt",
                content="nope",
            )
        assert result.success is False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Quick-route regex matching
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuickRouteFilesystem:
    """_quick_route must catch folder+file creation patterns."""

    def _route(self, text: str) -> dict | None:
        from bantz.core.brain import Brain
        return Brain._quick_route(text, text)

    def test_create_stark_folder_and_file(self):
        """'create a folder named Stark and write notes.txt in it'"""
        r = self._route("create a folder named Stark and write notes.txt in it")
        assert r is not None
        assert r["tool"] == "filesystem"
        assert r["args"]["action"] == "create_folder_and_file"
        assert "stark" in r["args"]["folder_path"].lower()
        assert r["args"]["file_name"] == "notes.txt"

    def test_make_folder_and_put_file(self):
        """'make a folder called Projects and put readme.md in it with hello world'"""
        r = self._route("make a folder called Projects and put readme.md in it with hello world")
        assert r is not None
        assert r["tool"] == "filesystem"
        assert "projects" in r["args"]["folder_path"].lower()
        assert r["args"]["file_name"] == "readme.md"
        assert "hello world" in r["args"]["content"]

    def test_create_directory_then_create_file(self):
        """'create a directory named reports then create notes.txt'"""
        r = self._route("create a directory named reports then create notes.txt")
        assert r is not None
        assert r["tool"] == "filesystem"
        assert "reports" in r["args"]["folder_path"].lower()
        assert r["args"]["file_name"] == "notes.txt"

    def test_create_folder_with_content(self):
        """'create a folder named test and write data.txt in it with some test data'"""
        r = self._route("create a folder named test and write data.txt in it with some test data")
        assert r is not None
        assert r["args"]["content"] == "some test data"

    def test_default_desktop_path(self):
        """When no path specified, folder should default to ~/Desktop/."""
        r = self._route("create a folder named Stark and write test.txt in it")
        assert r is not None
        assert "~/Desktop/" in r["args"]["folder_path"]

    def test_no_match_for_simple_create_folder(self):
        """'create a folder named X' (without file) should NOT match combo regex."""
        r = self._route("create a folder named TestOnly")
        # Should NOT match the combo regex — falls through to _generate
        if r is not None:
            assert r["tool"] != "filesystem" or r["args"].get("action") != "create_folder_and_file"

    def test_no_match_for_email_create(self):
        """'create a mail ...' must NOT match filesystem combo."""
        r = self._route("create a folder and write a mail to john")
        # Should NOT be filesystem
        if r is not None and r["tool"] == "filesystem":
            assert r["args"].get("action") != "create_folder_and_file"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Schema / description / COT
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaAndCOT:
    """Tool schema and COT_SYSTEM must advertise create_folder_and_file."""

    def test_tool_description_mentions_create_folder_and_file(self):
        """FilesystemTool.description must mention create_folder_and_file."""
        from bantz.tools.filesystem import FilesystemTool
        tool = FilesystemTool()
        assert "create_folder_and_file" in tool.description

    def test_cot_system_includes_create_folder_and_file(self):
        """COT_SYSTEM must list create_folder_and_file as a filesystem action."""
        from bantz.core.intent import COT_SYSTEM
        assert "create_folder_and_file" in COT_SYSTEM

    def test_cot_system_includes_folder_path_param(self):
        """COT_SYSTEM filesystem section must mention folder_path parameter."""
        from bantz.core.intent import COT_SYSTEM
        assert "folder_path" in COT_SYSTEM

    def test_cot_system_includes_file_name_param(self):
        """COT_SYSTEM filesystem section must mention file_name parameter."""
        from bantz.core.intent import COT_SYSTEM
        assert "file_name" in COT_SYSTEM
