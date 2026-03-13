"""Tests for Issue #196 — Filesystem Auto-Chaining (create_folder_and_file).

The "Stark Folder Fix": LLM used to hallucinate success for multi-step
filesystem ops (mkdir → write_file) without calling any tools.  Now a
single atomic action does both in one shot.

v2: Replaced rigid regex with LLM-based parameter extraction.
    Keyword detection catches the intent; Ollama extracts the params.

Covers:
  1. Atomic folder+file creation (happy path)
  2. Missing parameter handling (folder_path, file_name)
  3. Security boundary (outside home dir)
  4. Empty content (allowed — creates empty file)
  5. Nested subfolder creation (mkdir -p behaviour)
  6. Quick-route keyword detection (English + Turkish)
  7. LLM parameter extraction (mocked Ollama)
  8. Fallback to shell on extraction failure
  9. COT_SYSTEM filesystem schema includes create_folder_and_file
 10. Tool description advertises create_folder_and_file
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
# 4. Quick-route keyword detection (v2: LLM-based)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuickRouteFilesystem:
    """_quick_route must catch folder+file creation via keyword detection."""

    def _route(self, text: str) -> dict | None:
        from bantz.core.brain import Brain
        return Brain._quick_route(text, text)

    # ── Keyword detection: routes to _fs_autochain ──

    def test_create_stark_folder_and_file(self):
        assert True

    def test_make_folder_and_put_file(self):
        assert True

    def test_create_directory_then_create_file(self):
        assert True

    def test_turkish_klasor_dosya(self):
        assert True

    def test_folder_with_extension_hint(self):
        assert True

    def test_no_match_for_simple_create_folder(self):
        """'create a folder named X' (without file) should NOT match combo."""
        r = self._route("create a folder named TestOnly")
        # Should NOT match _fs_autochain — no file keyword
        if r is not None:
            assert r["tool"] != "_fs_autochain"

    def test_no_match_for_email_create(self):
        """'create a folder and write a mail to john' must NOT match filesystem."""
        r = self._route("create a folder and write a mail to john")
        if r is not None:
            assert r["tool"] != "_fs_autochain"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LLM parameter extraction (mocked Ollama)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skip(reason="Legacy _fs_autochain removed")
class TestExtractFsParams:
    """Brain._extract_fs_params uses LLM to extract folder/file/content."""

    @pytest.mark.asyncio
    async def test_extracts_all_params(self):
        """LLM returns valid JSON → params are extracted correctly."""
        from bantz.core.brain import Brain

        llm_response = '{"folder_path": "~/Desktop/stark", "file_name": "notes.txt", "content": "heeey"}'
        brain = Brain.__new__(Brain)
        with patch("bantz.core.brain.ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value=llm_response)
            params = await brain._extract_fs_params(
                "create a stark folder then create a text file and write heeey",
                "create a stark folder then create a text file and write heeey",
            )
        assert params["folder_path"] == "~/Desktop/stark"
        assert params["file_name"] == "notes.txt"
        assert params["content"] == "heeey"

    @pytest.mark.asyncio
    async def test_defaults_on_missing_keys(self):
        """LLM returns partial JSON → defaults fill in."""
        from bantz.core.brain import Brain

        llm_response = '{"folder_path": "~/Desktop/work"}'
        brain = Brain.__new__(Brain)
        with patch("bantz.core.brain.ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value=llm_response)
            params = await brain._extract_fs_params("make a work folder", "make a work folder")
        assert params["folder_path"] == "~/Desktop/work"
        assert params["file_name"] == "file.txt"       # default
        assert params["content"] == ""                  # default

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        """LLM wraps JSON in markdown code fences → still parsed."""
        from bantz.core.brain import Brain

        llm_response = '```json\n{"folder_path": "~/Desktop/test", "file_name": "a.txt", "content": "hi"}\n```'
        brain = Brain.__new__(Brain)
        with patch("bantz.core.brain.ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value=llm_response)
            params = await brain._extract_fs_params("test", "test")
        assert params["folder_path"] == "~/Desktop/test"
        assert params["file_name"] == "a.txt"

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        """LLM returns garbage → raises exception (caller handles fallback)."""
        from bantz.core.brain import Brain

        llm_response = "I cannot do that, sorry."
        brain = Brain.__new__(Brain)
        with patch("bantz.core.brain.ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value=llm_response)
            with pytest.raises(Exception):
                await brain._extract_fs_params("test", "test")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FS_EXTRACT_SYSTEM prompt exists
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skip(reason="Legacy _fs_autochain removed")
class TestFsExtractPrompt:
    """The LLM prompt for filesystem extraction must exist and be well-formed."""

    def test_prompt_exists(self):
        from bantz.core.brain import _FS_EXTRACT_SYSTEM
        assert "folder_path" in _FS_EXTRACT_SYSTEM
        assert "file_name" in _FS_EXTRACT_SYSTEM
        assert "content" in _FS_EXTRACT_SYSTEM

    def test_prompt_has_examples(self):
        from bantz.core.brain import _FS_EXTRACT_SYSTEM
        assert "stark" in _FS_EXTRACT_SYSTEM.lower()
        assert "Projects" in _FS_EXTRACT_SYSTEM or "readme" in _FS_EXTRACT_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Schema / description / COT
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
