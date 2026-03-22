"""Tests for Issue #258 — PDF reading, context bloat safeguard, binary rejection.

Covers:
  1. PDF happy path (text extraction via mocked fitz)
  2. Encrypted PDF → success=False
  3. Empty PDF (scanned images) → success=False
  4. Long PDF truncation at MAX_PDF_CHARS
  5. Binary file rejection (.exe, .jpg, .zip, etc.)
  6. PyMuPDF missing → graceful error
  7. Plain text reading still works (regression check)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest



# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — mock fitz objects
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_doc(pages_text: list[str], encrypted: bool = False):
    """Build a mock fitz.Document with given page texts."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = encrypted
    mock_doc.__len__ = lambda self: len(pages_text)

    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.get_text.return_value = text
        mock_pages.append(page)

    mock_doc.__iter__ = lambda self: iter(mock_pages)
    mock_doc.close = MagicMock()
    return mock_doc


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PDF happy path
# ═══════════════════════════════════════════════════════════════════════════════


class TestPDFReading:
    """PDF text extraction via PyMuPDF."""

    @pytest.mark.asyncio
    async def test_reads_pdf_text(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_doc(["Page 1 content.", "Page 2 content."])
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is True
        assert "Page 1 content." in result.output
        assert "Page 2 content." in result.output
        assert result.data["pages"] == 2
        assert result.data["truncated"] is False

    @pytest.mark.asyncio
    async def test_pdf_page_count_in_data(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "three_pages.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_doc(["A", "B", "C"])
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.data["pages"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Encrypted PDF
# ═══════════════════════════════════════════════════════════════════════════════


class TestEncryptedPDF:
    """Password-protected PDFs must return success=False."""

    @pytest.mark.asyncio
    async def test_encrypted_pdf_rejected(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "secret.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_doc([], encrypted=True)
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is False
        assert "password protected" in result.output.lower() or "encrypted" in result.output.lower()
        mock_doc.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Empty PDF (scanned images)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyPDF:
    """PDFs with no extractable text (image-only scans)."""

    @pytest.mark.asyncio
    async def test_empty_pdf_returns_failure(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_doc(["", "   ", ""])
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is False
        assert "no readable text" in result.output.lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_pdf_is_empty(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "blank.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_doc(["\n\n", "\t  \n"])
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is False
        assert "scanned images" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PDF truncation (context bloat safeguard)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPDFTruncation:
    """Long PDFs must be truncated at MAX_PDF_CHARS."""

    @pytest.mark.asyncio
    async def test_long_pdf_truncated(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool, MAX_PDF_CHARS

        pdf_file = tmp_path / "thesis.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        giant_text = "A" * 50_000
        mock_doc = _make_mock_doc([giant_text])
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is True
        assert result.data["truncated"] is True
        assert "[SYSTEM WARNING" in result.output
        assert "truncated" in result.output.lower()
        # Text before the warning should be exactly MAX_PDF_CHARS
        warning_idx = result.output.index("\n\n[SYSTEM WARNING")
        assert warning_idx == MAX_PDF_CHARS

    @pytest.mark.asyncio
    async def test_short_pdf_not_truncated(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "short.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_doc(["Short content."])
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is True
        assert result.data["truncated"] is False
        assert "[SYSTEM WARNING" not in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Binary file rejection
# ═══════════════════════════════════════════════════════════════════════════════


class TestBinaryRejection:
    """Non-text binary files must be rejected with success=False."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ext", [".exe", ".jpg", ".zip", ".mp3", ".mp4", ".dll", ".sqlite"])
    async def test_binary_extensions_rejected(self, tmp_path, ext):
        from bantz.tools.filesystem import FilesystemTool

        bin_file = tmp_path / f"file{ext}"
        bin_file.write_bytes(b"\x00\x01\x02\x03")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(action="read", path=str(bin_file))

        assert result.success is False
        assert "binary" in result.output.lower()

    @pytest.mark.asyncio
    async def test_text_file_not_rejected(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Hello world", encoding="utf-8")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(action="read", path=str(txt_file))

        assert result.success is True
        assert "Hello world" in result.output

    @pytest.mark.asyncio
    async def test_unknown_extension_reads_as_text(self, tmp_path):
        """Unknown extensions (e.g., .cfg, .log) should still be read as text."""
        from bantz.tools.filesystem import FilesystemTool

        cfg_file = tmp_path / "config.cfg"
        cfg_file.write_text("key=value", encoding="utf-8")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(action="read", path=str(cfg_file))

        assert result.success is True
        assert "key=value" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PyMuPDF missing
# ═══════════════════════════════════════════════════════════════════════════════


class TestFitzMissing:
    """Graceful error when PyMuPDF is not installed."""

    @pytest.mark.asyncio
    async def test_fitz_missing_returns_helpful_error(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=None):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is False
        assert "pymupdf" in result.output.lower() or "pip install" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PDF fitz.open failure
# ═══════════════════════════════════════════════════════════════════════════════


class TestPDFOpenFailure:
    """Corrupted PDFs that fitz can't open."""

    @pytest.mark.asyncio
    async def test_corrupt_pdf_returns_error(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        pdf_file = tmp_path / "corrupt.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = RuntimeError("not a valid PDF")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path), \
             patch("bantz.tools.filesystem._get_fitz", return_value=mock_fitz):
            result = await tool.execute(action="read", path=str(pdf_file))

        assert result.success is False
        assert "Failed to open PDF" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Regression — plain text still works
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlainTextRegression:
    """Ensure the new PDF/binary logic doesn't break normal text reading."""

    @pytest.mark.asyncio
    async def test_read_python_file(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        py_file = tmp_path / "script.py"
        py_file.write_text("print('hello')", encoding="utf-8")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(action="read", path=str(py_file))

        assert result.success is True
        assert "print('hello')" in result.output

    @pytest.mark.asyncio
    async def test_read_json_file(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value"}', encoding="utf-8")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(action="read", path=str(json_file))

        assert result.success is True
        assert '"key"' in result.output

    @pytest.mark.asyncio
    async def test_large_text_file_truncated(self, tmp_path):
        from bantz.tools.filesystem import FilesystemTool

        big_file = tmp_path / "big.txt"
        big_file.write_text("X" * 100_000, encoding="utf-8")

        tool = FilesystemTool()
        with patch("bantz.tools.filesystem.SAFE_ROOT", tmp_path):
            result = await tool.execute(action="read", path=str(big_file))

        assert result.success is True
        assert result.data["truncated"] is True
        assert "truncated" in result.output.lower()
