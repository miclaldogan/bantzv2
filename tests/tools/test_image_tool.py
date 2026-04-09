"""Tests for ImageTool (#290)."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from bantz.tools.image_tool import (
    CACHE_DIR,
    CACHE_TTL,
    ImageTool,
    ImageToolError,
    _chafa_available,
    _ensure_cache_dir,
    download,
    purge_stale_cache,
    read_bytes,
    render_terminal,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Override CACHE_DIR to a temporary directory."""
    import bantz.tools.image_tool as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def fake_image(tmp_path):
    """Create a fake image file."""
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG header
    return img


# ── Tests: cache management ───────────────────────────────────────────────────

class TestCacheManagement:
    def test_ensure_cache_dir(self, tmp_cache):
        import bantz.tools.image_tool as mod
        # Remove and recreate
        if tmp_cache.exists():
            import shutil
            shutil.rmtree(tmp_cache)
        assert not tmp_cache.exists()
        _ensure_cache_dir()
        assert tmp_cache.is_dir()

    def test_purge_stale_cache(self, tmp_cache):
        # Create files: one fresh, one stale
        fresh = tmp_cache / "fresh_file"
        stale = tmp_cache / "stale_file"
        fresh.write_bytes(b"data")
        stale.write_bytes(b"data")

        # Make stale file old
        old_time = time.time() - CACHE_TTL - 100
        import os
        os.utime(stale, (old_time, old_time))

        purged = purge_stale_cache()
        assert purged == 1
        assert fresh.exists()
        assert not stale.exists()

    def test_purge_empty_cache(self, tmp_cache):
        assert purge_stale_cache() == 0

    def test_purge_nonexistent_cache(self, monkeypatch):
        import bantz.tools.image_tool as mod
        monkeypatch.setattr(mod, "CACHE_DIR", Path("/nonexistent/path"))
        assert purge_stale_cache() == 0


# ── Tests: download ───────────────────────────────────────────────────────────

class TestDownload:
    def test_download_success(self, tmp_cache):
        """Mock httpx.stream to simulate a successful download."""
        fake_data = b"\xff\xd8\xff\xe0image_data"

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes = MagicMock(return_value=[fake_data])

        with patch("bantz.tools.image_tool.httpx.stream", return_value=mock_response):
            path = download("https://example.com/test.jpg")

        assert path.exists()
        assert path.read_bytes() == fake_data
        assert path.parent == tmp_cache

    def test_download_cached(self, tmp_cache):
        """Second download of same URL returns cached file without HTTP."""
        import hashlib
        url = "https://example.com/cached.jpg"
        dest = tmp_cache / hashlib.sha256(url.encode()).hexdigest()
        dest.write_bytes(b"cached_data")

        # Should not call httpx at all
        with patch("bantz.tools.image_tool.httpx.stream") as mock_stream:
            path = download(url)
            mock_stream.assert_not_called()

        assert path == dest

    def test_download_http_error(self, tmp_cache):
        """HTTP error raises ImageToolError and cleans up temp file."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            )
        )

        with patch("bantz.tools.image_tool.httpx.stream", return_value=mock_response):
            with pytest.raises(ImageToolError, match="Failed to download"):
                download("https://example.com/404.jpg")

        # No .tmp files left behind
        assert not list(tmp_cache.glob("*.tmp"))

    def test_download_sha256_filename(self, tmp_cache):
        """Cache filename is sha256 of URL."""
        import hashlib
        url = "https://example.com/image.png"
        expected_name = hashlib.sha256(url.encode()).hexdigest()

        fake_data = b"png_data"
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes = MagicMock(return_value=[fake_data])

        with patch("bantz.tools.image_tool.httpx.stream", return_value=mock_response):
            path = download(url)

        assert path.name == expected_name


# ── Tests: render_terminal ────────────────────────────────────────────────────

class TestRenderTerminal:
    def test_render_success(self, fake_image):
        """Successful chafa rendering."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="█▀▄ ANSI ART ▄▀█\n", stderr=""
        )
        with patch("bantz.tools.image_tool.subprocess.run", return_value=mock_result):
            output = render_terminal(str(fake_image))

        assert "ANSI ART" in output

    def test_render_chafa_not_installed(self, fake_image):
        """Returns empty string when chafa is not installed."""
        with patch(
            "bantz.tools.image_tool.subprocess.run",
            side_effect=FileNotFoundError("chafa not found"),
        ):
            output = render_terminal(str(fake_image))

        assert output == ""

    def test_render_chafa_timeout(self, fake_image):
        """Raises error when chafa times out."""
        with patch(
            "bantz.tools.image_tool.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="chafa", timeout=10),
        ):
            with pytest.raises(ImageToolError, match="timed out"):
                render_terminal(str(fake_image))

    def test_render_chafa_failure(self, fake_image):
        """Raises error when chafa exits non-zero."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Unknown format"
        )
        with patch("bantz.tools.image_tool.subprocess.run", return_value=mock_result):
            with pytest.raises(ImageToolError, match="chafa failed"):
                render_terminal(str(fake_image))

    def test_render_file_not_found(self):
        """Raises error for missing file."""
        with pytest.raises(ImageToolError, match="not found"):
            render_terminal("/nonexistent/image.jpg")

    def test_render_custom_size_and_colors(self, fake_image):
        """Custom size and colors are passed to chafa."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="art\n", stderr=""
        )
        with patch("bantz.tools.image_tool.subprocess.run", return_value=mock_result) as mock_run:
            render_terminal(str(fake_image), size="80x40", colors="full")

        call_args = mock_run.call_args[0][0]
        assert "--size" in call_args
        assert "80x40" in call_args
        assert "--colors" in call_args
        assert "full" in call_args


# ── Tests: read_bytes ─────────────────────────────────────────────────────────

class TestReadBytes:
    def test_read_local_file(self, fake_image):
        data = read_bytes(str(fake_image))
        assert data.startswith(b"\xff\xd8\xff\xe0")
        assert len(data) == 104  # 4 header + 100 zeros

    def test_read_missing_file(self):
        with pytest.raises(ImageToolError, match="not found"):
            read_bytes("/nonexistent/missing.jpg")


# ── Tests: chafa_available ────────────────────────────────────────────────────

class TestChafaAvailable:
    def test_chafa_installed(self):
        mock_result = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("bantz.tools.image_tool.subprocess.run", return_value=mock_result):
            assert _chafa_available() is True

    def test_chafa_not_installed(self):
        with patch(
            "bantz.tools.image_tool.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert _chafa_available() is False


# ── Tests: ImageTool actions ──────────────────────────────────────────────────

class TestImageToolActions:
    @pytest.mark.asyncio
    async def test_no_source(self):
        tool = ImageTool()
        result = await tool.execute(action="render")
        assert not result.success
        assert "url" in result.error.lower() or "path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_purge_action(self, tmp_cache):
        tool = ImageTool()
        result = await tool.execute(action="purge")
        assert result.success
        assert "Purged" in result.output

    @pytest.mark.asyncio
    async def test_render_action(self, fake_image):
        tool = ImageTool()
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ANSI\n", stderr=""
        )
        with patch("bantz.tools.image_tool.subprocess.run", return_value=mock_result):
            result = await tool.execute(
                action="render", path=str(fake_image), size="40x20"
            )
        assert result.success
        assert "ANSI" in result.output

    @pytest.mark.asyncio
    async def test_download_action(self, fake_image):
        tool = ImageTool()
        result = await tool.execute(action="download", path=str(fake_image))
        assert result.success
        assert "path" in result.data

    @pytest.mark.asyncio
    async def test_raw_action(self, fake_image):
        tool = ImageTool()
        result = await tool.execute(action="raw", path=str(fake_image))
        assert result.success
        assert "image_bytes" in result.data
        assert len(result.data["image_bytes"]) > 0

    @pytest.mark.asyncio
    async def test_render_chafa_missing_fallback(self, fake_image):
        """When chafa is missing, render still succeeds with info message."""
        tool = ImageTool()
        with patch(
            "bantz.tools.image_tool.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = await tool.execute(action="render", path=str(fake_image))
        assert result.success
        assert "chafa" in result.output.lower()


# Need httpx for HTTP error test
try:
    import httpx
except ImportError:
    httpx = None
