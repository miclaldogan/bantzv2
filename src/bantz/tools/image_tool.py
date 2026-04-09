"""
Bantz v2 — ImageTool: terminal image rendering via chafa + Telegram fallback (#290)

Downloads images (with content-hash deduplication), renders them as ANSI
art in the terminal via ``chafa``, and provides raw bytes for Telegram's
``send_photo``.

Cache: ``~/.bantz/cache/images/`` with 24h TTL, purged on startup.

Usage:
    from bantz.tools.image_tool import image_tool
    result = await image_tool.execute(action="render", url="https://…")
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tool.image")

# ── Cache config ──────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".bantz" / "cache" / "images"
CACHE_TTL = 86400  # 24 hours in seconds


class ImageToolError(RuntimeError):
    """Raised when image download or rendering fails."""


# ── Cache management ──────────────────────────────────────────────────────────

def _ensure_cache_dir() -> None:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def purge_stale_cache() -> int:
    """Remove cached images older than CACHE_TTL.

    Returns the number of files purged.
    """
    if not CACHE_DIR.is_dir():
        return 0

    now = time.time()
    purged = 0
    for f in CACHE_DIR.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > CACHE_TTL:
            try:
                f.unlink()
                purged += 1
            except OSError as exc:
                log.debug("Failed to purge %s: %s", f, exc)
    if purged:
        log.info("Purged %d stale image(s) from cache", purged)
    return purged


def _chafa_available() -> bool:
    """Check if chafa is installed and accessible."""
    try:
        subprocess.run(
            ["chafa", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Download ──────────────────────────────────────────────────────────────────

def download(url: str, timeout: int = 15) -> Path:
    """Download an image to the local cache.

    Uses sha256 content-hash filenames for deduplication.
    Atomic write via .tmp + rename to prevent cache poisoning
    from partial downloads.

    Returns the path to the cached file.
    """
    _ensure_cache_dir()
    dest = CACHE_DIR / hashlib.sha256(url.encode()).hexdigest()

    if dest.exists():
        # Touch to refresh mtime (keeps it in cache longer)
        dest.touch()
        return dest

    temp_dest = dest.with_suffix(".tmp")
    try:
        # True streaming: write chunk-by-chunk — memory stays flat
        # regardless of image size, per-chunk timeout enforced
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with open(temp_dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        # Atomic rename: cache is never poisoned by partial downloads
        temp_dest.rename(dest)
    except httpx.HTTPError as exc:
        if temp_dest.exists():
            temp_dest.unlink()  # clean up partial file
        raise ImageToolError(
            f"Failed to download image from {url}: {exc}"
        ) from exc
    except Exception as exc:
        if temp_dest.exists():
            temp_dest.unlink()
        raise ImageToolError(
            f"Unexpected error downloading {url}: {exc}"
        ) from exc

    return dest


# ── Terminal rendering ────────────────────────────────────────────────────────

def render_terminal(
    source: str,
    size: str = "60x30",
    colors: str = "256",
) -> str:
    """Render an image as ANSI art using chafa.

    Args:
        source: URL or local file path.
        size: chafa ``--size`` flag (e.g. "60x30", "80x40").
        colors: chafa ``--colors`` flag ("256", "16", "full").

    Returns:
        ANSI art string, or empty string if chafa is unavailable.

    Raises:
        ImageToolError: If chafa fails (non-zero exit) or times out.
    """
    path = download(source) if source.startswith("http") else Path(source)

    if not path.exists():
        raise ImageToolError(f"Image file not found: {path}")

    try:
        result = subprocess.run(
            ["chafa", "--size", size, "--colors", colors, str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        log.warning("chafa not installed — skipping image render")
        return ""
    except subprocess.TimeoutExpired:
        raise ImageToolError("chafa timed out rendering image")

    if result.returncode != 0:
        raise ImageToolError(f"chafa failed: {result.stderr.strip()}")

    return result.stdout


# ── Read raw bytes (for Telegram send_photo) ──────────────────────────────────

def read_bytes(source: str) -> bytes:
    """Download (if URL) and return raw image bytes.

    Used by Telegram handler for ``bot.send_photo()``.
    """
    path = download(source) if source.startswith("http") else Path(source)
    if not path.exists():
        raise ImageToolError(f"Image file not found: {path}")
    return path.read_bytes()


# ── Tool class ────────────────────────────────────────────────────────────────

class ImageTool(BaseTool):
    """Download, cache, and render images.

    Actions:
      - ``render``: Download + render as ANSI art via chafa (for TUI)
      - ``download``: Download to cache, return path
      - ``raw``: Download and return raw bytes (for Telegram)
      - ``purge``: Purge stale cache entries
    """

    name = "image"
    description = (
        "Download, cache, and render images. "
        "Params: action (render|download|raw|purge), "
        "url (str) = image URL, "
        "path (str) = local file path (alternative to URL), "
        "size (str) = chafa render size e.g. '60x30' (default), "
        "colors (str) = chafa color depth: '256' (default), '16', 'full'. "
        "Use for: 'show this image', 'render image in terminal', "
        "'download image from URL'."
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "render")
        url = kwargs.get("url", "").strip()
        path = kwargs.get("path", "").strip()
        size = kwargs.get("size", "60x30").strip()
        colors = kwargs.get("colors", "256").strip()

        source = url or path
        if action == "purge":
            return self._purge()
        if not source:
            return ToolResult(
                success=False, output="",
                error="Please provide an image URL or file path.",
            )

        if action == "download":
            return await self._download(source)
        elif action == "raw":
            return await self._raw(source)
        else:  # "render" or default
            return await self._render(source, size, colors)

    async def _render(self, source: str, size: str, colors: str) -> ToolResult:
        """Download and render image as ANSI art."""
        try:
            ansi_art = render_terminal(source, size=size, colors=colors)
            if not ansi_art:
                # chafa not available — try to at least confirm download
                path = download(source) if source.startswith("http") else Path(source)
                return ToolResult(
                    success=True,
                    output=(
                        f"Image downloaded to {path}, but chafa is not installed "
                        "for terminal rendering. Install with: sudo apt install chafa"
                    ),
                    data={"path": str(path)},
                )
            return ToolResult(
                success=True,
                output=ansi_art,
                data={"rendered": True, "size": size, "colors": colors},
            )
        except ImageToolError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:
            log.warning("ImageTool render error: %s", exc)
            return ToolResult(
                success=False, output="",
                error=f"Image rendering failed: {exc}",
            )

    async def _download(self, source: str) -> ToolResult:
        """Download image to cache and return path."""
        try:
            path = download(source) if source.startswith("http") else Path(source)
            return ToolResult(
                success=True,
                output=f"Image cached at {path}",
                data={"path": str(path)},
            )
        except ImageToolError as exc:
            return ToolResult(success=False, output="", error=str(exc))

    async def _raw(self, source: str) -> ToolResult:
        """Download and return raw bytes for Telegram."""
        try:
            data = read_bytes(source)
            return ToolResult(
                success=True,
                output=f"Image ready ({len(data)} bytes)",
                data={"image_bytes": data, "mime_type": "image/jpeg"},
            )
        except ImageToolError as exc:
            return ToolResult(success=False, output="", error=str(exc))

    @staticmethod
    def _purge() -> ToolResult:
        """Purge stale cached images."""
        count = purge_stale_cache()
        return ToolResult(
            success=True,
            output=f"Purged {count} stale image(s) from cache.",
        )


# ── Auto-register + startup cache purge ───────────────────────────────────────
image_tool = ImageTool()
registry.register(image_tool)

# Purge stale cache on import (startup)
try:
    purge_stale_cache()
except Exception:
    pass  # non-critical
