"""
Bantz v3 — BrowserTool (#288)

Terminal-native web fetching via curl + pup + readability-cli.
No API keys, no rate limits, no external Python dependencies.

Pipeline:
  fetch(url)         → raw HTML via curl -sL
  extract_text(url)  → article body via readability-cli
  query(url, sel)    → CSS-selected elements via pup (stdin pipe)
  extract_images(url)→ list of absolute image URLs via pup

Prerequisites:
  sudo apt install curl
  npm install -g @mozilla/readability-cli   # or readability-cli
  go install github.com/ericchiang/pup@latest
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.browser_tool")


# ── Typed error ────────────────────────────────────────────────────────────────

class BrowserToolError(RuntimeError):
    """Raised by BrowserTool._run() on any subprocess failure."""


# ── Core subprocess runner ─────────────────────────────────────────────────────

class BrowserTool(BaseTool):
    name = "browser"
    description = (
        "Fetch and parse web pages using local CLI tools (curl / pup / readability-cli). "
        "Use fetch() for raw HTML, extract_text() for clean article body, "
        "query() for CSS-selected content, extract_images() for image URLs. "
        "No API key required. Requires a valid URL — NEVER fabricate or guess URLs."
    )
    risk_level = "safe"

    # ── Internal runner ────────────────────────────────────────────────────────

    def _run(
        self,
        cmd: str | list[str],
        input_data: str | None = None,
        timeout: int = 15,
    ) -> str:
        """Run a subprocess, centralising all OS-level failure modes.

        Args:
            cmd:        Shell command string or pre-split list.
            input_data: Optional stdin data (used for pup pipe).
            timeout:    Kill and raise after this many seconds.

        Raises:
            BrowserToolError: binary missing, timeout, or non-zero exit.
        """
        cmd_args = shlex.split(cmd) if isinstance(cmd, str) else cmd
        exe_name = cmd_args[0]

        try:
            proc = subprocess.run(
                cmd_args,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise BrowserToolError(
                f"'{exe_name}' not found. "
                f"Make sure it is installed and available in PATH.\n"
                f"  curl:             sudo apt install curl\n"
                f"  readability-cli:  npm install -g @mozilla/readability-cli\n"
                f"  pup:              go install github.com/ericchiang/pup@latest"
            )
        except subprocess.TimeoutExpired:
            raise BrowserToolError(
                f"'{exe_name}' timed out after {timeout}s."
            )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise BrowserToolError(
                f"'{exe_name}' failed (exit {proc.returncode}): {stderr}"
            )

        return proc.stdout

    # ── Public interface ───────────────────────────────────────────────────────

    def fetch(self, url: str) -> str:
        """Return raw HTML from *url* via curl."""
        return self._run(f"curl -sL --max-time 12 {url}")

    def extract_text(self, url: str) -> str:
        """Return clean article body via readability-cli."""
        return self._run(f"readability-cli {url}", timeout=20)

    def query(self, url: str, selector: str) -> str:
        """Return HTML elements matching *selector* (CSS) via pup."""
        html = self.fetch(url)
        return self._run(["pup", selector], input_data=html, timeout=10)

    def extract_images(self, url: str) -> list[str]:
        """Return list of absolute image src URLs via pup."""
        html = self.fetch(url)
        raw = self._run(["pup", "img attr{src}"], input_data=html, timeout=10)
        return [line.strip() for line in raw.splitlines() if line.strip()]

    # ── BaseTool.execute ───────────────────────────────────────────────────────

    async def execute(
        self,
        url: str = "",
        action: str = "extract_text",
        selector: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Dispatch to the appropriate pipeline method.

        Args:
            url:      Target URL (required).
            action:   One of fetch | extract_text | query | extract_images.
            selector: CSS selector — required when action=query.
        """
        if not url:
            return ToolResult(
                success=False, output="",
                error="No URL provided. Pass a 'url' parameter.",
            )
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False, output="",
                error=f"Invalid URL (must start with http:// or https://): {url}",
            )

        try:
            if action == "fetch":
                output = self.fetch(url)
            elif action == "query":
                if not selector:
                    return ToolResult(
                        success=False, output="",
                        error="action=query requires a 'selector' parameter.",
                    )
                output = self.query(url, selector)
            elif action == "extract_images":
                imgs = self.extract_images(url)
                output = "\n".join(imgs) if imgs else "(no images found)"
            else:  # default: extract_text
                output = self.extract_text(url)

        except BrowserToolError as exc:
            log.warning("browser_tool %s %s → %s", action, url, exc)
            return ToolResult(success=False, output="", error=str(exc))

        log.info("browser_tool %s %s → %d chars", action, url, len(output))
        return ToolResult(success=True, output=output, data={"url": url, "action": action})


# ── Register ───────────────────────────────────────────────────────────────────

registry.register(BrowserTool())
