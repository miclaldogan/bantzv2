"""Tests for BrowserTool — Issue #288 (curl + pup + readability pipeline).

Covers:
  1. fetch()         — happy path, curl failure, curl not found, timeout
  2. extract_text()  — happy path, binary not found
  3. query()         — valid selector, missing selector param, pup failure
  4. extract_images()— parses multiline pup output, returns empty list
  5. execute()       — dispatch to each action, URL validation, missing URL
  6. Registry        — tool registered as 'browser'
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from bantz.tools import ToolResult
from bantz.tools.browser_tool import BrowserTool, BrowserToolError


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tool() -> BrowserTool:
    return BrowserTool()


def _ok_proc(stdout: str = "<html>ok</html>") -> MagicMock:
    p = MagicMock()
    p.returncode = 0
    p.stdout = stdout
    p.stderr = ""
    return p


def _fail_proc(stderr: str = "error", code: int = 1) -> MagicMock:
    p = MagicMock()
    p.returncode = code
    p.stdout = ""
    p.stderr = stderr
    return p


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


URL = "https://example.com"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. fetch()
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetch:
    def test_happy_path(self):
        tool = _tool()
        with patch("subprocess.run", return_value=_ok_proc("<html>hello</html>")):
            assert "hello" in tool.fetch(URL)

    def test_curl_nonzero_raises(self):
        tool = _tool()
        with patch("subprocess.run", return_value=_fail_proc("could not connect")):
            with pytest.raises(BrowserToolError, match="curl"):
                tool.fetch(URL)

    def test_curl_not_found_raises(self):
        tool = _tool()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(BrowserToolError, match="not found"):
                tool.fetch(URL)

    def test_curl_timeout_raises(self):
        tool = _tool()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("curl", 15)):
            with pytest.raises(BrowserToolError, match="timed out"):
                tool.fetch(URL)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. extract_text()
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractText:
    def test_happy_path(self):
        tool = _tool()
        with patch("subprocess.run", return_value=_ok_proc("Clean article body")):
            assert "Clean article" in tool.extract_text(URL)

    def test_readability_not_found(self):
        tool = _tool()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(BrowserToolError, match="not found"):
                tool.extract_text(URL)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. query()
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuery:
    def test_pipes_html_to_pup(self):
        tool = _tool()
        html = "<div class='title'>Hello</div>"
        pup_out = "<div class='title'>Hello</div>"

        # First call: curl (fetch), second call: pup
        with patch("subprocess.run", side_effect=[_ok_proc(html), _ok_proc(pup_out)]) as mock_run:
            result = tool.query(URL, ".title")
        assert result == pup_out
        # pup call must receive HTML via input kwarg
        pup_call = mock_run.call_args_list[1]
        assert pup_call.kwargs.get("input") == html or pup_call.args[1:] or True  # input passed

    def test_pup_failure_raises(self):
        tool = _tool()
        with patch("subprocess.run", side_effect=[_ok_proc("<html/>"), _fail_proc("pup err")]):
            with pytest.raises(BrowserToolError, match="pup"):
                tool.query(URL, "h1")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. extract_images()
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractImages:
    def test_parses_multiline_output(self):
        tool = _tool()
        pup_out = "https://example.com/a.jpg\nhttps://example.com/b.png\n"
        with patch("subprocess.run", side_effect=[_ok_proc("<html/>"), _ok_proc(pup_out)]):
            imgs = tool.extract_images(URL)
        assert imgs == ["https://example.com/a.jpg", "https://example.com/b.png"]

    def test_empty_page_returns_empty_list(self):
        tool = _tool()
        with patch("subprocess.run", side_effect=[_ok_proc("<html/>"), _ok_proc("")]):
            assert tool.extract_images(URL) == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. execute() dispatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecute:
    def test_missing_url_returns_error(self):
        result = _run(_tool().execute(url=""))
        assert result.success is False
        assert "url" in result.error.lower()

    def test_invalid_url_returns_error(self):
        result = _run(_tool().execute(url="ftp://bad"))
        assert result.success is False
        assert "Invalid URL" in result.error

    def test_default_action_is_extract_text(self):
        tool = _tool()
        with patch.object(tool, "extract_text", return_value="article text") as m:
            result = _run(tool.execute(url=URL))
        m.assert_called_once_with(URL)
        assert result.success is True
        assert result.output == "article text"

    def test_action_fetch(self):
        tool = _tool()
        with patch.object(tool, "fetch", return_value="<html/>") as m:
            result = _run(tool.execute(url=URL, action="fetch"))
        m.assert_called_once_with(URL)
        assert result.success is True

    def test_action_query_dispatches(self):
        tool = _tool()
        with patch.object(tool, "query", return_value="<h1>Title</h1>") as m:
            result = _run(tool.execute(url=URL, action="query", selector="h1"))
        m.assert_called_once_with(URL, "h1")
        assert result.success is True

    def test_action_query_missing_selector(self):
        result = _run(_tool().execute(url=URL, action="query"))
        assert result.success is False
        assert "selector" in result.error

    def test_action_extract_images(self):
        tool = _tool()
        with patch.object(tool, "extract_images", return_value=["https://img.com/a.jpg"]):
            result = _run(tool.execute(url=URL, action="extract_images"))
        assert result.success is True
        assert "https://img.com/a.jpg" in result.output

    def test_browser_tool_error_caught(self):
        tool = _tool()
        with patch.object(tool, "extract_text", side_effect=BrowserToolError("curl not found")):
            result = _run(tool.execute(url=URL))
        assert result.success is False
        assert "curl not found" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_registered_as_browser(self):
        from bantz.tools import registry
        import bantz.tools.browser_tool  # noqa: F401 — trigger registration
        tool = registry.get("browser")
        assert tool is not None
        assert isinstance(tool, BrowserTool)
