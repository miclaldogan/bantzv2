"""Tests for Web Reader tool (Plan A: Deep Reading), Issue #182 citations,
and Issue #257 (bot detection / UA rotation / retry).

Covers:
  1. HTML stripping — scripts, styles, tags removed
  2. Text truncation at MAX_TEXT_LENGTH
  3. URL validation (missing, invalid)
  4. HTTP error handling
  5. Telegraph Reference footer appended
  6. Tool registered in registry
  7. UA rotation + 403 retry (#257)
  8. Empty/blocked content → success=False (#257)
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx



# ═══════════════════════════════════════════════════════════════════════════════
# 1. HTML stripping
# ═══════════════════════════════════════════════════════════════════════════════


class TestHTMLStripping:
    """strip_html must remove tags, scripts, styles and collapse whitespace."""

    def test_strips_basic_tags(self):
        from bantz.tools.web_reader import strip_html
        assert strip_html("<p>Hello <b>World</b></p>") == "Hello World"

    def test_strips_script_tags(self):
        from bantz.tools.web_reader import strip_html
        html = "<p>Before</p><script>alert('xss')</script><p>After</p>"
        result = strip_html(html)
        assert "alert" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_style_tags(self):
        from bantz.tools.web_reader import strip_html
        html = "<style>body{color:red}</style><p>Content</p>"
        result = strip_html(html)
        assert "color" not in result
        assert "Content" in result

    def test_strips_noscript_and_svg(self):
        from bantz.tools.web_reader import strip_html
        html = "<noscript>Enable JS</noscript><svg><path/></svg><p>OK</p>"
        result = strip_html(html)
        assert "Enable JS" not in result
        assert "OK" in result

    def test_strips_head_content(self):
        from bantz.tools.web_reader import strip_html
        html = "<html><head><title>Test</title><meta charset='utf-8'></head><body><p>Body</p></body></html>"
        result = strip_html(html)
        assert "Test" not in result
        assert "Body" in result

    def test_collapses_whitespace(self):
        from bantz.tools.web_reader import strip_html
        html = "<p>  Lots   of   spaces  </p>"
        result = strip_html(html)
        assert result == "Lots of spaces"

    def test_empty_html(self):
        from bantz.tools.web_reader import strip_html
        assert strip_html("") == ""

    def test_plain_text_passthrough(self):
        from bantz.tools.web_reader import strip_html
        assert strip_html("Just plain text") == "Just plain text"

    def test_nested_invisible_tags(self):
        """Nested script inside style — all content hidden."""
        from bantz.tools.web_reader import strip_html
        html = "<style><script>inner</script>css</style><p>Visible</p>"
        result = strip_html(html)
        assert "inner" not in result
        assert "css" not in result
        assert "Visible" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WebReaderTool.execute — success paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebReaderExecution:
    """WebReaderTool.execute fetches, strips, truncates, and cites."""

    @pytest.mark.asyncio
    async def test_fetches_and_strips_html(self):
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Article content here</p></body></html>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/article")

        assert result.success is True
        assert "Article content here" in result.output
        assert "Telegraph Reference: https://example.com/article" in result.output

    @pytest.mark.asyncio
    async def test_truncates_long_content(self):
        from bantz.tools.web_reader import WebReaderTool, MAX_TEXT_LENGTH

        long_text = "A" * 20_000
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = f"<p>{long_text}</p>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/long")

        assert result.success is True
        assert result.data["truncated"] is True
        # Output = truncated text + "\n\nTelegraph Reference: ..."
        text_part = result.output.split("\n\nTelegraph Reference:")[0]
        assert len(text_part) <= MAX_TEXT_LENGTH

    @pytest.mark.asyncio
    async def test_telegraph_reference_footer(self):
        """Output must end with 'Telegraph Reference: <url>'."""
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<p>Some content that is long enough to pass the minimum length check</p>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/page")

        assert result.output.rstrip().endswith("Telegraph Reference: https://example.com/page")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WebReaderTool.execute — error paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebReaderErrors:
    """Error handling for invalid URLs, missing params, HTTP failures."""

    @pytest.mark.asyncio
    async def test_no_url(self):
        from bantz.tools.web_reader import WebReaderTool
        result = await WebReaderTool().execute()
        assert result.success is False
        assert "No URL" in result.error

    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        from bantz.tools.web_reader import WebReaderTool
        result = await WebReaderTool().execute(url="ftp://bad.example.com")
        assert result.success is False
        assert "Invalid URL" in result.error

    @pytest.mark.asyncio
    async def test_http_error(self):
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = ""

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/missing")

        assert result.success is False
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_network_exception(self):
        from bantz.tools.web_reader import WebReaderTool

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("DNS resolution failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://unreachable.example.com")

        assert result.success is False
        assert "Failed to fetch" in result.error

    @pytest.mark.asyncio
    async def test_empty_page_content(self):
        """Page returns only scripts/styles with no readable text → success=False (#257)."""
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><script>var x=1;</script><style>body{}</style></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/empty")

        assert result.success is False
        assert "empty content" in result.output.lower() or "javascript" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Tool registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebReaderRegistration:
    """read_url must be registered in the global tool registry."""

    def test_tool_is_registered(self):
        from bantz.tools.web_reader import registry
        tool = registry.get("read_url")
        assert tool is not None
        assert tool.name == "read_url"

    def test_tool_schema(self):
        from bantz.tools.web_reader import registry
        tool = registry.get("read_url")
        schema = tool.schema()
        assert schema["name"] == "read_url"
        assert "URL" in schema["description"] or "url" in schema["description"].lower()
        assert schema["risk_level"] == "safe"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UA rotation + 403 retry (#257)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUARotation:
    """User-Agent rotation and retry on 401/403 blocks (#257)."""

    def test_browser_uas_pool_exists(self):
        from bantz.tools.web_reader import _BROWSER_UAS
        assert len(_BROWSER_UAS) >= 3
        for ua in _BROWSER_UAS:
            assert "Mozilla" in ua

    @pytest.mark.asyncio
    async def test_uses_browser_ua_not_bantz(self):
        """First request must NOT use the old 'Bantz/3.0' UA."""
        from bantz.tools.web_reader import WebReaderTool, _BROWSER_UAS

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<p>" + "A" * 100 + "</p>"

        captured_headers: list[dict] = []

        original_init = httpx.AsyncClient.__init__

        def spy_init(self_client, *args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            original_init(self_client, *args, **kwargs)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client) as mock_cls:
            await tool.execute(url="https://example.com/page")

        # Check that the UA passed to AsyncClient is from our pool
        call_kwargs = mock_cls.call_args_list[0][1]
        ua_used = call_kwargs["headers"]["User-Agent"]
        assert ua_used in _BROWSER_UAS
        assert "Bantz" not in ua_used

    @pytest.mark.asyncio
    async def test_retries_on_403_with_different_ua(self):
        """403 on attempt 1 → retry with a different UA → succeed on attempt 2."""
        from bantz.tools.web_reader import WebReaderTool, _BROWSER_UAS

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.text = "Forbidden"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = "<p>" + "Real article content here" * 5 + "</p>"

        # Two separate mock clients for two loop iterations
        mock_client_1 = AsyncMock()
        mock_client_1.get = AsyncMock(return_value=resp_403)
        mock_client_1.__aenter__ = AsyncMock(return_value=mock_client_1)
        mock_client_1.__aexit__ = AsyncMock(return_value=False)

        mock_client_2 = AsyncMock()
        mock_client_2.get = AsyncMock(return_value=resp_200)
        mock_client_2.__aenter__ = AsyncMock(return_value=mock_client_2)
        mock_client_2.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch(
            "bantz.tools.web_reader.httpx.AsyncClient",
            side_effect=[mock_client_1, mock_client_2],
        ) as mock_cls:
            result = await tool.execute(url="https://example.com/blocked")

        # Must have been called twice (two attempts)
        assert mock_cls.call_count == 2

        # Two different UAs used
        ua1 = mock_cls.call_args_list[0][1]["headers"]["User-Agent"]
        ua2 = mock_cls.call_args_list[1][1]["headers"]["User-Agent"]
        assert ua1 != ua2
        assert ua1 in _BROWSER_UAS
        assert ua2 in _BROWSER_UAS

        # Final result is success
        assert result.success is True
        assert "Real article content here" in result.output

    @pytest.mark.asyncio
    async def test_retries_on_401_same_as_403(self):
        """401 triggers the same retry logic as 403."""
        from bantz.tools.web_reader import WebReaderTool

        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.text = "Unauthorized"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = "<p>" + "Authorized content" * 5 + "</p>"

        mock_client_1 = AsyncMock()
        mock_client_1.get = AsyncMock(return_value=resp_401)
        mock_client_1.__aenter__ = AsyncMock(return_value=mock_client_1)
        mock_client_1.__aexit__ = AsyncMock(return_value=False)

        mock_client_2 = AsyncMock()
        mock_client_2.get = AsyncMock(return_value=resp_200)
        mock_client_2.__aenter__ = AsyncMock(return_value=mock_client_2)
        mock_client_2.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch(
            "bantz.tools.web_reader.httpx.AsyncClient",
            side_effect=[mock_client_1, mock_client_2],
        ) as mock_cls:
            result = await tool.execute(url="https://example.com/auth")

        assert mock_cls.call_count == 2
        assert result.success is True

    @pytest.mark.asyncio
    async def test_both_attempts_403_returns_failure(self):
        """If both attempts return 403, return success=False with descriptive message."""
        from bantz.tools.web_reader import WebReaderTool

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.text = "Forbidden"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp_403)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch(
            "bantz.tools.web_reader.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await tool.execute(url="https://cloudflare-protected.com")

        assert result.success is False
        assert "403" in result.output
        assert "blocking" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Empty / blocked content detection (#257)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyContentDetection:
    """200 OK but empty/short text must return success=False (#257)."""

    @pytest.mark.asyncio
    async def test_short_text_returns_failure(self):
        """< 20 chars after stripping → success=False."""
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>OK</body></html>"  # 2 chars of text

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/captcha")

        assert result.success is False
        assert "empty content" in result.output.lower() or "javascript" in result.output.lower()

    @pytest.mark.asyncio
    async def test_cookie_banner_only_returns_failure(self):
        """Page with only a cookie notice (< 20 chars) → failure."""
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<div>Cookies OK</div>"  # 10 chars

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/cookies")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_sufficient_text_returns_success(self):
        """Page with >= 20 chars of real text → success=True."""
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<p>This is a perfectly fine article with enough content to read.</p>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/good")

        assert result.success is True
        assert "perfectly fine article" in result.output

    @pytest.mark.asyncio
    async def test_js_challenge_message_in_error(self):
        """Empty page error output must mention JavaScript/Captcha for LLM context."""
        from bantz.tools.web_reader import WebReaderTool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        tool = WebReaderTool()
        with patch("bantz.tools.web_reader.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(url="https://example.com/turnstile")

        assert result.success is False
        assert "JavaScript" in result.output or "Captcha" in result.output
