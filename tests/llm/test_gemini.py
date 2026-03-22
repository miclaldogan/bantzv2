import pytest
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from bantz.llm.gemini import GeminiClient, _notify_gemini_health


# --- _notify_gemini_health tests ---

def test_notify_gemini_health_main_thread():
    # Mock textual App (need to mock where it's imported in the try block, or mock sys.modules)
    import sys
    mock_textual_app = MagicMock()
    mock_app = MagicMock()
    mock_textual_app.App.current = mock_app

    # Mock ServiceStatus
    mock_header = MagicMock()
    mock_header.ServiceStatus.UP = "UP"
    mock_header.ServiceStatus.DOWN = "DOWN"

    with patch.dict(sys.modules, {"textual.app": mock_textual_app, "bantz.interface.tui.panels.header": mock_header}), \
         patch("threading.current_thread", return_value=threading.main_thread()):

        _notify_gemini_health(True)

        mock_app.notify_service_health.assert_called_once_with("gemini", "UP")


def test_notify_gemini_health_worker_thread():
    # Mock textual App
    import sys
    mock_textual_app = MagicMock()
    mock_app = MagicMock()
    mock_textual_app.App.current = mock_app

    # Mock ServiceStatus
    mock_header = MagicMock()
    mock_header.ServiceStatus.UP = "UP"
    mock_header.ServiceStatus.DOWN = "DOWN"

    with patch.dict(sys.modules, {"textual.app": mock_textual_app, "bantz.interface.tui.panels.header": mock_header}), \
         patch("threading.current_thread", return_value=threading.Thread()), \
         patch("threading.main_thread", return_value=threading.main_thread()):

        _notify_gemini_health(False)

        mock_app.call_from_thread.assert_called_once_with(mock_app.notify_service_health, "gemini", "DOWN")


# --- GeminiClient tests ---

@pytest.fixture
def mock_config():
    with patch("bantz.llm.gemini.config") as config:
        config.gemini_api_key = "test_api_key"
        config.gemini_model = "test_model"
        config.gemini_enabled = True
        yield config


def test_init_and_is_enabled(mock_config):
    # Test enabled
    client = GeminiClient()
    assert client.is_enabled() is True
    assert client._api_key == "test_api_key"
    assert client._model == "test_model"

    # Test disabled (missing API key)
    mock_config.gemini_api_key = ""
    client_disabled_key = GeminiClient()
    assert client_disabled_key.is_enabled() is False

    # Test disabled (flag disabled)
    mock_config.gemini_api_key = "test_api_key"
    mock_config.gemini_enabled = False
    client_disabled_flag = GeminiClient()
    assert client_disabled_flag.is_enabled() is False


@pytest.mark.asyncio
async def test_chat_disabled_raises_error(mock_config):
    mock_config.gemini_enabled = False
    client = GeminiClient()
    with pytest.raises(RuntimeError, match="Gemini is not enabled"):
        await client.chat([])


@pytest.mark.asyncio
async def test_chat_success(mock_config):
    client = GeminiClient()

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello, world!"}]
                }
            }
        ]
    }

    # We must patch httpx.AsyncClient itself to intercept what its context manager yields
    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_resp

    # The async context manager magic
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = False

    with patch("bantz.llm.gemini._notify_gemini_health") as mock_notify, \
         patch("httpx.AsyncClient", return_value=mock_client_instance):

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]

        response_text = await client.chat(messages)

        assert response_text == "Hello, world!"
        mock_notify.assert_called_once_with(True)

        # Verify the API payload structure
        mock_client_instance.post.assert_called_once()
        _, kwargs = mock_client_instance.post.call_args
        payload = kwargs["json"]

        assert payload["systemInstruction"]["parts"][0]["text"] == "You are a helpful assistant."
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"][0]["text"] == "Hi"


@pytest.mark.asyncio
async def test_chat_invalid_response_raises_error(mock_config):
    client = GeminiClient()

    mock_resp = MagicMock()
    # Invalid structure (missing 'parts')
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {}
            }
        ]
    }

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = False

    with patch("bantz.llm.gemini._notify_gemini_health") as mock_notify, \
         patch("httpx.AsyncClient", return_value=mock_client_instance):

        with pytest.raises(RuntimeError, match="Unexpected Gemini response format:"):
            await client.chat([{"role": "user", "content": "Hi"}])

        mock_notify.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_chat_stream_disabled(mock_config):
    mock_config.gemini_enabled = False
    client = GeminiClient()
    with pytest.raises(RuntimeError, match="Gemini is not enabled"):
        async for _ in client.chat_stream([]):
            pass


@pytest.mark.asyncio
async def test_chat_stream_success(mock_config):
    client = GeminiClient()

    # Create dummy streaming events
    async def mock_aiter_lines():
        yield "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"chunk1 \"}]}}]}"
        yield "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"chunk2\"}]}}]}"
        yield "data: [DONE]"

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = mock_aiter_lines
    mock_resp.__aenter__.return_value = mock_resp
    mock_resp.__aexit__.return_value = False

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield mock_resp

    mock_client_instance = AsyncMock()
    mock_client_instance.stream = mock_stream
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        messages = [{"role": "user", "content": "Hi"}]

        chunks = []
        async for chunk in client.chat_stream(messages):
            chunks.append(chunk)

        assert chunks == ["chunk1 ", "chunk2"]


@pytest.mark.asyncio
async def test_is_available_success(mock_config):
    client = GeminiClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        assert await client.is_available() is True


@pytest.mark.asyncio
async def test_is_available_disabled(mock_config):
    mock_config.gemini_enabled = False
    client = GeminiClient()
    assert await client.is_available() is False


@pytest.mark.asyncio
async def test_is_available_exception(mock_config):
    client = GeminiClient()

    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = Exception("Network error")
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        assert await client.is_available() is False
