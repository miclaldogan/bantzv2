import json
import socket
import threading
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bantz.core.gps_server import (
    GPSServer,
    _ensure_relay_token,
    _get_local_ip,
    GPS_PORT,
    NTFY_BASE,
    TTL_SECONDS,
)

@pytest.fixture
def mock_config():
    # Since _ensure_relay_token catches exception and falls back if `from bantz.config import config` fails
    # or if we patch the exact getattr sequence, let's patch the entire bantz.config.config in sys.modules
    # or just patch it within the function itself using patch.dict
    with patch("bantz.core.gps_server.getattr") as mock_getattr:
        # We need to ensure that it returns the mocked token
        mock_getattr.return_value = "config-token-123"
        yield mock_getattr

@pytest.fixture
def mock_paths(tmp_path):
    location_file = tmp_path / "live_location.json"
    token_file = tmp_path / "gps_relay_token"
    with patch("bantz.core.gps_server.LOCATION_FILE", location_file), \
         patch("bantz.core.gps_server.TOKEN_FILE", token_file):
        yield location_file, token_file

def test_ensure_relay_token_from_config(mock_paths):
    with patch("bantz.core.gps_server.getattr", return_value="config-token-123"):
        assert _ensure_relay_token() == "config-token-123"

def test_ensure_relay_token_from_disk(mock_paths):
    _, token_file = mock_paths
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text("disk-token-456", encoding="utf-8")
    assert _ensure_relay_token() == "disk-token-456"

def test_ensure_relay_token_generated(mock_paths):
    _, token_file = mock_paths
    token = _ensure_relay_token()
    assert token.startswith("bantz-gps-")
    assert token_file.read_text(encoding="utf-8") == token

def test_get_local_ip_success():
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value = mock_instance
        mock_instance.getsockname.return_value = ("192.168.1.100", 12345)

        assert _get_local_ip() == "192.168.1.100"

def test_get_local_ip_failure():
    with patch("socket.socket", side_effect=Exception("Network error")):
        assert _get_local_ip() == "localhost"

@pytest.fixture
def gps_server(mock_paths):
    return GPSServer(port=9999)

def test_gps_server_init(gps_server):
    assert gps_server._port == 9999
    assert gps_server._server is None
    assert gps_server._thread is None
    assert gps_server._relay_thread is None
    assert gps_server._relay_running is False
    assert gps_server._latest is None
    assert gps_server._relay_token is None

def test_relay_topic_and_url(gps_server, mock_paths):
    with patch("bantz.core.gps_server.getattr", return_value="my-test-topic"):
        assert gps_server.relay_topic == "my-test-topic"
        assert gps_server.relay_url == f"{NTFY_BASE}/my-test-topic"

def test_load_from_disk_valid(gps_server, mock_paths):
    location_file, _ = mock_paths
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data = {"lat": 10.0, "lon": 20.0, "timestamp": now_iso}
    location_file.write_text(json.dumps(data), encoding="utf-8")

    gps_server._load_from_disk()
    assert gps_server._latest == data

def test_load_from_disk_expired(gps_server, mock_paths):
    location_file, _ = mock_paths
    old_time = datetime.now(timezone.utc) - timedelta(seconds=TTL_SECONDS + 100)
    old_iso = old_time.isoformat().replace("+00:00", "Z")
    data = {"lat": 10.0, "lon": 20.0, "timestamp": old_iso}
    location_file.write_text(json.dumps(data), encoding="utf-8")

    gps_server._load_from_disk()
    assert gps_server._latest is None

def test_save_location(gps_server, mock_paths):
    location_file, _ = mock_paths
    data = {"lat": 12.34, "lon": 56.78, "accuracy": 10.5, "timestamp": "2023-01-01T00:00:00Z"}

    with patch("bantz.core.places.places.update_gps", create=True) as mock_update_gps:
        mock_update_gps.return_value = None

        gps_server._save_location(data)

        assert gps_server._latest == data
        assert location_file.exists()
        saved_data = json.loads(location_file.read_text(encoding="utf-8"))
        assert saved_data == data
        mock_update_gps.assert_called_once_with(12.34, 56.78)

def test_latest_property_valid(gps_server):
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data = {"lat": 1.0, "lon": 2.0, "timestamp": now_iso}
    gps_server._latest = data
    assert gps_server.latest == data

def test_latest_property_expired(gps_server):
    old_time = datetime.now(timezone.utc) - timedelta(seconds=TTL_SECONDS + 100)
    old_iso = old_time.isoformat().replace("+00:00", "Z")
    data = {"lat": 1.0, "lon": 2.0, "timestamp": old_iso}
    gps_server._latest = data
    assert gps_server.latest is None

def test_lat_lon_property(gps_server):
    assert gps_server.lat_lon is None

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    gps_server._latest = {"lat": 45.0, "lon": -90.0, "timestamp": now_iso}
    assert gps_server.lat_lon == (45.0, -90.0)

@pytest.mark.asyncio
async def test_start_stop(gps_server):
    with patch("bantz.core.gps_server._GPSHTTPServer") as mock_http_server, \
         patch("threading.Thread") as mock_thread, \
         patch("bantz.core.gps_server._get_local_ip", return_value="127.0.0.1"), \
         patch.object(gps_server, "_send_command") as mock_send_command, \
         patch.object(gps_server, "_start_relay_listener") as mock_start_relay:

        mock_server_instance = MagicMock()
        mock_http_server.return_value = mock_server_instance

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Start
        success = await gps_server.start()
        assert success is True
        mock_http_server.assert_called_once()
        assert mock_thread.call_count == 1
        mock_thread_instance.start.assert_called_once()
        mock_start_relay.assert_called_once()
        mock_send_command.assert_called_with("start")

        # Stop
        await gps_server.stop()
        mock_send_command.assert_called_with("stop")
        assert gps_server._relay_running is False
        mock_server_instance.shutdown.assert_called_once()
        assert gps_server._server is None
        assert gps_server._thread is None

def test_send_command(gps_server, mock_paths):
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("bantz.core.gps_server.getattr", return_value="test-topic"):
        gps_server._send_command("start")
        mock_urlopen.assert_called_once()
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        assert req.full_url == f"{NTFY_BASE}/test-topic-cmd"
        assert json.loads(req.data.decode()) == {"command": "start"}
        assert req.method == "POST"

def test_relay_listener_loop(gps_server, mock_paths):
    gps_server._relay_running = True

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp

    msg_data = {
        "event": "message",
        "message": json.dumps({"lat": 40.0, "lon": -74.0})
    }

    def fake_iter():
        yield json.dumps(msg_data).encode("utf-8") + b"\n"
        gps_server._relay_running = False
        yield b"should not be processed\n"

    mock_resp.__iter__.return_value = fake_iter()

    with patch("urllib.request.urlopen", return_value=mock_resp), \
         patch.object(gps_server, "_save_location") as mock_save, \
         patch("bantz.core.gps_server.getattr", return_value="test-topic"):
        gps_server._relay_listener_loop()

        mock_save.assert_called_once_with({"lat": 40.0, "lon": -74.0})

def test_url_and_status_line(gps_server, mock_paths):
    with patch("bantz.core.gps_server._get_local_ip", return_value="10.0.0.5"), \
         patch("bantz.core.gps_server.getattr", return_value="my-test-topic"):
        assert gps_server.url == "http://10.0.0.5:9999"

        assert "GPS: waiting" in gps_server.status_line()
        assert "http://10.0.0.5:9999" in gps_server.status_line()
        assert "relay:my-test-topic" in gps_server.status_line()

        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        gps_server._latest = {"lat": 40.12345, "lon": -74.98765, "accuracy": 12.3, "timestamp": now_iso}

        status = gps_server.status_line()
        assert "GPS: 40.1234, -74.9877 (±12m)" in status
        assert "relay:my-test-topic" in status
