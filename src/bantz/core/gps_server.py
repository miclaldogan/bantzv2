"""
Bantz v2 — Phone GPS Receiver

Tiny HTTP server that serves a web page to your phone.
The page uses the browser Geolocation API to get GPS coordinates,
then POSTs them back to Bantz.

Usage:
    from bantz.core.gps_server import gps_server

    # Start in background (non-blocking)
    await gps_server.start()          # listens on 0.0.0.0:9777

    # Read latest location
    loc = gps_server.latest           # {"lat": 40.55, "lon": 34.95, "acc": 12.3, "ts": "..."}

    # Stop
    await gps_server.stop()

Open http://<laptop-ip>:9777 on your phone (same WiFi network).
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.gps")

GPS_PORT = 9777
LOCATION_FILE = Path.home() / ".local" / "share" / "bantz" / "live_location.json"
TTL_SECONDS = 1800   # 30 minutes — location expires after this

# ── HTML page served to phone ──────────────────────────────────────────────

_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bantz GPS</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0d1117; color: #e6edf3;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 100vh; padding: 20px;
  }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 32px; max-width: 400px; width: 100%; text-align: center;
  }
  h1 { font-size: 24px; margin-bottom: 8px; }
  .subtitle { color: #8b949e; font-size: 14px; margin-bottom: 24px; }
  .btn {
    background: #238636; border: none; color: white; padding: 14px 28px;
    border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%;
    font-weight: 600; transition: background 0.2s;
  }
  .btn:hover { background: #2ea043; }
  .btn:disabled { background: #30363d; cursor: not-allowed; }
  .status {
    margin-top: 20px; padding: 12px; border-radius: 8px;
    font-size: 14px; line-height: 1.5;
  }
  .success { background: #0d1f0d; border: 1px solid #238636; color: #3fb950; }
  .error { background: #1f0d0d; border: 1px solid #da3633; color: #f85149; }
  .info { background: #0d1520; border: 1px solid #1f6feb; color: #58a6ff; }
  .coords { font-family: monospace; font-size: 13px; color: #8b949e; margin-top: 8px; }
  .icon { font-size: 48px; margin-bottom: 16px; }
  .auto-label { color: #8b949e; font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<div class="card">
  <div class="icon">📍</div>
  <h1>Bantz GPS</h1>
  <p class="subtitle">Share your location with Bantz</p>
  <button class="btn" id="btn" onclick="sendLocation()">Share Location</button>
  <div id="status"></div>
  <label style="display:block;margin-top:16px;cursor:pointer;">
    <input type="checkbox" id="autoRefresh" onchange="toggleAuto()">
    <span style="color:#8b949e;font-size:13px;">Auto-refresh every 5 min</span>
  </label>
</div>
<script>
let watchId = null;
let autoInterval = null;

function sendLocation() {
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  btn.textContent = 'Getting GPS...';
  status.innerHTML = '<div class="status info">Requesting location permission...</div>';

  if (!navigator.geolocation) {
    status.innerHTML = '<div class="status error">Geolocation not supported</div>';
    btn.disabled = false;
    btn.textContent = 'Share Location';
    return;
  }

  navigator.geolocation.getCurrentPosition(
    function(pos) {
      const data = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
        altitude: pos.coords.altitude,
        speed: pos.coords.speed,
        timestamp: new Date().toISOString()
      };

      fetch('/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      })
      .then(r => r.json())
      .then(resp => {
        status.innerHTML =
          '<div class="status success">✓ Location sent to Bantz!</div>' +
          '<div class="coords">' + data.lat.toFixed(6) + ', ' + data.lon.toFixed(6) +
          ' (±' + Math.round(data.accuracy) + 'm)</div>';
        btn.disabled = false;
        btn.textContent = 'Update Location';
      })
      .catch(err => {
        status.innerHTML = '<div class="status error">Failed to send: ' + err + '</div>';
        btn.disabled = false;
        btn.textContent = 'Retry';
      });
    },
    function(err) {
      status.innerHTML = '<div class="status error">GPS error: ' + err.message + '</div>';
      btn.disabled = false;
      btn.textContent = 'Retry';
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
  );
}

function toggleAuto() {
  if (document.getElementById('autoRefresh').checked) {
    sendLocation();
    autoInterval = setInterval(sendLocation, 300000); // 5 min
  } else {
    if (autoInterval) clearInterval(autoInterval);
    autoInterval = null;
  }
}
</script>
</body>
</html>
"""


class _GPSHandler(BaseHTTPRequestHandler):
    """HTTP request handler for GPS receiver."""

    server: "_GPSHTTPServer"

    def do_GET(self):
        if self.path == "/" or self.path == "/location":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(_HTML_PAGE.encode())
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            loc = self.server.gps_server.latest
            self.wfile.write(json.dumps(loc or {"status": "no location"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/update":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                self.server.gps_server._save_location(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())
            except Exception as exc:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class _GPSHTTPServer(HTTPServer):
    """HTTPServer subclass that holds a reference to GPSServer."""
    gps_server: "GPSServer"


class GPSServer:
    """Background HTTP server that receives GPS from phone browser."""

    def __init__(self, port: int = GPS_PORT) -> None:
        self._port = port
        self._server: Optional[_GPSHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[dict] = None
        # Load any existing location from disk
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load last known location from disk (if not expired)."""
        if LOCATION_FILE.exists():
            try:
                data = json.loads(LOCATION_FILE.read_text(encoding="utf-8"))
                ts = data.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts)
                    age = (datetime.now() - dt).total_seconds()
                    if age < TTL_SECONDS:
                        self._latest = data
                        log.debug("Loaded GPS from disk: %.4f, %.4f (age %ds)",
                                  data.get("lat", 0), data.get("lon", 0), int(age))
            except Exception:
                pass

    def _save_location(self, data: dict) -> None:
        """Save received GPS data to memory and disk."""
        self._latest = data
        LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCATION_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("GPS updated: %.6f, %.6f (±%sm)",
                 data.get("lat", 0), data.get("lon", 0),
                 round(data.get("accuracy", 0)))

    @property
    def latest(self) -> Optional[dict]:
        """Latest GPS data, or None if no location / expired."""
        if not self._latest:
            return None
        ts = self._latest.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                age = (datetime.now() - dt).total_seconds()
                if age > TTL_SECONDS:
                    return None
            except Exception:
                pass
        return self._latest

    @property
    def lat_lon(self) -> Optional[tuple[float, float]]:
        """Return (lat, lon) tuple or None."""
        loc = self.latest
        if loc and "lat" in loc and "lon" in loc:
            return (loc["lat"], loc["lon"])
        return None

    async def start(self) -> bool:
        """Start the GPS receiver server in a background thread."""
        if self._thread and self._thread.is_alive():
            return True
        try:
            self._server = _GPSHTTPServer(("0.0.0.0", self._port), _GPSHandler)
            self._server.gps_server = self
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="bantz-gps",
            )
            self._thread.start()
            ip = _get_local_ip()
            log.info("GPS server started: http://%s:%d", ip, self._port)
            return True
        except Exception as exc:
            log.warning("GPS server failed to start: %s", exc)
            return False

    async def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        ip = _get_local_ip()
        return f"http://{ip}:{self._port}"

    def status_line(self) -> str:
        """One-line summary for --doctor."""
        loc = self.latest
        if not loc:
            return f"GPS: no location (open {self.url} on phone)"
        lat, lon = loc.get("lat", 0), loc.get("lon", 0)
        acc = round(loc.get("accuracy", 0))
        ts = loc.get("timestamp", "?")
        return f"GPS: {lat:.4f}, {lon:.4f} (±{acc}m) @ {ts}"


def _get_local_ip() -> str:
    """Get the local network IP (for display in URL)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# Singleton
gps_server = GPSServer()
