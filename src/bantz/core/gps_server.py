"""
Bantz v2 — Phone GPS Receiver (LAN + Relay)

Two-mode GPS receiver:
1. Direct (LAN): HTTP server on port 9777, phone & laptop on same WiFi
2. Relay (any network): ntfy.sh pub/sub, phone can be on mobile data

First visit: open http://<laptop-ip>:9777 on phone (same WiFi).
Download the standalone app for cross-network use -> works from anywhere.

Usage:
    from bantz.core.gps_server import gps_server

    await gps_server.start()       # HTTP server + relay listener
    loc = gps_server.latest        # latest GPS dict
    await gps_server.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import threading
import time as _time
import urllib.request
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.gps")

GPS_PORT = 9777
NTFY_BASE = "https://ntfy.sh"
LOCATION_FILE = Path.home() / ".local" / "share" / "bantz" / "live_location.json"
TOKEN_FILE = Path.home() / ".local" / "share" / "bantz" / "gps_relay_token"
TTL_SECONDS = 1800  # 30 minutes


def _ensure_relay_token() -> str:
    """Load or generate a unique relay channel token."""
    # env-var override
    try:
        from bantz.config import config
        token = getattr(config, "gps_relay_token", "") or ""
        if token:
            return token
    except Exception:
        pass
    # persisted token
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = "bantz-gps-" + secrets.token_urlsafe(8)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    log.info("Generated relay token: %s", token)
    return token


# ── HTML: main page (served on LAN) ───────────────────────────────────────

_HTML_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0d1117">
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
    padding: 32px; max-width: 420px; width: 100%; text-align: center;
  }
  h1 { font-size: 24px; margin-bottom: 8px; }
  .subtitle { color: #8b949e; font-size: 14px; margin-bottom: 24px; }
  .btn {
    background: #238636; border: none; color: white; padding: 14px 28px;
    border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%;
    font-weight: 600; transition: background 0.2s; margin-bottom: 8px;
  }
  .btn:hover { background: #2ea043; }
  .btn:disabled { background: #30363d; cursor: not-allowed; }
  .btn-sec {
    background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    padding: 10px 20px; border-radius: 8px; font-size: 14px;
    cursor: pointer; width: 100%; transition: background 0.2s;
  }
  .btn-sec:hover { background: #30363d; }
  .status {
    margin-top: 16px; padding: 12px; border-radius: 8px;
    font-size: 14px; line-height: 1.5;
  }
  .success { background: #0d1f0d; border: 1px solid #238636; color: #3fb950; }
  .error { background: #1f0d0d; border: 1px solid #da3633; color: #f85149; }
  .info { background: #0d1520; border: 1px solid #1f6feb; color: #58a6ff; }
  .coords { font-family: monospace; font-size: 13px; color: #8b949e; margin-top: 8px; }
  .icon { font-size: 48px; margin-bottom: 16px; }
  .sep { border-top: 1px solid #21262d; margin: 20px 0; }
  .conn {
    text-align: left; font-size: 12px; color: #8b949e;
    background: #0d1117; border-radius: 8px; padding: 12px; margin-top: 16px;
  }
  .dot { display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  .dg { background: #3fb950; }
  .dd { background: #484f58; }
  .hint { color: #6e7681; font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<div class="card">
  <div class="icon">&#x1F4CD;</div>
  <h1>Bantz GPS</h1>
  <p class="subtitle">Share your location with Bantz</p>
  <button class="btn" id="btn" onclick="sendLocation()">Share Location</button>
  <div id="status"></div>
  <label style="display:block;margin-top:12px;cursor:pointer;">
    <input type="checkbox" id="autoRefresh" onchange="toggleAuto()">
    <span style="color:#8b949e;font-size:13px;">Auto-refresh every 5 min</span>
  </label>

  <div class="sep"></div>

  <button class="btn-sec" onclick="location.href='/app'">
    &#x1F4F2; Download Phone App
  </button>
  <p class="hint">Standalone app &mdash; works from any network, no server needed</p>

  <div class="conn">
    <div><span class="dot dg" id="dd"></span>
      Direct: <span id="du">%%DIRECT_URL%%</span></div>
    <div style="margin-top:4px"><span class="dot dg" id="dr"></span>
      Relay: <span id="rt">%%RELAY_TOPIC%%</span></div>
  </div>
</div>
<script>
var RELAY_TOPIC = '%%RELAY_TOPIC%%';
var NTFY = 'https://ntfy.sh';
var autoInterval = null;

function sendLocation() {
  var btn = document.getElementById('btn');
  var st  = document.getElementById('status');
  btn.disabled = true;
  btn.textContent = 'Getting GPS...';
  st.innerHTML = '<div class="status info">Requesting location...</div>';
  if (!navigator.geolocation) {
    st.innerHTML = '<div class="status error">Geolocation not supported</div>';
    btn.disabled = false; btn.textContent = 'Share Location'; return;
  }
  navigator.geolocation.getCurrentPosition(
    function(pos) {
      var data = {
        lat: pos.coords.latitude, lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy, altitude: pos.coords.altitude,
        speed: pos.coords.speed, timestamp: new Date().toISOString()
      };
      var directOk = false, relayOk = false;
      var body = JSON.stringify(data);
      // Direct (same network)
      var p1 = fetch('/update', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: body
      }).then(function(r){ if(r.ok) directOk=true; }).catch(function(){});
      // Relay (any network)
      var p2 = fetch(NTFY+'/'+RELAY_TOPIC, {
        method:'POST', body: body
      }).then(function(r){ if(r.ok) relayOk=true; }).catch(function(){});
      Promise.all([p1,p2]).then(function(){
        var via = [];
        if(directOk) via.push('direct');
        if(relayOk) via.push('relay');
        var viaStr = via.join('+') || 'failed';
        if (directOk || relayOk) {
          st.innerHTML =
            '<div class="status success">&#x2713; Location sent! (' + viaStr + ')</div>' +
            '<div class="coords">' + data.lat.toFixed(6) + ', ' + data.lon.toFixed(6) +
            ' (&plusmn;' + Math.round(data.accuracy) + 'm)</div>';
          btn.textContent = 'Update Location';
        } else {
          st.innerHTML = '<div class="status error">Failed to send location</div>';
          btn.textContent = 'Retry';
        }
        btn.disabled = false;
        document.getElementById('dd').className = 'dot ' + (directOk?'dg':'dd');
        document.getElementById('dr').className = 'dot ' + (relayOk?'dg':'dd');
      });
    },
    function(err) {
      st.innerHTML = '<div class="status error">GPS error: ' + err.message + '</div>';
      btn.disabled = false; btn.textContent = 'Retry';
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
  );
}
function toggleAuto() {
  if (document.getElementById('autoRefresh').checked) {
    sendLocation();
    autoInterval = setInterval(sendLocation, 300000);
  } else {
    if (autoInterval) clearInterval(autoInterval);
    autoInterval = null;
  }
}
</script>
</body>
</html>
"""

# ── Standalone phone app (downloaded as .html file) ──────────────────────

_PHONE_APP_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0d1117">
<title>Bantz GPS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
background:#0d1117;color:#e6edf3;display:flex;flex-direction:column;align-items:center;
justify-content:center;min-height:100vh;padding:20px}
.c{background:#161b22;border:1px solid #30363d;border-radius:12px;
padding:32px;max-width:400px;width:100%;text-align:center}
h1{font-size:22px;margin-bottom:6px}
.sub{color:#8b949e;font-size:13px;margin-bottom:20px}
.btn{background:#238636;border:none;color:white;padding:14px 28px;border-radius:8px;
font-size:16px;cursor:pointer;width:100%;font-weight:600}
.btn:disabled{background:#30363d;cursor:not-allowed}
.st{margin-top:16px;padding:12px;border-radius:8px;font-size:14px;line-height:1.5}
.ok{background:#0d1f0d;border:1px solid #238636;color:#3fb950}
.er{background:#1f0d0d;border:1px solid #da3633;color:#f85149}
.in{background:#0d1520;border:1px solid #1f6feb;color:#58a6ff}
.co{font-family:monospace;font-size:13px;color:#8b949e;margin-top:8px}
.md{color:#8b949e;font-size:11px;margin-top:16px}
</style>
</head><body>
<div class="c">
<div style="font-size:48px;margin-bottom:16px">&#x1F4CD;</div>
<h1>Bantz GPS</h1>
<p class="sub">Relay mode &mdash; works from any network</p>
<button class="btn" id="b" onclick="go()">Share Location</button>
<div id="s"></div>
<label style="display:block;margin-top:12px;cursor:pointer">
<input type="checkbox" id="au" onchange="tog()">
<span style="color:#8b949e;font-size:13px">Auto-refresh (5 min)</span>
</label>
<p class="md">Channel: %%RELAY_TOPIC%%</p>
</div>
<script>
var T='%%RELAY_TOPIC%%',N='https://ntfy.sh';var iv=null;
function go(){
  var b=document.getElementById('b'),s=document.getElementById('s');
  b.disabled=true;b.textContent='Getting GPS...';
  s.innerHTML='<div class="st in">Requesting location...</div>';
  if(!navigator.geolocation){
    s.innerHTML='<div class="st er">Geolocation not supported</div>';
    b.disabled=false;b.textContent='Share Location';return;
  }
  navigator.geolocation.getCurrentPosition(
    function(p){
      var d={lat:p.coords.latitude,lon:p.coords.longitude,
        accuracy:p.coords.accuracy,altitude:p.coords.altitude,
        speed:p.coords.speed,timestamp:new Date().toISOString()};
      fetch(N+'/'+T,{method:'POST',body:JSON.stringify(d)})
      .then(function(r){
        if(r.ok){
          s.innerHTML='<div class="st ok">&#x2713; Sent to Bantz!</div>'+
            '<div class="co">'+d.lat.toFixed(6)+', '+d.lon.toFixed(6)+
            ' (&plusmn;'+Math.round(d.accuracy)+'m)</div>';
          b.textContent='Update';
        } else {
          s.innerHTML='<div class="st er">Server error</div>';
          b.textContent='Retry';
        }
        b.disabled=false;
      })
      .catch(function(e){
        s.innerHTML='<div class="st er">Send failed: '+e+'</div>';
        b.disabled=false;b.textContent='Retry';
      });
    },
    function(e){
      s.innerHTML='<div class="st er">GPS: '+e.message+'</div>';
      b.disabled=false;b.textContent='Retry';
    },
    {enableHighAccuracy:true,timeout:15000,maximumAge:0}
  );
}
function tog(){
  if(document.getElementById('au').checked){go();iv=setInterval(go,300000);}
  else{if(iv)clearInterval(iv);iv=null;}
}
</script>
</body></html>
"""


# ── HTTP Handler ──────────────────────────────────────────────────────────

class _GPSHandler(BaseHTTPRequestHandler):
    """HTTP handler: GPS page, phone-app download, direct GPS updates."""

    server: "_GPSHTTPServer"

    def do_GET(self):
        if self.path in ("/", "/location"):
            self._serve_main_page()
        elif self.path == "/app":
            self._serve_phone_app()
        elif self.path == "/status":
            loc = self.server.gps_server.latest
            self._json_response(loc or {"status": "no location"})
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_main_page(self):
        srv = self.server.gps_server
        html = (_HTML_PAGE_TEMPLATE
                .replace("%%RELAY_TOPIC%%", srv.relay_topic)
                .replace("%%DIRECT_URL%%", srv.url))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_phone_app(self):
        srv = self.server.gps_server
        html = _PHONE_APP_TEMPLATE.replace("%%RELAY_TOPIC%%", srv.relay_topic)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header(
            "Content-Disposition",
            'attachment; filename="bantz-gps.html"',
        )
        self.end_headers()
        self.wfile.write(html.encode())

    def _json_response(self, data: dict, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        if self.path == "/update":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                self.server.gps_server._save_location(data)
                self._json_response({"ok": True})
            except Exception as exc:
                self._json_response({"error": str(exc)}, 400)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


class _GPSHTTPServer(HTTPServer):
    """HTTPServer that holds a reference to the GPSServer instance."""
    gps_server: "GPSServer"


# ── GPS Server ────────────────────────────────────────────────────────────

class GPSServer:
    """GPS receiver: LAN HTTP server + ntfy.sh relay listener."""

    def __init__(self, port: int = GPS_PORT) -> None:
        self._port = port
        self._server: Optional[_GPSHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._relay_thread: Optional[threading.Thread] = None
        self._relay_running = False
        self._latest: Optional[dict] = None
        self._relay_token: Optional[str] = None
        self._load_from_disk()

    # ── Relay token ──────────────────────────────────────────────────

    @property
    def relay_topic(self) -> str:
        """Unique ntfy.sh topic for this Bantz instance."""
        if not self._relay_token:
            self._relay_token = _ensure_relay_token()
        return self._relay_token

    @property
    def relay_url(self) -> str:
        """Full ntfy.sh URL for the relay channel."""
        return f"{NTFY_BASE}/{self.relay_topic}"

    # ── Disk persistence ─────────────────────────────────────────────

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
                        log.debug(
                            "Loaded GPS from disk: %.4f, %.4f (age %ds)",
                            data.get("lat", 0), data.get("lon", 0), int(age),
                        )
            except Exception:
                pass

    def _save_location(self, data: dict) -> None:
        """Save received GPS data to memory and disk."""
        self._latest = data
        LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCATION_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(
            "GPS updated: %.6f, %.6f (±%sm)",
            data.get("lat", 0), data.get("lon", 0),
            round(data.get("accuracy", 0)),
        )

    # ── Latest data ──────────────────────────────────────────────────

    @property
    def latest(self) -> Optional[dict]:
        """Latest GPS data, or None if expired / empty."""
        if not self._latest:
            return None
        ts = self._latest.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if (datetime.now() - dt).total_seconds() > TTL_SECONDS:
                    return None
            except Exception:
                pass
        return self._latest

    @property
    def lat_lon(self) -> Optional[tuple[float, float]]:
        """Return (lat, lon) or None."""
        loc = self.latest
        if loc and "lat" in loc and "lon" in loc:
            return (loc["lat"], loc["lon"])
        return None

    # ── Start / stop ─────────────────────────────────────────────────

    async def start(self) -> bool:
        """Start HTTP server + relay listener."""
        if self._thread and self._thread.is_alive():
            return True
        try:
            self._server = _GPSHTTPServer(("0.0.0.0", self._port), _GPSHandler)
            self._server.gps_server = self
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True, name="bantz-gps",
            )
            self._thread.start()
            ip = _get_local_ip()
            log.info("GPS server: http://%s:%d", ip, self._port)
        except Exception as exc:
            log.warning("GPS HTTP server failed: %s", exc)
            return False

        # Start relay listener
        self._start_relay_listener()
        log.info("Relay topic: %s", self.relay_topic)
        return True

    async def stop(self) -> None:
        """Stop HTTP server and relay listener."""
        self._relay_running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        self._thread = None
        self._relay_thread = None

    # ── Relay listener ───────────────────────────────────────────────

    def _start_relay_listener(self) -> None:
        """Start background thread that listens to ntfy.sh."""
        self._relay_running = True
        self._relay_thread = threading.Thread(
            target=self._relay_listener_loop, daemon=True, name="bantz-gps-relay",
        )
        self._relay_thread.start()

    def _relay_listener_loop(self) -> None:
        """Subscribe to ntfy.sh JSON stream; reconnect on errors."""
        topic = self.relay_topic
        url = f"{NTFY_BASE}/{topic}/json"

        while self._relay_running:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Bantz/2.0"}
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    for raw in resp:
                        if not self._relay_running:
                            return
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            if msg.get("event") == "message":
                                gps = json.loads(msg.get("message", "{}"))
                                if "lat" in gps and "lon" in gps:
                                    self._save_location(gps)
                                    log.info(
                                        "Relay GPS: %.6f, %.6f",
                                        gps.get("lat", 0), gps.get("lon", 0),
                                    )
                        except (json.JSONDecodeError, ValueError):
                            pass
            except Exception as exc:
                if not self._relay_running:
                    return
                log.debug("Relay reconnect in 15s: %s", exc)
                for _ in range(15):
                    if not self._relay_running:
                        return
                    _time.sleep(1)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def url(self) -> str:
        """LAN URL for phone access."""
        return f"http://{_get_local_ip()}:{self._port}"

    def status_line(self) -> str:
        """One-line summary for --doctor."""
        loc = self.latest
        relay = f"relay:{self.relay_topic}"
        if not loc:
            return f"GPS: waiting ({self.url} | {relay})"
        lat, lon = loc.get("lat", 0), loc.get("lon", 0)
        acc = round(loc.get("accuracy", 0))
        return f"GPS: {lat:.4f}, {lon:.4f} (±{acc}m) | {relay}"


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
