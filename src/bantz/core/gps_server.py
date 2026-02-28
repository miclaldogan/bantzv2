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
from datetime import datetime, timezone
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
  <button class="btn-sec" style="margin-top:6px" onclick="location.href='/pwa'">
    &#x1F4F1; Open as PWA (installable)
  </button>
  <p class="hint">PWA: install on home screen for background tracking &amp; auto-start</p>

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
# v2: adaptive interval, battery indicator, PWA-ready, auto-start/stop (#74)

_PHONE_APP_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0d1117">
<link rel="manifest" href="/manifest.json">
<title>Bantz GPS Tracker</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
background:#0d1117;color:#e6edf3;display:flex;flex-direction:column;align-items:center;
justify-content:center;min-height:100vh;padding:16px;-webkit-user-select:none;user-select:none}
.c{background:#161b22;border:1px solid #30363d;border-radius:16px;
padding:28px;max-width:400px;width:100%;text-align:center}
h1{font-size:22px;margin-bottom:4px}
.sub{color:#8b949e;font-size:13px;margin-bottom:20px}
.btn{border:none;color:white;padding:16px 28px;border-radius:12px;
font-size:17px;cursor:pointer;width:100%;font-weight:700;transition:all 0.3s}
.btn-start{background:#238636}
.btn-start:hover{background:#2ea043}
.btn-stop{background:#da3633}
.btn-stop:hover{background:#f85149}
.btn:disabled{background:#30363d;cursor:not-allowed}
.live{display:none;margin-top:20px}
.live.on{display:block}
.pulse{display:inline-block;width:12px;height:12px;border-radius:50%;
background:#3fb950;margin-right:8px;vertical-align:middle;
animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
.stats{background:#0d1117;border-radius:10px;padding:14px;margin-top:14px;text-align:left}
.row{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;
border-bottom:1px solid #161b22}
.row:last-child{border:none}
.lbl{color:#8b949e}
.val{color:#e6edf3;font-family:monospace;font-size:12px}
.st{margin-top:14px;padding:10px;border-radius:8px;font-size:13px}
.ok{background:#0d1f0d;border:1px solid #238636;color:#3fb950}
.er{background:#1f0d0d;border:1px solid #da3633;color:#f85149}
.in{background:#0d1520;border:1px solid #1f6feb;color:#58a6ff}
.warn{background:#1f1a0d;border:1px solid #d29922;color:#e3b341}
.md{color:#6e7681;font-size:11px;margin-top:14px;line-height:1.4}
.cnt{font-size:32px;font-weight:700;color:#3fb950;margin:8px 0}
.batt{display:flex;align-items:center;gap:8px;justify-content:center;margin-top:10px}
.batt-bar{width:36px;height:16px;border:2px solid #8b949e;border-radius:3px;position:relative;
display:inline-block}
.batt-bar::after{content:'';position:absolute;right:-5px;top:3px;width:3px;height:8px;
background:#8b949e;border-radius:0 2px 2px 0}
.batt-fill{height:100%;border-radius:1px;transition:width 0.5s,background 0.5s}
.batt-text{font-size:12px;color:#8b949e;font-family:monospace}
.mode-tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;
font-weight:600;margin-left:8px}
.mode-moving{background:#0d2818;color:#3fb950;border:1px solid #238636}
.mode-stationary{background:#0d1520;color:#58a6ff;border:1px solid #1f6feb}
</style>
</head><body>
<div class="c">
<div style="font-size:44px;margin-bottom:12px">&#x1F4CD;</div>
<h1>Bantz GPS Tracker</h1>
<p class="sub">Adaptive location tracking</p>

<button class="btn btn-start" id="btn" onclick="toggle()">Start Tracking</button>

<div id="s"></div>

<div class="live" id="live">
  <div>
    <span class="pulse"></span>
    <span style="color:#3fb950;font-weight:600">LIVE TRACKING</span>
    <span class="mode-tag mode-stationary" id="modeTag">stationary</span>
  </div>
  <div class="cnt" id="cnt">0</div>
  <div style="color:#8b949e;font-size:12px">updates sent</div>

  <div class="batt" id="battRow" style="display:none">
    <div class="batt-bar"><div class="batt-fill" id="battFill" style="width:100%;background:#3fb950"></div></div>
    <span class="batt-text" id="battText">--%%</span>
  </div>

  <div class="stats">
    <div class="row"><span class="lbl">Latitude</span><span class="val" id="lat">-</span></div>
    <div class="row"><span class="lbl">Longitude</span><span class="val" id="lon">-</span></div>
    <div class="row"><span class="lbl">Accuracy</span><span class="val" id="acc">-</span></div>
    <div class="row"><span class="lbl">Speed</span><span class="val" id="spd">-</span></div>
    <div class="row"><span class="lbl">Interval</span><span class="val" id="ivl">-</span></div>
    <div class="row"><span class="lbl">Last sent</span><span class="val" id="ts">-</span></div>
    <div class="row"><span class="lbl">Next in</span><span class="val" id="nxt">-</span></div>
  </div>
</div>

<p class="md">Channel: %%RELAY_TOPIC%%<br>
Adaptive: 30s moving · 5min stationary</p>
</div>

<script>
var T='%%RELAY_TOPIC%%', N='https://ntfy.sh';
var watchId=null, sendIv=null, wakeLock=null, countdownIv=null, cmdIv=null;
var lastPos=null, sendCount=0;
var secondsLeft=0, currentInterval=60;
var batteryRef=null;

// Adaptive intervals: moving (speed > 1 m/s ≈ 3.6 km/h) vs stationary
var INTERVAL_MOVING=30, INTERVAL_STATIONARY=300;
var SPEED_THRESHOLD=1.0; // m/s

function isMoving(){
  if(!lastPos) return false;
  var sp=lastPos.coords.speed;
  return sp!==null && sp>=SPEED_THRESHOLD;
}

function adaptInterval(){
  var moving=isMoving();
  var ideal=moving?INTERVAL_MOVING:INTERVAL_STATIONARY;
  var tag=document.getElementById('modeTag');
  if(moving){
    tag.textContent='moving';
    tag.className='mode-tag mode-moving';
  } else {
    tag.textContent='stationary';
    tag.className='mode-tag mode-stationary';
  }
  document.getElementById('ivl').textContent=ideal+'s';
  if(ideal!==currentInterval){
    currentInterval=ideal;
    if(sendIv){clearInterval(sendIv);sendIv=null;}
    sendIv=setInterval(function(){if(lastPos) doSend();},currentInterval*1000);
    secondsLeft=Math.min(secondsLeft,currentInterval);
  }
}

function toggle(){
  if(watchId!==null) stopTracking();
  else startTracking();
}

function startTracking(){
  var b=document.getElementById('btn'), s=document.getElementById('s');
  if(!navigator.geolocation){
    s.innerHTML='<div class="st er">Geolocation not supported</div>';
    return;
  }
  b.textContent='Starting...'; b.disabled=true;
  s.innerHTML='<div class="st in">Requesting GPS permission...</div>';

  currentInterval=INTERVAL_STATIONARY;

  watchId=navigator.geolocation.watchPosition(
    function(pos){
      lastPos=pos;
      updateDisplay(pos);
      adaptInterval();
      if(sendCount===0) doSend();
    },
    function(err){
      s.innerHTML='<div class="st er">GPS: '+err.message+'</div>';
      b.disabled=false; b.textContent='Start Tracking';
      b.className='btn btn-start';
    },
    {enableHighAccuracy:true, timeout:20000, maximumAge:5000}
  );

  sendIv=setInterval(function(){
    if(lastPos) doSend();
  }, currentInterval*1000);

  secondsLeft=currentInterval;
  countdownIv=setInterval(function(){
    secondsLeft--;
    if(secondsLeft<=0) secondsLeft=currentInterval;
    document.getElementById('nxt').textContent=secondsLeft+'s';
  },1000);

  acquireWake();
  document.addEventListener('visibilitychange',function(){
    if(document.visibilityState==='visible' && watchId!==null) acquireWake();
  });

  initBattery();
  startCommandListener();

  b.disabled=false;
  b.textContent='Stop Tracking';
  b.className='btn btn-stop';
  s.innerHTML='';
  document.getElementById('live').className='live on';
}

function stopTracking(){
  var b=document.getElementById('btn');
  if(watchId!==null){navigator.geolocation.clearWatch(watchId);watchId=null;}
  if(sendIv){clearInterval(sendIv);sendIv=null;}
  if(countdownIv){clearInterval(countdownIv);countdownIv=null;}
  stopCommandListener();
  releaseWake();
  b.textContent='Start Tracking';
  b.className='btn btn-start';
  document.getElementById('live').className='live';
  document.getElementById('s').innerHTML='<div class="st in">Tracking stopped ('+sendCount+' updates sent)</div>';
}

function updateDisplay(pos){
  document.getElementById('lat').textContent=pos.coords.latitude.toFixed(6);
  document.getElementById('lon').textContent=pos.coords.longitude.toFixed(6);
  document.getElementById('acc').textContent='\\u00b1'+Math.round(pos.coords.accuracy)+'m';
  var sp=pos.coords.speed;
  document.getElementById('spd').textContent=(sp!==null&&sp>=0)?(sp*3.6).toFixed(1)+' km/h':'--';
}

function doSend(){
  if(!lastPos) return;
  var c=lastPos.coords;
  var d={lat:c.latitude,lon:c.longitude,accuracy:c.accuracy,
    altitude:c.altitude,speed:c.speed,timestamp:new Date().toISOString(),
    battery:batteryRef?Math.round(batteryRef.level*100):null,
    interval:currentInterval};
  fetch(N+'/'+T,{method:'POST',body:JSON.stringify(d)})
  .then(function(r){
    if(r.ok){
      sendCount++;
      document.getElementById('cnt').textContent=sendCount;
      document.getElementById('ts').textContent=new Date().toLocaleTimeString();
      secondsLeft=currentInterval;
    }
  }).catch(function(){});
}

// ── Battery API ──
function initBattery(){
  if(!navigator.getBattery) return;
  navigator.getBattery().then(function(batt){
    batteryRef=batt;
    updateBatteryUI();
    batt.addEventListener('levelchange',updateBatteryUI);
    batt.addEventListener('chargingchange',updateBatteryUI);
  }).catch(function(){});
}

function updateBatteryUI(){
  if(!batteryRef) return;
  var pct=Math.round(batteryRef.level*100);
  var charging=batteryRef.charging;
  document.getElementById('battRow').style.display='flex';
  document.getElementById('battText').textContent=pct+'%%'+(charging?' \\u26A1':'');
  var fill=document.getElementById('battFill');
  fill.style.width=pct+'%%';
  fill.style.background=pct>50?'#3fb950':pct>20?'#d29922':'#da3633';
  // Warn if low battery
  if(pct<=15 && !charging && watchId!==null){
    var s=document.getElementById('s');
    s.innerHTML='<div class="st warn">\\u26A0 Battery low ('+pct+'%%) — consider stopping</div>';
  }
}

// ── Command listener (auto-start/stop from Bantz) ──
function startCommandListener(){
  cmdIv=setInterval(pollCommands,30000);
  pollCommands();
}
function stopCommandListener(){
  if(cmdIv){clearInterval(cmdIv);cmdIv=null;}
}
function pollCommands(){
  fetch(N+'/'+T+'-cmd/json?poll=1',{headers:{'Accept':'application/json'}})
  .then(function(r){return r.text();})
  .then(function(txt){
    if(!txt.trim()) return;
    var lines=txt.trim().split('\\n');
    for(var i=0;i<lines.length;i++){
      try{
        var msg=JSON.parse(lines[i]);
        if(msg.event==='message'){
          var cmd=JSON.parse(msg.message||'{}');
          if(cmd.command==='stop' && watchId!==null) stopTracking();
        }
      }catch(e){}
    }
  }).catch(function(){});
}

// ── Wake Lock ──
async function acquireWake(){
  try{
    if('wakeLock' in navigator){
      wakeLock=await navigator.wakeLock.request('screen');
    }
  }catch(e){}
}
function releaseWake(){
  try{if(wakeLock){wakeLock.release();wakeLock=null;}}catch(e){}
}

// ── PWA Service Worker ──
if('serviceWorker' in navigator){
  navigator.serviceWorker.register('/sw.js').catch(function(){});
}

// ── Auto-start: check for pending start command on page load ──
(function(){
  fetch(N+'/'+T+'-cmd/json?poll=1',{headers:{'Accept':'application/json'}})
  .then(function(r){return r.text();})
  .then(function(txt){
    if(!txt.trim()) return;
    var lines=txt.trim().split('\\n');
    for(var i=lines.length-1;i>=0;i--){
      try{
        var msg=JSON.parse(lines[i]);
        if(msg.event==='message'){
          var cmd=JSON.parse(msg.message||'{}');
          if(cmd.command==='start' && watchId===null){
            startTracking();
            return;
          }
        }
      }catch(e){}
    }
  }).catch(function(){});
})();
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
        elif self.path == "/pwa":
            self._serve_pwa()
        elif self.path == "/manifest.json":
            self._serve_manifest()
        elif self.path == "/sw.js":
            self._serve_sw()
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

    def _serve_pwa(self):
        """Serve phone app as a normal page (PWA-installable, not download)."""
        srv = self.server.gps_server
        html = _PHONE_APP_TEMPLATE.replace("%%RELAY_TOPIC%%", srv.relay_topic)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_manifest(self):
        """PWA manifest for installability."""
        manifest = json.dumps({
            "name": "Bantz GPS Tracker",
            "short_name": "Bantz GPS",
            "description": "Adaptive GPS tracker for Bantz",
            "start_url": "/pwa",
            "display": "standalone",
            "background_color": "#0d1117",
            "theme_color": "#0d1117",
            "icons": [{
                "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📍</text></svg>",
                "sizes": "any",
                "type": "image/svg+xml",
            }],
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/manifest+json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(manifest.encode())

    def _serve_sw(self):
        """Minimal service worker for PWA offline + background sync."""
        sw_js = (
            "// Bantz GPS Service Worker\n"
            "const CACHE = 'bantz-gps-v1';\n"
            "self.addEventListener('install', e => {\n"
            "  e.waitUntil(caches.open(CACHE).then(c => c.addAll(['/pwa'])));\n"
            "  self.skipWaiting();\n"
            "});\n"
            "self.addEventListener('activate', e => {\n"
            "  e.waitUntil(self.clients.claim());\n"
            "});\n"
            "self.addEventListener('fetch', e => {\n"
            "  if(e.request.url.includes('ntfy.sh')) return;\n"
            "  e.respondWith(\n"
            "    caches.match(e.request).then(r => r || fetch(e.request))\n"
            "  );\n"
            "});\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript")
        self.send_header("Service-Worker-Allowed", "/")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(sw_js.encode())

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
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    age = (now - dt).total_seconds()
                    if age < TTL_SECONDS:
                        self._latest = data
                        log.debug(
                            "Loaded GPS from disk: %.4f, %.4f (age %ds)",
                            data.get("lat", 0), data.get("lon", 0), int(age),
                        )
            except Exception:
                pass

    def _save_location(self, data: dict) -> None:
        """Save received GPS data to memory and disk, feed place manager."""
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

        # Feed the place manager for geofence / stationary tracking
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is not None and lon is not None:
            try:
                from bantz.core.places import places
                transition = places.update_gps(lat, lon)
                if transition:
                    log.info("Place transition → %s", transition)
            except Exception:
                pass

    # ── Latest data ──────────────────────────────────────────────────

    @property
    def latest(self) -> Optional[dict]:
        """Latest GPS data, or None if expired / empty."""
        if not self._latest:
            return None
        ts = self._latest.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - dt).total_seconds() > TTL_SECONDS:
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

        # Signal phone app to start tracking (#74)
        self._send_command("start")
        return True

    async def stop(self) -> None:
        """Stop HTTP server and relay listener."""
        # Signal phone app to stop tracking (#74)
        self._send_command("stop")
        self._relay_running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        self._thread = None
        self._relay_thread = None

    def _send_command(self, command: str) -> None:
        """Publish a command to the phone app via ntfy.sh (#74)."""
        topic = f"{self.relay_topic}-cmd"
        url = f"{NTFY_BASE}/{topic}"
        data = json.dumps({"command": command}).encode()
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            log.info("Sent GPS command: %s → %s", command, topic)
        except Exception as exc:
            log.debug("GPS command send failed: %s", exc)

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
