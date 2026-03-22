"""Event-driven application state detector (#127, #220 Sprint 3 Part 4).

Emits ``app_changed`` on the EventBus when the focused application changes.
Backends tried in order: X11 PropertyNotify → D-Bus (GNOME/KWin) → slow-poll (5 s).
"""
from __future__ import annotations

import json, logging, os, re, subprocess, threading, time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from bantz.core.event_bus import bus

log = logging.getLogger(__name__)

class Activity(str, Enum):
    CODING = "coding"; BROWSING = "browsing"; ENTERTAINMENT = "entertainment"
    COMMUNICATION = "communication"; PRODUCTIVITY = "productivity"; IDLE = "idle"

@dataclass
class WindowInfo:
    name: str = ""; title: str = ""; pid: int = 0; wm_class: str = ""
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "pid": self.pid, "wm_class": self.wm_class}

@dataclass
class _CachedResult:
    value: Any = None; timestamp: float = 0.0
    def is_valid(self, ttl: float) -> bool:
        return self.value is not None and (time.time() - self.timestamp) < ttl

# -- Classification maps (built via comprehension for compactness) ----------
_A = Activity
_APP_CATEGORIES: dict[str, Activity] = {k: v for names, v in [
    (["code", "code - oss", "visual studio code", "vscodium", "neovim", "vim",
      "nvim", "emacs", "jetbrains", "intellij", "pycharm", "webstorm", "clion",
      "goland", "sublime_text", "kate", "gedit", "gnome-terminal", "konsole",
      "alacritty", "kitty", "wezterm", "foot", "xterm", "tilix", "terminator",
      "terminal", "tmux", "gitk", "meld", "docker desktop"], _A.CODING),
    (["firefox", "mozilla firefox", "chromium", "google-chrome", "brave",
      "brave-browser", "vivaldi", "opera", "epiphany", "qutebrowser",
      "midori"], _A.BROWSING),
    (["vlc", "mpv", "totem", "celluloid", "spotify", "rhythmbox", "elisa",
      "amberol", "steam", "lutris", "heroic"], _A.ENTERTAINMENT),
    (["thunderbird", "evolution", "geary", "telegram-desktop", "telegram",
      "discord", "slack", "signal", "teams", "microsoft teams", "zoom",
      "skype", "element"], _A.COMMUNICATION),
    (["libreoffice", "soffice", "lowriter", "localc", "loimpress", "obsidian",
      "notion", "joplin", "zettlr", "evince", "okular", "nautilus", "dolphin",
      "thunar", "nemo", "gnome-calculator", "gnome-calendar"], _A.PRODUCTIVITY),
] for k in names}

_BROWSER_TITLE_PATTERNS: list[tuple[re.Pattern, Activity, str]] = [
    (re.compile(p, re.I), a, c) for p, a, c in [
    (r"youtube", _A.ENTERTAINMENT, "watching YouTube"),
    (r"netflix", _A.ENTERTAINMENT, "watching Netflix"),
    (r"twitch\.tv", _A.ENTERTAINMENT, "watching Twitch"),
    (r"disney\+|disneyplus", _A.ENTERTAINMENT, "watching Disney+"),
    (r"prime video|primevideo", _A.ENTERTAINMENT, "watching Prime Video"),
    (r"spotify", _A.ENTERTAINMENT, "listening on Spotify"),
    (r"reddit\.com", _A.ENTERTAINMENT, "browsing Reddit"),
    (r"github\.com", _A.CODING, "on GitHub"),
    (r"gitlab\.com", _A.CODING, "on GitLab"),
    (r"stackoverflow\.com|stackexchange", _A.CODING, "on StackOverflow"),
    (r"docs\.python\.org", _A.CODING, "reading Python docs"),
    (r"developer\.mozilla|mdn", _A.CODING, "reading MDN"),
    (r"crates\.io|docs\.rs", _A.CODING, "reading Rust docs"),
    (r"pkg\.go\.dev", _A.CODING, "reading Go docs"),
    (r"npmjs\.com", _A.CODING, "on npm"),
    (r"mail\.google|gmail", _A.COMMUNICATION, "reading email"),
    (r"outlook\.(com|live)", _A.COMMUNICATION, "reading email"),
    (r"web\.whatsapp|whatsapp", _A.COMMUNICATION, "on WhatsApp"),
    (r"web\.telegram", _A.COMMUNICATION, "on Telegram"),
    (r"discord\.com", _A.COMMUNICATION, "on Discord"),
    (r"slack\.com", _A.COMMUNICATION, "on Slack"),
    (r"teams\.microsoft|teams\.live", _A.COMMUNICATION, "on Teams"),
    (r"meet\.google", _A.COMMUNICATION, "in Google Meet"),
    (r"zoom\.us", _A.COMMUNICATION, "in Zoom meeting"),
    (r"docs\.google|sheets\.google|slides\.google", _A.PRODUCTIVITY, "on Google Docs"),
    (r"notion\.so", _A.PRODUCTIVITY, "on Notion"),
    (r"trello\.com", _A.PRODUCTIVITY, "on Trello"),
    (r"calendar\.google", _A.PRODUCTIVITY, "checking calendar"),
]]

_BROWSER_NAMES = {"firefox", "mozilla firefox", "chromium", "google-chrome",
                  "brave", "brave-browser", "vivaldi", "opera", "epiphany",
                  "qutebrowser", "midori"}
_VSCODE_TITLE_RE = re.compile(
    r"^(?:●\s)?(.+?)\s+[-–—]\s+(.+?)\s+[-–—]\s+(?:Visual Studio Code|Code(?:\s*-\s*OSS)?)")
_JETBRAINS_TITLE_RE = re.compile(r"^(.+?)\s+[-–—]\s+(.+?)\s+[-–—]\s+\[(.+?)\]")
_TERMINAL_CWD_RE = re.compile(r"^(.+?)(?::\s*~?(/\S+))?$")

# -- Display server ---------------------------------------------------------
def _detect_display_server() -> str:
    s = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if s == "wayland": return "wayland"
    if s == "x11": return "x11"
    if os.environ.get("WAYLAND_DISPLAY"): return "wayland"
    if os.environ.get("DISPLAY"): return "x11"
    return "unknown"

def _run_cmd(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

# -- X11 backend ------------------------------------------------------------
def _x11_active_window() -> Optional[WindowInfo]:
    wid = _run_cmd(["xdotool", "getactivewindow"])
    if not wid: return None
    name = _run_cmd(["xdotool", "getactivewindow", "getwindowname"]) or ""
    pid_s = _run_cmd(["xdotool", "getactivewindow", "getwindowpid"]) or "0"
    wm = _run_cmd(["xprop", "-id", wid, "WM_CLASS"]) or ""
    m = re.search(r'"(.+?)"', wm)
    wmc = m.group(1) if m else ""
    app = wmc or name.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()
    return WindowInfo(name=app, title=name, pid=int(pid_s) if pid_s.isdigit() else 0, wm_class=wmc)

def _x11_running_apps() -> list[str]:
    out = _run_cmd(["wmctrl", "-l"])
    if out:
        apps: set[str] = set()
        for ln in out.splitlines():
            p = ln.split(None, 3)
            if len(p) >= 4:
                a = p[3].rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()
                if a: apps.add(a)
        return sorted(apps)
    out = _run_cmd(["xdotool", "search", "--name", ""])
    if out:
        apps = set()
        for wid in out.splitlines()[:50]:
            n = _run_cmd(["xdotool", "getwindowname", wid])
            if n:
                a = n.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()
                if a: apps.add(a)
        return sorted(apps)
    return []

# -- Wayland backend --------------------------------------------------------
def _wayland_active_window() -> Optional[WindowInfo]:
    out = _run_cmd(["swaymsg", "-t", "get_tree", "--raw"])
    if out: return _parse_sway_focused(out)
    out = _run_cmd(["hyprctl", "activewindow", "-j"])
    if out: return _parse_hyprctl(out)
    return None

def _parse_sway_focused(tree_json: str) -> Optional[WindowInfo]:
    try: tree = json.loads(tree_json)
    except json.JSONDecodeError: return None
    def _find(n: dict) -> Optional[dict]:
        if n.get("focused"): return n
        for ch in n.get("nodes", []) + n.get("floating_nodes", []):
            r = _find(ch)
            if r: return r
        return None
    f = _find(tree)
    if not f: return None
    aid = f.get("app_id") or ""
    title = f.get("name") or ""
    wmc = f.get("window_properties", {}).get("class", "") or aid
    app = aid or wmc or title.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()
    return WindowInfo(name=app, title=title, pid=f.get("pid", 0), wm_class=wmc)

def _parse_hyprctl(output: str) -> Optional[WindowInfo]:
    try: d = json.loads(output)
    except json.JSONDecodeError: return None
    return WindowInfo(name=d.get("class", "") or d.get("initialClass", ""),
                      title=d.get("title", ""), pid=d.get("pid", 0),
                      wm_class=d.get("class", ""))

def _wayland_running_apps() -> list[str]:
    out = _run_cmd(["swaymsg", "-t", "get_tree", "--raw"])
    if out: return _parse_sway_apps(out)
    out = _run_cmd(["hyprctl", "clients", "-j"])
    if out: return _parse_hyprctl_clients(out)
    return []

def _parse_sway_apps(tree_json: str) -> list[str]:
    try: tree = json.loads(tree_json)
    except json.JSONDecodeError: return []
    apps: set[str] = set()
    def _walk(n: dict) -> None:
        aid = n.get("app_id") or n.get("window_properties", {}).get("class", "")
        if aid and n.get("type") in ("con", "floating_con"): apps.add(aid)
        for ch in n.get("nodes", []) + n.get("floating_nodes", []): _walk(ch)
    _walk(tree)
    return sorted(apps)

def _parse_hyprctl_clients(output: str) -> list[str]:
    try: clients = json.loads(output)
    except json.JSONDecodeError: return []
    return sorted({c.get("class", "") or c.get("initialClass", "")
                    for c in clients if c.get("class") or c.get("initialClass")})

# -- AT-SPI / /proc fallbacks -----------------------------------------------
def _atspi_active_window() -> Optional[WindowInfo]:
    try:
        from bantz.tools.accessibility import _init_atspi, _atspi, _get_desktop
        if not _init_atspi(): return None
        desktop = _get_desktop()
        if not desktop: return None
        for i in range(desktop.get_child_count()):
            app = desktop.get_child_at_index(i)
            if not app: continue
            try:
                for j in range(app.get_child_count()):
                    win = app.get_child_at_index(j)
                    if not win: continue
                    ss = win.get_state_set()
                    if ss and ss.contains(_atspi.StateType.ACTIVE):
                        return WindowInfo(
                            name=app.get_name() or "", title=win.get_name() or "",
                            pid=app.get_process_id() if hasattr(app, "get_process_id") else 0)
            except Exception: continue
    except Exception as exc:
        log.debug("AT-SPI fallback failed: %s", exc)
    return None

def _atspi_running_apps() -> list[str]:
    try:
        from bantz.tools.accessibility import list_applications
        return list_applications()
    except Exception: return []

def _proc_running_apps() -> list[str]:
    known = {"firefox", "chromium", "chrome", "brave", "code", "vim", "nvim",
             "emacs", "thunderbird", "telegram-desktop", "discord", "spotify",
             "vlc", "mpv", "steam", "slack", "zoom", "obs", "gimp", "blender",
             "docker", "alacritty", "kitty", "wezterm", "foot", "konsole"}
    found: set[str] = set()
    try:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit(): continue
            try:
                c = (d / "comm").read_text().strip().lower()
                if c in known: found.add(c)
            except (OSError, PermissionError): continue
    except OSError: pass
    return sorted(found)

# -- Context parsers --------------------------------------------------------
def parse_browser_context(title: str) -> dict[str, str]:
    r: dict[str, str] = {"url_hint": "", "site": "", "activity": "browsing", "context": ""}
    if not title: return r
    parts = re.split(r"\s+[-–—|·]\s+", title)
    if len(parts) > 1:
        bp = parts[-1].strip().lower()
        if any(b in bp for b in ("firefox", "chrome", "chromium", "brave", "vivaldi", "opera")):
            parts = parts[:-1]
    if parts:
        r["url_hint"] = parts[0].strip()
        if len(parts) > 1: r["site"] = parts[-1].strip()
    for pat, act, ctx in _BROWSER_TITLE_PATTERNS:
        if pat.search(title):
            r["activity"], r["context"] = act.value, ctx; break
    gh = re.search(r"(\w+/\w+)\s*[-–—]\s*(?:GitHub|GitLab)", title)
    if gh:
        r["context"] = f"working on {gh.group(1)}"
        r["site"], r["activity"] = "GitHub", _A.CODING.value
    return r

def parse_ide_context(win: WindowInfo) -> dict[str, str]:
    r: dict[str, str] = {"ide": "", "project": "", "file": "", "branch": ""}
    nl = win.name.lower()
    if any(x in nl for x in ("code", "vscodium")):
        r["ide"] = "vscode"
        m = _VSCODE_TITLE_RE.match(win.title)
        if m: r["file"], r["project"] = m.group(1), m.group(2)
        if r["project"]: r["branch"] = _detect_git_branch(r["project"])
    elif any(x in nl for x in ("jetbrains", "intellij", "pycharm", "webstorm", "clion", "goland")):
        r["ide"] = nl
        m = _JETBRAINS_TITLE_RE.match(win.title)
        if m: r["file"], r["project"] = m.group(1), m.group(2)
    elif any(x in nl for x in ("terminal", "alacritty", "kitty", "wezterm", "foot",
                                "konsole", "tilix", "terminator", "gnome-terminal", "xterm")):
        r["ide"] = "terminal"
        if win.pid:
            cwd = _get_proc_cwd(win.pid)
            if cwd:
                r["project"] = cwd; r["branch"] = _detect_git_branch(cwd)
    return r

def _get_proc_cwd(pid: int) -> str:
    try: return os.readlink(f"/proc/{pid}/cwd")
    except (OSError, PermissionError): return ""

def _detect_git_branch(project_path: str) -> str:
    try:
        head = Path(project_path) / ".git" / "HEAD"
        if head.exists():
            c = head.read_text().strip()
            if c.startswith("ref: refs/heads/"): return c[16:]
        return ""
    except (OSError, PermissionError): return ""

def _get_docker_containers() -> list[dict[str, str]]:
    out = _run_cmd(["docker", "ps", "--format", "{{json .}}"])
    if not out: return []
    containers = []
    for ln in out.splitlines():
        try:
            c = json.loads(ln)
            containers.append({"name": c.get("Names", ""), "image": c.get("Image", ""),
                               "status": c.get("Status", "")})
        except json.JSONDecodeError: continue
    return containers

# -- Native event listeners (#220 Part 4) -----------------------------------
def _start_x11_listener(on_change: Callable[[], None], stop: threading.Event) -> bool:
    """X11 PropertyNotify listener via python-xlib. Zero CPU when idle."""
    try:
        from Xlib import X, display as xdisplay  # type: ignore[import-untyped]
    except ImportError:
        log.debug("python-xlib not available"); return False
    def _listen() -> None:
        try:
            import select
            d = xdisplay.Display(); root = d.screen().root
            net_active = d.intern_atom("_NET_ACTIVE_WINDOW")
            root.change_attributes(event_mask=X.PropertyChangeMask)
            log.debug("X11 PropertyNotify listener started")
            while not stop.is_set():
                if d.pending_events():
                    ev = d.next_event()
                    if ev.type == X.PropertyNotify and ev.atom == net_active: on_change()
                else:
                    rr, _, _ = select.select([d.fileno()], [], [], 1.0)
                    if rr and d.pending_events():
                        ev = d.next_event()
                        if ev.type == X.PropertyNotify and ev.atom == net_active: on_change()
        except Exception as exc:
            log.debug("X11 listener error: %s", exc)
    threading.Thread(target=_listen, name="bantz-x11-listener", daemon=True).start()
    return True

def _start_dbus_listener(on_change: Callable[[], None], stop: threading.Event) -> bool:
    """GNOME Shell / KWin D-Bus focus-change listener."""
    try:
        import asyncio as _aio
        from dbus_next.aio import MessageBus  # type: ignore[import-untyped]
    except ImportError:
        log.debug("dbus-next not available"); return False
    def _listen() -> None:
        async def _run() -> None:
            try:
                _dbus = await MessageBus().connect()
                try:
                    intro = await _dbus.introspect("org.gnome.Shell", "/org/gnome/Shell")
                    proxy = _dbus.get_proxy_object("org.gnome.Shell", "/org/gnome/Shell", intro)
                    proxy.get_interface("org.gnome.Shell").on_focus_app_changed(lambda _: on_change())
                    log.debug("D-Bus GNOME Shell listener active")
                except Exception: pass
                try:
                    intro = await _dbus.introspect("org.kde.KWin", "/KWin")
                    proxy = _dbus.get_proxy_object("org.kde.KWin", "/KWin", intro)
                    proxy.get_interface("org.kde.KWin").on_active_window_changed(lambda: on_change())
                    log.debug("D-Bus KWin listener active")
                except Exception: pass
                while not stop.is_set(): await _aio.sleep(1.0)
                _dbus.disconnect()
            except Exception as exc:
                log.debug("D-Bus listener error: %s", exc)
        loop = _aio.new_event_loop()
        try: loop.run_until_complete(_run())
        finally: loop.close()
    threading.Thread(target=_listen, name="bantz-dbus-listener", daemon=True).start()
    return True

def _start_slow_poll(on_change: Callable[[], None], stop: threading.Event,
                     interval: float = 5.0) -> bool:
    """Fallback: poll active window every *interval* seconds."""
    def _poll() -> None:
        last_wid = ""
        log.debug("Slow-poll fallback started (%.1fs interval)", interval)
        while not stop.is_set():
            stop.wait(interval)
            if stop.is_set(): break
            wid = _run_cmd(["xdotool", "getactivewindow"]) or ""
            if not wid:
                w = _wayland_active_window()
                wid = f"{w.pid}:{w.name}" if w else ""
            if wid != last_wid: last_wid = wid; on_change()
    threading.Thread(target=_poll, name="bantz-slow-poll", daemon=True).start()
    return True

# -- AppDetector main class -------------------------------------------------
class AppDetector:
    """Event-driven application state detector."""
    def __init__(self) -> None:
        self._display_server = "unknown"; self._cache_ttl = 5.0
        self._polling_interval = 5; self._initialized = False
        self._stop = threading.Event(); self._listener_type = "none"
        self._active_window_cache = _CachedResult()
        self._running_apps_cache = _CachedResult()
        self._docker_cache = _CachedResult()

    def init(self, *, cache_ttl: float = 5.0, polling_interval: int = 5) -> None:
        if self._initialized: return
        self._display_server = _detect_display_server()
        self._cache_ttl = cache_ttl; self._polling_interval = polling_interval
        self._initialized = True; self._start_listener()
        log.info("AppDetector: display=%s listener=%s", self._display_server, self._listener_type)

    def _start_listener(self) -> None:
        if self._display_server == "x11":
            if _start_x11_listener(self._on_window_change, self._stop):
                self._listener_type = "x11-propertynotify"; return
        if self._display_server == "wayland":
            if _start_dbus_listener(self._on_window_change, self._stop):
                self._listener_type = "dbus"; return
        _start_slow_poll(self._on_window_change, self._stop, float(self._polling_interval))
        self._listener_type = "slow-poll"

    def _on_window_change(self) -> None:
        self._active_window_cache = _CachedResult()
        win = self.get_active_window()
        data: dict[str, Any] = {"name": "", "title": "", "pid": 0,
                                "wm_class": "", "activity": _A.IDLE.value}
        if win: data.update(win.to_dict()); data["activity"] = self.classify_activity(win).value
        bus.emit_threadsafe("app_changed", **data)

    def stop(self) -> None:
        self._stop.set()

    @property
    def initialized(self) -> bool: return self._initialized
    @property
    def display_server(self) -> str: return self._display_server
    @property
    def listener_type(self) -> str: return self._listener_type

    def get_active_window(self) -> Optional[WindowInfo]:
        if not self._initialized: return None
        if self._active_window_cache.is_valid(self._cache_ttl):
            return self._active_window_cache.value
        win: Optional[WindowInfo] = None
        if self._display_server == "x11": win = _x11_active_window()
        elif self._display_server == "wayland": win = _wayland_active_window()
        if win is None: win = _atspi_active_window()
        self._active_window_cache = _CachedResult(value=win, timestamp=time.time())
        return win

    def get_running_apps(self) -> list[str]:
        if not self._initialized: return []
        if self._running_apps_cache.is_valid(self._cache_ttl):
            return self._running_apps_cache.value
        apps: list[str] = []
        if self._display_server == "x11": apps = _x11_running_apps()
        elif self._display_server == "wayland": apps = _wayland_running_apps()
        if not apps: apps = _atspi_running_apps()
        if not apps: apps = _proc_running_apps()
        seen: set[str] = set(); unique: list[str] = []
        for a in apps:
            k = a.lower()
            if k not in seen: seen.add(k); unique.append(a)
        self._running_apps_cache = _CachedResult(value=unique, timestamp=time.time())
        return unique

    def get_workspace_context(self) -> dict[str, Any]:
        if not self._initialized: return {}
        win = self.get_active_window()
        if not win: return {"activity": _A.IDLE.value, "apps": self.get_running_apps()}
        result: dict[str, Any] = {"active_window": win.to_dict(),
                                   "apps": self.get_running_apps(),
                                   "activity": self.classify_activity(win).value}
        ide_ctx = parse_ide_context(win)
        if ide_ctx["ide"]: result["ide"] = ide_ctx
        nl = win.name.lower()
        if nl in _BROWSER_NAMES or any(b in nl for b in _BROWSER_NAMES):
            result["browser"] = parse_browser_context(win.title)
        if not self._docker_cache.is_valid(30.0):
            self._docker_cache = _CachedResult(value=_get_docker_containers(), timestamp=time.time())
        if self._docker_cache.value: result["docker"] = self._docker_cache.value
        return result

    def classify_activity(self, win: Optional[WindowInfo] = None) -> Activity:
        if win is None: win = self.get_active_window()
        if win is None: return _A.IDLE
        nl = win.name.lower()
        if nl in _BROWSER_NAMES or any(b in nl for b in _BROWSER_NAMES):
            for pat, act, _ in _BROWSER_TITLE_PATTERNS:
                if pat.search(win.title): return act
            return _A.BROWSING
        for ak, act in _APP_CATEGORIES.items():
            if ak in nl or nl.startswith(ak.split()[0]): return act
        wl = win.wm_class.lower()
        if wl:
            for ak, act in _APP_CATEGORIES.items():
                if ak in wl: return act
        return _A.IDLE

    def get_activity_category(self) -> Activity:
        if not self._initialized: return _A.IDLE
        return self.classify_activity()

    def should_enable_focus(self, win: Optional[WindowInfo] = None) -> bool:
        return self.classify_activity(win) in (_A.CODING, _A.ENTERTAINMENT, _A.COMMUNICATION)

    def stats(self) -> dict[str, Any]:
        return {"initialized": self._initialized, "display_server": self._display_server,
                "cache_ttl": self._cache_ttl, "polling_interval": self._polling_interval,
                "listener_type": self._listener_type}

    def status_line(self) -> str:
        win = self.get_active_window(); activity = self.classify_activity(win)
        app_count = len(self.get_running_apps())
        return (f"display={self._display_server} active={win.name if win else 'none'} "
                f"activity={activity.value} apps={app_count}")

app_detector = AppDetector()
