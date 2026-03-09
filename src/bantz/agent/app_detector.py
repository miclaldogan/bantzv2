"""
Bantz — Application state detector (#127).

Detects currently open applications, active window, workspace context,
and user activity category for the RL engine's state vector.

Architecture:
    Display server detection → X11 (xdotool) / Wayland (swaymsg/hyprctl)
    Fallback → AT-SPI (#119) → /proc scanning

    Active window → title parsing → Activity category
    Running apps  → dedup + categorise
    IDE context   → project path + branch

    5-second TTL cache on all queries to avoid excessive polling.

Activity categories for RL state vector:
    CODING       — IDE, terminal, git
    BROWSING     — web browser (non-media)
    ENTERTAINMENT— media players, YouTube, gaming
    COMMUNICATION— email, chat, video calls
    PRODUCTIVITY — office suite, notes, planning
    IDLE         — screensaver, lock screen, no focus

Usage:
    from bantz.agent.app_detector import app_detector

    app_detector.init()
    win = app_detector.get_active_window()
    apps = app_detector.get_running_apps()
    ctx = app_detector.get_workspace_context()
    cat = app_detector.get_activity_category()
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Activity categories (for RL state vector)
# ═══════════════════════════════════════════════════════════════════════════


class Activity(str, Enum):
    """High-level activity category for RL state space."""
    CODING = "coding"
    BROWSING = "browsing"
    ENTERTAINMENT = "entertainment"
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    IDLE = "idle"


# ═══════════════════════════════════════════════════════════════════════════
# Window info
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WindowInfo:
    """Information about a single window."""
    name: str = ""         # Application name (e.g. "Firefox", "Code")
    title: str = ""        # Full window title
    pid: int = 0
    wm_class: str = ""     # X11 WM_CLASS or app identifier

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "pid": self.pid, "wm_class": self.wm_class}


# ═══════════════════════════════════════════════════════════════════════════
# Cached result wrapper
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _CachedResult:
    value: Any = None
    timestamp: float = 0.0

    def is_valid(self, ttl: float) -> bool:
        return self.value is not None and (time.time() - self.timestamp) < ttl


# ═══════════════════════════════════════════════════════════════════════════
# App classification maps
# ═══════════════════════════════════════════════════════════════════════════

# App name (lowercased) → Activity category
_APP_CATEGORIES: dict[str, Activity] = {
    # Coding
    "code": Activity.CODING,
    "code - oss": Activity.CODING,
    "visual studio code": Activity.CODING,
    "vscodium": Activity.CODING,
    "neovim": Activity.CODING,
    "vim": Activity.CODING,
    "nvim": Activity.CODING,
    "emacs": Activity.CODING,
    "jetbrains": Activity.CODING,
    "intellij": Activity.CODING,
    "pycharm": Activity.CODING,
    "webstorm": Activity.CODING,
    "clion": Activity.CODING,
    "goland": Activity.CODING,
    "sublime_text": Activity.CODING,
    "kate": Activity.CODING,
    "gedit": Activity.CODING,
    "gnome-terminal": Activity.CODING,
    "konsole": Activity.CODING,
    "alacritty": Activity.CODING,
    "kitty": Activity.CODING,
    "wezterm": Activity.CODING,
    "foot": Activity.CODING,
    "xterm": Activity.CODING,
    "tilix": Activity.CODING,
    "terminator": Activity.CODING,
    "terminal": Activity.CODING,
    "tmux": Activity.CODING,
    "gitk": Activity.CODING,
    "meld": Activity.CODING,
    "docker desktop": Activity.CODING,

    # Browsing (default for browsers — media detection overrides)
    "firefox": Activity.BROWSING,
    "mozilla firefox": Activity.BROWSING,
    "chromium": Activity.BROWSING,
    "google-chrome": Activity.BROWSING,
    "brave": Activity.BROWSING,
    "brave-browser": Activity.BROWSING,
    "vivaldi": Activity.BROWSING,
    "opera": Activity.BROWSING,
    "epiphany": Activity.BROWSING,
    "qutebrowser": Activity.BROWSING,
    "midori": Activity.BROWSING,

    # Entertainment
    "vlc": Activity.ENTERTAINMENT,
    "mpv": Activity.ENTERTAINMENT,
    "totem": Activity.ENTERTAINMENT,
    "celluloid": Activity.ENTERTAINMENT,
    "spotify": Activity.ENTERTAINMENT,
    "rhythmbox": Activity.ENTERTAINMENT,
    "elisa": Activity.ENTERTAINMENT,
    "amberol": Activity.ENTERTAINMENT,
    "steam": Activity.ENTERTAINMENT,
    "lutris": Activity.ENTERTAINMENT,
    "heroic": Activity.ENTERTAINMENT,

    # Communication
    "thunderbird": Activity.COMMUNICATION,
    "evolution": Activity.COMMUNICATION,
    "geary": Activity.COMMUNICATION,
    "telegram-desktop": Activity.COMMUNICATION,
    "telegram": Activity.COMMUNICATION,
    "discord": Activity.COMMUNICATION,
    "slack": Activity.COMMUNICATION,
    "signal": Activity.COMMUNICATION,
    "teams": Activity.COMMUNICATION,
    "microsoft teams": Activity.COMMUNICATION,
    "zoom": Activity.COMMUNICATION,
    "skype": Activity.COMMUNICATION,
    "element": Activity.COMMUNICATION,

    # Productivity
    "libreoffice": Activity.PRODUCTIVITY,
    "soffice": Activity.PRODUCTIVITY,
    "lowriter": Activity.PRODUCTIVITY,
    "localc": Activity.PRODUCTIVITY,
    "loimpress": Activity.PRODUCTIVITY,
    "obsidian": Activity.PRODUCTIVITY,
    "notion": Activity.PRODUCTIVITY,
    "joplin": Activity.PRODUCTIVITY,
    "zettlr": Activity.PRODUCTIVITY,
    "evince": Activity.PRODUCTIVITY,
    "okular": Activity.PRODUCTIVITY,
    "nautilus": Activity.PRODUCTIVITY,
    "dolphin": Activity.PRODUCTIVITY,
    "thunar": Activity.PRODUCTIVITY,
    "nemo": Activity.PRODUCTIVITY,
    "gnome-calculator": Activity.PRODUCTIVITY,
    "gnome-calendar": Activity.PRODUCTIVITY,
}

# Browser title patterns → Activity override
_BROWSER_TITLE_PATTERNS: list[tuple[re.Pattern, Activity, str]] = [
    # Entertainment (media)
    (re.compile(r"youtube", re.I), Activity.ENTERTAINMENT, "watching YouTube"),
    (re.compile(r"netflix", re.I), Activity.ENTERTAINMENT, "watching Netflix"),
    (re.compile(r"twitch\.tv", re.I), Activity.ENTERTAINMENT, "watching Twitch"),
    (re.compile(r"disney\+|disneyplus", re.I), Activity.ENTERTAINMENT, "watching Disney+"),
    (re.compile(r"prime video|primevideo", re.I), Activity.ENTERTAINMENT, "watching Prime Video"),
    (re.compile(r"spotify", re.I), Activity.ENTERTAINMENT, "listening on Spotify"),
    (re.compile(r"reddit\.com", re.I), Activity.ENTERTAINMENT, "browsing Reddit"),

    # Coding (dev sites)
    (re.compile(r"github\.com", re.I), Activity.CODING, "on GitHub"),
    (re.compile(r"gitlab\.com", re.I), Activity.CODING, "on GitLab"),
    (re.compile(r"stackoverflow\.com|stackexchange", re.I), Activity.CODING, "on StackOverflow"),
    (re.compile(r"docs\.python\.org", re.I), Activity.CODING, "reading Python docs"),
    (re.compile(r"developer\.mozilla|mdn", re.I), Activity.CODING, "reading MDN"),
    (re.compile(r"crates\.io|docs\.rs", re.I), Activity.CODING, "reading Rust docs"),
    (re.compile(r"pkg\.go\.dev", re.I), Activity.CODING, "reading Go docs"),
    (re.compile(r"npmjs\.com", re.I), Activity.CODING, "on npm"),

    # Communication
    (re.compile(r"mail\.google|gmail", re.I), Activity.COMMUNICATION, "reading email"),
    (re.compile(r"outlook\.(com|live)", re.I), Activity.COMMUNICATION, "reading email"),
    (re.compile(r"web\.whatsapp|whatsapp", re.I), Activity.COMMUNICATION, "on WhatsApp"),
    (re.compile(r"web\.telegram", re.I), Activity.COMMUNICATION, "on Telegram"),
    (re.compile(r"discord\.com", re.I), Activity.COMMUNICATION, "on Discord"),
    (re.compile(r"slack\.com", re.I), Activity.COMMUNICATION, "on Slack"),
    (re.compile(r"teams\.microsoft|teams\.live", re.I), Activity.COMMUNICATION, "on Teams"),
    (re.compile(r"meet\.google", re.I), Activity.COMMUNICATION, "in Google Meet"),
    (re.compile(r"zoom\.us", re.I), Activity.COMMUNICATION, "in Zoom meeting"),

    # Productivity (web apps)
    (re.compile(r"docs\.google|sheets\.google|slides\.google", re.I), Activity.PRODUCTIVITY, "on Google Docs"),
    (re.compile(r"notion\.so", re.I), Activity.PRODUCTIVITY, "on Notion"),
    (re.compile(r"trello\.com", re.I), Activity.PRODUCTIVITY, "on Trello"),
    (re.compile(r"calendar\.google", re.I), Activity.PRODUCTIVITY, "checking calendar"),
]

# Browser name set (lowercased) for title-based category detection
_BROWSER_NAMES = {
    "firefox", "mozilla firefox", "chromium", "google-chrome",
    "brave", "brave-browser", "vivaldi", "opera", "epiphany",
    "qutebrowser", "midori",
}

# ── IDE window title → project context patterns ──────────────────────────

_VSCODE_TITLE_RE = re.compile(
    r"^(?:●\s)?(.+?)\s+[-–—]\s+(.+?)\s+[-–—]\s+(?:Visual Studio Code|Code(?:\s*-\s*OSS)?)"
)
# Matches: "filename — project_folder — Visual Studio Code"

_JETBRAINS_TITLE_RE = re.compile(
    r"^(.+?)\s+[-–—]\s+(.+?)\s+[-–—]\s+\[(.+?)\]"
)
# Matches: "filename – project – [~/path]"

_TERMINAL_CWD_RE = re.compile(r"^(.+?)(?::\s*~?(/\S+))?$")
# Matches: "user@host:~/project" or "bash: /home/user/project"


# ═══════════════════════════════════════════════════════════════════════════
# Display server backends
# ═══════════════════════════════════════════════════════════════════════════


def _detect_display_server() -> str:
    """Detect display server type."""
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland":
        return "wayland"
    if session == "x11":
        return "x11"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def _run_cmd(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    """Run a subprocess command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


# ── X11 backend ────────────────────────────────────────────────────────


def _x11_active_window() -> Optional[WindowInfo]:
    """Get active window via xdotool (X11)."""
    wid = _run_cmd(["xdotool", "getactivewindow"])
    if not wid:
        return None

    name = _run_cmd(["xdotool", "getactivewindow", "getwindowname"]) or ""
    pid_str = _run_cmd(["xdotool", "getactivewindow", "getwindowpid"]) or "0"
    wm_class = _run_cmd(["xprop", "-id", wid, "WM_CLASS"]) or ""

    # Parse WM_CLASS: WM_CLASS(STRING) = "code", "Code"
    wm_match = re.search(r'"(.+?)"', wm_class)
    wm_class_clean = wm_match.group(1) if wm_match else ""

    # Derive app name from WM_CLASS or title
    app_name = wm_class_clean or name.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()

    return WindowInfo(
        name=app_name,
        title=name,
        pid=int(pid_str) if pid_str.isdigit() else 0,
        wm_class=wm_class_clean,
    )


def _x11_running_apps() -> list[str]:
    """List running windowed apps via wmctrl or xdotool (X11)."""
    # Try wmctrl first
    output = _run_cmd(["wmctrl", "-l"])
    if output:
        apps = set()
        for line in output.splitlines():
            # wmctrl -l: 0x02200003  0 hostname Title...
            parts = line.split(None, 3)
            if len(parts) >= 4:
                title = parts[3]
                app = title.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()
                if app:
                    apps.add(app)
        return sorted(apps)

    # Fallback: xdotool search
    output = _run_cmd(["xdotool", "search", "--name", ""])
    if output:
        apps = set()
        for wid in output.splitlines()[:50]:  # limit
            name = _run_cmd(["xdotool", "getwindowname", wid])
            if name:
                app = name.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()
                if app:
                    apps.add(app)
        return sorted(apps)

    return []


# ── Wayland backend ────────────────────────────────────────────────────


def _wayland_active_window() -> Optional[WindowInfo]:
    """Get active window on Wayland (swaymsg / hyprctl)."""
    # Try swaymsg (Sway)
    output = _run_cmd(["swaymsg", "-t", "get_tree", "--raw"])
    if output:
        return _parse_sway_focused(output)

    # Try hyprctl (Hyprland)
    output = _run_cmd(["hyprctl", "activewindow", "-j"])
    if output:
        return _parse_hyprctl(output)

    return None


def _parse_sway_focused(tree_json: str) -> Optional[WindowInfo]:
    """Parse swaymsg get_tree JSON to find focused window."""
    try:
        tree = json.loads(tree_json)
    except json.JSONDecodeError:
        return None

    def _find_focused(node: dict) -> Optional[dict]:
        if node.get("focused"):
            return node
        for child in node.get("nodes", []) + node.get("floating_nodes", []):
            result = _find_focused(child)
            if result:
                return result
        return None

    focused = _find_focused(tree)
    if not focused:
        return None

    app_id = focused.get("app_id") or ""
    title = focused.get("name") or ""
    pid = focused.get("pid", 0)
    wm_class = focused.get("window_properties", {}).get("class", "") or app_id

    app_name = app_id or wm_class or title.rsplit(" – ", 1)[-1].rsplit(" - ", 1)[-1].strip()

    return WindowInfo(name=app_name, title=title, pid=pid, wm_class=wm_class)


def _parse_hyprctl(output: str) -> Optional[WindowInfo]:
    """Parse hyprctl activewindow -j output."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None

    return WindowInfo(
        name=data.get("class", "") or data.get("initialClass", ""),
        title=data.get("title", ""),
        pid=data.get("pid", 0),
        wm_class=data.get("class", ""),
    )


def _wayland_running_apps() -> list[str]:
    """List running apps on Wayland."""
    # Try swaymsg
    output = _run_cmd(["swaymsg", "-t", "get_tree", "--raw"])
    if output:
        return _parse_sway_apps(output)

    # Try hyprctl
    output = _run_cmd(["hyprctl", "clients", "-j"])
    if output:
        return _parse_hyprctl_clients(output)

    return []


def _parse_sway_apps(tree_json: str) -> list[str]:
    """Extract unique app names from sway tree."""
    try:
        tree = json.loads(tree_json)
    except json.JSONDecodeError:
        return []

    apps: set[str] = set()

    def _walk(node: dict) -> None:
        app_id = node.get("app_id") or node.get("window_properties", {}).get("class", "")
        if app_id and node.get("type") in ("con", "floating_con"):
            apps.add(app_id)
        for child in node.get("nodes", []) + node.get("floating_nodes", []):
            _walk(child)

    _walk(tree)
    return sorted(apps)


def _parse_hyprctl_clients(output: str) -> list[str]:
    """Extract unique app names from hyprctl clients."""
    try:
        clients = json.loads(output)
    except json.JSONDecodeError:
        return []

    apps: set[str] = set()
    for client in clients:
        name = client.get("class", "") or client.get("initialClass", "")
        if name:
            apps.add(name)
    return sorted(apps)


# ── AT-SPI fallback ────────────────────────────────────────────────────


def _atspi_active_window() -> Optional[WindowInfo]:
    """Get active window via AT-SPI (#119 accessibility module)."""
    try:
        from bantz.tools.accessibility import _init_atspi, _atspi, _get_desktop
        if not _init_atspi():
            return None

        desktop = _get_desktop()
        if not desktop:
            return None

        # Walk applications to find the focused window
        for i in range(desktop.get_child_count()):
            app = desktop.get_child_at_index(i)
            if not app:
                continue
            try:
                for j in range(app.get_child_count()):
                    win = app.get_child_at_index(j)
                    if not win:
                        continue
                    state_set = win.get_state_set()
                    # Check if window is active/focused
                    if state_set and state_set.contains(_atspi.StateType.ACTIVE):
                        return WindowInfo(
                            name=app.get_name() or "",
                            title=win.get_name() or "",
                            pid=app.get_process_id() if hasattr(app, "get_process_id") else 0,
                            wm_class="",
                        )
            except Exception:
                continue
    except Exception as exc:
        log.debug("AT-SPI active window fallback failed: %s", exc)
    return None


def _atspi_running_apps() -> list[str]:
    """List running apps via AT-SPI."""
    try:
        from bantz.tools.accessibility import list_applications
        return list_applications()
    except Exception:
        return []


# ── /proc fallback ─────────────────────────────────────────────────────


def _proc_running_apps() -> list[str]:
    """List windowed applications by scanning /proc for common app names."""
    known = {
        "firefox", "chromium", "chrome", "brave", "code", "vim", "nvim",
        "emacs", "thunderbird", "telegram-desktop", "discord", "spotify",
        "vlc", "mpv", "steam", "slack", "zoom", "obs", "gimp", "blender",
        "docker", "alacritty", "kitty", "wezterm", "foot", "konsole",
    }
    found: set[str] = set()
    try:
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            try:
                comm = (pid_dir / "comm").read_text().strip().lower()
                if comm in known:
                    found.add(comm)
            except (OSError, PermissionError):
                continue
    except OSError:
        pass
    return sorted(found)


# ═══════════════════════════════════════════════════════════════════════════
# Browser title parsing (explainability + context)
# ═══════════════════════════════════════════════════════════════════════════


def parse_browser_context(title: str) -> dict[str, str]:
    """Parse a browser window title for contextual information.

    Returns:
        {"url_hint": ..., "site": ..., "activity": ..., "context": ...}
    """
    result: dict[str, str] = {
        "url_hint": "",
        "site": "",
        "activity": "browsing",
        "context": "",
    }

    if not title:
        return result

    # Split on common delimiters: " - ", " — ", " | ", " · "
    parts = re.split(r"\s+[-–—|·]\s+", title)

    # Last part is usually the browser name — strip it
    if len(parts) > 1:
        browser_part = parts[-1].strip().lower()
        if any(b in browser_part for b in ("firefox", "chrome", "chromium", "brave", "vivaldi", "opera")):
            parts = parts[:-1]

    if parts:
        result["url_hint"] = parts[0].strip()
        if len(parts) > 1:
            result["site"] = parts[-1].strip()

    # Check against known patterns
    for pattern, activity, context in _BROWSER_TITLE_PATTERNS:
        if pattern.search(title):
            result["activity"] = activity.value
            result["context"] = context
            break

    # Try to detect GitHub repo
    gh_match = re.search(r"(\w+/\w+)\s*[-–—]\s*(?:GitHub|GitLab)", title)
    if gh_match:
        result["context"] = f"working on {gh_match.group(1)}"
        result["site"] = "GitHub"
        result["activity"] = Activity.CODING.value

    return result


# ═══════════════════════════════════════════════════════════════════════════
# IDE / workspace context parsing
# ═══════════════════════════════════════════════════════════════════════════


def parse_ide_context(win: WindowInfo) -> dict[str, str]:
    """Extract IDE workspace context from window title.

    Returns:
        {"ide": ..., "project": ..., "file": ..., "branch": ...}
    """
    result: dict[str, str] = {
        "ide": "",
        "project": "",
        "file": "",
        "branch": "",
    }

    name_lower = win.name.lower()
    title = win.title

    # VS Code
    if any(x in name_lower for x in ("code", "vscodium")):
        result["ide"] = "vscode"
        m = _VSCODE_TITLE_RE.match(title)
        if m:
            result["file"] = m.group(1)
            result["project"] = m.group(2)
        # Try to get branch from git
        if result["project"]:
            result["branch"] = _detect_git_branch(result["project"])

    # JetBrains IDEs
    elif any(x in name_lower for x in ("jetbrains", "intellij", "pycharm", "webstorm", "clion", "goland")):
        result["ide"] = name_lower
        m = _JETBRAINS_TITLE_RE.match(title)
        if m:
            result["file"] = m.group(1)
            result["project"] = m.group(2)

    # Terminal — try to get CWD from /proc
    elif any(x in name_lower for x in ("terminal", "alacritty", "kitty", "wezterm", "foot", "konsole", "tilix", "terminator", "gnome-terminal", "xterm")):
        result["ide"] = "terminal"
        if win.pid:
            cwd = _get_proc_cwd(win.pid)
            if cwd:
                result["project"] = cwd
                result["branch"] = _detect_git_branch(cwd)

    return result


def _get_proc_cwd(pid: int) -> str:
    """Get the current working directory of a process via /proc."""
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
        return cwd
    except (OSError, PermissionError):
        return ""


def _detect_git_branch(project_path: str) -> str:
    """Detect the current git branch for a project path."""
    try:
        head_file = Path(project_path) / ".git" / "HEAD"
        if head_file.exists():
            content = head_file.read_text().strip()
            if content.startswith("ref: refs/heads/"):
                return content[16:]
        return ""
    except (OSError, PermissionError):
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Docker status
# ═══════════════════════════════════════════════════════════════════════════


def _get_docker_containers() -> list[dict[str, str]]:
    """List running Docker containers."""
    output = _run_cmd(["docker", "ps", "--format", "{{json .}}"])
    if not output:
        return []

    containers = []
    for line in output.splitlines():
        try:
            c = json.loads(line)
            containers.append({
                "name": c.get("Names", ""),
                "image": c.get("Image", ""),
                "status": c.get("Status", ""),
            })
        except json.JSONDecodeError:
            continue
    return containers


# ═══════════════════════════════════════════════════════════════════════════
# AppDetector — main class
# ═══════════════════════════════════════════════════════════════════════════


class AppDetector:
    """Detects open applications, active window, and workspace context.

    Caches results with configurable TTL to avoid excessive polling.
    Auto-detects display server and uses appropriate backend.
    Falls back to AT-SPI → /proc if native tools fail.
    """

    def __init__(self) -> None:
        self._display_server = "unknown"
        self._cache_ttl = 5.0
        self._polling_interval = 5
        self._initialized = False

        # Caches
        self._active_window_cache = _CachedResult()
        self._running_apps_cache = _CachedResult()
        self._docker_cache = _CachedResult()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init(self, *, cache_ttl: float = 5.0, polling_interval: int = 5) -> None:
        if self._initialized:
            return
        self._display_server = _detect_display_server()
        self._cache_ttl = cache_ttl
        self._polling_interval = polling_interval
        self._initialized = True
        log.info(
            "AppDetector initialized: display=%s cache_ttl=%.1fs",
            self._display_server,
            self._cache_ttl,
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def display_server(self) -> str:
        return self._display_server

    # ── Active window ─────────────────────────────────────────────────

    def get_active_window(self) -> Optional[WindowInfo]:
        """Get the currently focused window.

        Tries native backend (X11/Wayland) then AT-SPI fallback.
        Results cached for `cache_ttl` seconds.
        """
        if not self._initialized:
            return None

        if self._active_window_cache.is_valid(self._cache_ttl):
            return self._active_window_cache.value

        win: Optional[WindowInfo] = None

        if self._display_server == "x11":
            win = _x11_active_window()
        elif self._display_server == "wayland":
            win = _wayland_active_window()

        # Fallback: AT-SPI
        if win is None:
            win = _atspi_active_window()

        self._active_window_cache = _CachedResult(value=win, timestamp=time.time())
        return win

    # ── Running apps ──────────────────────────────────────────────────

    def get_running_apps(self) -> list[str]:
        """List unique running application names.

        Tries native backend → AT-SPI → /proc scan.
        """
        if not self._initialized:
            return []

        if self._running_apps_cache.is_valid(self._cache_ttl):
            return self._running_apps_cache.value

        apps: list[str] = []

        if self._display_server == "x11":
            apps = _x11_running_apps()
        elif self._display_server == "wayland":
            apps = _wayland_running_apps()

        # Fallback chain
        if not apps:
            apps = _atspi_running_apps()
        if not apps:
            apps = _proc_running_apps()

        # Deduplicate (case-insensitive) while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for app in apps:
            key = app.lower()
            if key not in seen:
                seen.add(key)
                unique.append(app)
        apps = unique

        self._running_apps_cache = _CachedResult(value=apps, timestamp=time.time())
        return apps

    # ── Workspace context ─────────────────────────────────────────────

    def get_workspace_context(self) -> dict[str, Any]:
        """Get rich context about what the user is doing.

        Combines active window, IDE context, browser context,
        and Docker status into a comprehensive snapshot.
        """
        if not self._initialized:
            return {}

        win = self.get_active_window()
        if not win:
            return {"activity": Activity.IDLE.value, "apps": self.get_running_apps()}

        result: dict[str, Any] = {
            "active_window": win.to_dict(),
            "apps": self.get_running_apps(),
            "activity": self.classify_activity(win).value,
        }

        # IDE context
        ide_ctx = parse_ide_context(win)
        if ide_ctx["ide"]:
            result["ide"] = ide_ctx

        # Browser context
        name_lower = win.name.lower()
        if name_lower in _BROWSER_NAMES or any(b in name_lower for b in _BROWSER_NAMES):
            browser_ctx = parse_browser_context(win.title)
            result["browser"] = browser_ctx

        # Docker containers (longer TTL)
        if not self._docker_cache.is_valid(30.0):
            self._docker_cache = _CachedResult(
                value=_get_docker_containers(),
                timestamp=time.time(),
            )
        if self._docker_cache.value:
            result["docker"] = self._docker_cache.value

        return result

    # ── Activity classification ───────────────────────────────────────

    def classify_activity(self, win: Optional[WindowInfo] = None) -> Activity:
        """Classify current user activity for RL state vector.

        Priority: browser title patterns → app category map → IDLE
        """
        if win is None:
            win = self.get_active_window()
        if win is None:
            return Activity.IDLE

        name_lower = win.name.lower()

        # Check if it's a browser first → parse title for media/coding/etc.
        if name_lower in _BROWSER_NAMES or any(b in name_lower for b in _BROWSER_NAMES):
            for pattern, activity, _ in _BROWSER_TITLE_PATTERNS:
                if pattern.search(win.title):
                    return activity
            return Activity.BROWSING

        # Check app category map (partial/prefix match)
        for app_key, activity in _APP_CATEGORIES.items():
            if app_key in name_lower or name_lower.startswith(app_key.split()[0]):
                return activity

        # Check WM_CLASS too
        wm_lower = win.wm_class.lower()
        if wm_lower:
            for app_key, activity in _APP_CATEGORIES.items():
                if app_key in wm_lower:
                    return activity

        return Activity.IDLE

    def get_activity_category(self) -> Activity:
        """Shorthand: get current activity for RL state."""
        if not self._initialized:
            return Activity.IDLE
        return self.classify_activity()

    # ── Focus mode integration (#126) ─────────────────────────────────

    def should_enable_focus(self, win: Optional[WindowInfo] = None) -> bool:
        """Determine if focus mode should be auto-enabled.

        Returns True when the user appears to be in a flow state:
        - Coding in an IDE
        - Fullscreen media/gaming
        - In a video call
        """
        activity = self.classify_activity(win)
        return activity in (Activity.CODING, Activity.ENTERTAINMENT, Activity.COMMUNICATION)

    # ── Stats / Doctor ────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "display_server": self._display_server,
            "cache_ttl": self._cache_ttl,
            "polling_interval": self._polling_interval,
        }

    def status_line(self) -> str:
        win = self.get_active_window()
        activity = self.classify_activity(win)
        app_count = len(self.get_running_apps())
        win_name = win.name if win else "none"
        return (
            f"display={self._display_server} "
            f"active={win_name} "
            f"activity={activity.value} "
            f"apps={app_count}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

app_detector = AppDetector()
