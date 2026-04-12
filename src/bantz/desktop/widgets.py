"""
Bantz — Widget Data Provider (#365)

Produces JSON output for eww widget polling and waybar custom modules.
Each ``get_*`` method returns a JSON string suitable for eww ``defpoll``
or waybar ``custom/*`` modules.

The ``bantz-widget-data`` CLI entry point calls these methods based on
the first argument (news, weather, calendar, todos, cpu, ram, disk, gpu,
network, status).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("bantz.desktop.widgets")


class WidgetDataProvider:
    """Provide live data for Bantz Hyprland desktop widgets."""

    def __init__(self, *, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else (
            Path.home() / ".local" / "share" / "bantz"
        )

    # ── CPU ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_cpu() -> str:
        """Return CPU usage percentage as a plain number string."""
        try:
            import psutil
            return str(int(psutil.cpu_percent(interval=0.5)))
        except ImportError:
            # Fallback: /proc/stat
            try:
                with open("/proc/stat") as f:
                    line = f.readline()
                parts = line.split()
                idle = int(parts[4])
                total = sum(int(p) for p in parts[1:])
                # Approximate — single sample
                usage = 100 - (idle * 100 // max(total, 1))
                return str(max(0, min(100, usage)))
            except (OSError, IndexError):
                return "0"

    # ── RAM ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_ram() -> str:
        """Return RAM usage percentage as a plain number string."""
        try:
            import psutil
            return str(int(psutil.virtual_memory().percent))
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    lines = {
                        l.split(":")[0]: int(l.split()[1])
                        for l in f if ":" in l
                    }
                total = lines.get("MemTotal", 1)
                avail = lines.get("MemAvailable", 0)
                pct = 100 - (avail * 100 // max(total, 1))
                return str(max(0, min(100, pct)))
            except (OSError, KeyError):
                return "0"

    # ── Disk ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_disk() -> str:
        """Return root disk usage percentage."""
        try:
            import psutil
            return str(int(psutil.disk_usage("/").percent))
        except ImportError:
            try:
                st = os.statvfs("/")
                used = (st.f_blocks - st.f_bfree) * st.f_frsize
                total = st.f_blocks * st.f_frsize
                return str(int(used * 100 / max(total, 1)))
            except OSError:
                return "0"

    # ── GPU ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_gpu() -> str:
        """Return GPU info string (NVIDIA via pynvml, fallback to empty)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            pynvml.nvmlShutdown()
            return f"{util.gpu}% | {used_gb:.1f}/{total_gb:.1f}G | {temp}°C"
        except Exception:
            # Try nvidia-smi fallback
            if shutil.which("nvidia-smi"):
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        timeout=3,
                    ).decode().strip()
                    parts = [p.strip() for p in out.split(",")]
                    if len(parts) >= 4:
                        return f"{parts[0]}% | {int(parts[1])/1024:.1f}/{int(parts[2])/1024:.1f}G | {parts[3]}°C"
                except (subprocess.SubprocessError, ValueError):
                    pass
            return "N/A"

    # ── Network ───────────────────────────────────────────────────────────

    @staticmethod
    def get_network() -> str:
        """Return network throughput summary."""
        try:
            import psutil
            counters = psutil.net_io_counters()
            sent_mb = counters.bytes_sent / (1024 ** 2)
            recv_mb = counters.bytes_recv / (1024 ** 2)
            return f"↑ {sent_mb:.0f}M  ↓ {recv_mb:.0f}M"
        except ImportError:
            return "N/A"

    # ── Weather ───────────────────────────────────────────────────────────

    def get_weather(self) -> str:
        """Return weather summary from Bantz DB or wttr.in fallback."""
        # Try cached weather from Bantz data
        cache_file = self.data_dir / "weather_cache.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                if "summary" in data:
                    return data["summary"]
            except (json.JSONDecodeError, OSError):
                pass
        # Lightweight fallback: wttr.in one-liner
        try:
            out = subprocess.check_output(
                ["curl", "-s", "wttr.in/?format=%c+%t+%w"],
                timeout=5,
            ).decode().strip()
            return out if out and "Unknown" not in out else "Weather unavailable"
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Weather unavailable"

    # ── News ──────────────────────────────────────────────────────────────

    def get_news(self) -> str:
        """Return recent news headlines from Bantz feed cache."""
        cache_file = self.data_dir / "news_cache.json"
        if cache_file.exists():
            try:
                articles = json.loads(cache_file.read_text())
                if isinstance(articles, list):
                    lines = []
                    for a in articles[:6]:
                        title = a.get("title", "")
                        source = a.get("source", "")
                        if title:
                            lines.append(f"• {title}" + (f"  ({source})" if source else ""))
                    return "\n".join(lines) if lines else "No recent news"
            except (json.JSONDecodeError, OSError):
                pass
        return "No news data — run bantz to populate"

    # ── Calendar ──────────────────────────────────────────────────────────

    def get_calendar(self) -> str:
        """Return upcoming calendar events from Bantz cache."""
        cache_file = self.data_dir / "calendar_cache.json"
        if cache_file.exists():
            try:
                events = json.loads(cache_file.read_text())
                if isinstance(events, list):
                    lines = []
                    for e in events[:5]:
                        time_str = e.get("time", "")
                        summary = e.get("summary", "?")
                        lines.append(f"  {time_str}  {summary}")
                    return "\n".join(lines) if lines else "No upcoming events"
            except (json.JSONDecodeError, OSError):
                pass
        return "No calendar data — bantz --setup google calendar"

    # ── Todos ─────────────────────────────────────────────────────────────

    def get_todos(self) -> str:
        """Return active todos / reminders from Bantz DB."""
        cache_file = self.data_dir / "todos_cache.json"
        if cache_file.exists():
            try:
                todos = json.loads(cache_file.read_text())
                if isinstance(todos, list):
                    lines = []
                    for t in todos[:8]:
                        status = "✓" if t.get("done") else "○"
                        text = t.get("text", "")
                        lines.append(f"  {status} {text}")
                    return "\n".join(lines) if lines else "No todos"
            except (json.JSONDecodeError, OSError):
                pass
        return "No todos yet"

    # ── Bantz status ──────────────────────────────────────────────────────

    def get_status(self) -> str:
        """Return Bantz process status for waybar / eww."""
        # Check if bantz is running
        try:
            import psutil
            for proc in psutil.process_iter(["name", "cmdline"]):
                cmdline = proc.info.get("cmdline") or []
                if any("bantz" in str(c) for c in cmdline):
                    return "active"
        except ImportError:
            # Fallback: pgrep
            try:
                ret = subprocess.call(
                    ["pgrep", "-f", "bantz"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if ret == 0:
                    return "active"
            except FileNotFoundError:
                pass
        return "idle"

    def get_status_json(self) -> str:
        """Return Bantz status as waybar JSON module format."""
        status = self.get_status()
        icon = "🤖" if status == "active" else "💤"
        return json.dumps({
            "text": f"{icon} {status.title()}",
            "tooltip": f"Bantz is {status}",
            "class": status,
        })

    # ── Dispatch by name ──────────────────────────────────────────────────

    def get(self, name: str) -> str:
        """Get widget data by name (used by CLI entry point)."""
        dispatch: dict[str, Any] = {
            "cpu": self.get_cpu,
            "ram": self.get_ram,
            "disk": self.get_disk,
            "gpu": self.get_gpu,
            "network": self.get_network,
            "weather": self.get_weather,
            "news": self.get_news,
            "calendar": self.get_calendar,
            "todos": self.get_todos,
            "status": self.get_status_json,
        }
        fn = dispatch.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown widget: {name}"})
        try:
            return fn()
        except Exception as exc:
            log.exception("Widget %s failed", name)
            return json.dumps({"error": str(exc)})


# ── CLI entry points ──────────────────────────────────────────────────────────

def widget_data_cli() -> None:
    """CLI: ``bantz-widget-data <name>`` — called by eww defpoll."""
    if len(sys.argv) < 2:
        print("Usage: bantz-widget-data <cpu|ram|disk|gpu|network|weather|news|calendar|todos|status>")
        sys.exit(1)
    provider = WidgetDataProvider()
    print(provider.get(sys.argv[1]))


def widget_status_cli() -> None:
    """CLI: ``bantz-widget-status`` — waybar custom module (JSON)."""
    provider = WidgetDataProvider()
    print(provider.get_status_json())
