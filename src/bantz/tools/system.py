"""
Bantz v2 — System Tool
Takes CPU, RAM, disk, uptime information. Uses psutil, no API calls.
"""
from __future__ import annotations

from typing import Any

import psutil

from bantz.tools import BaseTool, ToolResult, registry


def _bytes_to_gb(b: int) -> float:
    return round(b / (1024 ** 3), 1)


class SystemTool(BaseTool):
    name = "system"
    description = (
        "Returns system information: CPU usage, RAM, disk usage, uptime. "
        "Used for performance-related questions to understand system status."
    )
    risk_level = "safe"

    async def execute(self, metric: str = "all", **kwargs: Any) -> ToolResult:
        """
        metric: "all" | "cpu" | "ram" | "disk" | "uptime"
        """
        try:
            data: dict[str, Any] = {}
            lines: list[str] = []

            if metric in ("all", "cpu"):
                cpu = psutil.cpu_percent(interval=0.5)
                data["cpu_percent"] = cpu
                lines.append(f"CPU: %{cpu:.0f}")

            if metric in ("all", "ram"):
                mem = psutil.virtual_memory()
                data["ram_used_gb"] = _bytes_to_gb(mem.used)
                data["ram_total_gb"] = _bytes_to_gb(mem.total)
                data["ram_percent"] = mem.percent
                lines.append(
                    f"RAM: %{mem.percent:.0f} "
                    f"({_bytes_to_gb(mem.used)}/{_bytes_to_gb(mem.total)} GB)"
                )

            if metric in ("all", "disk"):
                disk = psutil.disk_usage("/")
                data["disk_used_gb"] = _bytes_to_gb(disk.used)
                data["disk_total_gb"] = _bytes_to_gb(disk.total)
                data["disk_percent"] = disk.percent
                lines.append(
                    f"Disk: %{disk.percent:.0f} "
                    f"({_bytes_to_gb(disk.used)}/{_bytes_to_gb(disk.total)} GB)"
                )

            if metric in ("all", "uptime"):
                import time
                boot = psutil.boot_time()
                uptime_s = int(time.time() - boot)
                h, remainder = divmod(uptime_s, 3600)
                m, s = divmod(remainder, 60)
                uptime_str = f"{h}s {m}d {s}sn"
                data["uptime_seconds"] = uptime_s
                data["uptime_str"] = uptime_str
                lines.append(f"Uptime: {uptime_str}")

            return ToolResult(success=True, output="\n".join(lines), data=data)

        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


registry.register(SystemTool())