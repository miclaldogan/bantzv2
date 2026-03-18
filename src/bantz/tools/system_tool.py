"""
Bantz — SystemTool: unified subprocess + process management interface (#291)

Low-level infrastructure class (not a BaseTool) used by other tools that need
to run shell commands, list/kill processes, or launch applications.

Provides:
  ShellResult          — structured result dataclass
  DangerousCommandError — raised on denylist hits in safe_mode
  SystemTool            — the main utility class
  system_tool           — module-level singleton

The existing ``shell`` tool (shell.py / ShellTool) continues to handle
LLM-facing shell execution via the tool registry.  SystemTool is the
lower-level plumbing that other tools can import directly.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("bantz.system_tool")

# ── Denylist patterns ─────────────────────────────────────────────────────────
# These are regex patterns matched against the full command string.
# Matches in safe_mode → DangerousCommandError raised before execution.

DENYLIST: list[str] = [
    r"rm\s+-rf\s+/",        # rm -rf /  (recursive root deletion)
    r"rm\s+.*--no-preserve-root",
    r"dd\s+if=",            # disk-level writes
    r"mkfs",                # filesystem formatting
    r">\s*/dev/sd",         # direct writes to block device
    r":\(\)\s*\{",          # fork bomb  :(){:|:&};:
    r"chmod\s+-R\s+777\s+/",# world-write root
    r">\s*/etc/passwd",     # overwrite passwd
    r">\s*/etc/shadow",     # overwrite shadow
]

_COMPILED_DENYLIST: list[re.Pattern] = [re.compile(p) for p in DENYLIST]

# ── Audit log ─────────────────────────────────────────────────────────────────
_AUDIT_LOG: Path = Path.home() / ".bantz" / "logs" / "system_audit.log"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShellResult:
    """Structured result returned by SystemTool.run()."""
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Convenience: stdout if success, stderr otherwise."""
        return self.stdout if self.success else (self.stderr or self.stdout)


class DangerousCommandError(RuntimeError):
    """Raised when a command matches the denylist in safe_mode."""


# ── SystemTool ────────────────────────────────────────────────────────────────

class SystemTool:
    """Unified subprocess + process management interface.

    All methods are synchronous.  Import and use as a utility:

        from bantz.tools.system_tool import system_tool
        result = system_tool.run("ls -la /tmp")
    """

    AUDIT_LOG: Path = _AUDIT_LOG

    # ── Command execution ─────────────────────────────────────────────────

    def run(
        self,
        cmd: str,
        timeout: int = 30,
        safe_mode: bool = True,
    ) -> ShellResult:
        """Run a shell command and return a structured ShellResult.

        Args:
            cmd:        Shell command string (passed to bash -c).
            timeout:    Maximum execution time in seconds.
            safe_mode:  If True, commands matching DENYLIST raise
                        DangerousCommandError before execution.

        Returns:
            ShellResult with stdout, stderr, returncode, duration_ms.

        Raises:
            DangerousCommandError: if safe_mode=True and cmd is dangerous.
            subprocess.TimeoutExpired: re-raised after logging.
        """
        if safe_mode:
            for pattern in _COMPILED_DENYLIST:
                if pattern.search(cmd):
                    log.warning("SystemTool.run: BLOCKED dangerous command: %s", cmd)
                    raise DangerousCommandError(
                        f"Command blocked by safety denylist: {cmd!r}"
                    )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            result = ShellResult(
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                returncode=proc.returncode,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - t0) * 1000)
            self._audit(cmd, -1, duration_ms, note="TIMEOUT")
            raise

        self._audit(cmd, result.returncode, result.duration_ms)
        return result

    # ── Process listing ───────────────────────────────────────────────────

    def list_processes(self) -> list[dict[str, Any]]:
        """Return a snapshot of running processes.

        Each entry:  {"pid": int, "name": str, "cpu_pct": float, "mem_pct": float}

        Requires ``psutil``.  Returns empty list if psutil is unavailable.
        """
        try:
            import psutil
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = p.info
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"] or "",
                        "cpu_pct": round(info["cpu_percent"] or 0.0, 1),
                        "mem_pct": round(info["memory_percent"] or 0.0, 2),
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return procs
        except ImportError:
            log.warning("SystemTool.list_processes: psutil not available")
            return []
        except Exception as exc:
            log.warning("SystemTool.list_processes: %s", exc)
            return []

    # ── Process termination ───────────────────────────────────────────────

    def kill(self, pid: int) -> bool:
        """Terminate a process by PID.

        Sends SIGTERM first; if the process is still alive after 3 s, sends
        SIGKILL.

        Returns:
            True if the process was terminated, False if not found.

        Requires ``psutil``.
        """
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                name = proc.name()
            except psutil.NoSuchProcess:
                log.warning("SystemTool.kill: pid %d not found", pid)
                return False

            log.info("SystemTool.kill: terminating pid=%d name=%s", pid, name)
            proc.terminate()

            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                log.warning("SystemTool.kill: pid %d did not exit — sending SIGKILL", pid)
                proc.kill()

            self._audit(f"kill pid={pid} ({name})", 0, 0)
            return True

        except ImportError:
            log.warning("SystemTool.kill: psutil not available — falling back to os.kill")
            try:
                import signal
                os.kill(pid, signal.SIGTERM)
                self._audit(f"kill pid={pid}", 0, 0)
                return True
            except ProcessLookupError:
                return False
        except Exception as exc:
            log.warning("SystemTool.kill: %s", exc)
            return False

    # ── Application launching ─────────────────────────────────────────────

    def open_app(self, app_name: str) -> bool:
        """Resolve *app_name* to a binary and launch it detached.

        Tries ``shutil.which`` first, then common aliases.  The process is
        started detached (no stdout/stderr captured) so it survives the
        caller's lifetime.

        Returns:
            True if the binary was found and launched, False otherwise.
        """
        # Common aliases: UI name → binary name(s)
        _ALIASES: dict[str, list[str]] = {
            "firefox": ["firefox", "firefox-esr"],
            "chrome": ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
            "vscode": ["code", "code-oss"],
            "terminal": ["gnome-terminal", "xterm", "konsole", "xfce4-terminal"],
            "files": ["nautilus", "thunar", "dolphin", "nemo"],
            "spotify": ["spotify"],
            "vlc": ["vlc"],
            "gimp": ["gimp"],
        }

        candidates: list[str] = _ALIASES.get(app_name.lower(), [app_name])

        binary: str | None = None
        for candidate in candidates:
            found = shutil.which(candidate)
            if found:
                binary = found
                break

        if binary is None:
            log.warning("SystemTool.open_app: binary not found for %r", app_name)
            return False

        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            log.info("SystemTool.open_app: launched %s (%s)", app_name, binary)
            self._audit(f"open_app {app_name} → {binary}", 0, 0)
            return True
        except Exception as exc:
            log.warning("SystemTool.open_app: failed to launch %s: %s", binary, exc)
            return False

    # ── Audit logging ─────────────────────────────────────────────────────

    def _audit(
        self,
        cmd: str,
        returncode: int,
        duration_ms: int,
        note: str = "",
    ) -> None:
        """Append one audit line to AUDIT_LOG.

        Format::

            2024-01-15T14:32:01  rc=0  42ms  ls -la /tmp
        """
        try:
            self.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            note_part = f"  [{note}]" if note else ""
            line = f"{ts}  rc={returncode}  {duration_ms}ms{note_part}  {cmd}\n"
            with self.AUDIT_LOG.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:
            log.debug("SystemTool._audit: could not write audit log: %s", exc)


# ── Module singleton ──────────────────────────────────────────────────────────

system_tool = SystemTool()
