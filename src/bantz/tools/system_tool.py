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
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("bantz.system_tool")

# ── Denylist patterns ─────────────────────────────────────────────────────────

# Commands that are strictly forbidden as the primary executable
BLOCKED_EXECUTABLES: frozenset[str] = frozenset({
    "mkfs", "dd", "fdisk", "parted", "shred", "format",
})

# Dangerous argument combinations (regex on the parsed argument list)
# We check if any argument matches these.
BLOCKED_ARG_PATTERNS: list[re.Pattern] = [
    re.compile(r"^--no-preserve-root$"),
    re.compile(r"^:(\(\)\s*\{|.*\(\)\s*\{)"), # Potential fork bomb start
]

# We still use some global regex for patterns that shlex might split but are still dangerous
# e.g. redirection to sensitive files (if we were using shell=True)
# Since we are moving to shell=False, many of these won't work anyway,
# but we keep them for defense-in-depth during validation.
GLOBAL_DENYLIST: list[re.Pattern] = [
    re.compile(r"rm\s+.*-rf\s+/"),          # rm -rf /
    re.compile(r"chmod\s+.*-R\s+777\s+/"),  # world-write root
    re.compile(r">\s*/etc/(passwd|shadow)"), # overwrite sensitive files
    re.compile(r">\s*/dev/sd"),             # direct write to block device
]


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
        """Run a command and return a structured ShellResult.

        Args:
            cmd:        Command string (parsed via shlex.split).
            timeout:    Maximum execution time in seconds.
            safe_mode:  If True, commands matching security checks raise
                        DangerousCommandError before execution.

        Returns:
            ShellResult with stdout, stderr, returncode, duration_ms.

        Raises:
            DangerousCommandError: if safe_mode=True and cmd is dangerous.
            subprocess.TimeoutExpired: re-raised after logging.
        """
        if not cmd.strip():
            return ShellResult(
                stdout="",
                stderr="Command cannot be empty.",
                returncode=1,
                duration_ms=0,
            )

        try:
            args = shlex.split(cmd)
        except ValueError as exc:
            log.warning("SystemTool.run: shlex.split failed for cmd: %s", cmd)
            if safe_mode:
                raise DangerousCommandError(f"Invalid command string: {exc}") from exc
            # Fallback for safe_mode=False
            args = cmd.split()

        if not args:
             return ShellResult(
                stdout="",
                stderr="No command arguments found.",
                returncode=1,
                duration_ms=0,
            )

        if safe_mode:
            self._validate_command(cmd, args)

        t0 = time.monotonic()
        try:
            # shell=False is preferred for security.
            # args is already a list from shlex.split.
            proc = subprocess.run(
                args,
                shell=False,
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
        except FileNotFoundError as exc:
            # Executable not found
            duration_ms = int((time.monotonic() - t0) * 1000)
            result = ShellResult(
                stdout="",
                stderr=str(exc),
                returncode=127,
                duration_ms=duration_ms,
            )

        self._audit(cmd, result.returncode, result.duration_ms)
        return result

    def _validate_command(self, cmd: str, args: list[str]) -> None:
        """Check command and arguments against security policies."""
        if not args:
            return

        exe = args[0]
        # 1. Blocked executables
        if any(blocked in exe for blocked in BLOCKED_EXECUTABLES):
            self._block(cmd, f"Executable {exe!r} is blocked")

        # 2. Argument patterns
        for arg in args:
            for pattern in BLOCKED_ARG_PATTERNS:
                if pattern.search(arg):
                    self._block(cmd, f"Argument {arg!r} matches blocked pattern")

        # 3. rm -rf / check (more robustly)
        if exe == "rm":
            combined_args = "".join(args[1:])
            if ("r" in combined_args and "f" in combined_args) or "-recursive" in args:
                if "/" in args or "//" in args:
                    self._block(cmd, "Attempted recursive root deletion")

        # 4. Global regex patterns (fallback/defense-in-depth)
        for pattern in GLOBAL_DENYLIST:
            if pattern.search(cmd):
                self._block(cmd, f"Command matches blocked pattern: {pattern.pattern}")

    def _block(self, cmd: str, reason: str) -> None:
        log.warning("SystemTool.run: BLOCKED dangerous command: %s (Reason: %s)", cmd, reason)
        raise DangerousCommandError(f"Command blocked: {reason}")

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
