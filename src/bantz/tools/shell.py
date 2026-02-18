"""
Bantz v2 — Shell Tool
Runs Bash/zsh commands securely.
Destructive commands require confirmation, which is triggered by brain.py.
"""
from __future__ import annotations

import asyncio
import shlex
from typing import Any

from bantz.config import config
from bantz.tools import BaseTool, ToolResult, registry

# ── Security Lists ───────────────────────────────────────────────────────────

# These commands always require confirmation (if shell_confirm_destructive is True)
DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "rmdir", "sudo", "chmod", "chown", "kill", "killall",
    "pkill", "mv", "dd", "mkfs", "fdisk", "parted", "shred",
    "truncate", "poweroff", "reboot", "shutdown", "halt",
})

# These commands are never allowed, even with confirmation
BLOCKED_COMMANDS: frozenset[str] = frozenset({
    ":(){:|:&};:",  # fork bomb
    "fork",
    "wget",        # network download — separate tool
    "curl",        # network download — separate tool
})


def _first_word(cmd: str) -> str:
    """Returns the first word of the command (for security checks)."""
    try:
        parts = shlex.split(cmd.strip())
        return parts[0] if parts else ""
    except ValueError:
        return cmd.split()[0] if cmd.split() else ""


def is_destructive(cmd: str) -> bool:
    return _first_word(cmd) in DESTRUCTIVE_COMMANDS


def is_blocked(cmd: str) -> bool:
    first = _first_word(cmd)
    return first in BLOCKED_COMMANDS or any(b in cmd for b in BLOCKED_COMMANDS)


# ── Tool ─────────────────────────────────────────────────────────────────────

class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Runs Bash commands in the terminal. "
        "Used for file listing, process management, text processing, and similar tasks."
    )
    risk_level = "moderate"

    async def execute(self, command: str = "", **kwargs: Any) -> ToolResult:
        if not command:
            return ToolResult(success=False, output="", error="Command cannot be empty.")

        if is_blocked(command):
            return ToolResult(
                success=False,
                output="",
                error=f"This command is blocked for security reasons: `{command}`",
            )

        # Destructive flag — brain.py can request confirmation based on this
        # But if confirmation has already been obtained, we reach this point with confidence
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=config.shell_timeout_seconds,
            )

            out = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    output=out or "(command executed successfully, no output)",
                    data={"returncode": 0, "command": command},
                )
            else:
                return ToolResult(
                    success=False,
                    output=out,
                    error=err or f"Command failed (code: {proc.returncode})",
                    data={"returncode": proc.returncode, "command": command},
                )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=f"Command did not complete within {config.shell_timeout_seconds}s.",
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


# Register
registry.register(ShellTool())