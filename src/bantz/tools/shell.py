"""
Bantz v2 — Shell Tool
Runs Bash/zsh commands securely.
Destructive commands require confirmation, which is triggered by brain.py.
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex
from typing import Any

from bantz.config import config
from bantz.tools import BaseTool, ToolResult, registry

_HOME = os.path.expanduser("~")

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


# Command separators (chaining / pipes / newlines) and wrapper programs that
# carry the *real* command as a later argument.
_CMD_SEPARATORS = re.compile(r"[;&|\n]+")
_WRAPPERS: frozenset[str] = frozenset({
    "sudo", "env", "nohup", "setsid", "time", "nice", "ionice",
    "xargs", "watch", "bash", "sh", "zsh",
})


def _command_heads(cmd: str) -> set[str]:
    """All plausible command-head words in ``cmd``, looking through shell
    separators, wrapper programs, and ``-c`` payloads.

    First-word-only matching (the old ``is_destructive``) was trivially
    bypassed by ``bash -c "rm -rf ~"``, ``sudo rm ...``, ``foo && rm bar``,
    or ``ls | xargs rm`` (audit S2). This walks all of those.
    """
    heads: set[str] = set()
    for segment in _CMD_SEPARATORS.split(cmd):
        seg = segment.strip()
        if not seg:
            continue
        try:
            parts = shlex.split(seg)
        except ValueError:
            parts = seg.split()
        i = 0
        while i < len(parts):
            word = os.path.basename(parts[i])
            heads.add(word)
            if word in _WRAPPERS:
                # `bash -c "<payload>"` / `sh -c "..."` — recurse into payload.
                if word in {"bash", "sh", "zsh"}:
                    for j in range(i + 1, len(parts)):
                        if parts[j] == "-c" and j + 1 < len(parts):
                            heads |= _command_heads(parts[j + 1])
                            break
                i += 1  # skip the wrapper, keep scanning for the real head
                continue
            break
    return heads


def is_destructive(cmd: str) -> bool:
    """Deterministic destructive-command detection. Looks through wrappers,
    chaining, and quoted payloads — not just the first word."""
    return bool(_command_heads(cmd) & DESTRUCTIVE_COMMANDS)


def is_blocked(cmd: str) -> bool:
    first = _first_word(cmd)
    return first in BLOCKED_COMMANDS or any(b in cmd for b in BLOCKED_COMMANDS)


# ── Tool ─────────────────────────────────────────────────────────────────────

class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Execute a Bash shell command in the terminal. "
        "Params: command (str) = the full bash command to run. "
        f"Use for: running commands, listing files, managing processes, package management. "
        f"The user's home directory is {_HOME}. Always use absolute paths. "
        "If the user types a literal command (ls, df -h, top, etc.), this is the right tool. "
        "Do not use this tool for GUI interaction (clicking, hovering) — use visual_click for that. "
        "WRONG: shell(command='click the OK button'). RIGHT: visual_click(target='OK button')."
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
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
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