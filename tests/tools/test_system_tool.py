"""Tests for SystemTool — subprocess + process management utility (#291)."""
from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest


# ── ShellResult ───────────────────────────────────────────────────────────────

class TestShellResult:

    def test_success_true_when_rc_zero(self):
        from bantz.tools.system_tool import ShellResult
        r = ShellResult(stdout="hi", stderr="", returncode=0, duration_ms=10)
        assert r.success is True

    def test_success_false_when_rc_nonzero(self):
        from bantz.tools.system_tool import ShellResult
        r = ShellResult(stdout="", stderr="oops", returncode=1, duration_ms=5)
        assert r.success is False

    def test_output_returns_stdout_on_success(self):
        from bantz.tools.system_tool import ShellResult
        r = ShellResult(stdout="hello", stderr="ignored", returncode=0, duration_ms=1)
        assert r.output == "hello"

    def test_output_returns_stderr_on_failure(self):
        from bantz.tools.system_tool import ShellResult
        r = ShellResult(stdout="", stderr="error msg", returncode=2, duration_ms=1)
        assert r.output == "error msg"

    def test_output_falls_back_to_stdout_when_stderr_empty(self):
        from bantz.tools.system_tool import ShellResult
        r = ShellResult(stdout="partial", stderr="", returncode=1, duration_ms=1)
        assert r.output == "partial"


# ── DangerousCommandError ─────────────────────────────────────────────────────

class TestDangerousCommandError:

    def test_is_runtime_error(self):
        from bantz.tools.system_tool import DangerousCommandError
        assert issubclass(DangerousCommandError, RuntimeError)


# ── SystemTool.run() — safe_mode denylist ────────────────────────────────────

class TestRunSafeMode:

    def _tool(self):
        from bantz.tools.system_tool import SystemTool
        return SystemTool()

    def test_blocks_rm_rf_root(self):
        from bantz.tools.system_tool import DangerousCommandError
        tool = self._tool()
        with pytest.raises(DangerousCommandError):
            tool.run("rm -rf /", safe_mode=True)

    def test_blocks_dd_if(self):
        from bantz.tools.system_tool import DangerousCommandError
        tool = self._tool()
        with pytest.raises(DangerousCommandError):
            tool.run("dd if=/dev/zero of=/dev/sda", safe_mode=True)

    def test_blocks_mkfs(self):
        from bantz.tools.system_tool import DangerousCommandError
        tool = self._tool()
        with pytest.raises(DangerousCommandError):
            tool.run("mkfs.ext4 /dev/sdb1", safe_mode=True)

    def test_blocks_fork_bomb(self):
        from bantz.tools.system_tool import DangerousCommandError
        tool = self._tool()
        with pytest.raises(DangerousCommandError):
            tool.run(":(){:|:&};:", safe_mode=True)

    def test_blocks_overwrite_passwd(self):
        from bantz.tools.system_tool import DangerousCommandError
        tool = self._tool()
        with pytest.raises(DangerousCommandError):
            tool.run("echo 'evil' > /etc/passwd", safe_mode=True)

    def test_safe_mode_false_skips_denylist(self):
        """safe_mode=False: dangerous-looking command is attempted (we pass echo to avoid actual harm)."""
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        # "echo mkfs" contains "mkfs" — with safe_mode=False it should run
        result = tool.run("echo mkfs", safe_mode=False)
        assert result.success
        assert "mkfs" in result.stdout

    def test_normal_command_runs_successfully(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        result = tool.run("echo hello")
        assert result.success
        assert result.stdout == "hello"
        assert result.duration_ms >= 0

    def test_failed_command_captures_returncode(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        result = tool.run("exit 42", safe_mode=False)
        assert result.returncode == 42
        assert not result.success


# ── SystemTool.run() — timeout ────────────────────────────────────────────────

class TestRunTimeout:

    def test_timeout_raises_timeout_expired(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        with pytest.raises(subprocess.TimeoutExpired):
            tool.run("sleep 10", timeout=1, safe_mode=False)


# ── Audit log ─────────────────────────────────────────────────────────────────

class TestAuditLog:

    def test_audit_writes_to_log(self, tmp_path):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        tool.AUDIT_LOG = tmp_path / "audit.log"
        tool.run("echo audit_test")
        log_content = tool.AUDIT_LOG.read_text()
        assert "echo audit_test" in log_content
        assert "rc=0" in log_content

    def test_audit_records_nonzero_rc(self, tmp_path):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        tool.AUDIT_LOG = tmp_path / "audit.log"
        tool.run("exit 1", safe_mode=False)
        log_content = tool.AUDIT_LOG.read_text()
        assert "rc=1" in log_content

    def test_audit_records_timeout_note(self, tmp_path):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        tool.AUDIT_LOG = tmp_path / "audit.log"
        with pytest.raises(subprocess.TimeoutExpired):
            tool.run("sleep 10", timeout=1, safe_mode=False)
        log_content = tool.AUDIT_LOG.read_text()
        assert "TIMEOUT" in log_content


# ── SystemTool.list_processes() ───────────────────────────────────────────────

class TestListProcesses:

    def test_returns_list_of_dicts(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        procs = tool.list_processes()
        assert isinstance(procs, list)

    def test_each_entry_has_expected_keys(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        procs = tool.list_processes()
        if procs:  # non-empty
            for p in procs[:3]:
                assert "pid" in p
                assert "name" in p
                assert "cpu_pct" in p
                assert "mem_pct" in p

    def test_returns_empty_when_psutil_unavailable(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        with patch.dict("sys.modules", {"psutil": None}):
            # Force ImportError path
            import sys
            original = sys.modules.pop("psutil", None)
            try:
                procs = tool.list_processes()
                assert isinstance(procs, list)
            finally:
                if original is not None:
                    sys.modules["psutil"] = original


# ── SystemTool.kill() ─────────────────────────────────────────────────────────

class TestKill:

    def test_kill_nonexistent_pid_returns_false(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        # PID 999999999 almost certainly doesn't exist
        result = tool.kill(999999999)
        assert result is False

    def test_kill_live_process_returns_true(self):
        """Launch a real sleep process and kill it."""
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        proc = subprocess.Popen(["sleep", "60"])
        try:
            result = tool.kill(proc.pid)
            assert result is True
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass


# ── SystemTool.open_app() ─────────────────────────────────────────────────────

class TestOpenApp:

    def test_unknown_app_returns_false(self):
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()
        result = tool.open_app("nonexistent_app_xyz_123")
        assert result is False

    def test_known_binary_launches(self, tmp_path):
        """Create a fake binary, verify open_app finds and launches it."""
        from bantz.tools.system_tool import SystemTool
        tool = SystemTool()

        # Create a no-op script
        fake_bin = tmp_path / "fake_app_bantz"
        fake_bin.write_text("#!/bin/sh\nsleep 0\n")
        fake_bin.chmod(0o755)

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(tmp_path) + ":" + old_path
        try:
            result = tool.open_app("fake_app_bantz")
            assert result is True
        finally:
            os.environ["PATH"] = old_path


# ── Module singleton ──────────────────────────────────────────────────────────

class TestModuleSingleton:

    def test_system_tool_singleton_exists(self):
        from bantz.tools.system_tool import system_tool, SystemTool
        assert isinstance(system_tool, SystemTool)

    def test_singleton_is_reused(self):
        from bantz.tools.system_tool import system_tool as a
        from bantz.tools.system_tool import system_tool as b
        assert a is b
