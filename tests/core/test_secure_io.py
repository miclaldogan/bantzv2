"""Tests for secure_write_text — files must be created 0o600, no TOCTOU window."""
from __future__ import annotations

import json
import os
import stat

import pytest

from bantz.core.secure_io import secure_write_text


def _assert_owner_only(path) -> None:
    # POSIX permission bits are not represented on Windows (fchmod is absent
    # and st_mode reports 0o666 regardless) — mode checks only apply on posix.
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_creates_file_owner_only(tmp_path):
    p = tmp_path / "secret.json"
    secure_write_text(p, json.dumps({"lat": 38.68, "lon": 39.22}))
    _assert_owner_only(p)
    assert json.loads(p.read_text())["lat"] == 38.68


def test_overwrite_keeps_restrictive_mode(tmp_path):
    p = tmp_path / "secret.json"
    p.write_text("old")          # pre-existing, possibly world-readable
    os.chmod(p, 0o644)
    secure_write_text(p, "new")  # O_TRUNC overwrite
    _assert_owner_only(p)
    assert p.read_text() == "new"


def test_accepts_str_path(tmp_path):
    p = tmp_path / "s.txt"
    secure_write_text(str(p), "hello")
    assert p.read_text() == "hello"
    _assert_owner_only(p)


def test_unicode_roundtrip(tmp_path):
    p = tmp_path / "u.json"
    secure_write_text(p, json.dumps({"city": "Elâzığ"}, ensure_ascii=False))
    assert json.loads(p.read_text())["city"] == "Elâzığ"
