"""
Bantz — secure local file writes.

Sensitive data files (GPS coordinates, saved places, user profile, session
state) must never exist on disk — even momentarily — with world-readable
permissions. ``Path.write_text()`` creates files with the process umask
(typically ``0o644``), then any ``chmod`` happens afterwards, leaving a brief
Time-of-Check-to-Time-of-Use (TOCTOU) window where another local user can read
the contents.

``secure_write_text`` creates the file with ``0o600`` from the very first
syscall, mirroring the pattern already used in ``auth/token_store.py``.
"""
from __future__ import annotations

import os
from pathlib import Path


def secure_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *path*, creating it owner-only (``0o600``) atomically.

    Uses ``os.open(O_CREAT|O_WRONLY|O_TRUNC, 0o600)`` + ``os.fchmod`` so the
    file is never momentarily world-readable — closing the TOCTOU window that
    ``Path.write_text()`` followed by ``chmod`` leaves open.
    """
    fd = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        f = os.fdopen(fd, "w", encoding=encoding)
    except BaseException:
        # fdopen never took ownership of the descriptor — close it ourselves.
        os.close(fd)
        raise
    with f:
        f.write(text)
