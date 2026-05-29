"""Paper-1 label storage.

The eval needs human verdicts on finalizer outputs. We store them in a
``paper1_labels`` table keyed by ``messages.id`` (the assistant message
that the labeled response belongs to). One label per message.

Label values
------------
* ``faithful``    — response is fully supported by the tool output.
* ``hallucinated``— response contains fabricated content not in the tool output.
* ``partial``     — response is mostly grounded but adds one or more
                    unsupported claims.
* ``unsure``      — labeler cannot determine; row is excluded from metrics.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("paper1_eval.labels")

LABEL_VALUES = ("faithful", "hallucinated", "partial", "unsure")

_LABELS_DDL = """
CREATE TABLE IF NOT EXISTS paper1_labels (
  message_id INTEGER PRIMARY KEY,
  label      TEXT NOT NULL CHECK(label IN ('faithful','hallucinated','partial','unsure')),
  notes      TEXT,
  labeler    TEXT,
  labeled_at TEXT NOT NULL
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_LABELS_DDL)
    return conn


def ensure_table(db_path: Path) -> None:
    """Create the paper1_labels table if missing. Idempotent."""
    _connect(db_path).close()


def upsert_label(
    db_path: Path,
    message_id: int,
    label: str,
    *,
    notes: Optional[str] = None,
    labeler: Optional[str] = None,
) -> None:
    """Insert or replace a label for ``message_id``."""
    if label not in LABEL_VALUES:
        raise ValueError(
            f"label must be one of {LABEL_VALUES}, got {label!r}"
        )
    now = datetime.now().isoformat(timespec="seconds")
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO paper1_labels"
            "(message_id, label, notes, labeler, labeled_at)"
            " VALUES (?,?,?,?,?)",
            (message_id, label, notes, labeler, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_label(db_path: Path, message_id: int) -> Optional[dict]:
    """Return the label row for ``message_id`` or None."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT message_id, label, notes, labeler, labeled_at"
            " FROM paper1_labels WHERE message_id = ?",
            (message_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def iter_labels(db_path: Path) -> Iterator[dict]:
    """Yield every label row, oldest first."""
    conn = _connect(db_path)
    try:
        for row in conn.execute(
            "SELECT message_id, label, notes, labeler, labeled_at"
            " FROM paper1_labels ORDER BY labeled_at"
        ):
            yield dict(row)
    finally:
        conn.close()


def label_counts(db_path: Path) -> dict[str, int]:
    """Return {label_value: count} across all labeled rows."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT label, COUNT(*) FROM paper1_labels GROUP BY label"
        ).fetchall()
    finally:
        conn.close()
    counts = {v: 0 for v in LABEL_VALUES}
    for r in rows:
        counts[r[0]] = r[1]
    return counts


def labeled_message_ids(db_path: Path) -> set[int]:
    """Return the set of ``message_id`` values that already have a label."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT message_id FROM paper1_labels"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}
