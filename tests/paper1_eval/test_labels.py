"""Tests for paper1_eval.labels CRUD."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from paper1_eval import labels


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "labels.db"
    labels.ensure_table(p)
    return p


def test_upsert_inserts_then_replaces(db):
    labels.upsert_label(db, 1, "faithful", labeler="alice")
    labels.upsert_label(db, 1, "hallucinated", notes="wrong date", labeler="alice")
    row = labels.get_label(db, 1)
    assert row["label"] == "hallucinated"
    assert row["notes"] == "wrong date"
    assert row["labeler"] == "alice"


def test_invalid_label_raises(db):
    with pytest.raises(ValueError):
        labels.upsert_label(db, 1, "bogus")


def test_label_counts_aggregates(db):
    labels.upsert_label(db, 1, "faithful")
    labels.upsert_label(db, 2, "hallucinated")
    labels.upsert_label(db, 3, "hallucinated")
    labels.upsert_label(db, 4, "partial")
    counts = labels.label_counts(db)
    assert counts == {
        "faithful": 1,
        "hallucinated": 2,
        "partial": 1,
        "unsure": 0,
    }


def test_labeled_message_ids_returns_set(db):
    labels.upsert_label(db, 5, "faithful")
    labels.upsert_label(db, 7, "partial")
    assert labels.labeled_message_ids(db) == {5, 7}


def test_iter_labels_orders_by_labeled_at(db):
    labels.upsert_label(db, 1, "faithful")
    labels.upsert_label(db, 2, "hallucinated")
    out = list(labels.iter_labels(db))
    assert [r["message_id"] for r in out] == [1, 2]


def test_get_label_returns_none_for_unknown(db):
    assert labels.get_label(db, 999) is None


def test_table_constraint_enforces_label_values(db):
    """The DB CHECK constraint backs up the Python validation."""
    conn = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO paper1_labels(message_id, label, labeled_at)"
                " VALUES (1, 'bogus', '2026-01-01T00:00:00')"
            )
    finally:
        conn.close()
