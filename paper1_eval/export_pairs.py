"""Dump joined paper-1 eval rows to JSONL.

Joins :data:`paper1_eval` rows from :mod:`bantz.core.eval_view` with any
existing :data:`paper1_labels` rows. Output is one JSON object per line.

Usage::

    python -m paper1_eval.export_pairs \\
        --db ~/.local/share/bantz/bantz.db \\
        --out paper1_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bantz.core import eval_view
from bantz.core.memory import memory

from paper1_eval.labels import iter_labels


def export(
    db_path: Path,
    out_path: Path,
    *,
    flagged_only: bool = False,
    tool_filter: str | None = None,
    limit: int | None = None,
) -> int:
    """Write JSONL rows to ``out_path``. Returns row count."""
    memory.init(db_path)

    # Load labels once and index by message_id.
    labels_by_msg: dict[int, dict] = {}
    for row in iter_labels(db_path):
        labels_by_msg[row["message_id"]] = row

    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in eval_view.fetch_eval_rows(
            flagged_only=flagged_only,
            tool_filter=tool_filter,
            limit=limit,
        ):
            label = labels_by_msg.get(row["message_id"])
            if label:
                row["label"] = label["label"]
                row["label_notes"] = label["notes"]
                row["labeler"] = label["labeler"]
                row["labeled_at"] = label["labeled_at"]
            else:
                row["label"] = None
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--db", required=True, type=Path,
        help="Path to bantz.db (the SQLite store).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output JSONL path.",
    )
    p.add_argument(
        "--flagged-only", action="store_true",
        help="Only export rows where finalizer_flagged = 1.",
    )
    p.add_argument(
        "--tool", default=None,
        help="Restrict to a specific tool name (e.g. weather, gmail).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of exported rows.",
    )
    args = p.parse_args(argv)

    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = export(
        args.db, args.out,
        flagged_only=args.flagged_only,
        tool_filter=args.tool,
        limit=args.limit,
    )
    print(f"wrote {n} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
