"""Terminal labeler for paper-1 eval.

Cycles through unlabeled rows in :data:`paper1_eval` and asks the user
for a verdict. Writes to :data:`paper1_labels` immediately so progress
survives an interrupted session.

Keys
----
``f`` faithful · ``h`` hallucinated · ``p`` partial · ``u`` unsure
``s`` skip (do not record) · ``b`` back (re-label previous)
``q`` quit and save

Usage::

    python -m paper1_eval.label_tui \\
        --db ~/.local/share/bantz/bantz.db \\
        --labeler memre
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Optional

from bantz.core import eval_view
from bantz.core.memory import memory

from paper1_eval.labels import (
    LABEL_VALUES,
    label_counts,
    labeled_message_ids,
    upsert_label,
)


_KEY_TO_LABEL = {
    "f": "faithful",
    "h": "hallucinated",
    "p": "partial",
    "u": "unsure",
}


def _truncate(text: Optional[str], width: int = 100, lines: int = 12) -> str:
    if not text:
        return "(empty)"
    out = []
    for line in text.splitlines()[:lines]:
        if len(line) > width:
            line = line[:width - 1] + "…"
        out.append(line)
    if len(text.splitlines()) > lines:
        out.append(f"… ({len(text.splitlines()) - lines} more lines)")
    return "\n".join(out)


def _render(row: dict, idx: int, total: int) -> str:
    bar = "─" * 78
    return textwrap.dedent(f"""
    ╭{bar}╮
    │ paper-1 labeler  [{idx + 1}/{total}]  message_id={row['message_id']}
    │ tool={row.get('finalizer_tool') or row.get('message_tool') or '?'}    mode={row.get('finalizer_mode') or '?'}    conf={row.get('finalizer_confidence')}    flagged={row.get('finalizer_flagged')}
    │ route={row.get('route') or '?'}  route_conf={row.get('route_confidence')}  source={row.get('route_source') or '?'}
    ╰{bar}╯

    USER INPUT
    {_truncate(row.get('route_en_input') or row.get('finalizer_user_input'))}

    TOOL OUTPUT
    {_truncate(row.get('tool_output'))}

    RESPONSE
    {_truncate(row.get('response_text'))}
    """).strip()


def _prompt() -> str:
    sys.stdout.write(
        "\n[f]aithful  [h]allucinated  [p]artial  [u]nsure  "
        "[s]kip  [b]ack  [q]uit > "
    )
    sys.stdout.flush()
    return sys.stdin.readline().strip().lower()


def _candidate_rows(
    db_path: Path,
    flagged_only: bool,
    tool_filter: Optional[str],
    limit: Optional[int],
) -> list[dict]:
    already = labeled_message_ids(db_path)
    rows = []
    for row in eval_view.fetch_eval_rows(
        flagged_only=flagged_only,
        tool_filter=tool_filter,
        limit=None,
    ):
        if row["message_id"] in already:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def run(
    db_path: Path,
    *,
    labeler: Optional[str] = None,
    flagged_only: bool = False,
    tool_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, int]:
    """Run the interactive labeler. Returns the final label_counts dict."""
    memory.init(db_path)
    rows = _candidate_rows(db_path, flagged_only, tool_filter, limit)
    total = len(rows)
    if total == 0:
        print("No unlabeled rows match the filter. Done.")
        return label_counts(db_path)

    print(f"Loaded {total} unlabeled candidate rows.\n")

    idx = 0
    history: list[int] = []
    while idx < total:
        row = rows[idx]
        print(_render(row, idx, total))
        key = _prompt()

        if key == "q":
            print("\nSaving and quitting.")
            break
        if key == "s":
            idx += 1
            continue
        if key == "b":
            if not history:
                print("(no previous row to re-label)")
                continue
            idx = history.pop()
            continue
        verdict = _KEY_TO_LABEL.get(key)
        if not verdict:
            print(f"(unknown key {key!r}; expected one of f/h/p/u/s/b/q)")
            continue
        upsert_label(
            db_path, row["message_id"], verdict, labeler=labeler,
        )
        history.append(idx)
        idx += 1
        print(f"→ labeled {row['message_id']} as {verdict}")

    counts = label_counts(db_path)
    total_labeled = sum(counts.values())
    print(f"\nFinal counts ({total_labeled} total): {counts}")
    return counts


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", required=True, type=Path)
    p.add_argument(
        "--labeler", default=None,
        help="Optional name/id for the labeler (stored alongside each label).",
    )
    p.add_argument(
        "--flagged-only", action="store_true",
        help="Only label rows the detector flagged.",
    )
    p.add_argument("--tool", default=None)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of unlabeled rows pulled into this session.",
    )
    args = p.parse_args(argv)

    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2

    run(
        args.db,
        labeler=args.labeler,
        flagged_only=args.flagged_only,
        tool_filter=args.tool,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
