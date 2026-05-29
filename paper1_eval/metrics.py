"""Paper-1 metrics computation.

Evaluates the finalizer hallucination detector against labeled rows.

Treats ``label='hallucinated'`` as the positive class; ``faithful`` and
``partial`` are negatives by default (``--strict`` flips ``partial`` to
positive). ``unsure`` labels are excluded.

Predictor: a finalizer row is "flagged" when
``finalizer_confidence < threshold``. We compute precision, recall, F1,
TP, FP, FN, TN at a single threshold, and an ROC sweep across many
thresholds.

Usage::

    python -m paper1_eval.metrics --db bantz.db
    python -m paper1_eval.metrics --db bantz.db --threshold 0.7 --strict
    python -m paper1_eval.metrics --db bantz.db --by-tool
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from bantz.core import eval_view
from bantz.core.memory import memory

from paper1_eval.labels import iter_labels


def labeled_eval_rows(db_path: Path) -> Iterator[dict]:
    """Yield eval rows that have a paper1_labels entry, with ``label`` attached."""
    memory.init(db_path)
    labels = {r["message_id"]: r["label"] for r in iter_labels(db_path)}
    if not labels:
        return
    for row in eval_view.fetch_eval_rows():
        lab = labels.get(row["message_id"])
        if lab is None:
            continue
        row["label"] = lab
        yield row


def confusion(
    rows: Iterable[dict],
    *,
    threshold: float = 0.8,
    strict: bool = False,
) -> dict:
    """Compute confusion matrix + derived metrics.

    A row is excluded if:
      - label == 'unsure', OR
      - finalizer_confidence is NULL (no detector verdict available)
    """
    tp = fp = fn = tn = 0
    excluded_unsure = 0
    excluded_no_conf = 0

    positives = {"hallucinated"}
    if strict:
        positives.add("partial")

    for row in rows:
        lab = row.get("label")
        if lab == "unsure":
            excluded_unsure += 1
            continue
        conf = row.get("finalizer_confidence")
        if conf is None:
            excluded_no_conf += 1
            continue
        actual_pos = lab in positives
        predicted_pos = conf < threshold
        if predicted_pos and actual_pos:
            tp += 1
        elif predicted_pos and not actual_pos:
            fp += 1
        elif not predicted_pos and actual_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n else 0.0

    return {
        "threshold": threshold,
        "strict": strict,
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "excluded_unsure": excluded_unsure,
        "excluded_no_confidence": excluded_no_conf,
    }


def roc_sweep(
    rows: Iterable[dict],
    *,
    thresholds: Iterable[float] | None = None,
    strict: bool = False,
) -> list[dict]:
    """Return a list of confusion-matrix dicts across thresholds."""
    materialised = list(rows)
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]  # 0.05 … 0.95
    return [
        confusion(materialised, threshold=t, strict=strict)
        for t in thresholds
    ]


def by_tool_breakdown(
    rows: Iterable[dict],
    *,
    threshold: float = 0.8,
    strict: bool = False,
) -> dict[str, dict]:
    """Group rows by tool and compute confusion per group."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        tool = row.get("finalizer_tool") or row.get("message_tool") or "(none)"
        grouped[tool].append(row)
    return {
        tool: confusion(grp, threshold=threshold, strict=strict)
        for tool, grp in grouped.items()
    }


def _fmt_row(d: dict) -> str:
    return (
        f"  t={d['threshold']:.2f}  "
        f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}  "
        f"Acc={d['accuracy']:.3f}  "
        f"TP={d['tp']}  FP={d['fp']}  FN={d['fn']}  TN={d['tn']}  "
        f"n={d['n']}"
    )


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", required=True, type=Path)
    p.add_argument(
        "--threshold", type=float, default=0.8,
        help="Detector threshold for the single-point report (default 0.8).",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Treat 'partial' labels as positive too (default: only 'hallucinated').",
    )
    p.add_argument(
        "--roc", action="store_true",
        help="Print a full ROC-like sweep across thresholds 0.05…0.95.",
    )
    p.add_argument(
        "--by-tool", action="store_true",
        help="Print per-tool confusion breakdowns at the chosen threshold.",
    )
    args = p.parse_args(argv)

    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2

    rows = list(labeled_eval_rows(args.db))
    if not rows:
        print("No labeled rows yet — run the label TUI first.")
        return 0

    print(f"Labeled rows: {len(rows)}")
    main = confusion(rows, threshold=args.threshold, strict=args.strict)
    print(_fmt_row(main))
    if main["excluded_unsure"] or main["excluded_no_confidence"]:
        print(
            f"  excluded: unsure={main['excluded_unsure']}"
            f"  no_confidence={main['excluded_no_confidence']}"
        )

    if args.roc:
        print("\nROC sweep:")
        for d in roc_sweep(rows, strict=args.strict):
            print(_fmt_row(d))

    if args.by_tool:
        print("\nPer-tool breakdown:")
        for tool, d in sorted(by_tool_breakdown(
            rows, threshold=args.threshold, strict=args.strict,
        ).items()):
            print(f"[{tool}]")
            print(_fmt_row(d))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
