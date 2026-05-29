"""Tests for paper1_eval.metrics confusion + ROC math."""
from __future__ import annotations

from paper1_eval import metrics


def _row(*, label: str, confidence: float, tool: str = "shell") -> dict:
    return {
        "message_id": 0,
        "finalizer_confidence": confidence,
        "finalizer_tool": tool,
        "message_tool": tool,
        "label": label,
    }


def test_confusion_basic_counts():
    rows = [
        _row(label="hallucinated", confidence=0.4),  # TP at t=0.8
        _row(label="hallucinated", confidence=0.9),  # FN at t=0.8
        _row(label="faithful", confidence=0.4),       # FP at t=0.8
        _row(label="faithful", confidence=0.9),       # TN at t=0.8
    ]
    d = metrics.confusion(rows, threshold=0.8)
    assert d["tp"] == 1
    assert d["fn"] == 1
    assert d["fp"] == 1
    assert d["tn"] == 1
    assert d["n"] == 4
    assert abs(d["precision"] - 0.5) < 1e-9
    assert abs(d["recall"] - 0.5) < 1e-9
    assert abs(d["f1"] - 0.5) < 1e-9
    assert abs(d["accuracy"] - 0.5) < 1e-9


def test_confusion_excludes_unsure_and_missing_confidence():
    rows = [
        _row(label="unsure", confidence=0.5),
        _row(label="hallucinated", confidence=None),
        _row(label="faithful", confidence=0.9),  # TN at t=0.8
    ]
    d = metrics.confusion(rows, threshold=0.8)
    assert d["n"] == 1
    assert d["excluded_unsure"] == 1
    assert d["excluded_no_confidence"] == 1
    assert d["tn"] == 1


def test_confusion_strict_treats_partial_as_positive():
    rows = [
        _row(label="partial", confidence=0.4),
        _row(label="partial", confidence=0.9),
    ]
    lenient = metrics.confusion(rows, threshold=0.8, strict=False)
    strict = metrics.confusion(rows, threshold=0.8, strict=True)
    # Lenient: 'partial' is negative → first is FP, second is TN
    assert (lenient["fp"], lenient["tn"]) == (1, 1)
    assert (lenient["tp"], lenient["fn"]) == (0, 0)
    # Strict: 'partial' is positive → first is TP, second is FN
    assert (strict["tp"], strict["fn"]) == (1, 1)
    assert (strict["fp"], strict["tn"]) == (0, 0)


def test_roc_sweep_returns_one_row_per_threshold():
    rows = [
        _row(label="hallucinated", confidence=0.3),
        _row(label="faithful", confidence=0.85),
    ]
    sweep = metrics.roc_sweep(rows)
    assert len(sweep) == 19  # 0.05 … 0.95 in 0.05 steps
    # At t=0.5 the hallucination (conf 0.3) is flagged → TP=1, FP=0
    mid = [d for d in sweep if abs(d["threshold"] - 0.5) < 1e-9][0]
    assert mid["tp"] == 1
    assert mid["tn"] == 1
    # At t=0.95 even the faithful row is flagged → TP=1, FP=1
    high = [d for d in sweep if abs(d["threshold"] - 0.95) < 1e-9][0]
    assert high["fp"] == 1


def test_by_tool_breakdown_groups_correctly():
    rows = [
        _row(label="hallucinated", confidence=0.4, tool="gmail"),
        _row(label="faithful", confidence=0.9, tool="gmail"),
        _row(label="hallucinated", confidence=0.4, tool="weather"),
    ]
    breakdown = metrics.by_tool_breakdown(rows, threshold=0.8)
    assert set(breakdown.keys()) == {"gmail", "weather"}
    assert breakdown["gmail"]["tp"] == 1
    assert breakdown["gmail"]["tn"] == 1
    assert breakdown["weather"]["tp"] == 1
    assert breakdown["weather"]["n"] == 1


def test_precision_recall_zero_when_no_positives():
    rows = [_row(label="faithful", confidence=0.9)]
    d = metrics.confusion(rows, threshold=0.8)
    assert d["precision"] == 0.0
    assert d["recall"] == 0.0
    assert d["f1"] == 0.0
