"""Tests for eval/loop_eval/analyze.py (issue #514)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import analyze  # noqa: E402
import schema  # noqa: E402


def _rates_valid(stats: dict) -> None:
    for s in stats["success_rate_per_condition"].values():
        assert s["success_rate"] is None or 0 <= s["success_rate"] <= 1
    for r in stats["recovery_by_failure_class"].values():
        assert r["recovery_fraction"] is None or 0 <= r["recovery_fraction"] <= 1


# ── acceptance: runs green on synthetic records day one ─────────────────────


def test_synthetic_self_check_green(capsys):
    rc = analyze.main(["--synthetic", "400"])
    assert rc == 0
    out = capsys.readouterr().out
    for section in ("Success rate per condition", "Paired per-task delta",
                    "Recovery fraction by failure class", "Headline gap",
                    "Never-retriggered residual", "Cost"):
        assert section in out, section


def test_synthetic_stats_shape():
    records = schema.synthetic_results(400)
    stats = analyze.analyze(records, {"files": 0, "invalid_schema": 0,
                                      "torn_lines": 0}, None)
    assert stats["n_records"] == 400
    _rates_valid(stats)
    # every failure class with eligible records gets a recovery number
    assert set(stats["recovery_by_failure_class"]) <= {
        "bad_args", "transient_error", "wrong_tool_first"}
    assert stats["thrash_on_unrecoverable"]["n"] > 0
    # synthetic data has no cross-condition task_id pairs — the analysis
    # must say so instead of inventing a CI
    for d in stats["paired_deltas_by_model"].values():
        assert d["n_pairs"] == 0
        assert d["ci95"] is None
        assert "CI withheld" in d["ci_note"]


# ── paired bootstrap CI on constructed pairs ─────────────────────────────────


def test_paired_delta_with_bootstrap_ci():
    base = [r for r in schema.synthetic_results(200)
            if r["condition"]["tool_loop_max_steps"] == 1][:40]
    records = []
    for r in base:
        r1 = copy.deepcopy(r)
        r1["success"] = False
        r1["recovery"] = False
        r1["loop_triggered"] = False
        records.append(r1)
        r3 = copy.deepcopy(r)
        r3["condition"] = dict(r3["condition"], tool_loop_max_steps=3)
        r3["success"] = True          # the loop condition wins every pair
        r3["loop_triggered"] = True
        r3["recovery"] = True
        records.append(r3)

    stats = analyze.paired_deltas(records)
    (model_stats,) = stats.values()
    assert model_stats["n_pairs"] == 40
    assert model_stats["mean_delta"] == 1.0
    lo, hi = model_stats["ci95"]
    assert lo == hi == 1.0  # constant deltas -> degenerate but honest CI


def test_recovery_fraction_definition():
    """recovery ÷ (recoverable AND iteration-1 failed) — §4 verbatim."""
    recs = schema.synthetic_results(400)
    stats = analyze.recovery_by_class(recs)
    for klass, entry in stats.items():
        eligible = [r for r in recs
                    if r["outcome_class"] in schema.DENOMINATOR_OUTCOMES
                    and r["recoverable"] is True
                    and r["failure_class"] == klass
                    and not r["iterations"][0]["result"]["success"]]
        assert entry["n_eligible"] == len(eligible)
        expect = sum(r["recovery"] for r in eligible) / len(eligible)
        assert abs(entry["recovery_fraction"] - expect) < 1e-9


# ── partial batches: named, never silent ─────────────────────────────────────


def test_partial_batch_reports_missing_tasks(tmp_path):
    records = schema.synthetic_results(60)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with open(results_dir / "cond.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write('{"torn": \n')  # a torn line — counted, not fatal

    loaded, report = analyze.load_records(results_dir)
    assert len(loaded) == 60
    assert report["torn_lines"] == 1

    expected = sorted({r["task_id"] for r in loaded}) + ["ghost_task.base"]
    stats = analyze.analyze(loaded, report, expected)
    md = analyze.to_markdown(stats)
    assert "INCOMPLETE" in md  # named loudly, never silent
    for cond in stats["completeness"].values():
        assert cond["complete"] is False
        assert "ghost_task.base" in cond["missing"]


def test_invalid_records_skipped_and_counted(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    good = schema.synthetic_results(10)
    bad = dict(good[0], schema_version="0.9")  # contract violation
    with open(results_dir / "cond.jsonl", "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(bad) + "\n")
    loaded, report = analyze.load_records(results_dir)
    assert len(loaded) == 10
    assert report["invalid_schema"] == 1
