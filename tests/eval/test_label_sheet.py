"""Tests for eval/loop_eval/label_sheet.py + TAXONOMY.md (issue #516)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import label_sheet  # noqa: E402
import schema  # noqa: E402


def _results_dir(tmp_path: Path, records) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    with open(d / "cond.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return d


def test_scope_selects_only_residual_failures(tmp_path):
    records = schema.synthetic_results(200)
    d = _results_dir(tmp_path, records)
    scoped = label_sheet.residual_failures(d)
    assert scoped, "synthetic data must contain residual failures"
    for r in scoped:
        assert r["success"] is False
        assert r["outcome_class"] in ("completed", "honest_giveup")
        assert r["condition"]["tool_loop_max_steps"] > 1


def test_sheet_has_evidence_columns_and_empty_labels(tmp_path):
    records = schema.synthetic_results(200)
    d = _results_dir(tmp_path, records)
    out = tmp_path / "sheet.csv"
    assert label_sheet.main(["--results", str(d), "--out", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert set(label_sheet.COLUMNS) == set(rows[0].keys())
    for row in rows:
        assert row["label"] == "" and row["notes"] == ""
        assert row["iteration_summary"]  # transcript evidence inline


def test_stratified_sample_is_reproducible_and_covers_classes(tmp_path):
    records = schema.synthetic_results(400)
    d = _results_dir(tmp_path, records)
    out1, out2 = tmp_path / "s1.csv", tmp_path / "s2.csv"
    label_sheet.main(["--results", str(d), "--sample", "20", "--out", str(out1)])
    label_sheet.main(["--results", str(d), "--sample", "20", "--out", str(out2)])
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    with open(out1, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    classes = {r["failure_class"] for r in rows}
    assert len(classes) >= 3  # stratification reached multiple classes


def test_agreement_kappa(tmp_path):
    header = ",".join(label_sheet.COLUMNS)
    blank = "," * (len(label_sheet.COLUMNS) - 12)

    def sheet(path, labels):
        lines = [header]
        for i, lab in enumerate(labels):
            lines.append(f"t{i},c,none,,completed,False,1,x,resp,tp,{lab},"
                         + blank)
        path.write_text("\n".join(lines), encoding="utf-8")

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    labels_a = ["honest-give-up"] * 10 + ["other"] * 10
    sheet(a, labels_a)
    sheet(b, labels_a)  # perfect agreement
    report = label_sheet.agreement(a, b)
    assert report["raw_agreement"] == 1.0
    assert report["cohens_kappa"] == 1.0
    assert "PASS" in report["gate"]

    labels_b = ["honest-give-up"] * 10 + ["honest-give-up"] * 10
    sheet(b, labels_b)  # b collapses to one label -> poor kappa
    report = label_sheet.agreement(a, b)
    assert report["cohens_kappa"] < 0.7
    assert "FAIL" in report["gate"]
    assert len(report["disagreements"]) == 10


def test_taxonomy_doc_pre_registered_and_ordered():
    text = (EVAL_DIR / "TAXONOMY.md").read_text(encoding="utf-8")
    for label in label_sheet.LABELS:
        assert label in text, label
    assert "IN ORDER" in text                 # ordering IS the tie-break
    assert "κ ≥ 0.7" in text                  # agreement gate
    assert "version bump" in text.lower()     # pre-registration discipline
