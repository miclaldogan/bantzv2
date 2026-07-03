"""Labeling sheet for the residual-failure taxonomy (issue #516).

Consumes runner results directly and emits a CSV one row per residual
failure, ready for the two-person labeling pass described in TAXONOMY.md:

    python eval/loop_eval/label_sheet.py --results results/BATCH --out sheet.csv
    python eval/loop_eval/label_sheet.py --results results/BATCH --sample 20 --out sheet.csv
    python eval/loop_eval/label_sheet.py --agreement sheetA.csv sheetB.csv

Scope rule (from TAXONOMY.md): outcome_class ∈ {completed, honest_giveup},
success = false, tool_loop_max_steps > 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SAMPLE_SEED = 516  # fixed: the stratified sample must be reproducible

COLUMNS = ["task_id", "condition", "failure_class", "recoverable",
           "outcome_class", "loop_triggered", "iterations_used",
           "iteration_summary", "final_response_excerpt", "transcript_path",
           "label", "notes"]

LABELS = ["fabricated-success", "never-observed-the-real-error",
          "re-selected-same-wrong-tool",
          "correct-re-decision-but-arg-degradation",
          "budget-exhausted-mid-recovery", "honest-give-up", "other"]


def residual_failures(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("outcome_class") in ("completed", "honest_giveup")
                    and rec.get("success") is False
                    and rec.get("condition", {}).get(
                        "tool_loop_max_steps", 0) > 1):
                records.append(rec)
    return records


def _iteration_summary(rec: dict) -> str:
    parts = []
    for it in rec.get("iterations", []):
        result = it.get("result", {})
        state = "ok" if result.get("success") else \
            f"ERR:{(result.get('error') or it.get('exception') or '?')[:60]}"
        args = ",".join(f"{k}={str(v)[:20]}" for k, v in
                        sorted(it.get("tool_args", {}).items())[:3])
        parts.append(f"[{it.get('index')}] {it.get('tool_name') or it.get('route')}"
                     f"({args}) -> {state}")
    return " | ".join(parts)


def _stratified_sample(records: list[dict], n: int) -> list[dict]:
    """≥3 per failure class where available (TAXONOMY protocol), fixed seed."""
    rng = random.Random(SAMPLE_SEED)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["failure_class"]].append(r)
    picked: list[dict] = []
    for klass in sorted(by_class):
        pool = sorted(by_class[klass], key=lambda r: r["task_id"])
        rng.shuffle(pool)
        picked.extend(pool[:3])
    remaining = sorted((r for r in records if r not in picked),
                       key=lambda r: r["task_id"])
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, n - len(picked))])
    return picked[:n]


def write_sheet(records: list[dict], out: Path) -> None:
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r["task_id"]):
            cond = rec["condition"]
            writer.writerow({
                "task_id": rec["task_id"],
                "condition": f"{cond['model']}|steps{cond['tool_loop_max_steps']}",
                "failure_class": rec["failure_class"],
                "recoverable": rec["recoverable"],
                "outcome_class": rec["outcome_class"],
                "loop_triggered": rec["loop_triggered"],
                "iterations_used": rec["iterations_used"],
                "iteration_summary": _iteration_summary(rec),
                "final_response_excerpt": rec.get("final_response_excerpt", ""),
                "transcript_path": rec.get("transcript_path", ""),
                "label": "",   # one of LABELS — see TAXONOMY.md, rules in order
                "notes": "",
            })


def agreement(sheet_a: Path, sheet_b: Path) -> dict:
    """Raw agreement + Cohen's κ between two labeled sheets."""
    def load(path: Path) -> dict[str, str]:
        with open(path, encoding="utf-8", newline="") as f:
            return {row["task_id"]: row["label"].strip()
                    for row in csv.DictReader(f) if row["label"].strip()}

    a, b = load(sheet_a), load(sheet_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no commonly-labeled rows between the two sheets")
    pairs = [(a[t], b[t]) for t in common]
    po = sum(x == y for x, y in pairs) / len(pairs)
    # chance agreement from marginal label distributions
    labels = sorted({x for p in pairs for x in p})
    pe = sum(
        (sum(x == lab for x, _ in pairs) / len(pairs))
        * (sum(y == lab for _, y in pairs) / len(pairs))
        for lab in labels
    )
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    disagreements = [t for t in common if a[t] != b[t]]
    return {"n": len(pairs), "raw_agreement": round(po, 3),
            "cohens_kappa": round(kappa, 3),
            "gate": "PASS (κ ≥ 0.7)" if kappa >= 0.7 else
                    "FAIL — revise rubric (version bump), relabel 10 fresh",
            "disagreements": disagreements}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="results/{batch} directory")
    parser.add_argument("--out", default="label_sheet.csv")
    parser.add_argument("--sample", type=int, default=0,
                        help="stratified sample size (0 = all)")
    parser.add_argument("--agreement", nargs=2, metavar=("A", "B"),
                        help="two labeled sheets -> raw agreement + κ")
    args = parser.parse_args(argv)

    if args.agreement:
        report = agreement(Path(args.agreement[0]), Path(args.agreement[1]))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["cohens_kappa"] >= 0.7 else 1

    if not args.results:
        parser.error("pass --results DIR or --agreement A B")
    records = residual_failures(Path(args.results))
    if not records:
        print("no residual failures in scope — nothing to label")
        return 0
    if args.sample:
        records = _stratified_sample(records, args.sample)
    write_sheet(records, Path(args.out))
    print(f"{len(records)} row(s) -> {args.out}  (labels: {', '.join(LABELS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
