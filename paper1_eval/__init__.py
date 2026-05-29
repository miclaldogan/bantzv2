"""Paper-1 evaluation harness for Bantz hallucination detection.

Modules
-------
labels        — paper1_labels SQLite table + CRUD
export_pairs  — dump joined eval rows (+ labels if present) to JSONL
label_tui     — terminal labeler that cycles unlabeled pairs
metrics       — precision/recall/F1, ROC sweeps, per-tool breakdown

All modules are runnable as scripts: ``python -m paper1_eval.<name> --help``.
"""
