"""Figures + tables generation — ACL two-column, grayscale-safe (issue #515).

Produces camera-ready assets from ``analyze.py`` output. One command
regenerates everything when real data lands:

    python eval/loop_eval/figures.py --synthetic 400 --out-dir figures/
    python eval/loop_eval/figures.py --stats results/b1/stats.json --out-dir figures/

Outputs (under --out-dir):

    fig1_success_by_class.pdf   single-shot vs looped success per failure
                                class — the money figure
    fig2_selection_gap.pdf      selection accuracy vs end-to-end success
    table_recovery_cost.tex     per-class recovery fraction + cost/recovery
    figures.tex                 \\includegraphics + caption stubs, ready to
                                \\input into the paper

Design constraints baked in (do not "fix" them):
- **Grayscale-legible** (ACL accessibility): series identity is carried by
  LIGHTNESS + HATCH, never hue. Printed on a mono laser it reads the same.
- **Column width 7.7 cm** (3.03 in); fonts sized so nothing dips below ~7 pt
  at final size; captions live in LaTeX at 10 pt, figures carry no titles.
- **Vector PDF, TrueType embedding** (``pdf.fonttype=42``) — ACL rejects
  Type-3 fonts.
- Bars start at 0, one axis, recessive grid, legend present (2 series),
  selective direct value labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COL_WIDTH_IN = 3.03  # 7.7 cm ACL column width

# Grayscale series encoding: lightness + hatch, not hue.
SINGLE_SHOT = {"color": "#c9c9c9", "hatch": "", "label": "single-shot (steps=1)"}
LOOPED = {"color": "#4a4a4a", "hatch": "///", "label": "looped (steps>1)"}
EDGE = "#1a1a1a"

CLASS_ORDER = ["none", "bad_args", "transient_error", "wrong_tool_first",
               "unrecoverable"]
CLASS_LABELS = {"none": "none\n(base)", "bad_args": "bad\nargs",
                "transient_error": "trans-\nient", "wrong_tool_first":
                "wrong\ntool", "unrecoverable": "unre-\ncoverable"}


def _mpl():
    try:
        import matplotlib
    except ImportError:
        raise SystemExit(
            "figures.py needs matplotlib: pip install -e '.[eval]'")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,  # no Type-3 fonts (ACL)
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#dddddd",
        "grid.linewidth": 0.5, "axes.axisbelow": True,
        "hatch.linewidth": 0.5,
    })
    return plt


def _split_conditions(stats: dict) -> tuple[str | None, str | None]:
    """(steps=1 condition key, steps>1 condition key) — first of each."""
    single = looped = None
    for key in stats["success_rate_per_condition"]:
        steps = int(key.rsplit("steps", 1)[1])
        if steps == 1 and single is None:
            single = key
        elif steps > 1 and looped is None:
            looped = key
    return single, looped


# ── Fig 1: the money figure ──────────────────────────────────────────────────

def fig1_success_by_class(stats: dict, out: Path) -> Path:
    plt = _mpl()
    single_key, looped_key = _split_conditions(stats)
    by_class = stats["success_by_class_per_condition"]
    classes = [c for c in CLASS_ORDER
               if c in by_class.get(single_key, {})
               or c in by_class.get(looped_key, {})]

    def rates(key):
        return [by_class.get(key, {}).get(c, {}).get("success_rate")
                for c in classes]

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 1.9))
    x = range(len(classes))
    width = 0.38
    for offset, series, key in ((-width / 2, SINGLE_SHOT, single_key),
                                (width / 2, LOOPED, looped_key)):
        if key is None:
            continue
        vals = [v if v is not None else 0.0 for v in rates(key)]
        bars = ax.bar([i + offset for i in x], vals, width,
                      color=series["color"], hatch=series["hatch"],
                      edgecolor=EDGE, linewidth=0.6, label=series["label"])
        # Selective direct labels: value on top of each bar (2 series ×
        # ≤5 classes stays uncluttered; text in ink, not series color).
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}".lstrip("0"),
                        (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=6, color="#1a1a1a")

    ax.set_xticks(list(x))
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in classes])
    ax.set_ylabel("end-to-end success rate")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # Legend OUTSIDE the axes (above) — tall bars own the full plot area.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.4, ncol=2, columnspacing=1.0, borderaxespad=0)
    fig.tight_layout(pad=0.4)
    path = out / "fig1_success_by_class.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Fig 2: selection accuracy vs end-to-end success (the gap) ────────────────

def fig2_selection_gap(stats: dict, out: Path) -> Path:
    plt = _mpl()
    conds = stats["success_rate_per_condition"]
    keys = sorted(conds)
    sel = [conds[k]["selection_accuracy"] or 0 for k in keys]
    succ = [conds[k]["success_rate"] or 0 for k in keys]
    labels = [f"steps={k.rsplit('steps', 1)[1]}" for k in keys]

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 1.7))
    x = range(len(keys))
    width = 0.38
    bars_sel = ax.bar([i - width / 2 for i in x], sel, width,
                      color="#c9c9c9", edgecolor=EDGE, linewidth=0.6,
                      label="tool selection correct")
    bars_succ = ax.bar([i + width / 2 for i in x], succ, width,
                       color="#4a4a4a", hatch="///", edgecolor=EDGE,
                       linewidth=0.6, label="end-to-end success")
    for bars, vals in ((bars_sel, sel), (bars_succ, succ)):
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}".lstrip("0"),
                        (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=6, color="#1a1a1a")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.4, ncol=2, columnspacing=1.0, borderaxespad=0)
    fig.tight_layout(pad=0.4)
    path = out / "fig2_selection_gap.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Table: per-class recovery + cost per recovery ────────────────────────────

def table_recovery_cost(stats: dict, out: Path) -> Path:
    rec = stats["recovery_by_failure_class"]
    thrash = stats["thrash_on_unrecoverable"]
    # cost per recovery is per condition; take the looped condition
    _, looped_key = _split_conditions(stats)
    cost = stats["cost"].get(looped_key, {}) if looped_key else {}

    def fmt(v, digits=2):
        return "--" if v is None else f"{v:.{digits}f}"

    rows = []
    for klass in ("bad_args", "transient_error", "wrong_tool_first"):
        if klass in rec:
            r = rec[klass]
            rows.append(f"  {klass.replace('_', ' ')} & {r['n_eligible']} & "
                        f"{fmt(r['recovery_fraction'])} \\\\")
    lines = [
        "% Generated by eval/loop_eval/figures.py (#515) — do not hand-edit.",
        "\\begin{table}[t]",
        "  \\centering\\small",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    failure class & $n$ & recovery \\\\",
        "    \\midrule",
        *rows,
        "    \\midrule",
        f"    thrash on unrecoverable & {thrash['n']} & "
        f"{fmt(thrash['thrash_rate'])} \\\\",
        f"    tokens / recovery & \\multicolumn{{2}}{{r}}{{"
        f"{fmt(cost.get('cost_per_recovery_tokens'), 1)}}} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Recovery fraction per injected failure class "
        "(recoverable tasks whose first tool call failed), thrash rate on "
        "the unrecoverable class, and mean loop-overhead tokens per "
        "successful recovery.}",
        "  \\label{tab:recovery}",
        "\\end{table}",
    ]
    path = out / "table_recovery_cost.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def latex_snippets(out: Path) -> Path:
    snippet = r"""% Generated by eval/loop_eval/figures.py (#515) — \input{figures/figures}
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig1_success_by_class}
  \caption{End-to-end success by injected failure class, single-shot
  (steps=1) vs.\ looped (steps$>$1). Bars are grayscale- and
  CVD-safe (lightness + hatching).}
  \label{fig:success-by-class}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig2_selection_gap}
  \caption{Correct tool selection vs.\ end-to-end success per condition —
  the necessary-but-not-sufficient gap.}
  \label{fig:selection-gap}
\end{figure}

\input{figures/table_recovery_cost}
"""
    path = out / "figures.tex"
    path.write_text(snippet, encoding="utf-8")
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", help="stats JSON produced by analyze.py")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="build from N synthetic records (day one)")
    parser.add_argument("--out-dir", default=str(HERE / "figures"))
    args = parser.parse_args(argv)

    if args.synthetic:
        import analyze
        import schema
        records = schema.synthetic_results(args.synthetic)
        stats = analyze.analyze(records, {"files": 0, "invalid_schema": 0,
                                          "torn_lines": 0}, None)
    elif args.stats:
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    else:
        parser.error("pass --stats FILE or --synthetic N")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced = [
        fig1_success_by_class(stats, out),
        fig2_selection_gap(stats, out),
        table_recovery_cost(stats, out),
        latex_snippets(out),
    ]
    for p in produced:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
