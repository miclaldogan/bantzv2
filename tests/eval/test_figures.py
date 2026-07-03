"""Tests for eval/loop_eval/figures.py (issue #515)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO / "eval" / "loop_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

matplotlib = pytest.importorskip(
    "matplotlib", reason="figures need the [eval] extra: matplotlib")

import figures  # noqa: E402


def test_synthetic_generates_all_assets(tmp_path):
    """Acceptance: generated from synthetic records day one, one command."""
    rc = figures.main(["--synthetic", "400", "--out-dir", str(tmp_path)])
    assert rc == 0

    fig1 = tmp_path / "fig1_success_by_class.pdf"
    fig2 = tmp_path / "fig2_selection_gap.pdf"
    table = tmp_path / "table_recovery_cost.tex"
    snippets = tmp_path / "figures.tex"
    for f in (fig1, fig2, table, snippets):
        assert f.exists(), f.name
    # real vector PDFs, not stubs
    assert fig1.stat().st_size > 5_000
    assert fig2.stat().st_size > 5_000
    assert fig1.read_bytes()[:5] == b"%PDF-"


def test_latex_snippets_are_column_width_and_complete(tmp_path):
    figures.main(["--synthetic", "200", "--out-dir", str(tmp_path)])
    snippets = (tmp_path / "figures.tex").read_text(encoding="utf-8")
    assert snippets.count("\\includegraphics[width=\\columnwidth]") == 2
    assert "fig1_success_by_class" in snippets
    assert "fig2_selection_gap" in snippets
    assert "\\input{figures/table_recovery_cost}" in snippets

    table = (tmp_path / "table_recovery_cost.tex").read_text(encoding="utf-8")
    for marker in ("\\toprule", "\\midrule", "\\bottomrule",
                   "recovery", "thrash on unrecoverable",
                   "tokens / recovery", "\\label{tab:recovery}"):
        assert marker in table, marker


def test_acl_font_and_grayscale_constraints():
    """No Type-3 fonts (ACL hard requirement); series identity is carried
    by lightness + hatch, never hue."""
    figures._mpl()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42

    # both series are pure grays (R==G==B) and far apart in lightness
    def gray_level(hexstr):
        r, g, b = (int(hexstr[i:i + 2], 16) for i in (1, 3, 5))
        assert r == g == b, f"{hexstr} is not a pure gray"
        return r

    light = gray_level(figures.SINGLE_SHOT["color"])
    dark = gray_level(figures.LOOPED["color"])
    assert abs(light - dark) > 80, "insufficient lightness separation"
    # the darker series additionally carries a hatch (CVD/print redundancy)
    assert figures.LOOPED["hatch"]
