"""Offline correctness checks for the cooling-break thesis core (no network).

Guards the load-bearing logic the paper rests on and that had no test: the
stoppage classifier (the detector itself -- a threshold change here moved the
headline break count), the pitch->grid mapping, and the learned xT surface.

    python tests/test_thesis.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze import NX, NY, cell, classify              # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def S(**kw):
    """a stoppage row with neutral defaults, overridable per test"""
    base = dict(goal_before=0, injury=0, card_near=0, gap_sec=0,
                half_min=20, period=1, subs_in=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_classify_drinks_break():
    # a long clean pause in the 25-31 detection window of a regulation half
    assert classify(S(gap_sec=120, half_min=28, period=2)) == "drinks_break"
    assert classify(S(gap_sec=120, half_min=25, period=1)) == "drinks_break"  # lower edge
    assert classify(S(gap_sec=120, half_min=32, period=1)) != "drinks_break"  # 32 excluded
    assert classify(S(gap_sec=89, half_min=28, period=1)) != "drinks_break"   # gap too short
    assert classify(S(gap_sec=300, half_min=28, period=3)) != "drinks_break"  # extra time excluded


def test_classify_context_precedence():
    # a break-shaped gap is NOT a drinks break if a goal/injury/card explains it
    assert classify(S(gap_sec=120, half_min=28, goal_before=1)) == "goal_restart"
    assert classify(S(gap_sec=120, half_min=28, injury=1)) == "injury"
    assert classify(S(gap_sec=120, half_min=28, card_near=1)) == "card_or_var"


def test_classify_other_and_substitution():
    assert classify(S(gap_sec=30, half_min=10)) == "other"
    assert classify(S(gap_sec=30, half_min=10, subs_in=2)) == "substitution"


def test_cell_stays_in_bounds():
    cx, cy = cell([0, 60, 120, 999], [0, 40, 80, -5])
    assert (cx >= 0).all() and (cx < NX).all()
    assert (cy >= 0).all() and (cy < NY).all()
    assert cx[0] == 0 and cy[0] == 0                     # own corner -> cell (0,0)


def test_xt_surface_is_sane():
    grid = np.load(os.path.join(PROC, "xt_grid.npy"))
    assert grid.shape == (NX, NY)
    assert np.isfinite(grid).all()
    assert (grid >= 0).all() and (grid <= 1).all()       # it is a probability
    # threat rises toward the goal being attacked (high x), the whole point of xT
    assert grid[-3:, :].mean() > grid[:3, :].mean()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))
