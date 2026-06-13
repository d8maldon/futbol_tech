"""Offline correctness checks for the WC predictor (no network).

Guards the parts most likely to be silently wrong: the 2026 head-to-head-first
group tiebreaker, the ranked-probability-score axis, and the Elo invariants.

    python tests/test_predictor.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import rps                                    # noqa: E402
from montecarlo import rank_group                           # noqa: E402
from ratings import (davidson_proba, elo_update, expected,  # noqa: E402
                     g_mult)


def test_h2h_first_tiebreaker():
    """2026 uses head-to-head BEFORE overall goal difference."""
    rt = {t: 1500 for t in "ABCD"}
    # A and B both finish on 6 pts; A has the far better overall GD (+7 vs +1),
    # but B won the head-to-head 1-0, so B must rank above A.
    results = [("A", "C", 5, 0), ("A", "D", 3, 0), ("B", "A", 1, 0),
               ("C", "B", 1, 0), ("B", "D", 1, 0), ("D", "C", 2, 0)]
    order, _ = rank_group(list("ABCD"), results, rt)
    assert order[:2] == ["B", "A"], order


def test_rps_axis():
    perfect = {"H": 1.0, "D": 0.0, "A": 0.0}
    assert abs(rps(perfect, "H")) < 1e-9
    worst = {"H": 1.0, "D": 0.0, "A": 0.0}
    assert abs(rps(worst, "A") - 1.0) < 1e-9          # confident & opposite
    # ordinal: missing by one step (predict away, draw happens) beats missing by
    # two (predict away, home wins)
    away = {"H": 0.0, "D": 0.0, "A": 1.0}
    assert rps(away, "D") < rps(away, "H")


def test_elo_invariants():
    assert abs(expected(0) - 0.5) < 1e-9
    assert expected(200) > 0.5 > expected(-200)
    assert (g_mult(1), g_mult(2), g_mult(3)) == (1.0, 1.5, 1.75)
    # zero-sum: a win moves both ratings by the same amount, opposite signs
    nh, na = elo_update(1500, 1500, 2, 0, 60, ha=0)
    assert abs((nh - 1500) + (na - 1500)) < 1e-9 and nh > 1500 > na


def test_davidson_proba():
    for dr in (-300, 0, 250):
        p = davidson_proba(dr, 0.78)
        assert abs(sum(p.values()) - 1.0) < 1e-9
    even = davidson_proba(0, 0.78)
    assert abs(even["H"] - even["A"]) < 1e-9           # symmetric at dr=0
    assert 0.24 < even["D"] < 0.32                     # draw rate near base rate


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))
