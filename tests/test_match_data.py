"""Offline checks for the dashboard's win-prob / score core (no data/, no network).

match_data.wp() and calibrate() compute the headline live win-probability the
dashboard shows; they had zero coverage. These guard the distribution, lead
monotonicity, the full-time degenerate cases, and the pre-match anchoring claim
in the module docstring.

    python tests/test_match_data.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import match_data as MD  # noqa: E402


def test_wp_is_a_distribution():
    for d in (-2, -1, 0, 1, 2):
        for t in (0, 10, 45, 80, 89):
            w, dr, l = MD.wp(d, t, 1.6, 1.1)
            assert abs((w + dr + l) - 1.0) < 1e-9
            assert min(w, dr, l) >= 0.0


def test_wp_lead_helps_home():
    # at a fixed minute, a larger goal difference never lowers P(home win)
    t, mh, ma = 60, 1.6, 1.1
    ph = [MD.wp(d, t, mh, ma)[0] for d in (-2, -1, 0, 1, 2)]
    assert all(ph[i] <= ph[i + 1] + 1e-12 for i in range(len(ph) - 1))


def test_wp_terminal_is_degenerate():
    assert MD.wp(1, 90, 1.6, 1.1) == (1.0, 0.0, 0.0)    # leading at FT -> certain win
    assert MD.wp(0, 95, 1.6, 1.1) == (0.0, 1.0, 0.0)    # level past FT -> certain draw
    assert MD.wp(-1, 90, 1.6, 1.1) == (0.0, 0.0, 1.0)   # trailing at FT -> certain loss


def test_calibrate_reproduces_prematch_and_anchors():
    from scipy.stats import skellam
    p_h, p_d = 0.5, 0.27
    mh, ma = MD.calibrate(p_h, p_d)
    win = float(skellam.sf(0, mh, ma)); draw = float(skellam.pmf(0, mh, ma))
    assert abs(win - p_h) < 0.05 and abs(draw - p_d) < 0.05   # within the 0.05 grid step
    # the in-game curve is anchored: minute-0 win prob ~= the pre-match number
    assert abs(MD.wp(0, 0, mh, ma)[0] - p_h) < 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))
