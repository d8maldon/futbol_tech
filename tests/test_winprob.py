"""Offline sanity checks for the in-game win-probability model (no network).

Loads the trained coefficients and checks the predictions behave like
probabilities and move in the right direction; pins the chess-engine eval bar
that live_eval and board both drive.

    python tests/test_winprob.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from winprob import predict                              # noqa: E402

MODEL = json.load(open(os.path.join(os.path.dirname(__file__), "..", "wc2026",
                                    "winprob_model.json"), encoding="utf-8"))


def test_predict_is_a_distribution():
    for gd, xgd, mad, minute in [(0, 0, 0, 1), (2, 1.5, 1, 85), (-1, -0.5, -1, 70)]:
        p = predict(MODEL, gd, xgd, mad, minute)
        assert set(p) == {"H", "D", "A"}
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in p.values())


def test_leading_team_favoured_late():
    assert predict(MODEL, 2, 1.0, 0, 88)["H"] > 0.8      # home +2 in the 88th
    assert predict(MODEL, -2, -1.0, 0, 88)["A"] > 0.8    # away +2 in the 88th


def test_level_late_is_a_draw():
    p = predict(MODEL, 0, 0.0, 0, 89)                    # 0-0 at 89'
    assert p["D"] > p["H"] and p["D"] > p["A"]


def test_an_extra_goal_helps():
    base = predict(MODEL, 0, 0.0, 0, 45)
    ahead = predict(MODEL, 1, 0.0, 0, 45)
    assert ahead["H"] > base["H"] and ahead["A"] < base["A"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))
