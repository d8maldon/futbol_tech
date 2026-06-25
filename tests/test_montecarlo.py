"""Offline checks for the Monte-Carlo sim engine (no data/ -- constructed inputs).

Guards the pieces the champion odds rest on and that had no test: the Poisson
score sampler, the 2026 H2H-first group ranking, the 32-team bracket
reconstruction, and the knockout reducing to a single champion.

    python tests/test_montecarlo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import montecarlo as MC  # noqa: E402


def test_sample_score_nonneg_and_favours_stronger():
    rng = np.random.default_rng(0)
    hs, as_ = [], []
    for _ in range(4000):
        h, a = MC.sample_score(2000, 1600, 0.0, rng)   # home 400 Elo stronger
        assert h >= 0 and a >= 0 and isinstance(h, int)
        hs.append(h); as_.append(a)
    assert np.mean(hs) > np.mean(as_)                   # the stronger side scores more


def test_rank_group_points_ladder():
    teams = ["A", "B", "C", "D"]; rt = {t: MC.BASE for t in teams}
    results = [("A", "B", 1, 0), ("A", "C", 1, 0), ("A", "D", 1, 0),
               ("B", "C", 1, 0), ("B", "D", 1, 0), ("C", "D", 1, 0)]   # A9 B6 C3 D0
    order, _ = MC.rank_group(teams, results, rt)
    assert order == ["A", "B", "C", "D"]


def test_rank_group_h2h_equal_falls_to_overall_gd():
    teams = ["A", "B", "C"]; rt = {t: MC.BASE for t in teams}
    # A and B draw (tied on points, H2H even), both beat C, A by more -> A above B
    results = [("A", "B", 0, 0), ("A", "C", 3, 0), ("B", "C", 1, 0)]
    order, _ = MC.rank_group(teams, results, rt)
    assert order == ["A", "B", "C"]


def test_build_bracket_is_32_unique_and_avoids_same_group_first_round():
    groups = [chr(ord("A") + i) for i in range(12)]
    winners = {g: g + "1" for g in groups}
    runners = {g: g + "2" for g in groups}
    thirds = [(g + "3", g, 3, 0, 0) for g in groups[:8]]
    slots = MC.build_bracket(winners, runners, thirds)
    assert len(slots) == 32 and len(set(slots)) == 32
    for i in range(0, 16, 2):                           # winners A..H vs their thirds
        assert slots[i][0] != slots[i + 1][0]          # different group letter


def test_sim_knockout_returns_one_entrant():
    rng = np.random.default_rng(1)
    slots = ["T{}".format(i) for i in range(8)]
    rt = {t: MC.BASE for t in slots}
    assert MC.sim_knockout(slots, rt, rng) in slots


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))
