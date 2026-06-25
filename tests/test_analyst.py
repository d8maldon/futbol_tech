"""Offline checks for the rule-based match analyst (the always-on insight engine).

Runs analyst.analyze() minute-by-minute over the synthetic Argentina 3-0 Algeria
match and asserts the deterministic behaviour: goals/VAR fire at their minutes, a
turning point fires on the opener's win-prob swing, every insight targets a known
panel, and no goal is announced twice.

    python tests/test_analyst.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import _preview_data as P   # noqa: E402
import analyst              # noqa: E402


def _run():
    m = P.build_m()
    mem, fired = {}, []
    for ti in range(0, 91):
        ins, mem = analyst.analyze(m, ti, None, mem)
        for x in ins:
            fired.append((ti, x["kind"], x["panel"], x["text"]))
    return fired


def test_goals_and_var_fire_at_their_minutes():
    by_kind = {}
    for ti, kind, _, _ in _run():
        by_kind.setdefault(kind, []).append(ti)
    assert by_kind.get("goal") == [23, 47, 64]          # Messi hat-trick minutes
    assert 8 in by_kind.get("var", [])                  # disallowed-offside VAR at 8'
    assert 23 in by_kind.get("turning_point", [])       # opener swings the win-prob


def test_every_insight_targets_a_known_panel():
    for _, _, panel, _ in _run():
        assert panel in analyst.PANELS


def test_no_duplicate_goal_callouts():
    goal_mins = [ti for ti, kind, _, _ in _run() if kind == "goal"]
    assert len(goal_mins) == len(set(goal_mins))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("all {} tests passed".format(len(fns)))
