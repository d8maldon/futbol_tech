"""Offline checks for the dashboard theme layer: theme registry, the playhead
mapping, the polygon-area helper, and the Voronoi pitch-control split. All run
with no ffmpeg/GPU/network (Voronoi uses mplsoccer, a hard dep).

    python tests/test_dashboard_themes.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dashboard_themes as T  # noqa: E402


def test_registry_resolves_three_themes():
    assert set(T.names()) == {"broadcast", "editorial", "telemetry"}
    for n in T.names():
        mod = T.get(n)
        assert hasattr(mod, "make_axes") and hasattr(mod, "draw_frame")
        assert isinstance(mod.BG, str)
    with pytest.raises(SystemExit):
        T.get("nope")


def test_compute_frame_min_advances_only_on_pitch_frames():
    states = [{"note": ""}, {"note": "no pitch view"}, {"note": ""}, {"note": ""}]
    fm = T.compute_frame_min(states)
    assert len(fm) == 4
    assert fm[0] == fm[1]                       # a blank frame does not advance the playhead
    assert fm[2] > fm[1] and fm[3] > fm[2]      # pitch frames advance it
    assert (fm >= 0).all() and (fm <= 98).all()
    ov = T.compute_frame_min(states, override=[10, 200, -5, 50])
    assert ov[1] == 98 and ov[2] == 0           # override is clipped, not recomputed


def test_poly_area_shoelace():
    assert abs(T._poly_area([[0, 0], [1, 0], [1, 1], [0, 1]]) - 1.0) < 1e-9   # unit square
    assert abs(T._poly_area([[0, 0], [4, 0], [0, 3]]) - 6.0) < 1e-9           # 3-4-5 triangle


def test_voronoi_regions_split_and_degenerate():
    # 3 home on the left, 3 away on the right -> a roughly even territory split
    tracks = [[20, 30, 0, 1, True, 1, 1, 0], [20, 40, 0, 1, True, 1, 1, 0], [30, 34, 0, 1, True, 1, 1, 0],
              [85, 30, 1, 1, True, 1, 1, 0], [85, 40, 1, 1, True, 1, 1, 0], [75, 34, 1, 1, True, 1, 1, 0]]
    th, ta, share = T.voronoi_regions(tracks)
    assert th is not None and ta is not None
    assert 0.0 <= share <= 1.0
    assert 0.35 < share < 0.65
    assert T.voronoi_regions([])[2] == 0.5      # < 2 players -> neutral 0.5, no crash


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn(); print("ok", fn.__name__)
        except pytest.skip.Exception as e:
            print("skip", fn.__name__, "-", e)
    print("done ({} checks)".format(len(fns)))
