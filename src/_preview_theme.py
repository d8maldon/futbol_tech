"""Render a single static dashboard frame for any theme module, from the preview
harness, to _frames_review/preview_<theme>.png -- verify a theme with no
ffmpeg/GPU/network.  Usage:  python _preview_theme.py broadcast"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _preview_data as P
import dashboard_themes as T


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "broadcast"
    mode = sys.argv[2] if len(sys.argv) > 2 else "live"   # live | blank | est
    th = T.get(name)
    m = P.build_m(); d = P.build_state(); ti = P.PREVIEW_MIN
    if mode == "blank":           # genuinely no pitch in view (graphic / replay)
        d = dict(d, note="no pitch view", tracks=[], ball=None, conf=0.0, cam="other")
    elif mode == "est":           # close-up: holding last shape, faded
        d = dict(d, est=True, conf=0.0, tracks=[[t[0], t[1], t[2], 0.5, False, t[5], t[6], t[7]] for t in d["tracks"]])
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(th.BG)
    axes = th.make_axes(fig)
    th.draw_frame(axes, d, m, ti, P.TEAM_RGB, label="visual-AI dashboard")
    suffix = "" if mode == "live" else "_" + mode
    out = os.path.join(os.path.dirname(__file__), "..", "_frames_review", "preview_{}{}.png".format(name, suffix))
    fig.savefig(out, facecolor=th.BG); plt.close(fig)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
