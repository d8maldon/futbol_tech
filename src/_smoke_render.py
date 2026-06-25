"""Mimic dashboard_themes.render()'s animation loop WITHOUT ffmpeg: make_axes once,
then draw_frame across a live -> no-pitch -> estimated -> live sequence reusing the
same axes (the real FuncAnimation behaviour). Catches state-transition / axis-reuse
bugs the single-frame _preview_theme can't. Usage: python _smoke_render.py <theme>"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _preview_data as P
import dashboard_themes as T


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "broadcast"
    th = T.get(name)
    m = P.build_m()
    live = P.build_state()
    blank = dict(live, note="no pitch view", tracks=[], ball=None, conf=0.0, cam="other")
    est = dict(live, est=True, conf=0.0,
               tracks=[[t[0], t[1], t[2], 0.5, False, t[5], t[6], t[7]] for t in live["tracks"]])
    states = [live, blank, est, live]
    frame_min = T.compute_frame_min(states)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(th.BG)
    axes = th.make_axes(fig)
    for i, s in enumerate(states):
        th.draw_frame(axes, s, m, int(round(float(frame_min[i]))), P.TEAM_RGB, label="visual-AI dashboard")
    out = os.path.join(P.ROOT, "_frames_review", "smoke_{}.png".format(name))
    fig.savefig(out, facecolor=th.BG); plt.close(fig)
    print("OK {:<10} frame_min={} -> {}".format(name, [round(float(x), 1) for x in frame_min], out))


if __name__ == "__main__":
    main()
