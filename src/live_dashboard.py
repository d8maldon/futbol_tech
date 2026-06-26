"""Real-time-capable dashboard runner.

Drives a theme's draw_frame from a LIVE frame source and paints through a reusable
matplotlib Agg canvas, encoding to mp4 with OpenCV's own writer (NO ffmpeg-on-PATH
needed) and/or showing a live window.

Frame sources (each yields (state, match_data, minute) tuples):
  synthetic : loops the preview harness, no GPU/camera   -> for testing the loop
  states    : a precomputed cv_pass cache (_states.pkl)   -> re-time an offline clip
  video     : a video file, CV pass per frame            -> needs the GPU detector
  screen    : (scaffold) a screen region via mss          -> wire the CV pass, see source_video
  camera    : (scaffold) a webcam / phone-as-webcam       -> wire the CV pass, see source_video

Matplotlib paints ~6-10 fps (fine for an analyst overlay or a recording). For true
30fps+ broadcast overlay, port draw_frame's primitives to a cv2/OpenGL backend --
the make_axes/draw_frame split keeps the layout logic reusable as-is.

    python src/live_dashboard.py --theme broadcast --source synthetic --out figures/_live_demo.mp4
    python src/live_dashboard.py --theme telemetry --source camera --device 1 --show
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

import dashboard_themes as T

ROOT = os.path.join(os.path.dirname(__file__), "..")


# ----------------------------------------------------------------- renderer
class Renderer:
    """One persistent figure + axes; paint a frame and return a BGR ndarray."""

    def __init__(self, theme_name, team_rgb):
        self.th = T.get(theme_name)
        self.team_rgb = team_rgb
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        self.fig.patch.set_facecolor(self.th.BG)
        self.axes = self.th.make_axes(self.fig)

    def paint(self, state, m, ti, label="visual-AI dashboard"):
        self.th.draw_frame(self.axes, state, m, int(ti), self.team_rgb, label=label)
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# ----------------------------------------------------------------- sources
def source_synthetic(limit=160):
    """Loop the preview harness as a fake live feed: advance the minute, wobble the
    players so it looks alive, and inject an occasional graphic/replay (no-pitch)."""
    import _preview_data as P
    m = P.build_m()
    base = P.build_state()
    for i in range(limit):
        ti = int(np.clip(i / max(limit - 1, 1) * 90.0, 0, 98))
        if i % 27 == 26:
            d = dict(base, note="no pitch view", tracks=[], ball=None, conf=0.0, cam="other")
        else:
            wob = 0.7 * np.sin(i / 6.0)
            d = dict(base, tracks=[[t[0] + wob * np.cos(j), t[1] + wob * np.sin(j * 1.3),
                                    t[2], 1.0, True, t[5], t[6], t[7]]
                                   for j, t in enumerate(base["tracks"])])
        yield d, m, ti


def source_states(name, match_id="4667812"):
    """Re-time an offline clip's cached cv_pass states as if live."""
    import pickle
    import match_data as MD
    cache = os.path.join(ROOT, "data", "clips", name, "_states.pkl")
    with open(cache, "rb") as f:
        blob = pickle.load(f)
    states = blob["states"]
    m = MD.load(match_id)
    fmin = T.compute_frame_min(states)
    for s, ti in zip(states, fmin):
        yield s, m, ti


# Real CV sources (need the GPU detector / a device) -- wired, not run in CI.
def source_video(path, fps, match_id="4667812"):
    """Per-frame CV pass over a video file. Needs the detector weights (data/models)."""
    import match_data as MD
    import visual_ai as VA
    frames = VA.extract_frames(path, os.path.join(ROOT, "data", "clips", "_live_tmp"), fps)
    cen, team_rgb = VA.fit_teams(frames)
    m = MD.load(match_id)
    states = VA.cv_pass(frames, cen)
    fmin = T.compute_frame_min(states)
    for s, ti in zip(states, fmin):
        yield s, m, ti


# ----------------------------------------------------------------- runner
def run(theme, source, out_path=None, show=False, fps=8, limit=0, team_rgb=None):
    if team_rgb is None:
        import _preview_data as P
        team_rgb = P.TEAM_RGB
    r = Renderer(theme, team_rgb)
    writer, n = None, 0
    try:
        for state, m, ti in source:
            frame = r.paint(state, m, ti)
            if out_path:
                if writer is None:
                    h, w = frame.shape[:2]
                    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("futbol_tech dashboard", frame)
                if cv2.waitKey(1) & 0xFF == 27:   # Esc
                    break
            n += 1
            if limit and n >= limit:
                break
    finally:                                   # always release the writer/figure, even mid-frame error
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        plt.close(r.fig)
    return n


def _make_source(args):
    if args.source == "synthetic":
        return source_synthetic(limit=args.limit or 160)
    if args.source == "states":
        return source_states(args.name)
    if args.source == "video":
        return source_video(args.video, args.fps)
    if args.source == "camera":
        cap = cv2.VideoCapture(args.device)
        raise SystemExit("camera/screen sources need the CV pass wired to your live "
                         "feed; use source_video as the template (cap is open: %s)" % cap.isOpened())
    raise SystemExit("unknown source " + args.source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="broadcast")
    ap.add_argument("--source", default="synthetic",
                    help="synthetic | states | video  (screen/camera are scaffolds: wire your CV pass like source_video)")
    ap.add_argument("--name", default="argentina_full", help="states-cache name")
    ap.add_argument("--video", default="")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    src = _make_source(args)
    n = run(args.theme, src, out_path=args.out or None, show=args.show, fps=args.fps, limit=args.limit)
    print("rendered {} frames [{}] source={}{}".format(
        n, args.theme, args.source, " -> " + args.out if args.out else ""))


if __name__ == "__main__":
    main()
