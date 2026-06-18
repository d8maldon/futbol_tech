"""Live mode: prove the system runs AS THE BROADCAST AIRS, at the operating point
the research nailed down -- ~10 fps, 1-2 s behind the feed (SkillCorner Live's
deliberate point, not 30 fps).

It streams frames in order (simulating a live feed), and per frame:
  1. classify camera state (wide / tight / other) and GATE -- only wide frames
     are tracked, so we never emit a garbage homography on a close-up/graphic;
  2. on wide frames: detect + homography + warp + team assignment + a per-frame
     calibration CONFIDENCE, and a rolling tactical readout (compactness/block);
  3. on non-wide frames: hold last state / fall back, with a staleness note.

It also benchmarks throughput: if mean per-frame time < 100 ms it clears the
10 fps live budget on this GPU. (Demonstrated on the recorded match; a true live
stream is the same loop fed by yt-dlp/ffmpeg.)

    python src/live_mode.py --count 600 --emit 10
"""
import argparse
import glob
import os
import time

import numpy as np

import camera_state as cs
import homography as hg
import uncertainty as uq
from broadcast_track import detect
from tactical import jersey_color
from tactical_metrics import team_shape, plausible

ROOT = os.path.join(os.path.dirname(__file__), "..")
FRAMES = os.path.join(ROOT, "data", "clips", "argentina_full")


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--start", type=int, default=0, help="start frame (skip pre-match graphics)")
    ap.add_argument("--emit", type=int, default=10, help="emit a live readout every N frames")
    ap.add_argument("--fps", type=float, default=6.0, help="source fps (for the clock)")
    args = ap.parse_args()
    frames = sorted(glob.glob(os.path.join(FRAMES, "f_*.jpg")))[args.start:args.start + args.count]

    # lock team kit colours once from early wide frames
    from sklearn.cluster import KMeans
    cols = []
    for fp in frames[::5]:
        H, _, _ = hg.keypoint_homography(fp)
        if H is None:
            continue
        img = cv2.imread(fp)
        for _, _, b, _ in detect(fp)[0]:
            c = jersey_color(img, b)
            if c is not None:
                cols.append(c)
        if len(cols) > 200:
            break
    cen = KMeans(n_clusters=2, n_init=5, random_state=0).fit(np.array(cols)).cluster_centers_

    def team_of(img, box):
        c = jersey_color(img, box)
        return int(np.argmin([np.linalg.norm(c - cen[k]) for k in range(2)])) if c is not None else -1

    gate = {"wide": 0, "tight": 0, "other": 0}
    last_state = "(none yet)"
    stale = 0
    times = []
    print("LIVE MODE  (gate on camera state, ~10 fps target, hold on non-wide)\n")
    for i, fp in enumerate(frames):
        t0 = time.time()
        img = cv2.imread(fp)
        players, ball, (h, w) = detect(fp)
        H, ip, pp = hg.keypoint_homography(fp)
        state, feat = cs.classify(img, H is not None, len(players))
        gate[state] += 1
        readout = None
        if state == "wide":
            fc = uq.frame_confidence(H, ip, pp)
            shapes = {}
            for c in (0, 1):
                pts = [hg.warp(H, [[fx, fy]])[0] for fx, fy, b, _ in players if team_of(img, b) == c]
                pts = [p for p in pts if 0 <= p[0] <= hg.PL and 0 <= p[1] <= hg.PW]
                shapes[c] = team_shape(pts) if plausible(pts) else None
            a, b = shapes[0], shapes[1]
            readout = "WIDE conf={:.2f} | {} tracked | ARG {} | ALG {}".format(
                fc["conf"], len(players),
                "block {:.0f}m compact {:.0f}m2".format(a["cx"], a["area"]) if a else "n/a",
                "block {:.0f}m compact {:.0f}m2".format(b["cx"], b["area"]) if b else "n/a")
            last_state = readout; stale = 0
        else:
            stale += 1
            tag = "TIGHT (close-up) - holding last shape" if state == "tight" else "OTHER (graphic/crowd) - data-layer only"
            readout = "{}  [stale {:.1f}s]".format(tag, stale / args.fps)
        times.append(time.time() - t0)
        if i % args.emit == 0:
            print("[t={:5.1f}s] {}".format(i / args.fps, readout))

    times = np.array(times) * 1000.0
    print("\n--- throughput ---")
    print("frames: {} | mean {:.0f} ms/frame | p90 {:.0f} ms | achieved {:.1f} fps".format(
        len(times), times.mean(), np.percentile(times, 90), 1000.0 / times.mean()))
    print("gate: {} | live budget (<100 ms = 10 fps): {}".format(
        gate, "PASS" if times.mean() < 100 else "FAIL (needs TensorRT/lower-res)"))


if __name__ == "__main__":
    main()
