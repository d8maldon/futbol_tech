"""Temporal fusion: track players over time and persist them through misses.

A single frame loses any player the detector misses (occlusion, edge of frame).
Fusing OVER TIME fixes the short gaps: detect players each frame, keep a stable
id with ByteTrack, warp every detection to the top-down pitch through the
per-frame homography, and run a constant-velocity Kalman filter per player in
PITCH METRES. When a player is seen, the filter is corrected by the measurement
(MEASURED); when a player is briefly missed, the filter PREDICTS their position
and its uncertainty grows (INFERRED ghost) instead of dropping them.

Tracking is done in WORLD (pitch) coordinates, not image space: every detection
is warped to the pitch first, then associated to existing tracks by nearest
position. A stationary player keeps the same world coordinate no matter how the
camera pans, so this sidesteps the camera-motion problem that breaks image-space
trackers (the council's L3) -- association + homography (L4) + Kalman (L5) on a
real broadcast clip. Honest scope: it persists players through SHORT gaps; a
player NEVER in view cannot be tracked (that needs the formation-prior layer,
which single-frame tests showed is weak). Off-screen positions are inferred with
growing uncertainty, never claimed as measured.

    python src/track_fuse.py --frames-dir data/clips/track
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import homography as hg
from broadcast_track import DEFAULT_WEIGHTS, PERSON, SPORTS_BALL

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
PL, PW = 120.0, 80.0
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"
MEAS = "#5e9bff"; GHOST = "#ff7a1a"; BALLC = "#ffd23f"


def appearance(img, box):
    """L1-normalised HSV torso colour histogram -- a cheap kit-colour ReID cue"""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    crop = img[max(y1 + int(0.15 * h), 0):y1 + int(0.55 * h),
               max(x1 + int(0.2 * w), 0):x1 + int(0.8 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
    s = hist.sum()
    return hist / s if s > 0 else None


def appdist(a, b):
    return 0.5 if a is None or b is None else 1.0 - float(np.minimum(a, b).sum())


def ema_app(a, b, alpha=0.3):
    if b is None:
        return a
    if a is None:
        return b
    c = (1 - alpha) * a + alpha * b
    s = c.sum()
    return c / s if s > 0 else c


class Kalman:
    """constant-velocity KF in metres: state [x, y, vx, vy]"""

    def __init__(self, xy, dt):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([4., 4., 25., 25.])
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.Q = np.diag([0.25, 0.25, 4., 4.]) * dt
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.R = np.diag([2., 2.])              # ~1.4 m measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def radius(self):
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


def main():
    os.makedirs(FIG, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/clips/track")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    from scipy.optimize import linear_sum_assignment
    import cv2
    from broadcast_track import detect
    dt = 1.0 / args.fps
    GATE = 4.0           # m: max association distance (player moves <1m/frame)
    MAX_MISS = 6         # frames a track may be Kalman-ghosted before dropping
    APP_W = 3.0          # weight (in metres) of kit-colour distance in matching

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    tracks = []          # {"kf":, "pts":[(x,y,measured,radius)], "miss":, "alive":}
    ball_xy = []
    miss_fills = 0
    used = 0
    Hs = None            # EMA-smoothed homography (L7): kills per-frame H jitter
    A = 0.35
    for fp in frames:
        H, _, _ = hg.keypoint_homography(fp)
        if H is None:
            continue                              # cut / no pitch: hold state
        Hs = H if Hs is None else A * H + (1 - A) * Hs
        H = Hs / Hs[2, 2]                          # use the smoothed homography
        used += 1
        # soccer detector (player/gk, no crowd) + foot points; warp on-pitch ones
        players, ball, _ = detect(fp, conf=args.conf)
        img = cv2.imread(fp)
        dets, dets_app = [], []
        for fx, fy, box, _ in players:
            w = hg.warp(H, [[fx, fy]])[0]
            if 0 <= w[0] <= PL and 0 <= w[1] <= PW:
                dets.append(w); dets_app.append(appearance(img, box))
        dets = np.array(dets) if dets else np.empty((0, 2))
        if ball is not None:
            ball_xy.append(hg.warp(H, [[ball[0], ball[1]]])[0])

        alive = [t for t in tracks if t["alive"]]
        for t in alive:
            t["kf"].predict()
        # associate by world distance + kit-colour (ReID): keeps identities stable
        # through crossings/occlusion so a player isn't re-spawned as a new track
        matched_d, matched_t = set(), set()
        if alive and len(dets):
            pred = np.array([[t["kf"].x[0], t["kf"].x[1]] for t in alive])
            wd = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=2)
            app = np.array([[appdist(t["app"], da) for da in dets_app] for t in alive])
            cost = wd + APP_W * app
            cost[wd > GATE] = 1e6
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    alive[a]["kf"].update(dets[b])
                    alive[a]["miss"] = 0
                    alive[a]["app"] = ema_app(alive[a]["app"], dets_app[b])
                    alive[a]["pts"].append((alive[a]["kf"].x[0], alive[a]["kf"].x[1],
                                            True, alive[a]["kf"].radius()))
                    matched_d.add(b); matched_t.add(a)
        # unmatched alive tracks -> Kalman ghost (predict-only), drop if too long
        for a, t in enumerate(alive):
            if a in matched_t:
                continue
            t["miss"] += 1
            if t["miss"] > MAX_MISS:
                t["alive"] = False
                continue
            t["pts"].append((t["kf"].x[0], t["kf"].x[1], False, t["kf"].radius()))
            miss_fills += 1
        # unmatched dets -> new tracks
        for j in range(len(dets)):
            if j not in matched_d:
                kf = Kalman(dets[j], dt)
                tracks.append({"kf": kf, "miss": 0, "alive": True,
                               "app": dets_app[j],
                               "pts": [(dets[j][0], dets[j][1], True, kf.radius())]})

    def n_meas(t):
        return sum(1 for p in t["pts"] if p[2])
    # a CONFIRMED track has enough real detections (drops jitter-spawned ghosts)
    conf = [t for t in tracks if n_meas(t) >= 6 and len(t["pts"]) >= 10]
    longtracks = {i: t for i, t in enumerate(conf)}
    print("frames used: {}/{}  | raw tracks: {}  | confirmed (>=6 detections): {}".format(
        used, len(frames), len(tracks), len(conf)))
    print("Kalman gap-fills on confirmed tracks (persisted through misses): {}".format(
        sum(sum(1 for p in t["pts"] if not p[2]) for t in conf)))
    print("NOTE 480p + panning broadcast -> fragmentation; production needs ReID + L3 camera-comp (SOTA ~64 GS-HOTA)")

    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=170)
    fig.patch.set_facecolor(BG)
    hg.draw_pitch(ax)
    for t in longtracks.values():
        pts = np.array([(x, y) for x, y, *_ in t["pts"]])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=MEAS, lw=1.0, alpha=0.5, zorder=3)
        # ghost (predicted) points
        gx = [p[0] for p in t["pts"] if not p[2]]
        gy = [p[1] for p in t["pts"] if not p[2]]
        if gx:
            ax.scatter(gx, gy, s=20, c=GHOST, alpha=0.5, marker="x", zorder=4)
        # current position + uncertainty
        lx, ly, lm, lr = t["pts"][-1]
        ax.scatter([lx], [ly], s=120, c=MEAS if lm else GHOST, edgecolors=BG,
                   lw=1.2, zorder=6)
        if not lm:
            ax.add_patch(plt.Circle((lx, ly), lr, color=GHOST, fill=False,
                                    lw=1.0, alpha=0.5, ls=(0, (2, 2))))
    if ball_xy:
        b = np.array(ball_xy)
        ax.plot(b[:, 0], b[:, 1], "-", color=BALLC, lw=1.0, alpha=0.5)
        ax.scatter([b[-1, 0]], [b[-1, 1]], s=70, c=BALLC, edgecolors=BG, lw=1, zorder=7)
    ax.scatter([], [], s=120, c=MEAS, label="measured (detected)")
    ax.scatter([], [], s=60, c=GHOST, marker="x", label="Kalman-filled (missed frame)")
    ax.scatter([], [], s=70, c=BALLC, label="ball")
    ax.legend(loc="upper center", ncol=3, frameon=False, labelcolor=INK,
              bbox_to_anchor=(0.5, 1.09), prop={"family": "Bahnschrift", "size": 9})
    ax.set_title("Tracking players over 6s of broadcast: world-space fusion (homography + Kalman)",
                 color=INK, loc="left", pad=30, fontfamily="Bahnschrift",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "trails on the top-down pitch | gaps filled by Kalman (uncertainty ring) | measured != inferred | 480p+yolov8n -> heavy gap-fill; needs higher-res + ReID | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontsize=7.0, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(FIG, "wc2026_track_fuse.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("figure: figures/wc2026_track_fuse.png")


if __name__ == "__main__":
    main()
