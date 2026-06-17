"""Animated tactical side-by-side, CONTINUOUS real-time tracking.

Every consecutive broadcast frame is shown -- nothing is skipped. The homography
is EMA-smoothed and HELD through frames where the pitch keypoints momentarily
drop out (so the clip never stutters), and players are associated across frames
in world coordinates with a constant-velocity Kalman filter and persisted
through brief detection misses (ghosts). The result is the top-down dots gliding
with the play instead of popping frame to frame -- the live tracking capability,
end to end on one continuous passage.

Honest scope: one ball-chasing broadcast camera, so only the visible players are
tracked (heatmap-grade homography, ~5 m).

    python src/tactical_clip.py --dir data/clips/brazil_morocco_1080 --start 388 --count 66 --fps 12
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import homography as hg
from broadcast_track import detect
from tactical import BG, FIG, INK, MUT, PL, PW, jersey_color
from track_fuse import Kalman, appearance, appdist, ema_app

GATE, APP_W, MAX_MISS, DT = 4.0, 3.0, 8, 0.1


def team_centroids(frames):
    import cv2
    from sklearn.cluster import KMeans
    cols = []
    for fp in frames[:8]:
        img = cv2.imread(fp)
        for _, _, b, _ in detect(fp)[0]:
            c = jersey_color(img, b)
            if c is not None:
                cols.append(c)
    km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(np.array(cols))
    cen = km.cluster_centers_
    rgb = []
    for c in range(2):
        lab = cen[c].astype(np.uint8).reshape(1, 1, 3)
        b = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
        rgb.append((b[2] / 255, b[1] / 255, b[0] / 255))
    return cen, rgb


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/clips/brazil_morocco_1080")
    ap.add_argument("--start", type=int, default=388)
    ap.add_argument("--count", type=int, default=66)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    frames = sorted(glob.glob(os.path.join(args.dir, "*.png")))[args.start:args.start + args.count]
    cen, team_rgb = team_centroids(frames)

    def team_of(img, box):
        c = jersey_color(img, box)
        return int(np.argmin([np.linalg.norm(c - cen[k]) for k in range(2)])) if c is not None else -1

    tracks, states, Hs, A = [], [], None, 0.35
    held = 0
    for fp in frames:
        H, _, _ = hg.keypoint_homography(fp)
        if H is not None:
            Hs = H if Hs is None else A * H + (1 - A) * Hs
            held = 0
        else:
            held += 1
        Huse = Hs / Hs[2, 2] if Hs is not None else None
        img = cv2.imread(fp)
        players, ball, (h, w) = detect(fp)
        boxes = [(b, team_of(img, b)) for _, _, b, _ in players]
        dets, dets_app, dets_team = [], [], []
        if Huse is not None:
            for fx, fy, b, _ in players:
                p = hg.warp(Huse, [[fx, fy]])[0]
                if 0 <= p[0] <= PL and 0 <= p[1] <= PW:
                    dets.append(p); dets_app.append(appearance(img, b)); dets_team.append(team_of(img, b))
        alive = [t for t in tracks if t["alive"]]
        for t in alive:
            t["kf"].predict()
        md, mt = set(), set()
        if alive and dets:
            pred = np.array([[t["kf"].x[0], t["kf"].x[1]] for t in alive])
            D = np.array(dets)
            wd = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            app = np.array([[appdist(t["app"], da) for da in dets_app] for t in alive])
            cost = wd + APP_W * app
            cost[wd > GATE] = 1e6
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    alive[a]["kf"].update(D[b]); alive[a]["miss"] = 0
                    alive[a]["app"] = ema_app(alive[a]["app"], dets_app[b])
                    alive[a]["votes"][dets_team[b]] = alive[a]["votes"].get(dets_team[b], 0) + 1
                    alive[a]["n"] += 1
                    md.add(b); mt.add(a)
        for a, t in enumerate(alive):
            if a not in mt:
                t["miss"] += 1
                if t["miss"] > MAX_MISS:
                    t["alive"] = False
        for j in range(len(dets)):
            if j not in md:
                tracks.append({"kf": Kalman(dets[j], DT), "app": dets_app[j], "miss": 0,
                               "alive": True, "votes": {dets_team[j]: 1}, "n": 1, "trail": []})
        cur = []
        for t in tracks:
            if t["alive"] and t["n"] >= 3:                 # confirmed: drop jitter-spawned
                tm = max(t["votes"], key=t["votes"].get)
                t["trail"].append((t["kf"].x[0], t["kf"].x[1]))
                cur.append((t["kf"].x[0], t["kf"].x[1], tm, t["miss"] > 0, t["trail"][-7:]))
        ballw = hg.warp(Huse, [[ball[0], ball[1]]])[0] if (ball is not None and Huse is not None) else None
        states.append({"img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "boxes": boxes,
                       "tracks": cur, "ball": ballw, "wh": (w, h), "held": held})
    print("frames:", len(states), "| confirmed tracks:", sum(1 for t in tracks if t["n"] >= 3))

    fig, (axi, axp) = plt.subplots(1, 2, figsize=(13, 4.7), dpi=90)
    fig.patch.set_facecolor(BG)
    F = {"fontfamily": "Bahnschrift"}

    def draw(i):
        d = states[i]
        axi.clear(); axp.clear()
        axi.imshow(d["img"])
        for box, tm in d["boxes"]:
            col = team_rgb[tm] if tm >= 0 else (0.6, 0.6, 0.6)
            x1, y1, x2, y2 = box
            axi.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.5))
        w, h = d["wh"]
        axi.set_xlim(0, w); axi.set_ylim(h, 0); axi.axis("off")
        axi.set_title("broadcast (every frame){}".format("  - homography held" if d["held"] else ""),
                      color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        hg.draw_pitch(axp)
        for x, y, tm, ghost, trail in d["tracks"]:
            col = team_rgb[tm] if tm >= 0 else (0.6, 0.6, 0.6)
            if len(trail) > 1:
                tr = np.array(trail)
                axp.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=1.0, alpha=0.35, zorder=3)
            axp.scatter([x], [y], s=130, facecolor=col if not ghost else "none",
                        edgecolors=col if ghost else BG, lw=1.6 if ghost else 1.2, alpha=0.95, zorder=5)
        if d["ball"] is not None and 0 <= d["ball"][0] <= PL and 0 <= d["ball"][1] <= PW:
            axp.scatter([d["ball"][0]], [d["ball"][1]], s=70, c="#ffd23f", edgecolors=BG, lw=1, zorder=7)
        axp.set_title("top-down: continuous Kalman tracking (ghost = predicted through a miss)",
                      color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        fig.suptitle("WC 2026  Brazil v Morocco  -- live broadcast to top-down tracking, every frame (visible players)",
                     color=INK, x=0.5, fontsize=12.5, fontweight="bold", **F)

    a = manim.FuncAnimation(fig, draw, frames=len(states), interval=1000 / args.fps)
    gif = os.path.join(FIG, "wc2026_tactical_clip.gif")
    a.save(gif, writer=manim.PillowWriter(fps=args.fps))
    plt.close(fig)
    print("wrote", gif)


if __name__ == "__main__":
    main()
