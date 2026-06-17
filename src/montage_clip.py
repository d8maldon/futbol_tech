"""Tracking on a HIGHLIGHTS montage: continuous footage is cut into shots, so we
detect scene cuts (frame-difference) and reset the tracker at each one, and we
blank the top-down on frames with no pitch view (replays/closeups/graphics where
the homography fails). The result tracks each continuous broadcast shot and
honestly says "scene change" / "no pitch view" in between -- the best you can do
on a montage, and a live demonstration of why continuous footage is better.

    python src/montage_clip.py --dir data/clips/argentina_algeria --fps 8
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

GATE, APP_W, MAX_MISS, DT, CUT = 4.0, 3.0, 6, 0.12, 28.0


def team_centroids(frames):
    import cv2
    from sklearn.cluster import KMeans
    cols = []
    for fp in frames[::max(len(frames) // 12, 1)]:
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
    ap.add_argument("--dir", default="data/clips/argentina_algeria")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    frames = sorted(glob.glob(os.path.join(args.dir, "f_*.png")))
    cen, team_rgb = team_centroids(frames)

    def team_of(img, box):
        c = jersey_color(img, box)
        return int(np.argmin([np.linalg.norm(c - cen[k]) for k in range(2)])) if c is not None else -1

    tracks, states, prev_small = [], [], None
    for fp in frames:
        img = cv2.imread(fp)
        small = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY).astype(float)
        cut = prev_small is not None and float(np.abs(small - prev_small).mean()) > CUT
        prev_small = small
        if cut:
            tracks = []                                   # scene change: drop all tracks
        H, _, _ = hg.keypoint_homography(fp)
        players, ball, (h, w) = detect(fp)
        boxes = [(b, team_of(img, b)) for _, _, b, _ in players]
        note = "scene change" if cut else ("" if H is not None else "no pitch view")
        if H is None:
            tracks = []                                   # off the pitch: nothing to track
            states.append({"img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "boxes": boxes,
                           "tracks": [], "ball": None, "wh": (w, h), "note": note or "no pitch view"})
            continue
        dets, dapp, dteam = [], [], []
        for fx, fy, b, _ in players:
            p = hg.warp(H, [[fx, fy]])[0]
            if 0 <= p[0] <= PL and 0 <= p[1] <= PW:
                dets.append(p); dapp.append(appearance(img, b)); dteam.append(team_of(img, b))
        alive = [t for t in tracks if t["alive"]]
        for t in alive:
            t["kf"].predict()
        md, mt = set(), set()
        if alive and dets:
            pred = np.array([[t["kf"].x[0], t["kf"].x[1]] for t in alive]); D = np.array(dets)
            wd = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            app = np.array([[appdist(t["app"], da) for da in dapp] for t in alive])
            cost = wd + APP_W * app; cost[wd > GATE] = 1e6
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    alive[a]["kf"].update(D[b]); alive[a]["miss"] = 0
                    alive[a]["app"] = ema_app(alive[a]["app"], dapp[b])
                    alive[a]["votes"][dteam[b]] = alive[a]["votes"].get(dteam[b], 0) + 1
                    alive[a]["n"] += 1; md.add(b); mt.add(a)
        for a, t in enumerate(alive):
            if a not in mt:
                t["miss"] += 1
                if t["miss"] > MAX_MISS:
                    t["alive"] = False
        for j in range(len(dets)):
            if j not in md:
                tracks.append({"kf": Kalman(dets[j], DT), "app": dapp[j], "miss": 0,
                               "alive": True, "votes": {dteam[j]: 1}, "n": 1})
        cur = [(t["kf"].x[0], t["kf"].x[1], max(t["votes"], key=t["votes"].get), t["miss"] > 0)
               for t in tracks if t["alive"] and t["n"] >= 2]
        bw = hg.warp(H, [[ball[0], ball[1]]])[0] if ball is not None else None
        states.append({"img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "boxes": boxes,
                       "tracks": cur, "ball": bw, "wh": (w, h), "note": note})
    trackable = sum(1 for s in states if s["tracks"])
    print("frames:", len(states), "| with tracking:", trackable, "| no-pitch/cut:", len(states) - trackable)

    fig, (axi, axp) = plt.subplots(1, 2, figsize=(13, 4.7), dpi=88)
    fig.patch.set_facecolor(BG)
    F = {"fontfamily": "Bahnschrift"}

    def draw(i):
        d = states[i]
        axi.clear(); axp.clear()
        axi.imshow(d["img"])
        for box, tm in d["boxes"]:
            col = team_rgb[tm] if tm >= 0 else (0.6, 0.6, 0.6); x1, y1, x2, y2 = box
            axi.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.5))
        w, h = d["wh"]; axi.set_xlim(0, w); axi.set_ylim(h, 0); axi.axis("off")
        axi.set_title("Argentina v Algeria highlights -- players detected", color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        hg.draw_pitch(axp)
        for x, y, tm, ghost in d["tracks"]:
            col = team_rgb[tm] if tm >= 0 else (0.6, 0.6, 0.6)
            axp.scatter([x], [y], s=130, facecolor="none" if ghost else col,
                        edgecolors=col, lw=1.8 if ghost else 1.2, zorder=5)
        if d["ball"] is not None and 0 <= d["ball"][0] <= PL and 0 <= d["ball"][1] <= PW:
            axp.scatter([d["ball"][0]], [d["ball"][1]], s=70, c="#ffd23f", edgecolors=BG, lw=1, zorder=7)
        if d["note"]:
            axp.text(60, 40, d["note"].upper(), ha="center", va="center", color="#ffb347", fontsize=15, fontweight="bold", **F)
        axp.set_title("top-down: tracking resets at each cut (montage)", color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        fig.suptitle("WC 2026 Argentina v Algeria -- tracking a highlights montage (cuts detected, tracker reset per shot)",
                     color=INK, x=0.5, fontsize=12, fontweight="bold", **F)

    a = manim.FuncAnimation(fig, draw, frames=len(states), interval=1000 / args.fps)
    gif = os.path.join(FIG, "wc2026_argentina_montage.gif")
    a.save(gif, writer=manim.PillowWriter(fps=args.fps))
    plt.close(fig)
    print("wrote", gif)


if __name__ == "__main__":
    main()
