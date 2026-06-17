"""Animated tactical side-by-side: broadcast (left) | top-down (right) across an
attacking sequence -- the snapshot pipeline run frame by frame into a short clip.
Team colours are fixed once (from the opening frames) so they do not flicker as
k-means re-runs. Honest scope: visible players only, heatmap-grade homography --
the ceiling of a single ball-chasing broadcast camera.

    python src/tactical_clip.py --dir data/clips/brazil_morocco_1080 --start 388 --end 440 --step 3
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np

import homography as hg
from broadcast_track import detect
from tactical import BG, FIG, INK, MUT, PL, PW, hull, jersey_color


def fixed_team_centroids(frames):
    """two team LAB centroids from the first frames, so colours stay stable"""
    import cv2
    from sklearn.cluster import KMeans
    cols = []
    for fp in frames[:6]:
        img = cv2.imread(fp)
        players, _, _ = detect(fp)
        for _, _, b, _ in players:
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
    ap.add_argument("--end", type=int, default=440)
    ap.add_argument("--step", type=int, default=3)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.dir, "*.png")))[args.start:args.end:args.step]
    cen, team_rgb = fixed_team_centroids(frames)

    data = []
    for fp in frames:
        H, _, _ = hg.keypoint_homography(fp)
        if H is None:
            continue
        img = cv2.imread(fp)
        players, ball, (h, w) = detect(fp)
        if len(players) < 3:
            continue
        boxes, labs, foot = [], [], []
        for fx, fy, box, _ in players:
            c = jersey_color(img, box)
            lab = int(np.argmin([np.linalg.norm(c - cen[k]) for k in range(2)])) if c is not None else -1
            boxes.append(box); labs.append(lab); foot.append([fx, fy])
        top = hg.warp(H, np.array(foot, np.float32))
        bw = hg.warp(H, [[ball[0], ball[1]]])[0] if ball is not None else None
        data.append({"img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "boxes": boxes,
                     "labs": np.array(labs), "top": top,
                     "on": np.array([(0 <= x <= PL and 0 <= y <= PW) for x, y in top]),
                     "ball": bw, "wh": (w, h)})
    print("usable frames:", len(data))

    fig, (axi, axp) = plt.subplots(1, 2, figsize=(13, 4.7), dpi=88)
    fig.patch.set_facecolor(BG)
    F = {"fontfamily": "Bahnschrift"}

    def draw(i):
        d = data[i]
        axi.clear(); axp.clear()
        axi.imshow(d["img"])
        for box, lab in zip(d["boxes"], d["labs"]):
            col = team_rgb[lab] if lab >= 0 else (0.5, 0.5, 0.5)
            x1, y1, x2, y2 = box
            axi.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.6))
        w, h = d["wh"]
        axi.set_xlim(0, w); axi.set_ylim(h, 0); axi.axis("off")
        axi.set_title("broadcast: players detected, teams by kit colour", color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        hg.draw_pitch(axp)
        for c in range(2):
            sel = (d["labs"] == c) & d["on"]
            pts = d["top"][sel]
            if len(pts):
                axp.scatter(pts[:, 0], pts[:, 1], s=130, c=[team_rgb[c]], edgecolors=BG, lw=1.2, zorder=5)
                hull(axp, pts, team_rgb[c])
        if d["ball"] is not None and 0 <= d["ball"][0] <= PL and 0 <= d["ball"][1] <= PW:
            axp.scatter([d["ball"][0]], [d["ball"][1]], s=70, c="#ffd23f", edgecolors=BG, lw=1, zorder=7)
        axp.set_title("top-down: visible team shapes (heatmap-grade)", color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        fig.suptitle("WC 2026  Brazil v Morocco  -- broadcast to top-down, live across one attack (visible players only)",
                     color=INK, x=0.5, fontsize=12.5, fontweight="bold", **F)

    a = manim.FuncAnimation(fig, draw, frames=len(data), interval=320)
    out = os.path.join(FIG, "wc2026_tactical_clip.gif")
    a.save(out, writer=manim.PillowWriter(fps=4))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
