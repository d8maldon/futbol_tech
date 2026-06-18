"""Tactical snapshot from a broadcast frame: teams + top-down formation shape.

On a wide tactical frame we can do more than detect players: cluster their kit
colours into the two TEAMS, warp them to the top-down pitch through the
homography, and read the visible team SHAPE (line height, width, where they
are). Optionally overlays instance-segmentation masks if a seg model is present.

Honest scope: highlights show only part of the pitch and never all 22 players,
so this is a SNAPSHOT of the visible players' shape, not a full 90-minute
formation; team clustering can misfire on similar kits, keepers and referees.

    python src/tactical.py --frame data/clips/frames/out_0021.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import homography as hg
from broadcast_track import detect

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
SEG_MODEL = os.path.join(ROOT, "data", "models", "yolov8n-seg.pt")
PL, PW = 105.0, 68.0          # real FIFA metres (matches homography canonical)
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"


def jersey_color(img, box):
    """median kit colour from a player's torso, with grass (green) removed"""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    # torso band: middle horizontally, upper-third vertically
    cx1, cx2 = x1 + int(0.2 * w), x1 + int(0.8 * w)
    cy1, cy2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
    crop = img[max(cy1, 0):cy2, max(cx1, 0):cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    px = crop[~grass]
    if len(px) < 10:
        px = crop.reshape(-1, 3)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0)


def assign_teams(img, players):
    """k-means kit colours into 2 teams; returns labels + each team's BGR colour"""
    import cv2
    from sklearn.cluster import KMeans
    cols = [jersey_color(img, b) for _, _, b, _ in players]
    idx = [i for i, c in enumerate(cols) if c is not None]
    X = np.array([cols[i] for i in idx])
    if len(X) < 2:
        return np.zeros(len(players), int), [(0, 0, 255), (255, 0, 0)]
    km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(X)
    labels = np.full(len(players), -1)
    for k, i in enumerate(idx):
        labels[i] = km.labels_[k]
    team_bgr = []
    for c in range(2):
        lab = km.cluster_centers_[c].astype(np.uint8).reshape(1, 1, 3)
        team_bgr.append(tuple(int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]))
    return labels, team_bgr


def hull(ax, pts, color):
    if len(pts) < 3:
        return
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts)
        poly = pts[np.append(h.vertices, h.vertices[0])]
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=1.4, alpha=0.7)
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.10)
    except Exception:
        pass


def main():
    os.makedirs(FIG, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    args = ap.parse_args()
    import cv2

    img = cv2.imread(args.frame)
    H, _, _ = hg.keypoint_homography(args.frame)
    if H is None:
        print("no homography on this frame (not a wide pitch view)")
        return
    players, ball, (h, w) = detect(args.frame)
    if len(players) < 4:
        print("too few players detected")
        return
    labels, team_bgr = assign_teams(img, players)
    team_rgb = [(b[2] / 255, b[1] / 255, b[0] / 255) for b in team_bgr]

    foot = np.array([[fx, fy] for fx, fy, *_ in players], np.float32)
    top = hg.warp(H, foot)
    on = np.array([(0 <= x <= PL and 0 <= y <= PW) for x, y in top])

    # optional segmentation masks
    masks = None
    if os.path.exists(SEG_MODEL) and os.path.getsize(SEG_MODEL) > 5e6:
        try:
            from ultralytics import YOLO
            r = YOLO(SEG_MODEL)(args.frame, classes=[0], verbose=False)[0]
            if r.masks is not None:
                masks = r.masks.xy
        except Exception:
            masks = None

    fig, (axi, axp) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor(BG)
    axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if masks is not None:
        for poly in masks:
            axi.fill(poly[:, 0], poly[:, 1], color="#5e9bff", alpha=0.18)
    for (fx, fy, box, _), lab in zip(players, labels):
        col = team_rgb[lab] if lab >= 0 else (0.5, 0.5, 0.5)
        x1, y1, x2, y2 = box
        axi.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=col, lw=1.8))
    axi.set_xlim(0, w); axi.set_ylim(h, 0); axi.axis("off")
    axi.set_title("teams by kit colour{}".format("  + segmentation masks" if masks is not None else ""),
                  color=INK, loc="left", fontsize=11, fontfamily="Bahnschrift", fontweight="bold")

    hg.draw_pitch(axp)
    for c in range(2):
        sel = (labels == c) & on
        pts = top[sel]
        if len(pts):
            axp.scatter(pts[:, 0], pts[:, 1], s=150, c=[team_rgb[c]],
                        edgecolors=BG, lw=1.3, zorder=5)
            hull(axp, pts, team_rgb[c])
            # honest shape descriptors
            line_x = pts[:, 0].mean()
            width = np.ptp(pts[:, 1])
            axp.text(2, 76 - c * 6, "team {}: {} shown | width {:.0f}m | mean x {:.0f}m".format(
                c + 1, len(pts), width, line_x), color=team_rgb[c],
                fontsize=8.5, fontfamily="Bahnschrift")
    bw = hg.warp(H, [[ball[0], ball[1]]])[0] if ball is not None else None
    if bw is not None and 0 <= bw[0] <= PL and 0 <= bw[1] <= PW:
        axp.scatter([bw[0]], [bw[1]], s=70, c="#ffd23f", edgecolors=BG, lw=1, zorder=7)
    axp.set_title("top-down: visible team shapes (not a full-22 formation)",
                  color=INK, loc="left", fontsize=11, fontfamily="Bahnschrift", fontweight="bold")
    fig.suptitle("Tactical snapshot: teams + visible formation from one broadcast frame",
                 color=INK, x=0.5, fontsize=14, fontfamily="Bahnschrift", fontweight="bold")
    fig.text(0.5, 0.01, "SNAPSHOT of visible players only | team clustering can misfire on similar kits/keepers | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontsize=8, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(FIG, "_tactical_snapshot.png")   # broadcast pixels -> local
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("teams: {} vs {} | on-pitch {}/{} | masks {}".format(
        int((labels == 0).sum()), int((labels == 1).sum()), int(on.sum()),
        len(players), "yes" if masks is not None else "no"))
    print("figure:", out)


if __name__ == "__main__":
    main()
