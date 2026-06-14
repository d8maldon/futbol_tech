"""Validate the broadcast -> top-down pipeline in METRES against ground truth.

The honest accuracy test the user asked for. SoccerNet Game State Reconstruction
(SN-GSR-2025) ships real broadcast frames WITH each player's true pitch position
in metres. We run our own pipeline on those exact frames -- soccer detector ->
foot point -> pitch-keypoint homography -> top-down (x,y) -- and compare to the
ground truth, reporting the error in metres (mean/median, and Accuracy@d, the
GS-HOTA convention uses a 5 m tolerance).

Critical fix: our normal homography targets the roboflow pitch model (120x70 with
distorted box depths), so here we re-pair the 32 detected keypoints to REAL-metre
pitch coordinates (105x68, real box dimensions) so the warp is in true metres and
directly comparable to the GT. Frames are pulled on demand from the HuggingFace
zip with remotezip (no 11 GB download).

    python src/validate_topdown.py            # validate on cached SN-GSR clip
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import homography as hg
from broadcast_track import detect

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
CLIP = os.path.join(ROOT, "data", "sngs", "SNGS-021")
HALF_L, HALF_W = 52.5, 34.0       # real pitch half-length / half-width (105x68)
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; OUR = "#5e9bff"; GT = "#3fb950"


def real_vertices():
    """the 32 roboflow keypoints recomputed on a REAL 105x68 pitch (metres)"""
    W, L = 68.0, 105.0
    pbw, pbl, gbw, gbl, ccr, psd = 40.32, 16.5, 18.32, 5.5, 9.15, 11.0
    v = [
        (0, 0), (0, (W - pbw) / 2), (0, (W - gbw) / 2), (0, (W + gbw) / 2),
        (0, (W + pbw) / 2), (0, W), (gbl, (W - gbw) / 2), (gbl, (W + gbw) / 2),
        (psd, W / 2), (pbl, (W - pbw) / 2), (pbl, (W - gbw) / 2),
        (pbl, (W + gbw) / 2), (pbl, (W + pbw) / 2), (L / 2, 0),
        (L / 2, W / 2 - ccr), (L / 2, W / 2 + ccr), (L / 2, W),
        (L - pbl, (W - pbw) / 2), (L - pbl, (W - gbw) / 2), (L - pbl, (W + gbw) / 2),
        (L - pbl, (W + pbw) / 2), (L - psd, W / 2), (L - gbl, (W - gbw) / 2),
        (L - gbl, (W + gbw) / 2), (L, 0), (L, (W - pbw) / 2), (L, (W - gbw) / 2),
        (L, (W + gbw) / 2), (L, (W + pbw) / 2), (L, W),
        (L / 2 - ccr, W / 2), (L / 2 + ccr, W / 2),
    ]
    return np.array(v, np.float32)


REAL = real_vertices()


def homography_metres(frame):
    """H mapping image pixels -> REAL metres (corner origin), via keypoints"""
    import cv2
    res = hg.model()(frame, imgsz=640, verbose=False)[0]
    if res.keypoints is None or res.keypoints.data.shape[0] == 0:
        return None
    kp = res.keypoints.data[0].cpu().numpy()
    keep = kp[:, 2] > 0.5
    img_pts = kp[keep, :2].astype(np.float32)
    if len(img_pts) < 4 or not hg.spread_ok(img_pts):
        return None
    H, _ = cv2.findHomography(img_pts, REAL[keep], cv2.USAC_MAGSAC, 5.0)
    return H


def gt_positions(labels_by_img, image_id):
    out = []
    for a in labels_by_img.get(image_id, []):
        if a.get("attributes", {}).get("role") in ("player", "goalkeeper"):
            bp = a.get("bbox_pitch")
            if bp and bp.get("x_bottom_middle") is not None:
                out.append((bp["x_bottom_middle"], bp["y_bottom_middle"]))
    return np.array(out, float) if out else np.empty((0, 2))


def orient(pts, flip):
    """apply one of 4 global orientations (centre-origin), to align our frame
    with GT's (which end is +x / +y is a convention we don't know a priori)"""
    o = pts.copy()
    o[:, 0] *= flip[0]; o[:, 1] *= flip[1]
    return o


def match_err(ours, gt, gate=12.0):
    """Hungarian-match our points to GT; return matched distances (<= gate)"""
    if len(ours) == 0 or len(gt) == 0:
        return np.array([])
    d = np.linalg.norm(ours[:, None, :] - gt[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(d)
    e = d[ri, ci]
    return e[e <= gate]


def main():
    os.makedirs(FIG, exist_ok=True)
    labels = json.load(open(os.path.join(CLIP, "Labels-GameState.json"), encoding="utf-8"))
    by_img = {}
    for a in labels["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    file_to_id = {im["file_name"]: im["image_id"] for im in labels["images"]}

    frames = sorted(glob.glob(os.path.join(CLIP, "img1", "*.jpg")))
    # gather per-frame our-positions (centre-origin metres) + GT
    samples = []
    for fp in frames:
        iid = file_to_id.get(os.path.basename(fp))
        if iid is None:
            continue
        gt = gt_positions(by_img, iid)
        H = homography_metres(fp)
        if H is None or len(gt) == 0:
            continue
        players, _, _ = detect(fp)
        foot = np.array([[fx, fy] for fx, fy, *_ in players], np.float32)
        if len(foot) == 0:
            continue
        warp = hg.warp(H, foot) - [HALF_L, HALF_W]      # centre origin
        inb = (np.abs(warp[:, 0]) <= HALF_L + 3) & (np.abs(warp[:, 1]) <= HALF_W + 3)
        samples.append((fp, warp[inb], gt))
    if not samples:
        print("no usable frames (homography failed on all)")
        return

    # pick the single global orientation (of 4) that minimises total error
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    best_flip, best_tot = None, 1e18
    for f in flips:
        tot = sum(match_err(orient(o, f), g).sum() for _, o, g in samples)
        cnt = sum(len(match_err(orient(o, f), g)) for _, o, g in samples)
        score = tot / max(cnt, 1)
        if score < best_tot:
            best_tot, best_flip = score, f

    all_err = []
    per_frame = []
    for fp, o, g in samples:
        e = match_err(orient(o, best_flip), g)
        all_err += list(e)
        per_frame.append((fp, orient(o, best_flip), g, e))
    all_err = np.array(all_err)
    print("validated on {} frames of SN-GSR SNGS-021 (orientation {})".format(
        len(samples), best_flip))
    print("matched players: {}  | GT players/frame: {:.1f}  our on-pitch/frame: {:.1f}".format(
        len(all_err), np.mean([len(g) for _, _, g in samples]),
        np.mean([len(o) for _, o, _ in samples])))
    print("LOCALISATION ERROR (metres):  mean {:.2f}  median {:.2f}".format(
        all_err.mean(), np.median(all_err)))
    for d in (3, 5, 10):
        print("  within {:>2} m: {:.0%}".format(d, (all_err <= d).mean()))

    # figure: the frame with the most matches -- our (blue) vs GT (green)
    per_frame.sort(key=lambda x: -len(x[3]))
    fp, o, g, e = per_frame[0]
    fig, ax = plt.subplots(figsize=(9, 6.4), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor("#16341f")
    ax.plot([-HALF_L, -HALF_L, HALF_L, HALF_L, -HALF_L],
            [-HALF_W, HALF_W, HALF_W, -HALF_W, -HALF_W], color="#fff", lw=1.2, alpha=0.5)
    ax.plot([0, 0], [-HALF_W, HALF_W], color="#fff", lw=1, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(9.15 * np.cos(th), 9.15 * np.sin(th), color="#fff", lw=1, alpha=0.4)
    ax.scatter(g[:, 0], g[:, 1], s=150, facecolors="none", edgecolors=GT, lw=2,
               label="ground truth (SN-GSR)", zorder=4)
    ax.scatter(o[:, 0], o[:, 1], s=90, c=OUR, edgecolors=BG, lw=1, alpha=0.85,
               label="our pipeline", zorder=5)
    d = np.linalg.norm(o[:, None, :] - g[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(d)
    for a, b in zip(ri, ci):
        if d[a, b] <= 12:
            ax.plot([o[a, 0], g[b, 0]], [o[a, 1], g[b, 1]], color="#ffb347", lw=0.8, alpha=0.6)
    ax.set_xlim(-HALF_L - 3, HALF_L + 3); ax.set_ylim(-HALF_W - 3, HALF_W + 3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, labelcolor=INK,
              bbox_to_anchor=(0.5, 1.07), prop={"family": "Bahnschrift", "size": 9})
    ax.set_title("Top-down validated vs ground truth: mean {:.1f} m error ({} frames)".format(
        all_err.mean(), len(samples)), color=INK, loc="left", pad=28,
        fontfamily="Bahnschrift", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "SoccerNet GSR-2025 ground truth | our detector->homography->top-down | orange = match error | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(FIG, "wc2026_topdown_validation.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("figure: figures/wc2026_topdown_validation.png")


if __name__ == "__main__":
    main()
