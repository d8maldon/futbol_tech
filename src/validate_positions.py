"""Validate that the top-down positions are ACCURATE and match the video frame.

Three independent checks, because "looks right" is not enough:

1. REPROJECTION RMSE (px): project the pitch-model landmarks back onto the frame
   through H^-1 and compare to where the keypoint model actually detected them.
   Low = the homography is self-consistent with the painted lines it locked onto.

2. LEAVE-ONE-OUT keypoint error (METRES) -- the real accuracy test: for each
   frame, drop one detected pitch keypoint, fit the homography on the REST, then
   predict the dropped keypoint's pitch position and compare to its KNOWN true
   pitch coordinate. The held-out point never informed the fit, so this measures
   how accurately the homography places a point of known ground-truth location.
   Reported in metres on a real 105x68 m pitch.

3. VISUAL overlays: reproject the pitch lines onto the broadcast (they must land
   on the painted lines) and show numbered players in the frame vs the same
   numbers on the flipped top-down -- so a human/agent can confirm left/right,
   near/far and per-player placement against the actual video.

Context: validate_topdown.py already measured ~5.1 m vs SoccerNet ground truth on
comparable broadcast footage; this confirms the same pipeline holds on THIS match.

    python src/validate_positions.py
"""
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import homography as hg
from broadcast_track import detect

ROOT = os.path.join(os.path.dirname(__file__), "..")
FRAMES = os.path.join(ROOT, "data", "clips", "argentina_full")
FIG = os.path.join(ROOT, "figures")
PL, PW = 120.0, 80.0
MX, MY = 105.0 / PL, 68.0 / PW       # StatsBomb units -> real metres
INK = "#e6edf3"; BG = "#0d1117"

# pitch lines in StatsBomb 120x80 units, for the reprojection overlay
def pitch_lines():
    L = [[(0, 0), (120, 0)], [(120, 0), (120, 80)], [(120, 80), (0, 80)], [(0, 80), (0, 0)],
         [(60, 0), (60, 80)],
         [(0, 18), (18, 18)], [(18, 18), (18, 62)], [(18, 62), (0, 62)],
         [(120, 18), (102, 18)], [(102, 18), (102, 62)], [(102, 62), (120, 62)]]
    th = np.linspace(0, 2 * np.pi, 40)
    circ = [[(60 + 10 * np.cos(a), 40 + 10 * np.sin(a)),
             (60 + 10 * np.cos(b), 40 + 10 * np.sin(b))] for a, b in zip(th[:-1], th[1:])]
    return L + circ


def loo_errors_m(img_pts, pitch_pts):
    """leave-one-out: predict each held-out keypoint's pitch pos from the rest"""
    errs = []
    n = len(img_pts)
    if n < 6:
        return errs
    for i in range(n):
        tr_i = np.delete(img_pts, i, 0); tr_p = np.delete(pitch_pts, i, 0)
        if not hg.spread_ok(tr_i):
            continue
        H, _ = cv2.findHomography(tr_i, tr_p, cv2.USAC_MAGSAC, 5.0)
        if H is None:
            continue
        pred = hg.warp(H, [img_pts[i]])[0]
        d = pred - pitch_pts[i]
        errs.append(float(np.hypot(d[0] * MX, d[1] * MY)))
    return errs


def overlay(fp, k):
    img = cv2.imread(fp)
    H, img_pts, pitch_pts = hg.keypoint_homography(fp)
    if H is None:
        return None
    Hinv = np.linalg.inv(H)
    players, ball, (h, w) = detect(fp)
    fig, (axi, axp) = plt.subplots(1, 2, figsize=(16, 5), dpi=104)
    fig.patch.set_facecolor(BG)
    axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # reprojected pitch lines (should sit on the painted lines)
    for (a, b) in pitch_lines():
        pa, pb = hg.warp(Hinv, [a, b])
        axi.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#ff3bd6", lw=1.4, alpha=0.85)
    # numbered players
    on = []
    for j, (fx, fy, b, _) in enumerate(players):
        p = hg.warp(H, [[fx, fy]])[0]
        ok = 0 <= p[0] <= PL and 0 <= p[1] <= PW
        axi.plot(fx, fy, "o", ms=5, mfc="red" if ok else "gray", mec="k")
        axi.text(fx + 4, fy, str(j), color="yellow", fontsize=10, fontweight="bold")
        if ok:
            on.append((j, p))
    axi.set_xlim(0, w); axi.set_ylim(h, 0); axi.axis("off")
    axi.set_title("broadcast + reprojected pitch lines (pink should sit on the painted lines)",
                  color=INK, fontsize=10, loc="left")
    hg.draw_pitch(axp)
    for j, p in on:
        axp.scatter([p[0]], [PW - p[1]], s=120, c="red", zorder=5)   # PW-y: same flip as the dashboard
        axp.text(p[0] + 1, PW - p[1], str(j), color="white", fontsize=10, fontweight="bold")
    axp.set_title("top-down (x-axis flipped, as in the dashboard) -- same player numbers",
                  color=INK, fontsize=10, loc="left")
    out = os.path.join(FIG, "_val_{:02d}.png".format(k))
    fig.tight_layout(); fig.savefig(out, facecolor=BG); plt.close(fig)
    return out


def main():
    fs = sorted(glob.glob(os.path.join(FRAMES, "f_*.jpg")))
    sample = fs[::40]
    rmse_all, loo_all, used = [], [], []
    for fp in sample:
        H, img_pts, pitch_pts = hg.keypoint_homography(fp)
        if H is None or img_pts is None or len(img_pts) < 6:
            continue
        Hinv = np.linalg.inv(H)
        reproj = hg.warp(Hinv, pitch_pts)
        rmse_all.append(float(np.sqrt(((reproj - img_pts) ** 2).sum(1).mean())))
        loo_all += loo_errors_m(img_pts, pitch_pts)
        used.append(fp)
    rmse = np.array(rmse_all); loo = np.array(loo_all)
    print("frames validated: {} (of {} sampled)".format(len(used), len(sample)))
    print("reprojection RMSE (px):  median {:.1f}  mean {:.1f}  p90 {:.1f}".format(
        np.median(rmse), rmse.mean(), np.percentile(rmse, 90)))
    print("LEAVE-ONE-OUT keypoint error (m):  median {:.2f}  mean {:.2f}  p90 {:.2f}  (n={})".format(
        np.median(loo), loo.mean(), np.percentile(loo, 90), len(loo)))
    for thr in (1, 2, 3, 5):
        print("   within {} m: {:.0%}".format(thr, (loo <= thr).mean()))
    # visual overlays across diverse frames
    k = 0
    for fp in used[::max(len(used) // 8, 1)][:8]:
        if overlay(fp, k):
            k += 1
    print("overlays written: figures/_val_00..{:02d}.png".format(k - 1))


if __name__ == "__main__":
    main()
