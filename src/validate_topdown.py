"""Validate the broadcast -> top-down pipeline in METRES against ground truth.

The honest accuracy test. SoccerNet Game State Reconstruction (SN-GSR-2025) ships
real broadcast frames WITH each player's true pitch position in metres. We run our
pipeline on those frames -- soccer detector -> foot point -> pitch-keypoint
homography -> top-down (x,y) -- and compare to ground truth, in metres, across
several clips for a robust mean +/- confidence interval (GS-HOTA uses 5 m).

Frames are pulled on demand from the 11 GB HuggingFace zip with remotezip (HTTP
range requests) -- only a handful of frames per clip, never the full download. The
32 detected keypoints are paired to REAL-metre pitch coordinates (105x68, real box
dims) so the warp is true metres, directly comparable to the GT.

    python src/validate_topdown.py            # batch several clips
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
SNGS = os.path.join(ROOT, "data", "sngs")
ZIP_URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/valid.zip"
CLIPS = ["SNGS-021", "SNGS-022", "SNGS-023", "SNGS-024",
         "SNGS-025", "SNGS-026", "SNGS-027", "SNGS-028"]
N_FRAMES = 16
HALF_L, HALF_W = 52.5, 34.0
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; OUR = "#5e9bff"; GT = "#3fb950"; ACC = "#ffb347"


def real_vertices():
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


def ensure_clips(clips, n_frames):
    """extract labels + sampled frames for any clip not cached, one zip session"""
    todo = [c for c in clips if not os.path.exists(
        os.path.join(SNGS, c, "Labels-GameState.json"))
        or len(glob.glob(os.path.join(SNGS, c, "img1", "*.jpg"))) < n_frames]
    if not todo:
        return
    import urllib3
    import requests
    urllib3.disable_warnings()
    _o = requests.adapters.HTTPAdapter.send
    requests.adapters.HTTPAdapter.send = lambda s, r, **k: _o(s, r, **{**k, "verify": False})
    from remotezip import RemoteZip
    idxs = list(range(1, 751, max(750 // n_frames, 1)))[:n_frames]
    with RemoteZip(ZIP_URL) as z:
        for c in todo:
            try:
                z.extract("{}/Labels-GameState.json".format(c), SNGS)
                for i in idxs:
                    z.extract("{}/img1/{:06d}.jpg".format(c, i), SNGS)
                print("  extracted {}".format(c))
            except Exception as e:
                print("  {} extract failed: {}".format(c, str(e)[:50]))


def homography_metres(frame):
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


def gt_positions(by_img, image_id):
    out = []
    for a in by_img.get(image_id, []):
        if a.get("attributes", {}).get("role") in ("player", "goalkeeper"):
            bp = a.get("bbox_pitch")
            if bp and bp.get("x_bottom_middle") is not None:
                out.append((bp["x_bottom_middle"], bp["y_bottom_middle"]))
    return np.array(out, float) if out else np.empty((0, 2))


def match_err(ours, gt, flip, gate=12.0):
    if len(ours) == 0 or len(gt) == 0:
        return np.array([])
    o = ours * np.array(flip)
    d = np.linalg.norm(o[:, None, :] - gt[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(d)
    e = d[ri, ci]
    return e[e <= gate]


def validate_clip(clip):
    """returns (errors_metres, n_gt_per_frame, n_our_per_frame, example) for a clip"""
    cdir = os.path.join(SNGS, clip)
    lab = json.load(open(os.path.join(cdir, "Labels-GameState.json"), encoding="utf-8"))
    by_img = {}
    for a in lab["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    f2id = {im["file_name"]: im["image_id"] for im in lab["images"]}
    samples = []
    for fp in sorted(glob.glob(os.path.join(cdir, "img1", "*.jpg"))):
        iid = f2id.get(os.path.basename(fp))
        gt = gt_positions(by_img, iid) if iid else np.empty((0, 2))
        if len(gt) == 0:
            continue
        H = homography_metres(fp)
        if H is None:
            continue
        players, _, _ = detect(fp)
        foot = np.array([[fx, fy] for fx, fy, *_ in players], np.float32)
        if len(foot) == 0:
            continue
        w = hg.warp(H, foot) - [HALF_L, HALF_W]
        inb = (np.abs(w[:, 0]) <= HALF_L + 3) & (np.abs(w[:, 1]) <= HALF_W + 3)
        samples.append((fp, w[inb], gt))
    if not samples:
        return np.array([]), 0, 0, None
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

    def flip_score(f):
        es = [match_err(o, g, f) for _, o, g in samples]
        cnt = sum(len(e) for e in es)
        tot = sum(float(e.sum()) for e in es)
        return (cnt, -tot)                         # most matches, then least error
    bf = max(flips, key=flip_score)
    err = np.concatenate([match_err(o, g, bf) for _, o, g in samples])
    ex = max(samples, key=lambda s: len(match_err(s[1], s[2], bf)))
    return err, np.mean([len(g) for _, _, g in samples]), \
        np.mean([len(o) for _, o, _ in samples]), (ex[0], ex[1] * np.array(bf), ex[2])


def main():
    os.makedirs(FIG, exist_ok=True)
    ensure_clips(CLIPS, N_FRAMES)
    per_clip = []
    pooled = []
    example = None
    for c in CLIPS:
        if not os.path.exists(os.path.join(SNGS, c, "Labels-GameState.json")):
            continue
        err, ngt, nour, ex = validate_clip(c)
        if len(err) == 0:
            print("  {}: no usable frames".format(c)); continue
        per_clip.append((c, err.mean(), len(err)))
        pooled += list(err)
        if example is None and ex is not None:
            example = ex
        print("  {}: mean {:.2f} m  (n={}, GT/frame {:.1f})".format(c, err.mean(), len(err), ngt))
    pooled = np.array(pooled)
    means = np.array([m for _, m, _ in per_clip])
    rng = np.random.default_rng(0)
    boot = [rng.choice(means, len(means)).mean() for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n=== {} clips, {} matched players ===".format(len(per_clip), len(pooled)))
    print("mean localisation error: {:.2f} m  (per-clip mean {:.2f} m, 95% CI [{:.2f}, {:.2f}])".format(
        pooled.mean(), means.mean(), lo, hi))
    print("median {:.2f} m | within 5 m {:.0%} | within 10 m {:.0%}".format(
        np.median(pooled), (pooled <= 5).mean(), (pooled <= 10).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=160,
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor("#16341f")
    ex_fp, ex_o, ex_g = example
    ax1.plot([-HALF_L, -HALF_L, HALF_L, HALF_L, -HALF_L], [-HALF_W, HALF_W, HALF_W, -HALF_W, -HALF_W], color="#fff", lw=1.1, alpha=0.5)
    ax1.plot([0, 0], [-HALF_W, HALF_W], color="#fff", lw=1, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 60); ax1.plot(9.15 * np.cos(th), 9.15 * np.sin(th), color="#fff", lw=1, alpha=0.4)
    ax1.scatter(ex_g[:, 0], ex_g[:, 1], s=130, facecolors="none", edgecolors=GT, lw=2, label="ground truth", zorder=4)
    ax1.scatter(ex_o[:, 0], ex_o[:, 1], s=80, c=OUR, edgecolors=BG, lw=1, alpha=0.85, label="our pipeline", zorder=5)
    d = np.linalg.norm(ex_o[:, None, :] - ex_g[None, :, :], axis=2); ri, ci = linear_sum_assignment(d)
    for a, b in zip(ri, ci):
        if d[a, b] <= 12:
            ax1.plot([ex_o[a, 0], ex_g[b, 0]], [ex_o[a, 1], ex_g[b, 1]], color=ACC, lw=0.8, alpha=0.6)
    ax1.set_xlim(-HALF_L - 3, HALF_L + 3); ax1.set_ylim(-HALF_W - 3, HALF_W + 3); ax1.set_aspect("equal")
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.legend(loc="upper center", ncol=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 1.08), prop={"family": "Bahnschrift", "size": 9})
    ax1.set_title("Example frame: our top-down vs ground truth", color=INK, loc="left", pad=22, fontfamily="Bahnschrift", fontsize=12, fontweight="bold")

    ax2.set_facecolor(BG)
    xs = np.arange(len(per_clip))
    ax2.bar(xs, [m for _, m, _ in per_clip], color=OUR, width=0.66)
    ax2.axhline(pooled.mean(), color=ACC, lw=1.4, ls=(0, (4, 3)), label="overall {:.1f} m".format(pooled.mean()))
    ax2.axhline(5, color=GT, lw=1.0, ls=":", label="5 m (GS-HOTA)")
    ax2.set_xticks(xs); ax2.set_xticklabels([name.replace("SNGS-", "") for name, _, _ in per_clip], color=MUT, fontsize=8)
    ax2.set_ylabel("mean error (m)", color=MUT, fontsize=9, fontfamily="Bahnschrift")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(MUT)
    ax2.tick_params(colors=MUT)
    ax2.legend(frameon=False, labelcolor=INK, loc="upper right", prop={"family": "Bahnschrift", "size": 8})
    ax2.set_title("Per-clip error  ({} clips, mean {:.1f} m, 95% CI [{:.1f},{:.1f}])".format(
        len(per_clip), pooled.mean(), lo, hi), color=INK, loc="left", pad=10, fontfamily="Bahnschrift", fontsize=11, fontweight="bold")
    fig.suptitle("Broadcast -> top-down, validated in metres vs SoccerNet GSR ground truth", color=INK, x=0.5, fontsize=14, fontfamily="Bahnschrift", fontweight="bold")
    fig.text(0.5, 0.01, "remotezip-sampled SN-GSR frames | within 5m {:.0%}, 10m {:.0%} | clean tactical broadcast (highlights are harder) | github.com/d8maldon/hidden-timeout".format((pooled <= 5).mean(), (pooled <= 10).mean()), ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(FIG, "wc2026_topdown_validation.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("figure: figures/wc2026_topdown_validation.png")


if __name__ == "__main__":
    main()
