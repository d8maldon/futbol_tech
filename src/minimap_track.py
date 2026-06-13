"""Reconstruct player tracking from a broadcast minimap.

The FIFA/FUSSBALL broadcast paints player positions onto a top-down minimap.
A minimap is already a bird's-eye projection, so no perspective/homography
correction is needed: recovering positions is colour segmentation + blob
detection + a linear pixel->pitch scale. This is dramatically easier than
extracting positions from the main broadcast camera.

This module proves the pipeline end to end. With no argument it generates a
synthetic minimap that mimics the broadcast (green pitch, mowing stripes,
white pitch lines, 11 red + 11 white player discs, a ball) and shows that the
detector recovers all 22 players and scales them back to the pitch within a
small error. Point it at a real frame with --frame and a crop box and it runs
the identical detector on real pixels.

    python src/minimap_track.py                      # synthetic proof
    python src/minimap_track.py --frame shot.png --box 40,300,300,420
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy import ndimage

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
PITCH_L, PITCH_W = 105.0, 68.0

BG = "#0d1117"
INK = "#e6edf3"
MUT = "#7d8590"
HOME_C = "#e63946"   # red dots
AWAY_C = "#f1faee"   # white dots


# ----------------------------------------------------------- synthetic frame
def disc(img, cx, cy, r, color):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    img[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = color


def make_synthetic(path, w=560, h=360):
    """a broadcast-style top-down minimap with a known ground truth"""
    img = np.zeros((h, w, 3), np.uint8)
    for i in range(0, w, 40):                       # mowing stripes
        img[:, i:i + 20] = (28, 132, 50)
        img[:, i + 20:i + 40] = (33, 146, 56)
    line = (235, 235, 235)
    img[8:h - 8, 8:10] = line                       # pitch outline
    img[8:h - 8, w - 10:w - 8] = line
    img[8:10, 8:w - 8] = line
    img[h - 10:h - 8, 8:w - 8] = line
    img[8:h - 8, w // 2 - 1:w // 2 + 1] = line      # halfway line
    yy, xx = np.ogrid[:h, :w]                        # centre circle
    ring = np.abs((xx - w / 2) ** 2 + (yy - h / 2) ** 2 - 44 ** 2) < 360
    img[ring] = line

    rng = np.random.default_rng(7)
    # 4-3-3-ish home (attacking right) and away (attacking left), in pitch m
    home = [(8, 34), (22, 12), (22, 27), (22, 41), (22, 56),
            (40, 20), (40, 34), (40, 48), (58, 16), (58, 34), (58, 52)]
    away = [(97, 34), (83, 12), (83, 27), (83, 41), (83, 56),
            (66, 22), (66, 34), (66, 46), (50, 18), (50, 34), (50, 50)]
    truth = []
    for team, pts, col in (("home", home, (224, 49, 53)), ("away", away, (245, 250, 246))):
        for mx, my in pts:
            mx += rng.normal(0, 1.2)
            my += rng.normal(0, 1.2)
            px = int(8 + mx / PITCH_L * (w - 16))
            py = int(8 + (PITCH_W - my) / PITCH_W * (h - 16))
            disc(img, px, py, 6, col)
            truth.append((team, mx, my, px, py))
    disc(img, w // 2 + 30, h // 2 - 10, 3, (250, 210, 40))  # ball
    Image.fromarray(img).save(path)
    return truth


# ------------------------------------------------------------------ detector
def detect(img, box=None):
    """returns list of (team, pitch_x, pitch_y, px, py) from a minimap image"""
    if box is not None:
        x0, y0, x1, y1 = box
        crop = img[y0:y1, x0:x1]
        ox, oy = x0, y0
    else:
        crop = img
        ox, oy = 0, 0
    h, w = crop.shape[:2]
    hsv = rgb_to_hsv(crop / 255.0)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    red = ((H < 0.045) | (H > 0.95)) & (S > 0.45) & (V > 0.4)
    white = (S < 0.18) & (V > 0.75)
    # discs are compact; pitch lines are long and thin, removed by area + aspect
    disc_area = np.pi * 6 ** 2
    lo, hi = 0.35 * disc_area, 4.0 * disc_area

    out = []
    for team, mask in (("home", red), ("away", white)):
        labels, n = ndimage.label(mask)
        for lab in range(1, n + 1):
            ys, xs = np.where(labels == lab)
            area = xs.size
            if not (lo <= area <= hi):
                continue
            bw, bh = np.ptp(xs) + 1, np.ptp(ys) + 1
            if max(bw, bh) / max(min(bw, bh), 1) > 2.5:   # reject line fragments
                continue
            cx, cy = xs.mean(), ys.mean()
            pitch_x = cx / w * PITCH_L
            pitch_y = (1 - cy / h) * PITCH_W
            out.append((team, pitch_x, pitch_y, cx + ox, cy + oy))
    return out


# -------------------------------------------------------------------- render
def pitch_ax(ax):
    ax.set_facecolor("#16341f")
    ax.set_xlim(-2, PITCH_L + 2)
    ax.set_ylim(-2, PITCH_W + 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    lc = dict(color="#ffffff", lw=1.2, alpha=0.6)
    ax.plot([0, 0, PITCH_L, PITCH_L, 0], [0, PITCH_W, PITCH_W, 0, 0], **lc)
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], **lc)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(PITCH_L / 2 + 9.15 * np.cos(th), PITCH_W / 2 + 9.15 * np.sin(th), **lc)
    for x0 in (0, PITCH_L - 16.5):
        ax.plot([x0, x0 + 16.5, x0 + 16.5, x0], [13.84, 13.84, 54.16, 54.16], **lc)


def render(detected, truth, path):
    fig, axes = plt.subplots(1, 2 if truth else 1, figsize=(12 if truth else 6.5, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    pitch_ax(ax)
    for team, x, y, *_ in detected:
        ax.plot(x, y, "o", ms=11, mfc=HOME_C if team == "home" else AWAY_C,
                mec=BG, mew=1.2)
    nh = sum(1 for d in detected if d[0] == "home")
    na = len(detected) - nh
    ax.set_title("recovered from the minimap: {} red + {} white".format(nh, na),
                 color=INK, loc="left", pad=8,
                 fontfamily="Bahnschrift", fontsize=12, fontweight="bold")

    if truth:
        # match each detection to nearest same-team truth, measure error (m)
        errs = []
        for team, x, y, *_ in detected:
            cands = [(tx, ty) for tm, tx, ty, *_ in truth if tm == team]
            if cands:
                d = min(np.hypot(x - tx, y - ty) for tx, ty in cands)
                errs.append(d)
        ax2 = axes[1]
        pitch_ax(ax2)
        for team, x, y, *_ in truth:
            ax2.plot(x, y, "o", ms=12, mfc="none",
                     mec=HOME_C if team == "home" else AWAY_C, mew=1.6)
        for team, x, y, *_ in detected:
            ax2.plot(x, y, "x", ms=7, color="#ffb347", mew=2)
        ax2.set_title("detected (x) vs truth (o)  |  {}/22 found, mean err {:.2f} m".format(
            len(detected), np.mean(errs) if errs else float("nan")),
            color=INK, loc="left", pad=8,
            fontfamily="Bahnschrift", fontsize=12, fontweight="bold")

    fig.suptitle("Minimap -> tracking: feasibility check", color=INK, x=0.075, ha="left",
                 fontfamily="Bahnschrift", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", help="path to a real minimap PNG/JPG")
    ap.add_argument("--box", help="crop box x0,y0,x1,y1 of the minimap within the frame")
    args = ap.parse_args()

    if args.frame:
        img = np.asarray(Image.open(args.frame).convert("RGB"))
        box = tuple(int(v) for v in args.box.split(",")) if args.box else None
        det = detect(img, box)
        render(det, None, os.path.join(FIG, "minimap_real.png"))
        print("detected {} players from {}".format(len(det), args.frame))
    else:
        synth = os.path.join(FIG, "_synthetic_minimap.png")
        truth = make_synthetic(synth)
        img = np.asarray(Image.open(synth).convert("RGB"))
        det = detect(img)
        errs = []
        for team, x, y, *_ in det:
            cands = [(tx, ty) for tm, tx, ty, *_ in truth if tm == team]
            if cands:
                errs.append(min(np.hypot(x - tx, y - ty) for tx, ty in cands))
        render(det, truth, os.path.join(FIG, "minimap_feasibility.png"))
        print("synthetic minimap: placed 22, recovered {}".format(len(det)))
        print("mean position error: {:.2f} m  (pitch is {}x{} m)".format(
            np.mean(errs), int(PITCH_L), int(PITCH_W)))
        print("figure: figures/minimap_feasibility.png")


if __name__ == "__main__":
    main()
