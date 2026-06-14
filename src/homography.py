"""Broadcast -> pitch homography: the missing piece that puts players on a map.

broadcast_track.py detects players and gives each a foot point in IMAGE pixels.
To turn that into a top-down position we need the homography H mapping image
pixels -> pitch coordinates. This finds H automatically from a pitch-keypoint
model: a 32-point football-pitch pose model (HuggingFace, reachable here) marks
known pitch landmarks (box corners, halfway line, centre circle), each of which
has a known real pitch coordinate, and cv2.findHomography solves for H.

Canonical frame is StatsBomb 120x80 (what pitch_control.py and the 360 data use),
so warped points feed the existing analytics. We gate on >=4 well-spread
keypoints and draw a reprojection check (project the pitch model back onto the
frame -- if the lines land on the painted lines, H is good).

Honest scope: this MEASURES the players the camera can see. Off-screen players
are a separate (hard) inference problem (see fuse_eval.py).

    python src/homography.py --frame data/clips/frames/out_0021.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
KP_MODEL = os.path.join(ROOT, "data", "models", "pitch_kp.pt")
PL, PW = 120.0, 80.0
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"
ATT = "#5e9bff"; DEF = "#ff7a1a"; BALL = "#ffd23f"


def _pitch_vertices():
    """the 32 roboflow pitch landmarks (cm, 120x70) -> StatsBomb 120x80"""
    W, L = 7000.0, 12000.0
    pbw, pbl, gbw, gbl, ccr, psd = 4100., 2015., 1832., 550., 915., 1100.
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
    return np.array([(x / L * PL, y / W * PW) for x, y in v], np.float32)


PITCH = _pitch_vertices()
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(KP_MODEL):
            raise FileNotFoundError(
                "pitch-keypoint model missing. Fetch it (140MB, not in git):\n"
                "  curl -k -L -o data/models/pitch_kp.pt "
                "https://huggingface.co/martinjolif/yolo-football-pitch-detection"
                "/resolve/main/yolo-football-pitch-detection.pt")
        from ultralytics import YOLO
        _MODEL = YOLO(KP_MODEL)
    return _MODEL


def spread_ok(pts):
    """reject near-collinear / tiny-region point sets (unstable homography)"""
    if len(pts) < 4:
        return False
    p = pts - pts.mean(0)
    s = np.linalg.svd(p, compute_uv=False)
    return s[-1] > 1e-3 and (np.ptp(pts[:, 0]) > 40 and np.ptp(pts[:, 1]) > 40)


def keypoint_homography(img, conf=0.5):
    """returns (H_img->pitch, img_pts_used, pitch_pts_used) or (None, ..)"""
    import cv2
    res = model()(img, verbose=False)[0]
    if res.keypoints is None or res.keypoints.data.shape[0] == 0:
        return None, None, None
    kp = res.keypoints.data[0].cpu().numpy()      # (32, 3): x, y, conf
    keep = kp[:, 2] > conf
    img_pts = kp[keep, :2].astype(np.float32)
    pitch_pts = PITCH[keep]
    if not spread_ok(img_pts):
        return None, img_pts, pitch_pts
    H, mask = cv2.findHomography(img_pts, pitch_pts, cv2.USAC_MAGSAC, 5.0)
    return H, img_pts, pitch_pts


def warp(H, pts):
    import cv2
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def draw_pitch(ax):
    ax.set_facecolor("#16341f")
    ax.plot([0, 0, PL, PL, 0], [0, PW, PW, 0, 0], color="#fff", lw=1.2, alpha=0.55)
    ax.plot([PL / 2, PL / 2], [0, PW], color="#fff", lw=1, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(PL / 2 + 10 * np.cos(th), PW / 2 + 10 * np.sin(th), color="#fff", lw=1, alpha=0.4)
    for x0 in (0, PL - 18):
        ax.plot([x0, x0 + 18, x0 + 18, x0], [18, 18, 62, 62], color="#fff", lw=1, alpha=0.4)
    ax.set_xlim(-3, PL + 3); ax.set_ylim(-3, PW + 3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    os.makedirs(FIG, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()
    import cv2
    from broadcast_track import detect

    img = cv2.imread(args.frame)
    H, img_pts, pitch_pts = keypoint_homography(args.frame, args.conf)
    if H is None:
        n = 0 if img_pts is None else len(img_pts)
        print("homography FAILED: only {} usable keypoints (need >=4 well spread)".format(n))
        return
    print("keypoints used: {}".format(len(img_pts)))
    # reprojection error (pitch->image via H_inv, compare to detected keypoints)
    Hinv = np.linalg.inv(H)
    reproj = warp(Hinv, pitch_pts)
    rmse = float(np.sqrt(((reproj - img_pts) ** 2).sum(1).mean()))
    print("reprojection RMSE: {:.1f} px".format(rmse))

    players, ball, (h, w) = detect(args.frame)
    foot = np.array([[fx, fy] for fx, fy, *_ in players], np.float32)
    top = warp(H, foot) if len(foot) else np.empty((0, 2))
    on = [(0 <= x <= PL and 0 <= y <= PW) for x, y in top]
    print("players warped onto pitch: {}/{}".format(sum(on), len(on)))

    fig, (axi, axp) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor(BG)
    axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # reprojected pitch lines on the frame = the H check
    for a, b in [(0,5),(25,29),(0,25),(5,29),(13,16),(9,12),(17,20),(25,26),(28,29)]:
        if (pitch_pts == PITCH[a]).all(1).any() and (pitch_pts == PITCH[b]).all(1).any():
            pa, pb = warp(Hinv, PITCH[[a, b]])
            axi.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#ff3bd6", lw=1.5, alpha=0.8)
    axi.scatter(img_pts[:, 0], img_pts[:, 1], s=30, c="#ff3bd6", edgecolors="w", lw=0.5, zorder=5)
    for (fx, fy, *_), o in zip(players, on):
        axi.plot(fx, fy, "o", ms=5, mfc=ATT if o else MUT, mec=BG, mew=0.6)
    axi.set_xlim(0, w); axi.set_ylim(h, 0); axi.axis("off")
    axi.set_title("broadcast frame + detected pitch keypoints (pink) | RMSE {:.0f}px".format(rmse),
                  color=INK, loc="left", fontsize=11, fontfamily="Bahnschrift", fontweight="bold")
    draw_pitch(axp)
    if len(top):
        ok = np.array(on)
        axp.scatter(top[ok, 0], top[ok, 1], s=150, c=ATT, edgecolors=BG, lw=1.3, zorder=5)
    axp.set_title("warped to top-down pitch ({} players placed)".format(sum(on)),
                  color=INK, loc="left", fontsize=11, fontfamily="Bahnschrift", fontweight="bold")
    fig.suptitle("Broadcast -> pitch: automatic homography (32-keypoint model)",
                 color=INK, x=0.5, fontsize=14, fontfamily="Bahnschrift", fontweight="bold")
    fig.text(0.5, 0.01, "MEASURES visible players only | pitch keypoints from a HuggingFace pose model | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, fontsize=8, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(FIG, "_homography_check.png")   # has broadcast pixels -> local
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("figure: {}".format(out))


if __name__ == "__main__":
    main()
