"""Calibrated uncertainty for the top-down positions -- the honesty moat.

We never quote a single global accuracy number. Instead, per FRAME we score how
trustworthy the homography is, and per PLAYER we attach a covariance ellipse that
grows with depth (distance from camera) and toward the image edges, exactly where
a broadcast homography is least certain.

  frame_confidence(H, img_pts, pitch_pts) -> dict with:
     mre_px      mean reprojection error of the detected keypoints (px)
     n_kp        number of keypoints used (>=4 needed, more = better conditioned)
     cond        condition number of H (ill-conditioned = unstable)
     conf        a 0..1 confidence; gate/abstain below ~0.35

  player_cov(H, foot_xy, sigma_px) -> (cov 2x2 in metres, (sx, sy, angle_deg))
     propagates an assumed sigma_px foot-point error through the local Jacobian
     of the perspective map, so far/edge players get bigger ellipses.

This mirrors how broadcast-tracking vendors expose a detected-vs-extrapolated flag
and a per-point confidence; here positions never seen are ABSENT, not estimated.

    python src/uncertainty.py
"""
import os

import numpy as np


def frame_confidence(H, img_pts, pitch_pts):
    if H is None or img_pts is None or len(img_pts) < 4:
        return {"mre_px": float("inf"), "n_kp": 0 if img_pts is None else len(img_pts),
                "cond": float("inf"), "conf": 0.0}
    import cv2
    Hinv = np.linalg.inv(H)
    reproj = cv2.perspectiveTransform(np.asarray(pitch_pts, np.float32).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    mre = float(np.sqrt(((reproj - img_pts) ** 2).sum(1).mean()))
    n = len(img_pts)
    cond = float(np.linalg.cond(H))
    conf = (np.clip(1.0 - mre / 25.0, 0, 1)       # low reprojection error
            * np.clip((n - 4) / 8.0, 0, 1)        # enough well-spread keypoints
            * (1.0 if cond < 5e5 else 0.3))       # well-conditioned transform
    return {"mre_px": mre, "n_kp": n, "cond": cond, "conf": float(conf)}


def player_cov(H, foot_xy, sigma_px=8.0):
    """propagate sigma_px image-point uncertainty to a pitch-metre covariance via
    the local Jacobian of the perspective warp (numeric finite-difference). Pass
    sigma_px = the frame's reprojection error (calibration uncertainty, which
    DOMINATES detection noise) so the ellipse reflects true positional uncertainty,
    not just detector jitter -- and it grows with depth/edge through the Jacobian."""
    import cv2
    p = np.asarray(foot_xy, np.float32).reshape(1, 1, 2)

    def w(pt):
        return cv2.perspectiveTransform(np.asarray(pt, np.float32).reshape(1, 1, 2), H).reshape(2)
    du = (w(foot_xy + np.array([1.0, 0.0])) - w(foot_xy - np.array([1.0, 0.0]))) / 2.0
    dv = (w(foot_xy + np.array([0.0, 1.0])) - w(foot_xy - np.array([0.0, 1.0]))) / 2.0
    J = np.array([[du[0], dv[0]], [du[1], dv[1]]])      # d(pitch)/d(image)
    cov = J @ (sigma_px ** 2 * np.eye(2)) @ J.T
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    ang = float(np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)])))
    sx, sy = float(np.sqrt(vals.max())), float(np.sqrt(vals.min()))
    return cov, (sx, sy, ang)


def main():
    import glob
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import homography as hg
    from broadcast_track import detect
    ROOT = os.path.join(os.path.dirname(__file__), "..")
    frames = sorted(glob.glob(os.path.join(ROOT, "data", "clips", "argentina_full", "f_*.jpg")))
    confs = []
    example = None
    for fp in frames[::40]:
        H, ip, pp = hg.keypoint_homography(fp)
        fc = frame_confidence(H, ip, pp)
        confs.append(fc["conf"])
        if H is not None and fc["conf"] > 0.4 and example is None:
            n_on = sum(1 for fx, fy, *_ in detect(fp)[0]
                       if 0 <= hg.warp(H, [[fx, fy]])[0][0] <= hg.PL
                       and 0 <= hg.warp(H, [[fx, fy]])[0][1] <= hg.PW)
            if n_on >= 8:
                example = (fp, H, fc["mre_px"])
    confs = np.array(confs)
    print("frame confidence over {} frames: median {:.2f} mean {:.2f} | abstain(<0.35) {:.0%}".format(
        len(confs), np.median(confs), confs.mean(), (confs < 0.35).mean()))

    if example:
        fp, H, mre = example
        sig = max(5.0, mre)                      # calibration uncertainty in px
        players, _, (h, w) = detect(fp)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=130); fig.patch.set_facecolor("#0d1117")
        hg.draw_pitch(ax)
        for fx, fy, *_ in players:
            p = hg.warp(H, [[fx, fy]])[0]
            if not (0 <= p[0] <= hg.PL and 0 <= p[1] <= hg.PW):
                continue
            _, (sx, sy, ang) = player_cov(H, np.array([fx, fy]), sigma_px=sig)
            ax.add_patch(Ellipse((p[0], hg.PW - p[1]), 2 * sx, 2 * sy, angle=-ang,
                         facecolor="#5e9bff", edgecolor="w", lw=0.6, alpha=0.4))
            ax.scatter([p[0]], [hg.PW - p[1]], s=20, c="w", zorder=5)
        ax.set_title("per-player 1-sigma uncertainty ellipses (sigma_px={:.0f}, grow with depth/edge)".format(sig),
                     color="#e6edf3", fontsize=11, loc="left")
        out = os.path.join(ROOT, "figures", "_uncertainty.png")
        fig.savefig(out, facecolor="#0d1117"); plt.close(fig)
        print("ellipse demo:", out)


if __name__ == "__main__":
    main()
