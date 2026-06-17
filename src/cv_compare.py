"""480p vs 1080p: how much does input resolution sharpen the tactical layer?

Runs the identical CV pipeline -- detect players -> pitch-keypoint homography ->
team assignment -> top-down -- on the best wide frame at each resolution, and
puts them side by side with the hard numbers (pitch keypoints found, homography
reprojection error, players detected, players landed on-pitch). Player boxes are
drawn with the roboflow `supervision` annotators (BoxAnnotator + LabelAnnotator),
coloured by team.

    python src/cv_compare.py --lo data/clips/brazil_morocco --hi data/clips/brazil_morocco_1080
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import homography as hg
from broadcast_track import detect
from tactical import assign_teams

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
PL, PW = 120.0, 80.0
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"


def analyse(frame, imgsz):
    """homography + detection + reprojection error for one frame at given imgsz"""
    H, img_pts, pitch_pts = hg.keypoint_homography(frame, imgsz=imgsz)
    if H is None:
        return None
    players, _, (h, w) = detect(frame, conf=0.3, imgsz=imgsz)
    foot = np.array([[fx, fy] for fx, fy, *_ in players], np.float32)
    top = hg.warp(H, foot) if len(foot) else np.empty((0, 2))
    on = (top[:, 0] >= 0) & (top[:, 0] <= PL) & (top[:, 1] >= 0) & (top[:, 1] <= PW) \
        if len(top) else np.array([], bool)
    Hinv = np.linalg.inv(H)
    rmse = float(np.sqrt(((hg.warp(Hinv, pitch_pts) - img_pts) ** 2).sum(1).mean()))
    return {"frame": frame, "H": H, "players": players, "top": top, "on": on,
            "rmse": rmse, "nkp": len(img_pts), "hw": (h, w), "imgsz": imgsz}


def best(folder, imgsz, tries=16):
    fs = sorted(glob.glob(os.path.join(folder, "*.png")))
    best_r, best_n = None, -1
    for fp in fs[::max(len(fs) // tries, 1)]:
        r = analyse(fp, imgsz)
        if r and int(r["on"].sum()) > best_n:
            best_r, best_n = r, int(r["on"].sum())
    return best_r


def sv_annotate(frame_path, players, labels, team_bgr):
    """team-coloured labeled boxes via supervision; matplotlib fallback handled by caller"""
    import cv2
    import supervision as sv
    img = cv2.imread(frame_path)
    xyxy = np.array([b for _, _, b, _ in players], float)
    cid = np.array([max(int(l), 0) for l in labels])
    det = sv.Detections(xyxy=xyxy, class_id=cid)
    pal = sv.ColorPalette([sv.Color(r=int(b[2]), g=int(b[1]), b=int(b[0])) for b in team_bgr])
    out = sv.BoxAnnotator(color=pal, color_lookup=sv.ColorLookup.CLASS, thickness=2).annotate(img.copy(), det)
    out = sv.LabelAnnotator(color=pal, color_lookup=sv.ColorLookup.CLASS,
                            text_scale=0.4, text_thickness=1).annotate(
        out, det, labels=["T{}".format(c + 1) for c in cid])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_row(axf, axt, res, tag):
    import cv2
    img = cv2.imread(res["frame"])
    labels, team_bgr = assign_teams(img, res["players"])
    team_rgb = [(b[2] / 255, b[1] / 255, b[0] / 255) for b in team_bgr]
    try:
        annotated = sv_annotate(res["frame"], res["players"], labels, team_bgr)
        axf.imshow(annotated)
    except Exception as e:
        axf.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for (fx, fy, box, _), lab in zip(res["players"], labels):
            x1, y1, x2, y2 = box
            axf.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                          edgecolor=team_rgb[lab] if lab >= 0 else (.6, .6, .6), lw=1.6))
        print("  (supervision annotate fell back: {})".format(str(e)[:60]))
    axf.axis("off")
    h, w = res["hw"]
    axf.set_title("{}  {}x{}, imgsz {} | {} keypoints | reproj {:.2f}% | {} players, {} on pitch".format(
        tag, w, h, res["imgsz"], res["nkp"], res["rmse"] / w * 100,
        len(res["players"]), int(res["on"].sum())),
        color=INK, loc="left", fontsize=10, fontfamily="Bahnschrift", fontweight="bold")
    hg.draw_pitch(axt)
    for (x, y), lab, o in zip(res["top"], labels, res["on"]):
        if o:
            axt.scatter([x], [y], s=130, c=[team_rgb[lab] if lab >= 0 else (.6, .6, .6)],
                        edgecolors=BG, lw=1.3, zorder=5)
    axt.set_title("top-down (team-coloured)", color=INK, loc="left",
                  fontsize=10, fontfamily="Bahnschrift", fontweight="bold")


def main():
    os.makedirs(FIG, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", required=True, help="480p frames dir")
    ap.add_argument("--hi", required=True, help="1080p frames dir")
    args = ap.parse_args()

    lo, hi = best(args.lo, 640), best(args.hi, 1920)
    if lo is None or hi is None:
        print("could not find a usable wide frame in one of the folders")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor(BG)
    draw_row(axes[0, 0], axes[0, 1], lo, "480p (highlight)")
    draw_row(axes[1, 0], axes[1, 1], hi, "1080p (highlight)")
    fig.suptitle("It's not the file resolution -- it's the inference size: 480p@640 vs 1080p@1920",
                 color=INK, x=0.5, fontsize=14, fontfamily="Bahnschrift", fontweight="bold")
    fig.text(0.5, 0.01, "detect -> pitch-keypoint homography -> team assignment -> top-down | boxes via supervision annotators | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontsize=8, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(FIG, "_cv_resolution_compare.png")   # broadcast pixels -> local
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("480p@640 : {} players, {} on-pitch, reproj {:.2f}% width".format(
        len(lo["players"]), int(lo["on"].sum()), lo["rmse"] / lo["hw"][1] * 100))
    print("1080p@1920: {} players, {} on-pitch, reproj {:.2f}% width".format(
        len(hi["players"]), int(hi["on"].sum()), hi["rmse"] / hi["hw"][1] * 100))
    print("figure:", out)


if __name__ == "__main__":
    main()
