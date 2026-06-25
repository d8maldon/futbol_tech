"""Player tracking from a real broadcast frame (the main-camera route).

Ordinary highlight reels are the broadcast program feed, not the top-down
tactical minimap, so the colour-segmentation trick in minimap_track.py does not
apply. To recover player positions from a normal broadcast we need the harder,
more general pipeline:

    detect players + ball  (YOLO)
        -> foot point of each box
        -> homography from the broadcast camera onto a top-down pitch
        -> player (x, y) in pitch coordinates

The (x, y) it emits is the same coordinate that pitch_control.py already
consumes from StatsBomb 360, so the analytics layer is unchanged; only the data
front-end swaps. This module is the front-end.

Stage 1 (this file, run with --detect) proves detection works on the actual
broadcast pixels: it runs a COCO-pretrained YOLO over the extracted frames and
draws every person + ball it finds, with the foot point marked. Stage 2 adds
the homography to top-down coordinates.

    python src/broadcast_track.py --detect --frame data/clips/frames/out_000010.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
# COCO fallback detector. Prefer a local copy under data/models/, else let
# ultralytics fetch yolov8n.pt by name (cached after first download) -- portable
# instead of a machine-specific path.
_LOCAL_COCO = os.path.join(ROOT, "data", "models", "yolov8n.pt")
DEFAULT_WEIGHTS = _LOCAL_COCO if os.path.exists(_LOCAL_COCO) else "yolov8n.pt"

BG = "#0d1117"
INK = "#e6edf3"
MUT = "#7d8590"
PLAYER_C = "#5e9bff"
BALL_C = "#ffd23f"

# COCO class ids we care about on a pitch
PERSON, SPORTS_BALL = 0, 32


def foot_point(box):
    """bottom-centre of an (x1,y1,x2,y2) box = where the player meets the grass"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


_DET = {}
# A soccer-tuned 4-class detector (ball/goalkeeper/player/referee) beats COCO
# 'person': it never fires on crowd/bench (no generic person class). Auto-used
# when present at the path below; otherwise detect() falls back to COCO. To
# enable it, drop a 4-class YOLO here as data/models/soccer_players.pt -- e.g. the
# Roboflow "football-players-detection" model used by the `supervision` sports
# examples (https://github.com/roboflow/sports), exported to .pt.
SOCCER_WEIGHTS = os.path.join(ROOT, "data", "models", "soccer_players.pt")
SOCCER_CLS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}


def detect(frame_path, weights=None, conf=0.25, imgsz=None):
    """returns (players, ball, hw) from one broadcast frame.

    players: list of (foot_x, foot_y, box, score); ball: (x,y) or None.
    Uses the soccer-tuned model when available (players = player + goalkeeper,
    crowd excluded by design); else falls back to COCO person+sports_ball.
    imgsz defaults to each model's native size (soccer 800, COCO 640); raising it
    on high-res input finds more small players but do NOT exceed what the model
    was trained at, or accuracy drops.
    """
    from ultralytics import YOLO
    soccer = weights is None and os.path.exists(SOCCER_WEIGHTS)
    w = SOCCER_WEIGHTS if soccer else (weights or DEFAULT_WEIGHTS)
    if w not in _DET:
        _DET[w] = YOLO(w)
    model = _DET[w]
    iz = imgsz or (800 if soccer else 640)
    res = model(frame_path, conf=conf, imgsz=iz, verbose=False)[0]
    h, ww = res.orig_shape
    player_cls = {SOCCER_CLS["player"], SOCCER_CLS["goalkeeper"]} if soccer else {PERSON}
    ball_cls = SOCCER_CLS["ball"] if soccer else SPORTS_BALL
    players, ball, ball_score = [], None, -1.0
    for b in res.boxes:
        cls = int(b.cls[0])
        score = float(b.conf[0])
        xyxy = [float(v) for v in b.xyxy[0]]
        if cls in player_cls:
            players.append((*foot_point(xyxy), xyxy, score))
        elif cls == ball_cls and score > ball_score:
            ball, ball_score = foot_point(xyxy), score
    return players, ball, (h, ww)


def render_detection(frame_path, players, ball, out_path):
    img = np.asarray(Image.open(frame_path).convert("RGB"))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 130.0, h / 130.0), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.imshow(img)
    for fx, fy, (x1, y1, x2, y2), score in players:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=PLAYER_C, lw=1.4, alpha=0.9))
        ax.plot(fx, fy, "o", ms=4, mfc=PLAYER_C, mec=BG, mew=0.6)
    if ball is not None:
        ax.plot(ball[0], ball[1], "o", ms=9, mfc=BALL_C, mec=BG, mew=1.2)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("YOLO on real broadcast pixels: {} people, ball {}".format(
        len(players), "found" if ball is not None else "not found"),
        color=INK, loc="left", pad=10, fontfamily="Bahnschrift",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true", help="run stage 1 detection")
    ap.add_argument("--frame", help="a single extracted frame PNG/JPG")
    ap.add_argument("--frames-dir", help="run on every frame in a directory")
    ap.add_argument("--weights", default=None, help="default: soccer model if present, else COCO")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if args.detect:
        frames = ([args.frame] if args.frame
                  else sorted(glob.glob(os.path.join(args.frames_dir, "*.png"))))
        for i, fr in enumerate(frames):
            players, ball, (h, w) = detect(fr, args.weights, args.conf)
            out = os.path.join(FIG, "_detect_{:03d}.png".format(i))
            render_detection(fr, players, ball, out)
            print("{}: {}x{}  people={}  ball={}  -> {}".format(
                os.path.basename(fr), w, h, len(players),
                "yes" if ball is not None else "no", out))


if __name__ == "__main__":
    main()
