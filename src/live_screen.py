"""Real-time analysis of a live game on your screen (screen-capture -> YOLO).

If a match is playing on one of your monitors, this grabs that region of the
screen many times a second, runs the YOLO player+ball detector on each frame on
the GPU, and draws the boxes back over a live window. It is the same detector
proven on broadcast clips in broadcast_track.py -- only the input changes, from
a downloaded file to your screen. On an RTX 4070 it runs faster than broadcast
frame-rate, so it keeps up with a live stream.

Scope (same as broadcast_track): it DETECTS players and the ball. Turning those
into top-down pitch (x,y) for pitch-control still needs the homography step
(pitch-keypoint detection), which is not built yet. Screen capture is for your
own personal viewing/analysis; don't redistribute the captured video.

    python src/live_screen.py --box 0,0,1280,720 --show   # analyse that region live
    python src/live_screen.py --selftest                  # verify the loop runs (no save)
"""
import argparse
import os
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_WEIGHTS = os.path.join(
    ROOT, "..", "Image-Processing", "Object Detect", "yolov8n.pt")
PERSON, SPORTS_BALL = 0, 32
PLAYER_BGR = (255, 155, 94)   # blue-ish (OpenCV is BGR)
BALL_BGR = (63, 210, 255)     # amber


def parse_box(s):
    left, top, w, h = (int(v) for v in s.split(","))
    return {"left": left, "top": top, "width": w, "height": h}


def detections(res):
    """yield (cls, x1, y1, x2, y2) for person + ball boxes"""
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls in (PERSON, SPORTS_BALL):
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            yield cls, x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", help="screen region left,top,width,height (default: primary monitor)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true", help="live annotated window (press q to quit)")
    ap.add_argument("--selftest", action="store_true", help="time the capture+detect loop, no save/display")
    ap.add_argument("--frames", type=int, default=40, help="frames for --selftest")
    args = ap.parse_args()

    import mss as mss_mod
    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights)
    dev = 0 if torch.cuda.is_available() else "cpu"

    with mss_mod.MSS() as sct:
        box = parse_box(args.box) if args.box else sct.monitors[1]

        if args.selftest:
            # capture the screen, run YOLO, report ONLY timing + counts.
            # nothing is saved or shown -- this verifies the loop without
            # snooping whatever is on screen.
            for _ in range(3):                      # warm up GPU + model
                model(np.asarray(sct.grab(box))[:, :, :3], verbose=False, device=dev)
            t0 = time.perf_counter()
            tp = tb = 0
            for _ in range(args.frames):
                img = np.asarray(sct.grab(box))[:, :, :3]
                res = model(img, conf=args.conf, verbose=False, device=dev)[0]
                for cls, *_ in detections(res):
                    if cls == PERSON:
                        tp += 1
                    else:
                        tb += 1
            dt = time.perf_counter() - t0
            print("captured {}x{} region, {} frames in {:.2f}s = {:.1f} FPS".format(
                box["width"], box["height"], args.frames, dt, args.frames / dt))
            print("avg per frame: {:.1f} people, {:.2f} ball  (0 is expected if no game is on screen)".format(
                tp / args.frames, tb / args.frames))
            print("real-time? broadcast is ~25-30 FPS, so {} keeps up.".format(
                "this" if args.frames / dt >= 25 else "borderline -- use a smaller --box or a faster GPU model"))
            return

        import cv2
        print("live analysis -- press q in the window to quit")
        last = time.perf_counter()
        fps = 0.0
        while True:
            img = np.ascontiguousarray(np.asarray(sct.grab(box))[:, :, :3])
            res = model(img, conf=args.conf, verbose=False, device=dev)[0]
            npl = nbl = 0
            for cls, x1, y1, x2, y2 in detections(res):
                if cls == PERSON:
                    cv2.rectangle(img, (x1, y1), (x2, y2), PLAYER_BGR, 2)
                    npl += 1
                else:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 7, BALL_BGR, -1)
                    nbl += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(img, "players {}  ball {}  {:.0f} FPS".format(npl, nbl, fps),
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
            cv2.imshow("hidden-timeout live", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
