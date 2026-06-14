"""Real-time game analysis from a live source: screen, camera, or video file.

If a match is on a monitor, this grabs the screen; if you point a phone (used as
a webcam via DroidCam / Iriun / Camo) at your TV, it reads that camera; or it can
run on a saved clip. Each frame goes to the YOLO player+ball detector on the GPU
and the boxes are drawn back live. Same detector proven on broadcast clips in
broadcast_track.py -- only the input changes. On an RTX 4070 it runs faster than
broadcast frame-rate, so it keeps up live.

Scope (same as broadcast_track): it DETECTS players and the ball. Top-down pitch
(x,y) still needs the homography step (not built yet) -- and a phone aimed at a
TV adds glare/angle/perspective, so detection is a bit noisier than a clean feed,
though still solid. Personal viewing/analysis only.

    python src/live_screen.py --screen --box 0,0,1280,720 --show   # a monitor region
    python src/live_screen.py --camera 1 --show                    # phone-as-webcam at the TV
    python src/live_screen.py --video clip.mp4 --show              # a saved clip
    python src/live_screen.py --camera 1 --selftest                # verify a source (no save)
"""
import argparse
import os
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_WEIGHTS = os.path.join(
    ROOT, "..", "Image-Processing", "Object Detect", "yolov8n.pt")
PERSON, SPORTS_BALL = 0, 32
PLAYER_BGR = (255, 155, 94)
BALL_BGR = (63, 210, 255)


def parse_box(s):
    left, top, w, h = (int(v) for v in s.split(","))
    return {"left": left, "top": top, "width": w, "height": h}


class Source:
    """unified frame source: screen (mss), camera or video (OpenCV)."""

    def __init__(self, args):
        self.kind = "camera" if args.camera is not None else (
            "video" if args.video else "screen")
        self.cap = self.sct = self.box = None
        if self.kind == "screen":
            import mss as mss_mod
            self.sct = mss_mod.MSS()
            self.box = parse_box(args.box) if args.box else self.sct.monitors[1]
            self.label = "screen {}x{}".format(self.box["width"], self.box["height"])
        else:
            import cv2
            if self.kind == "camera":
                # CAP_DSHOW opens Windows webcams (incl. phone-as-webcam) fast
                self.cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(args.video)
            if not self.cap.isOpened():
                raise RuntimeError(
                    "could not open {} {!r} -- if a phone webcam, start the "
                    "DroidCam/Iriun/Camo app and try --camera 0/1/2".format(
                        self.kind, args.camera if self.kind == "camera" else args.video))
            w = int(self.cap.get(3)); h = int(self.cap.get(4))
            self.label = "{} {}x{}".format(self.kind, w, h)

    def grab(self):
        if self.kind == "screen":
            return np.ascontiguousarray(np.asarray(self.sct.grab(self.box))[:, :, :3])
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.sct is not None:
            self.sct.close()


def detections(res):
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls in (PERSON, SPORTS_BALL):
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            yield cls, x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="capture the screen (default)")
    ap.add_argument("--camera", type=int, help="webcam index (phone-as-webcam at the TV)")
    ap.add_argument("--video", help="path to a video file")
    ap.add_argument("--box", help="screen region left,top,width,height (with --screen)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true", help="live annotated window (q to quit)")
    ap.add_argument("--selftest", action="store_true", help="time the loop, no save/display")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights)
    dev = 0 if torch.cuda.is_available() else "cpu"
    src = Source(args)

    try:
        if args.selftest:
            for _ in range(3):                      # warm up
                f = src.grab()
                if f is not None:
                    model(f, verbose=False, device=dev)
            t0 = time.perf_counter()
            tp = tb = got = 0
            for _ in range(args.frames):
                f = src.grab()
                if f is None:
                    break
                got += 1
                for cls, *_ in detections(model(f, conf=args.conf, verbose=False, device=dev)[0]):
                    tp += cls == PERSON
                    tb += cls == SPORTS_BALL
            dt = time.perf_counter() - t0
            print("source: {}".format(src.label))
            print("{} frames in {:.2f}s = {:.1f} FPS".format(got, dt, got / max(dt, 1e-6)))
            print("avg per frame: {:.1f} people, {:.2f} ball  (0 is fine if no game in view)".format(
                tp / max(got, 1), tb / max(got, 1)))
            print("real-time? {} (broadcast is ~25-30 FPS)".format(
                "yes, keeps up" if got / max(dt, 1e-6) >= 25 else "borderline -- smaller region or lighter model"))
            return

        import cv2
        print("live on {} -- press q to quit".format(src.label))
        last = time.perf_counter(); fps = 0.0
        while True:
            img = src.grab()
            if img is None:
                break
            npl = nbl = 0
            for cls, x1, y1, x2, y2 in detections(model(img, conf=args.conf, verbose=False, device=dev)[0]):
                if cls == PERSON:
                    cv2.rectangle(img, (x1, y1), (x2, y2), PLAYER_BGR, 2); npl += 1
                else:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 7, BALL_BGR, -1); nbl += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)); last = now
            cv2.putText(img, "players {}  ball {}  {:.0f} FPS  [{}]".format(
                npl, nbl, fps, src.kind), (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (230, 230, 230), 2)
            cv2.imshow("hidden-timeout live", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    finally:
        src.close()


if __name__ == "__main__":
    main()
