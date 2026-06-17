"""Visual-AI tracking over a WHOLE broadcast/highlights video.

Three synced panels, the "Second-Spectrum" view, driven entirely by the real CV
pipeline:
  - left   : the broadcast frame with each detected player boxed in team colour
  - top-r  : top-down pitch with the VISIBLE team SHAPES (convex hulls) + ball
  - bottom : a live PITCH-CONTROL probability map (which team owns each patch of
             grass), recomputed every frame from the tracked positions

It is built for a highlights MONTAGE: scene cuts are detected (frame-difference)
and the tracker + homography are reset at each one; frames with no usable pitch
view (graphics, tight replays, crowd) are honestly blanked "NO PITCH VIEW".

Anti-flicker (the thing that makes it watchable):
  - the homography is EMA-smoothed WITHIN a shot (reset at cuts) so the top-down
    does not jitter frame to frame (raw per-frame H is the main flicker source)
  - only CONFIRMED tracks are drawn (enough real detections + a few frames old),
    so jitter-spawned blips never appear
  - a missed detection is Kalman-PREDICTED and fades, instead of blinking out
  - tracks fade IN over their first frames; the control surface is EMA-smoothed
  - team colours are locked once from a sample of real pitch frames

Two stages, with a cache, because the render is the slow part and gets iterated:
  stage A (CV + tracking)  -> data/clips/<name>/_states.pkl   (cheap to redo)
  stage B (render)         -> figures/<name>.mp4

Honest scope: WC2026 has no public tracking data. These positions come ONLY
from broadcast CV -- visible players only, heatmap-grade, off-screen players
unrecoverable. The pitch-control map is therefore computed from whoever the
camera shows, not a full 22, and is masked to where players are actually seen.

    python src/visual_ai.py --video data/clips/_arg_full.mp4 --name argentina_full --fps 6
    python src/visual_ai.py --name argentina_full --render-only   # reuse the cache
"""
import argparse
import glob
import os
import pickle
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

import homography as hg
from broadcast_track import detect
from tactical import hull, jersey_color
from track_fuse import Kalman, appearance, appdist, ema_app

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLIPS = os.path.join(ROOT, "data", "clips")
FIG = os.path.join(ROOT, "figures")
PL, PW = 120.0, 80.0
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; BALLC = "#ffd23f"

# tracking / smoothing knobs
GATE = 4.0          # m, max association distance per frame
APP_W = 3.0         # weight of kit-colour distance in association
MAX_MISS = 6        # frames a track is Kalman-ghosted before it dies
A_HOMO = 0.35       # homography EMA factor within a shot
CUT = 28.0          # mean abs gray-diff (64x36) above which it is a scene cut
N_CONF = 4          # real detections needed before a track is drawn
AGE_CONF = 4        # frames alive before a track is drawn (kills blips)
TAU = 6.0           # pitch-control falloff (m); larger = softer ownership
SEE = 26.0          # only colour control within this many m of a tracked player


def extract_frames(video, frame_dir, fps):
    os.makedirs(frame_dir, exist_ok=True)
    have = glob.glob(os.path.join(frame_dir, "f_*.jpg"))
    if have:
        return sorted(have)
    subprocess.run(["ffmpeg", "-y", "-i", video, "-vf", "fps={}".format(fps),
                    "-q:v", "3", os.path.join(frame_dir, "f_%05d.jpg")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(glob.glob(os.path.join(frame_dir, "f_*.jpg")))


def fit_teams(frames):
    """lock the two kit colours from a sample of REAL pitch frames (homography ok);
    fall back to ungated sampling, then to defaults, so it never crashes"""
    import cv2
    from sklearn.cluster import KMeans
    sample = frames[::max(len(frames) // 80, 1)]

    def collect(gate):
        cols = []
        for fp in sample:
            if gate and hg.keypoint_homography(fp)[0] is None:
                continue
            img = cv2.imread(fp)
            for _, _, b, _ in detect(fp)[0]:
                c = jersey_color(img, b)
                if c is not None:
                    cols.append(c)
            if len(cols) > 400:
                break
        return cols

    cols = collect(gate=True)
    if len(cols) < 8:
        cols = collect(gate=False)
    if len(cols) < 2:
        # default: Argentina light-blue vs Algeria green (BGR->RGB), LAB centroids unused
        return np.array([[200, 128, 110], [120, 110, 150]], float), [(0.4, 0.6, 0.95), (0.2, 0.6, 0.3)]
    cen = KMeans(n_clusters=2, n_init=5, random_state=0).fit(np.array(cols)).cluster_centers_
    rgb = []
    for c in range(2):
        lab = cen[c].astype(np.uint8).reshape(1, 1, 3)
        b = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
        rgb.append((float(b[2]) / 255, float(b[1]) / 255, float(b[0]) / 255))
    return cen, rgb


def cv_pass(frames, cen):
    """stage A: detect + EMA-homography + Kalman track; return per-frame states"""
    import cv2

    def team_of(img, box):
        c = jersey_color(img, box)
        return int(np.argmin([np.linalg.norm(c - cen[k]) for k in range(2)])) if c is not None else -1

    tracks, states, prev_small, Hs = [], [], None, None
    for n, fp in enumerate(frames):
        img = cv2.imread(fp)
        small = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY).astype(float)
        cut = prev_small is not None and float(np.abs(small - prev_small).mean()) > CUT
        prev_small = small
        if cut:
            tracks, Hs = [], None                       # scene change: hard reset
        H, _, _ = hg.keypoint_homography(fp)
        players, ball, (h, w) = detect(fp)
        boxes = [[float(v) for v in b] + [team_of(img, b)] for _, _, b, _ in players]
        note = "scene change" if cut else ("" if H is not None else "no pitch view")
        if H is None:
            tracks, Hs = [], None
            states.append({"fp": fp, "boxes": boxes, "tracks": [], "ball": None,
                           "wh": (w, h), "note": note or "no pitch view"})
            continue
        Hs = H if Hs is None else A_HOMO * H + (1 - A_HOMO) * Hs
        H = Hs / Hs[2, 2]                               # smoothed homography
        dets, dapp, dteam = [], [], []
        for fx, fy, b, _ in players:
            p = hg.warp(H, [[fx, fy]])[0]
            if 0 <= p[0] <= PL and 0 <= p[1] <= PW:
                dets.append(p); dapp.append(appearance(img, b)); dteam.append(team_of(img, b))
        alive = [t for t in tracks if t["alive"]]
        for t in alive:
            t["kf"].predict(); t["age"] += 1
        md, mt = set(), set()
        if alive and dets:
            pred = np.array([[t["kf"].x[0], t["kf"].x[1]] for t in alive]); D = np.array(dets)
            wd = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            app = np.array([[appdist(t["app"], da) for da in dapp] for t in alive])
            cost = wd + APP_W * app; cost[wd > GATE] = 1e6
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    alive[a]["kf"].update(D[b]); alive[a]["miss"] = 0; alive[a]["n"] += 1
                    alive[a]["app"] = ema_app(alive[a]["app"], dapp[b])
                    alive[a]["votes"][dteam[b]] = alive[a]["votes"].get(dteam[b], 0) + 1
                    md.add(b); mt.add(a)
        for a, t in enumerate(alive):
            if a not in mt:
                t["miss"] += 1
                if t["miss"] > MAX_MISS:
                    t["alive"] = False
        for j in range(len(dets)):
            if j not in md:
                tracks.append({"kf": Kalman(dets[j], 1.0), "app": dapp[j], "miss": 0,
                               "alive": True, "votes": {dteam[j]: 1}, "n": 1, "age": 0})
        cur = []
        for t in tracks:
            if not t["alive"] or t["n"] < N_CONF or t["age"] < AGE_CONF:
                continue
            measured = t["miss"] == 0
            fade_in = min(t["age"], 6) / 6.0                 # ramp in over 6 frames
            fade_out = 1.0 if measured else max(0.28, 1.0 - t["miss"] / MAX_MISS)
            alpha = float(np.clip(fade_in * fade_out, 0.0, 1.0))
            tm = max(t["votes"], key=t["votes"].get)
            cur.append([float(t["kf"].x[0]), float(t["kf"].x[1]), int(tm), alpha, bool(measured)])
        bw = hg.warp(H, [[ball[0], ball[1]]])[0] if ball is not None else None
        ball_xy = [float(bw[0]), float(bw[1])] if bw is not None and 0 <= bw[0] <= PL and 0 <= bw[1] <= PW else None
        states.append({"fp": fp, "boxes": boxes, "tracks": cur, "ball": ball_xy,
                       "wh": (w, h), "note": note})
    return states


# ---- pitch-control surface (proximity softmax per team, masked to visibility) ----
_GX, _GY = np.meshgrid(np.linspace(0, PL, 92), np.linspace(0, PW, 62))
_CELLS = np.stack([_GX.ravel(), _GY.ravel()], 1)


def control_surface(tracks):
    """returns (control[0..1, team0 owns=1], visible_mask) over the grid, or None"""
    if len(tracks) < 2:
        return None
    P = np.array([[t[0], t[1]] for t in tracks])
    tm = np.array([t[2] for t in tracks])
    d = np.linalg.norm(_CELLS[:, None, :] - P[None, :, :], axis=2)

    def infl(sel):
        return np.exp(-d[:, sel].min(1) / TAU) if sel.any() else np.zeros(len(_CELLS))

    ia, idf = infl(tm == 0), infl(tm == 1)
    ctrl = np.where(ia + idf > 0, ia / (ia + idf + 1e-9), 0.5)
    vis = d.min(1) < SEE
    return ctrl.reshape(_GX.shape), vis.reshape(_GX.shape)


def display_palette(team_rgb):
    """kit-colour clustering separates the teams, but the raw centroids can be
    muddy/similar on broadcast; remap to a fixed high-contrast pair (lighter kit
    -> cyan, darker -> orange) so the two teams are unmistakable on the pitch"""
    lum = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in team_rgb]
    light = int(np.argmax(lum))
    disp = [None, None]
    disp[light] = (0.30, 0.78, 1.0)         # lighter kit (Argentina) -> cyan
    disp[1 - light] = (1.0, 0.45, 0.20)     # darker kit (Algeria)   -> orange
    return disp


def render(states, team_rgb, out, fps, title):
    import cv2
    cmap = LinearSegmentedColormap.from_list(
        "ctrl", [team_rgb[1], (0.93, 0.93, 0.93), team_rgb[0]])
    fig = plt.figure(figsize=(12.8, 8.4), dpi=86)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], width_ratios=[1.05, 1.0],
                          left=0.015, right=0.985, top=0.9, bottom=0.055, hspace=0.16, wspace=0.06)
    axb = fig.add_subplot(gs[0, 0])      # broadcast
    axp = fig.add_subplot(gs[0, 1])      # top-down shapes
    axc = fig.add_subplot(gs[1, :])      # pitch-control map
    F = {"fontfamily": "Bahnschrift"}
    ema = {"c": None}

    def draw(i):
        d = states[i]
        for ax in (axb, axp, axc):
            ax.clear()
        # ---- broadcast + team-coloured boxes ----
        img = cv2.cvtColor(cv2.imread(d["fp"]), cv2.COLOR_BGR2RGB)
        axb.imshow(img)
        for box in d["boxes"]:
            x1, y1, x2, y2, tm = box
            col = team_rgb[int(tm)] if tm >= 0 else (0.6, 0.6, 0.6)
            axb.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.4))
        w, h = d["wh"]; axb.set_xlim(0, w); axb.set_ylim(h, 0); axb.axis("off")
        axb.set_title("broadcast — players detected & teamed", color=INK, loc="left",
                      fontsize=10.5, fontweight="bold", **F)
        # ---- top-down team shapes ----
        hg.draw_pitch(axp)
        tr = d["tracks"]
        for c in range(2):
            pts = np.array([[t[0], t[1]] for t in tr if t[2] == c])
            if len(pts):
                for t in [t for t in tr if t[2] == c]:
                    axp.scatter([t[0]], [t[1]], s=120, facecolor=team_rgb[c], edgecolors=BG,
                                lw=1.1, alpha=t[3], zorder=5)
                hull(axp, pts, team_rgb[c])
        if d["ball"] is not None:
            axp.scatter([d["ball"][0]], [d["ball"][1]], s=66, c=BALLC, edgecolors=BG, lw=1, zorder=7)
        if d["note"]:
            axp.text(60, 40, d["note"].upper(), ha="center", va="center",
                     color="#ffb347", fontsize=15, fontweight="bold", **F)
            ema["c"] = None
        axp.set_title("top-down — visible team shapes", color=INK, loc="left",
                      fontsize=10.5, fontweight="bold", **F)
        # ---- live pitch-control probability map ----
        hg.draw_pitch(axc); axc.set_title(
            "live pitch control — probability each team owns the space",
            color=INK, loc="left", fontsize=10.5, fontweight="bold", **F)
        surf = None if d["note"] else control_surface(tr)
        if surf is None:
            ema["c"] = None
            if d["note"]:
                axc.text(60, 40, d["note"].upper(), ha="center", va="center",
                         color="#ffb347", fontsize=15, fontweight="bold", **F)
        else:
            ctrl, vis = surf
            ema["c"] = ctrl if ema["c"] is None else 0.5 * ctrl + 0.5 * ema["c"]
            shown = np.ma.masked_where(~vis, ema["c"])
            axc.imshow(shown, origin="lower", extent=[0, PL, 0, PW], cmap=cmap,
                       vmin=0, vmax=1, alpha=0.62, aspect="equal", zorder=1.5)
            for t in tr:
                tc = team_rgb[int(t[2])] if t[2] >= 0 else (0.6, 0.6, 0.6)
                axc.scatter([t[0]], [t[1]], s=34, facecolor=tc,
                            edgecolors=BG, lw=0.6, alpha=0.9 * t[3], zorder=4)
            if d["ball"] is not None:
                axc.scatter([d["ball"][0]], [d["ball"][1]], s=46, c=BALLC, edgecolors=BG, lw=0.8, zorder=6)
        mm, ss = divmod(int(i / fps), 60)
        fig.suptitle("{}   ·   {:d}:{:02d}".format(title, mm, ss), color=INK, x=0.5,
                     fontsize=13, fontweight="bold", **F)

    a = manim.FuncAnimation(fig, draw, frames=len(states), interval=1000 / fps)
    a.save(out, writer=manim.FFMpegWriter(fps=fps, bitrate=2600,
           extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--name", default="argentina_full")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--cv-only", action="store_true", help="build the state cache, skip render")
    ap.add_argument("--limit", type=int, default=0, help="cap frames (for a quick test)")
    ap.add_argument("--range", default="", help="A,B render only states[A:B] to a _testAB file")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    frame_dir = os.path.join(CLIPS, args.name)
    cache = os.path.join(frame_dir, "_states.pkl")
    title = "WC2026 Argentina v Algeria — visual-AI tracking (extended highlights)"

    if args.render_only and os.path.exists(cache):
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        states, team_rgb = blob["states"], blob["team_rgb"]
        print("loaded cache:", len(states), "frames")
    else:
        if not args.video:
            raise SystemExit("need --video the first time")
        frames = extract_frames(args.video, frame_dir, args.fps)
        if args.limit:
            frames = frames[:args.limit]
        print("frames:", len(frames))
        cen, team_rgb = fit_teams(frames)
        print("teams locked. running CV pass ...")
        states = cv_pass(frames, cen)
        with open(cache, "wb") as f:
            pickle.dump({"states": states, "team_rgb": team_rgb}, f)
        tracked = sum(1 for s in states if s["tracks"])
        blank = sum(1 for s in states if s["note"])
        print("CV done. frames {} | with tracking {} ({:.0%}) | blanked {} ({:.0%})".format(
            len(states), tracked, tracked / len(states), blank, blank / len(states)))
        if args.cv_only:
            return

    team_rgb = display_palette(team_rgb)
    if args.range:
        a, b = (int(v) for v in args.range.split(","))
        out = os.path.join(FIG, "wc2026_{}_test{}_{}.mp4".format(args.name, a, b))
        print("rendering test range [{}:{}] -> {}".format(a, b, out))
        render(states[a:b], team_rgb, out, args.fps, title)
    else:
        out = os.path.join(FIG, "wc2026_{}.mp4".format(args.name))
        print("rendering", len(states), "frames ->", out)
        render(states, team_rgb, out, args.fps, title)
    print("wrote", out)


if __name__ == "__main__":
    main()
