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
from matplotlib.patches import Ellipse
from scipy.optimize import linear_sum_assignment

import homography as hg
import match_data as MD
import camera_state as cstate
import uncertainty as uq
from broadcast_track import detect
from tactical import hull, jersey_color
from track_fuse import Kalman, appearance, appdist, ema_app
from tactical_metrics import team_shape

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLIPS = os.path.join(ROOT, "data", "clips")
FIG = os.path.join(ROOT, "figures")
PL, PW = 105.0, 68.0          # real FIFA metres (matches homography canonical)
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; BALLC = "#ffd23f"; GREEN = "#3fb950"
ARG_C, ALG_C = (0.30, 0.78, 1.0), (1.0, 0.45, 0.20)   # cyan / orange, matches palette
MATCH_ID = "4667812"                                  # Argentina 3-0 Algeria

# tracking / smoothing knobs
GATE = 4.0          # m, max association distance per frame
APP_W = 3.0         # weight of kit-colour distance in association
MAX_MISS = 6        # frames a track is Kalman-ghosted before it dies
A_HOMO = 0.35       # homography EMA factor within a shot
CUT = 55.0          # mean abs gray-diff (64x36) above which it is a HARD cut
                    # (measured: median diff ~16, pans ~40, real cuts >55)
HOLD = 10           # frames to keep ESTIMATING last positions through a no-pitch
                    # gap before the overlay finally goes to "no pitch view"
N_CONF = 3          # real detections needed before a track is drawn
AGE_CONF = 3        # frames alive before a track is drawn (kills blips)
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
    last_cur, last_ball, gap = [], None, 99             # for estimation through gaps
    for n, fp in enumerate(frames):
        img = cv2.imread(fp)
        small = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY).astype(float)
        cut = prev_small is not None and float(np.abs(small - prev_small).mean()) > CUT
        prev_small = small
        if cut:
            tracks, Hs, last_cur, gap = [], None, [], 99  # hard cut: reset identity,
            # do NOT hold positions across a real cut and do NOT blank for it
        H, ip, pp = hg.keypoint_homography(fp)
        players, ball, (h, w) = detect(fp)
        boxes = [[float(v) for v in b] + [team_of(img, b)] for _, _, b, _ in players]
        cam, _ = cstate.classify(img, H is not None, len(players))   # gate (camera_state)
        fc = uq.frame_confidence(H, ip, pp)                          # calibration confidence
        if H is None:
            gap += 1
            if gap <= HOLD and last_cur:                # ESTIMATE: hold last shape,
                dim = max(0.25, 1.0 - gap / (HOLD + 1.0))   # fading as the gap grows
                est = [[x, y, tm, a * dim, False, sx, sy, an] for x, y, tm, a, _, sx, sy, an in last_cur]
                states.append({"fp": fp, "boxes": boxes, "tracks": est, "ball": last_ball,
                               "wh": (w, h), "note": "estimated", "est": True, "cam": cam, "conf": 0.0})
            else:                                       # genuinely no pitch in view
                tracks, Hs = [], None
                states.append({"fp": fp, "boxes": boxes, "tracks": [], "ball": None,
                               "wh": (w, h), "note": "no pitch view", "cam": cam, "conf": 0.0})
            continue
        gap = 0
        Hs = H if Hs is None else A_HOMO * H + (1 - A_HOMO) * Hs
        H = Hs / Hs[2, 2]                               # smoothed homography
        Hinv = np.linalg.inv(H)
        sig = max(5.0, fc["mre_px"])                    # calibration uncertainty (px)
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
            px, py = float(t["kf"].x[0]), float(t["kf"].x[1])
            imgpt = hg.warp(Hinv, [[px, py]])[0]            # back to image for the Jacobian
            _, (sx, sy, ang) = uq.player_cov(H, imgpt, sigma_px=sig)
            cur.append([px, py, int(tm), alpha, bool(measured), float(sx), float(sy), float(ang)])
        bw = hg.warp(H, [[ball[0], ball[1]]])[0] if ball is not None else None
        ball_xy = [float(bw[0]), float(bw[1])] if bw is not None and 0 <= bw[0] <= PL and 0 <= bw[1] <= PW else None
        if cur:                                         # remember for gap estimation
            last_cur, last_ball = cur, ball_xy
        states.append({"fp": fp, "boxes": boxes, "tracks": cur, "ball": ball_xy,
                       "wh": (w, h), "note": "", "cam": cam, "conf": fc["conf"]})
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


def render(states, team_rgb, m, out, fps, title, label="", frame_min_override=None):
    import cv2
    cmap = LinearSegmentedColormap.from_list("ctrl", [team_rgb[1], (0.93, 0.93, 0.93), team_rgb[0]])
    F = {"fontfamily": "Bahnschrift"}
    mins = m["wp_mins"]
    if frame_min_override is not None:
        frame_min = np.clip(np.asarray(frame_min_override, float), 0, 98)
    else:
        # playhead: advance match-minute only on frames that SHOW the pitch, so the
        # dashboard reveals the match story progressively (approx; no clock OCR here)
        pitch = np.array([s["note"] != "no pitch view" for s in states], float)
        cum = np.cumsum(pitch)
        frame_min = np.clip(90.0 * cum / max(cum[-1], 1.0), 0, 98)

    fig = plt.figure(figsize=(16, 9), dpi=84); fig.patch.set_facecolor(BG)
    axw = fig.add_axes([0.02, 0.875, 0.96, 0.052])    # win-prob bar
    axb = fig.add_axes([0.02, 0.40, 0.455, 0.45])     # broadcast
    axp = fig.add_axes([0.49, 0.40, 0.265, 0.45])     # top-down shapes
    axc = fig.add_axes([0.775, 0.40, 0.205, 0.45])    # pitch control
    axg = fig.add_axes([0.05, 0.055, 0.32, 0.27])     # xG race
    axe = fig.add_axes([0.41, 0.055, 0.23, 0.295])    # event ticker
    axr = fig.add_axes([0.665, 0.055, 0.315, 0.295])  # ratings + prediction
    allax = [axw, axb, axp, axc, axg, axe, axr]
    ema = {"c": None}

    def draw(i):
        d = states[i]; blank = d["note"] == "no pitch view"; est = d.get("est", False)
        ti = int(round(float(frame_min[i])))
        for ax in allax:
            ax.clear()

        # ===== WIN-PROBABILITY BAR =====
        axw.set_xlim(0, 1); axw.set_ylim(0, 1); axw.axis("off")
        ph, pdr, pa = float(m["wp_home"][ti]), float(m["wp_draw"][ti]), float(m["wp_away"][ti])
        axw.add_patch(plt.Rectangle((0, 0), ph, 1, color=ARG_C))
        axw.add_patch(plt.Rectangle((ph, 0), pdr, 1, color=(0.42, 0.42, 0.46)))
        axw.add_patch(plt.Rectangle((ph + pdr, 0), pa, 1, color=ALG_C))
        axw.text(0.008, 0.5, "ARGENTINA  {:.0%}".format(ph), va="center", ha="left",
                 color=BG, fontsize=12, fontweight="bold", **F)
        axw.text(0.992, 0.5, "{:.0%}  ALGERIA".format(pa), va="center", ha="right",
                 color="white", fontsize=12, fontweight="bold", **F)
        if pdr > 0.05:
            axw.text(ph + pdr / 2, 0.5, "draw {:.0%}".format(pdr), va="center", ha="center",
                     color="white", fontsize=8, **F)
        xph = float(m["wp_xg"][ti])                       # xG-DESERVED ARG win% (white tick)
        axw.add_patch(plt.Rectangle((xph - 0.0015, -0.05), 0.003, 1.1, color="white", zorder=6, clip_on=False))
        axw.set_title("LIVE WIN PROBABILITY  ·  bar = score-based (OOS 0.82);  white tick = xG-deserved (ARG {:.0%})".format(xph),
                      color=MUT, loc="center", fontsize=9, **F)

        # ===== BROADCAST =====
        axb.imshow(cv2.cvtColor(cv2.imread(d["fp"]), cv2.COLOR_BGR2RGB))
        for x1, y1, x2, y2, tm in d["boxes"]:
            col = team_rgb[int(tm)] if tm >= 0 else (0.6, 0.6, 0.6)
            axb.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.4))
        w, h = d["wh"]; axb.set_xlim(0, w); axb.set_ylim(h, 0); axb.axis("off")
        axb.set_title("broadcast — players detected & teamed", color=INK, loc="left",
                      fontsize=10.5, fontweight="bold", **F)

        # ===== TOP-DOWN TEAM SHAPES =====
        # flip about the x-axis (y -> PW-y): the homography puts the camera-near
        # touchline at y=PW (top); flipping it puts near at the bottom so the
        # top-down matches what you see in the broadcast (also un-mirrors the map)
        hg.draw_pitch(axp); tr = d["tracks"]
        # flip about x-axis (y->PW-y, angle negated) + carry the covariance ellipse
        trd = [[t[0], PW - t[1], t[2], t[3], t[5], t[6], -t[7]] for t in tr]
        bd = [d["ball"][0], PW - d["ball"][1]] if d["ball"] is not None else None
        for c in range(2):
            grp = [t for t in trd if t[2] == c]
            for t in grp:                                # 1-sigma uncertainty ellipse
                axp.add_patch(Ellipse((t[0], t[1]), 2 * t[4], 2 * t[5], angle=t[6],
                              facecolor=team_rgb[c], edgecolor="none", alpha=0.16 * t[3], zorder=4))
                axp.scatter([t[0]], [t[1]], s=110, facecolor=team_rgb[c], edgecolors=BG,
                            lw=1.1, alpha=t[3], zorder=5)
            if len(grp) >= 3:
                hull(axp, np.array([[t[0], t[1]] for t in grp]), team_rgb[c])
        if bd is not None:
            axp.scatter([bd[0]], [bd[1]], s=60, c=BALLC, edgecolors=BG, lw=1, zorder=7)
        # live tactical readout (visible-block compactness)
        sa = team_shape([[t[0], t[1]] for t in trd if t[2] == 0])
        sb = team_shape([[t[0], t[1]] for t in trd if t[2] == 1])
        if sa and sb and not blank:
            axp.text(PL / 2, PW - 1.5, "compactness  ARG {:.0f}  ·  ALG {:.0f} m2".format(sa["area"], sb["area"]),
                     ha="center", va="top", color=MUT, fontsize=7.5, **F)
        if blank:
            lbl = "NO PITCH VIEW · graphic/replay" if d.get("cam") == "other" else "NO PITCH VIEW"
            axp.text(PL / 2, PW / 2, lbl, ha="center", va="center", color="#ffb347",
                     fontsize=13, fontweight="bold", **F)
        elif est:
            axp.text(PL / 2, 2, "CLOSE-UP · holding last shape (estimated)", ha="center", va="bottom",
                     color="#ffb347", fontsize=8, **F)
        axp.set_title("top-down — shapes + 1-sigma ellipses  ·  conf {:.2f}".format(d.get("conf", 0.0)),
                      color=INK, loc="left", fontsize=10.5, fontweight="bold", **F)

        # ===== PITCH CONTROL ===== (uses the same x-axis-flipped positions)
        hg.draw_pitch(axc)
        surf = None if blank else control_surface(trd)
        if surf is None:
            ema["c"] = None
            if blank:
                axc.text(PL / 2, PW / 2, "NO PITCH VIEW", ha="center", va="center", color="#ffb347",
                         fontsize=12, fontweight="bold", **F)
        else:
            ctrl, vis = surf
            ema["c"] = ctrl if ema["c"] is None else 0.5 * ctrl + 0.5 * ema["c"]
            axc.imshow(np.ma.masked_where(~vis, ema["c"]), origin="lower", extent=[0, PL, 0, PW],
                       cmap=cmap, vmin=0, vmax=1, alpha=0.62, aspect="equal", zorder=1.5)
            for t in trd:
                tc = team_rgb[int(t[2])] if t[2] >= 0 else (0.6, 0.6, 0.6)
                axc.scatter([t[0]], [t[1]], s=28, facecolor=tc, edgecolors=BG, lw=0.5,
                            alpha=0.9 * t[3], zorder=4)
        axc.set_title("live pitch control", color=INK, loc="left", fontsize=10.5, fontweight="bold", **F)

        # ===== xG RACE =====
        axg.set_facecolor("#0f1620")
        axg.plot(mins[:ti + 1], m["xg_h"][:ti + 1], color=ARG_C, lw=2.0)
        axg.plot(mins[:ti + 1], m["xg_a"][:ti + 1], color=ALG_C, lw=2.0)
        for s in m["shots"]:
            if s["min"] <= ti:
                cy = (m["xg_h"] if s["is_home"] else m["xg_a"])[min(s["min"], 98)]
                axg.scatter([s["min"]], [cy], s=80 if s["goal"] else 20,
                            marker="*" if s["goal"] else "o",
                            color=ARG_C if s["is_home"] else ALG_C, edgecolors=BG, lw=0.5, zorder=5)
        axg.axvline(ti, color=MUT, lw=0.8, ls=(0, (3, 3)))
        axg.set_xlim(0, 95); axg.set_ylim(0, max(1.5, float(m["xg_h"][-1]) + 0.25))
        axg.set_title("xG race  ·  ARG {:.2f}   ALG {:.2f}".format(float(m["xg_h"][ti]), float(m["xg_a"][ti])),
                      color=INK, loc="left", fontsize=10, fontweight="bold", **F)
        axg.tick_params(colors=MUT, labelsize=7)
        for sp in axg.spines.values():
            sp.set_color("#30363d")

        # ===== EVENT TICKER =====
        axe.set_xlim(0, 1); axe.set_ylim(0, 1); axe.axis("off")
        axe.text(0, 0.98, "MATCH EVENTS", color=INK, fontsize=10, fontweight="bold", va="top", **F)
        past = [e for e in m["events"] if e["min"] <= ti
                and not (e["type"] == "Substitution" and not e["player"])][-7:]
        for k, e in enumerate(reversed(past)):
            y = 0.85 - k * 0.118
            tag = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}.get(e["type"], e["type"])
            latest = e is past[-1]
            if e["type"] == "Goal" and e.get("score"):
                extra = "  {}-{}".format(*e["score"])
            elif e["type"] == "VAR" and e.get("note"):
                extra = " ({})".format(e["note"])
            else:
                extra = ""
            col = "#ffb347" if e["type"] == "VAR" else (ARG_C if e["is_home"] else ALG_C)
            axe.text(0.0, y, "{}'".format(e["min"]), color=MUT, fontsize=9, va="center", **F)
            axe.text(0.13, y, "{} {}{}".format(tag, e["player"], extra),
                     color=col if (latest or e["type"] == "VAR") else INK,
                     fontsize=9.5 if latest else 8.5,
                     fontweight="bold" if latest else "normal", va="center", **F)

        # ===== RATINGS + PREDICTION =====
        axr.set_xlim(0, 1); axr.set_ylim(0, 1); axr.axis("off")
        pm = m["pre_match"]
        axr.text(0, 0.98, "PRE-MATCH CALL  ·  our Elo model", color=INK, fontsize=10, fontweight="bold", va="top", **F)
        axr.text(0, 0.89, "ARG {:.0%}    draw {:.0%}    ALG {:.0%}".format(pm["p_h"], pm["p_d"], pm["p_a"]),
                 color=MUT, fontsize=9, va="top", **F)
        axr.text(0, 0.80, "our call: ARGENTINA  ·  final {}-{}  (correct)".format(m["final_h"], m["final_a"]),
                 color=GREEN, fontsize=9.5, fontweight="bold", va="top", **F)
        axr.text(0, 0.66, "TOP PLAYER RATINGS  ·  FotMob", color=INK, fontsize=10, fontweight="bold", va="top", **F)
        for k, r in enumerate(m["ratings"][:5]):
            y = 0.555 - k * 0.107; col = ARG_C if r["is_home"] else ALG_C
            axr.add_patch(plt.Rectangle((0.46, y - 0.03), 0.52 * r["rating"] / 10.0, 0.052,
                          color=col, alpha=0.45))
            axr.text(0.0, y, (r["name"][:18] + ("  POTM" if r["potm"] else "")),
                     color=(1.0, 0.84, 0.2) if r["potm"] else col, fontsize=9, va="center", **F)
            axr.text(0.985, y, "{:.2f}".format(r["rating"]), color=INK, fontsize=9.5,
                     ha="right", va="center", fontweight="bold", **F)

        # ===== HEADER =====
        sch, sca = int(m["sc_h"][ti]), int(m["sc_a"][ti])
        head = label if label else "visual-AI dashboard"
        fig.suptitle("{}    ·    {} {}-{} {}    ·    ~{}'".format(
            head, m["home"], sch, sca, m["away"], ti), color=INK, x=0.5, y=0.978,
            fontsize=15, fontweight="bold", **F)

    a = manim.FuncAnimation(fig, draw, frames=len(states), interval=1000 / fps)
    a.save(out, writer=manim.FFMpegWriter(fps=fps, codec="libx264",
           extra_args=["-crf", "24", "-pix_fmt", "yuv420p", "-preset", "veryfast"]))
    plt.close(fig)


def _render(theme, states, team_rgb, m, out, fps, title, label="", frame_min_override=None):
    """dispatch to a theme module (broadcast|editorial|telemetry) or the legacy look."""
    if theme == "legacy":
        return render(states, team_rgb, m, out, fps, title, label, frame_min_override)
    import dashboard_themes as T
    return T.render(T.get(theme), states, team_rgb, m, out, fps,
                    label=label, frame_min_override=frame_min_override)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--name", default="argentina_full")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--cv-only", action="store_true", help="build the state cache, skip render")
    ap.add_argument("--limit", type=int, default=0, help="cap frames (for a quick test)")
    ap.add_argument("--range", default="", help="A,B render only states[A:B] to a _testAB file")
    ap.add_argument("--theme", default="broadcast", help="broadcast | editorial | telemetry | legacy")
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
    m = MD.load(MATCH_ID)
    print("match data: {} {}-{} {} | pre-match P(H) {:.2f} | POTM {}".format(
        m["home"], m["final_h"], m["final_a"], m["away"], m["pre_match"]["p_h"],
        m["ratings"][0]["name"] if m["ratings"] else "?"))
    tag = "" if args.theme == "legacy" else "_" + args.theme
    if args.range:
        a, b = (int(v) for v in args.range.split(","))
        out = os.path.join(FIG, "wc2026_{}{}_test{}_{}.mp4".format(args.name, tag, a, b))
        print("rendering [{}] test range [{}:{}] -> {}".format(args.theme, a, b, out))
        _render(args.theme, states[a:b], team_rgb, m, out, args.fps, title)
    else:
        out = os.path.join(FIG, "wc2026_{}{}.mp4".format(args.name, tag))
        print("rendering [{}] {} frames -> {}".format(args.theme, len(states), out))
        _render(args.theme, states, team_rgb, m, out, args.fps, title)
    print("wrote", out)


if __name__ == "__main__":
    main()
