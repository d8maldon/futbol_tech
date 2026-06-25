"""Preview harness for iterating on the visual-AI dashboard LOOK without needing
ffmpeg, the GPU detector, the FotMob cache, or a video file.

It synthesises ONE representative dashboard frame's worth of inputs that match the
real contracts:
  - build_m()      -> the match-data dict exactly as match_data.load() returns
                      (Argentina 3-0 Algeria, Messi hat-trick, realistic curves)
  - build_state()  -> one per-frame state dict exactly as visual_ai.cv_pass emits
                      (a ~64' mid-block, cyan ARG vs orange ALG, ball, ellipses)
  - TEAM_RGB       -> the display palette [ARG cyan, ALG orange]
  - BROADCAST      -> a real broadcast still (cropped from the rendered video)
  - draw_pitch / hull / team_shape -> lightweight helpers (no torch / no repo deps)
  - PREVIEW_MIN    -> the match minute to freeze the dashboard at

Anyone (me or a redesign agent) writes a standalone script that imports this and
renders a single PNG, so the new styling is visible before re-encoding the real
MP4 on the GPU box.
"""
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
BROADCAST = os.path.join(ROOT, "_frames_review", "bcast_wide.png")
PREVIEW_MIN = 64

PL, PW = 105.0, 68.0
CYAN = (0.30, 0.78, 1.0)        # ARG (lighter kit) -> cyan, matches display_palette
ORANGE = (1.0, 0.45, 0.20)      # ALG (darker kit)  -> orange
TEAM_RGB = [CYAN, ORANGE]       # tm 0 = home/ARG, tm 1 = away/ALG


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_m():
    """match-data dict identical in shape to src/match_data.load()."""
    mins = np.arange(0, 99)
    goals = [(23, True), (47, True), (64, True)]            # Messi hat-trick, all home
    sc_h = np.array([sum(1 for gm, h in goals if h and gm <= t) for t in mins], int)
    sc_a = np.array([sum(1 for gm, h in goals if (not h) and gm <= t) for t in mins], int)
    lead = (sc_h - sc_a).astype(float)
    tf = mins / 90.0
    wp_home = _sig(1.05 + 1.7 * lead + 2.2 * lead * tf)     # sig(1.05)=0.74 at kickoff
    rem = 1.0 - wp_home
    wp_draw = rem * np.clip(0.30 - 0.25 * tf - 0.10 * np.abs(lead), 0.02, 0.40)
    wp_away = rem - wp_draw

    shots = [
        {"min": 8,  "is_home": False, "goal": False, "xg": 0.28, "player": "Faris Chaibi"},
        {"min": 15, "is_home": True,  "goal": False, "xg": 0.09, "player": "Nicolás González"},
        {"min": 23, "is_home": True,  "goal": True,  "xg": 0.34, "player": "Lionel Messi"},
        {"min": 31, "is_home": True,  "goal": False, "xg": 0.12, "player": "Enzo Fernández"},
        {"min": 39, "is_home": False, "goal": False, "xg": 0.06, "player": "Riyad Mahrez"},
        {"min": 47, "is_home": True,  "goal": True,  "xg": 0.22, "player": "Lionel Messi"},
        {"min": 54, "is_home": False, "goal": False, "xg": 0.05, "player": "Islam Slimani"},
        {"min": 61, "is_home": True,  "goal": False, "xg": 0.41, "player": "Julián Álvarez"},
        {"min": 64, "is_home": True,  "goal": True,  "xg": 0.27, "player": "Lionel Messi"},
        {"min": 78, "is_home": True,  "goal": False, "xg": 0.15, "player": "Paulo Dybala"},
        {"min": 83, "is_home": False, "goal": False, "xg": 0.04, "player": "Aïssa Mandi"},
    ]
    xg_h = np.array([sum(s["xg"] for s in shots if s["is_home"] and s["min"] <= t) for t in mins])
    xg_a = np.array([sum(s["xg"] for s in shots if (not s["is_home"]) and s["min"] <= t) for t in mins])
    xgd = xg_h - xg_a
    wp_xg = _sig(1.05 + 1.4 * xgd + 1.0 * xgd * tf)

    events = [
        {"min": 8,  "type": "VAR",          "player": "Faris Chaibi",   "is_home": False, "note": "goal ruled out, offside"},
        {"min": 23, "type": "Goal",         "player": "Lionel Messi",   "is_home": True,  "score": (1, 0)},
        {"min": 35, "type": "Card",         "player": "Aïssa Mandi",    "is_home": False},
        {"min": 47, "type": "Goal",         "player": "Lionel Messi",   "is_home": True,  "score": (2, 0)},
        {"min": 58, "type": "Substitution", "player": "Julián Álvarez", "is_home": True},
        {"min": 64, "type": "Goal",         "player": "Lionel Messi",   "is_home": True,  "score": (3, 0)},
        {"min": 72, "type": "Substitution", "player": "Paulo Dybala",   "is_home": True},
    ]
    ratings = [
        {"name": "Lionel Messi",      "rating": 9.66, "is_home": True,  "potm": True},
        {"name": "Rodrigo De Paul",   "rating": 8.22, "is_home": True,  "potm": False},
        {"name": "Lisandro Martínez", "rating": 7.67, "is_home": True,  "potm": False},
        {"name": "Enzo Fernández",    "rating": 7.61, "is_home": True,  "potm": False},
        {"name": "Nicolás González",  "rating": 7.44, "is_home": True,  "potm": False},
    ]
    return {"home": "Argentina", "away": "Algeria", "events": events, "shots": shots,
            "ratings": ratings, "pre_match": {"p_h": 0.74, "p_d": 0.08, "p_a": 0.18},
            "wp_mins": mins, "wp_home": wp_home, "wp_draw": wp_draw, "wp_away": wp_away,
            "wp_xg": wp_xg, "sc_h": sc_h, "sc_a": sc_a, "xg_h": xg_h, "xg_a": xg_a,
            "final_h": 3, "final_a": 0}


def build_state():
    """one per-frame state dict identical in shape to visual_ai.cv_pass output.
    tracks = [px, py, team, alpha, measured, sigma_x, sigma_y, angle_deg]."""
    arg = [(8, 34), (33, 20), (36, 42), (40, 55), (50, 30), (53, 48),
           (60, 22), (63, 40), (66, 53), (72, 35), (74, 46)]            # cyan, pushed up
    alg = [(98, 34), (82, 18), (84, 30), (85, 44), (86, 56), (78, 24),
           (80, 40), (80, 52), (76, 34)]                                # orange, deep block
    tracks = []
    for i, (x, y) in enumerate(arg):
        tracks.append([float(x), float(y), 0, 1.0, True, 1.3, 0.9, float((i * 37) % 180)])
    for i, (x, y) in enumerate(alg):
        tracks.append([float(x), float(y), 1, 1.0, True, 1.2, 0.95, float((i * 53) % 180)])
    return {"fp": BROADCAST, "boxes": [], "tracks": tracks, "ball": [70.0, 38.0],
            "wh": (606, 332), "note": "", "est": False, "cam": "wide", "conf": 0.74}


def draw_pitch(ax, line="#ffffff", lw=1.1, face="#16341f", alpha=0.5):
    ax.set_facecolor(face)
    ax.plot([0, 0, PL, PL, 0], [0, PW, PW, 0, 0], color=line, lw=lw, alpha=alpha)
    ax.plot([PL / 2, PL / 2], [0, PW], color=line, lw=lw * 0.85, alpha=alpha * 0.8)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(PL / 2 + 9.15 * np.cos(th), PW / 2 + 9.15 * np.sin(th), color=line, lw=lw * 0.85, alpha=alpha * 0.8)
    for x0 in (0, PL - 16.5):
        ax.plot([x0, x0 + 16.5, x0 + 16.5, x0], [13.84, 13.84, 54.16, 54.16], color=line, lw=lw * 0.85, alpha=alpha * 0.8)
    ax.set_xlim(-3, PL + 3); ax.set_ylim(-3, PW + 3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def hull(ax, pts, color, alpha=0.12, lw=1.4):
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return
    try:
        h = ConvexHull(pts)
    except Exception:
        return
    poly = pts[h.vertices]
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=2)
    pp = np.vstack([poly, poly[:1]])
    ax.plot(pp[:, 0], pp[:, 1], color=color, lw=lw, alpha=0.7, zorder=3)


def team_shape(pts):
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
    except Exception:
        return None
    return {"area": float(h.volume)}     # 2-D ConvexHull.volume == polygon area
