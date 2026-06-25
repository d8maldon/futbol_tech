"""Theme registry + shared primitives for the visual-AI dashboard.

A THEME is a module (theme_broadcast / theme_editorial / theme_telemetry) exposing:

    BG : str                                  # figure background hex
    make_axes(fig) -> axes                    # build the theme's axes layout on a
                                              # 19.2x10.8 (16:9) fig; return a handle
                                              # (dict) the draw step reuses every frame
    draw_frame(axes, d, m, ti, team_rgb, label="")
                                              # render ONE frame. MUST handle:
                                              #   - live frame  (d["note"]=="")        : full tracking + data row
                                              #   - no pitch    (d["note"]=="no pitch view"): blank top-down + control
                                              #       with a label; broadcast still shows d["fp"]; data row stays live
                                              #   - estimated   (d.get("est"))         : "holding last shape" hint
                                              # index minute-keyed series (m["wp_*"], m["xg_*"], m["sc_*"]) at ti.

The animation loop (render, below) owns the playhead/frame_min; themes get ti (int minute).
This keeps the three audiences (fans / analysts / technical staff) as three skins over
ONE data path, so a single CV+match-data pass renders any of them.
"""
import importlib
import numpy as np

PL, PW = 105.0, 68.0
_REG = {"broadcast": "theme_broadcast", "editorial": "theme_editorial", "telemetry": "theme_telemetry"}


def get(name):
    if name not in _REG:
        raise SystemExit("unknown theme '{}', choose from {} (or 'legacy')".format(name, list(_REG)))
    return importlib.import_module(_REG[name])


def names():
    return list(_REG)


def draw_pitch(ax, line="#ffffff", lw=1.1, face="#16341f", alpha=0.5, goals=True, equal=True):
    """COMPLETE real-geometry FIFA 105x68 m pitch (IFAB markings): outline, halfway,
    centre circle + spot, both penalty boxes (40.32x16.5) + 6-yard boxes (18.32x5.5),
    penalty spots (11 m) + penalty arcs ('the D', r=9.15), corner arcs (r=1), and the
    GOALS behind each goal-line (7.32 m wide). Pass equal=False to stretch into a wide
    short panel (territory strip)."""
    ax.set_facecolor(face)
    cy = PW / 2
    a2 = alpha * 0.85
    def L(xs, ys, w=lw * 0.85, al=a2):
        ax.plot(xs, ys, color=line, lw=w, alpha=al, solid_capstyle="round", zorder=2.5)
    L([0, 0, PL, PL, 0], [0, PW, PW, 0, 0], w=lw, al=alpha)          # outline
    L([PL / 2, PL / 2], [0, PW])                                     # halfway
    th = np.linspace(0, 2 * np.pi, 100)
    L(PL / 2 + 9.15 * np.cos(th), cy + 9.15 * np.sin(th))            # centre circle
    ax.plot([PL / 2], [cy], marker="o", ms=max(2.0, lw * 1.6), color=line, alpha=alpha * 0.9, zorder=2.5)
    arc = np.arccos(5.5 / 9.15)
    for sx, spot, sgn in ((0.0, 11.0, 1.0), (PL, PL - 11.0, -1.0)):
        L([sx, sx + sgn * 16.5, sx + sgn * 16.5, sx], [cy - 20.16, cy - 20.16, cy + 20.16, cy + 20.16])  # penalty box
        L([sx, sx + sgn * 5.5, sx + sgn * 5.5, sx], [cy - 9.16, cy - 9.16, cy + 9.16, cy + 9.16])        # 6-yard box
        ax.plot([spot], [cy], marker="o", ms=max(1.8, lw * 1.4), color=line, alpha=alpha * 0.9, zorder=2.5)  # pen spot
        t = np.linspace(-arc, arc, 40) if sgn > 0 else np.linspace(np.pi - arc, np.pi + arc, 40)
        L(spot + 9.15 * np.cos(t), cy + 9.15 * np.sin(t))            # penalty arc (D)
        if goals:
            L([sx, sx - sgn * 2.4, sx - sgn * 2.4, sx], [cy - 3.66, cy - 3.66, cy + 3.66, cy + 3.66], w=lw, al=alpha)  # goal
    for cxc, cyc, a0 in ((0, 0, 0), (PL, 0, 90), (PL, PW, 180), (0, PW, 270)):
        tt = np.linspace(np.radians(a0), np.radians(a0 + 90), 14)
        L(cxc + 1.0 * np.cos(tt), cyc + 1.0 * np.sin(tt), w=lw * 0.7, al=alpha * 0.7)  # corner arc
    ax.set_xlim(-4.5, PL + 4.5); ax.set_ylim(-3, PW + 3)
    if equal:
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def hull(ax, pts, color, alpha=0.12, lw=1.4):
    """filled convex hull of a team's points (silently skips < 3 points)."""
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


_VP = None


def _vpitch():
    global _VP
    if _VP is None:
        from mplsoccer import Pitch
        _VP = Pitch(pitch_type="uefa", pitch_length=PL, pitch_width=PW)
    return _VP


def mpl_pitch(ax, line="#ffffff", face="#16341f", lw=1.2, line_zorder=2, pad=2.0, goal_type="box"):
    """Draw a battle-tested mplsoccer pitch (UEFA 105x68: goals, 6-yd boxes, penalty
    arcs + spots, corner arcs) on ax and return the Pitch object (for voronoi)."""
    from mplsoccer import Pitch
    p = Pitch(pitch_type="uefa", pitch_length=PL, pitch_width=PW, line_color=line,
              pitch_color=face, linewidth=lw, goal_type=goal_type, corner_arcs=True,
              line_zorder=line_zorder, pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad)
    p.draw(ax=ax)
    return p


def _poly_area(poly):
    poly = np.asarray(poly, float)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def voronoi_regions(tracks):
    """mplsoccer Voronoi pitch-control in the flipped top-down frame (y -> PW-y).
    Returns (home_polys, away_polys, home_share); (None, None, 0.5) if < 2 players."""
    pts = [(t[0], PW - t[1], t[2]) for t in (tracks or [])]
    if len(pts) < 2:
        return None, None, 0.5
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    home = np.array([p[2] == 0 for p in pts])
    th, ta = _vpitch().voronoi(xs, ys, home)
    ah = sum(_poly_area(p) for p in th); aa = sum(_poly_area(p) for p in ta)
    return th, ta, ah / (ah + aa + 1e-9)


def compute_frame_min(states, override=None):
    """playhead: advance match-minute only on frames that SHOW the pitch, so the
    dashboard reveals the match story progressively (approx; no clock OCR)."""
    if override is not None:
        return np.clip(np.asarray(override, float), 0, 98)
    pitch = np.array([s["note"] != "no pitch view" for s in states], float)
    cum = np.cumsum(pitch)
    return np.clip(90.0 * cum / max(cum[-1], 1.0), 0, 98)


def render(theme, states, team_rgb, m, out, fps, label="", frame_min_override=None):
    """drive a theme module over a whole clip -> mp4 (needs ffmpeg, like the original)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    frame_min = compute_frame_min(states, frame_min_override)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(theme.BG)
    axes = theme.make_axes(fig)

    def draw(i):
        theme.draw_frame(axes, states[i], m, int(round(float(frame_min[i]))), team_rgb, label=label)

    a = manim.FuncAnimation(fig, draw, frames=len(states), interval=1000 / fps)
    a.save(out, writer=manim.FFMpegWriter(fps=fps, codec="libx264",
           extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-preset", "veryfast"]))
    plt.close(fig)
