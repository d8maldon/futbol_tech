"""THEME: telemetry  (PITCHWALL) -- sports-tech / F1 pit-wall console for staff.
Ported from _variant_c.py into the visual_ai theme interface. Charcoal grid,
monospaced numerals, restrained neon. Handles live/no-pitch/estimated frames.
"""
import numpy as np
import cv2
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, RegularPolygon
from matplotlib.colors import LinearSegmentedColormap

import dashboard_themes as T

PL, PW = T.PL, T.PW
W, H = 1920.0, 1080.0

INK_BG = "#0a0d12"; CARD = "#11161d"; CARD_HI = "#161d26"; HAIR = "#222c38"; HAIR_HI = "#2c3a49"
RULE = "#1a232d"; GRID = "#1c2530"; TXT_HI = "#f2f6fb"; TXT = "#aeb9c6"; TXT_MUT = "#6a7787"
TXT_FAINT = "#48535f"; ACCENT = "#39d0c8"; AMBER = "#f4b740"; GOOD = "#5fcf80"
CYAN_HX = "#4ec8ff"; ORANGE_HX = "#ff7333"
MONO = "DejaVu Sans Mono"; SANS = "Bahnschrift"; SANS2 = "DejaVu Sans"
BG = INK_BG

HX, HY, HW, HH = 28, 22, W - 56, 96
TOP = 134; GAP = 14
COLL_X = 28; COLL_W = 560
COLM_X = COLL_X + COLL_W + GAP; COLM_W = 660
COLR_X = COLM_X + COLM_W + GAP; COLR_W = W - 28 - COLR_X
BOT = H - 22

WP_BOX = (COLL_X, TOP, COLL_W + GAP + COLM_W, 118)
BC_BOX = (COLL_X, TOP + 118 + GAP, COLL_W, 250)
CL_BOX = (COLL_X, TOP + 118 + GAP + 250 + GAP, COLL_W, 116)
TD_BOX = (COLM_X, TOP + 118 + GAP, COLM_W, 250)
PC_BOX = (COLM_X, TOP + 118 + GAP + 250 + GAP, COLM_W, BOT - (TOP + 118 + GAP + 250 + GAP))
XG_BOX = (COLR_X, TOP + 118 + GAP, COLR_W, 232)
EV_BOX = (COLR_X, TOP + 118 + GAP + 232 + GAP, COLR_W, 250)
RT_BOX = (COLR_X, TOP + 118 + GAP + 232 + GAP + 250 + GAP, COLR_W, BOT - (TOP + 118 + GAP + 232 + GAP + 250 + GAP))


def _card_inner(x, y, w, h, pad=18, header_h=34):
    return (x + pad, y + header_h + 12, w - 2 * pad, h - header_h - 24)


def _inset(fig, rect):
    x, y, w, h = rect
    ix, iy, iw, ih = _card_inner(x, y, w, h)
    ax = fig.add_axes([ix / W, 1 - (iy + ih) / H, iw / W, ih / H])
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def _reset_inset(ax):
    ax.clear(); ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def _card(bg, x, y, w, h, title=None, kicker=None, tag=None, tag_color=ACCENT, header_h=34):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7",
                 fc=CARD, ec=HAIR, lw=1.2, zorder=2, mutation_aspect=1))
    bg.add_patch(Rectangle((x, y), 3.2, header_h, fc=tag_color, ec="none", zorder=4))
    if title is not None:
        bg.plot([x + 14, x + w - 14], [y + header_h, y + header_h], color=RULE, lw=1.0, zorder=3)
        if kicker:
            bg.text(x + 16, y + 12.5, kicker, color=TXT_MUT, fontsize=7.6, family=MONO,
                    fontweight="bold", va="center", ha="left", zorder=5)
        bg.text(x + 16, y + (24 if kicker else header_h / 2), title, color=TXT_HI, fontsize=11.5,
                family=SANS, fontweight="bold", va="center", ha="left", zorder=5)
        if tag:
            tw = 11 + len(tag) * 6.4
            bg.add_patch(FancyBboxPatch((x + w - 14 - tw, y + 8), tw, header_h - 16,
                         boxstyle="round,pad=0,rounding_size=3", fc="none", ec=tag_color, lw=1.1, zorder=4))
            bg.text(x + w - 14 - tw / 2, y + header_h / 2, tag, color=tag_color, fontsize=7.8, family=MONO,
                    fontweight="bold", va="center", ha="center", zorder=5)


def make_axes(fig):
    import matplotlib as mpl
    mpl.rcParams["font.family"] = SANS2
    mpl.rcParams["axes.unicode_minus"] = False
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H)
    bg.invert_yaxis(); bg.axis("off")
    return {
        "bg": bg,
        "wp": _inset(fig, WP_BOX), "bc": _inset(fig, BC_BOX), "cl": _inset(fig, CL_BOX),
        "td": _inset(fig, TD_BOX), "pc": _inset(fig, PC_BOX), "xg": _inset(fig, XG_BOX),
        "ev": _inset(fig, EV_BOX), "rt": _inset(fig, RT_BOX),
    }


def _backdrop(bg):
    bg.clear(); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.invert_yaxis(); bg.axis("off")
    bg.set_facecolor(INK_BG)
    for gx in range(0, int(W) + 1, 60):
        bg.plot([gx, gx], [0, H], color="#0e131a", lw=0.6, zorder=0)
    for gy in range(0, int(H) + 1, 60):
        bg.plot([0, W], [gy, gy], color="#0e131a", lw=0.6, zorder=0)


def _draw_header(bg, d, m, ti):
    sc_h, sc_a = int(m["sc_h"][ti]), int(m["sc_a"][ti])
    ARG, ALG = m["home"], m["away"]
    bg.add_patch(FancyBboxPatch((HX, HY), HW, HH, boxstyle="round,pad=0,rounding_size=8",
                 fc=CARD, ec=HAIR, lw=1.2, zorder=2))
    bg.add_patch(Rectangle((HX, HY), 3.2, HH, fc=ACCENT, ec="none", zorder=4))
    bg.text(HX + 26, HY + 30, "PITCHWALL", color=TXT_HI, fontsize=20, family=SANS, fontweight="bold",
            va="center", ha="left", zorder=5)
    bg.text(HX + 178, HY + 28, "LIVE  MATCH  TELEMETRY", color=ACCENT, fontsize=8.5, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=5)
    bg.text(HX + 26, HY + 64, "BROADCAST COMPUTER-VISION  /  WIN-PROBABILITY ENGINE", color=TXT_MUT,
            fontsize=8.2, family=MONO, va="center", ha="left", zorder=5)
    bg.plot([HX + 470, HX + 470], [HY + 16, HY + HH - 16], color=HAIR_HI, lw=1.1, zorder=4)
    cx = HX + HW / 2
    bg.add_patch(Circle((cx - 86, HY + 24), 4.2, fc="#ff4040", ec="none", zorder=6))
    bg.text(cx - 76, HY + 24, "LIVE", color="#ff5c5c", fontsize=9, family=MONO, fontweight="bold",
            va="center", ha="left", zorder=6)
    bg.add_patch(FancyBboxPatch((cx + 36, HY + 16), 56, 18, boxstyle="round,pad=0,rounding_size=3",
                 fc=CARD_HI, ec=HAIR_HI, lw=1.0, zorder=5))
    bg.text(cx + 64, HY + 25, "{}'".format(ti), color=TXT_HI, fontsize=10, family=MONO,
            fontweight="bold", va="center", ha="center", zorder=6)
    bg.text(cx - 158, HY + 60, ARG.upper(), color=TXT_HI, fontsize=23, family=SANS, fontweight="bold",
            va="center", ha="right", zorder=6)
    bg.add_patch(Rectangle((cx - 150, HY + 44), 7, 32, fc=CYAN_HX, ec="none", zorder=6))
    bg.text(cx - 14, HY + 60, "{}".format(sc_h), color=CYAN_HX, fontsize=34, family=MONO,
            fontweight="bold", va="center", ha="right", zorder=6)
    bg.text(cx, HY + 59, "-", color=TXT, fontsize=22, family=MONO, fontweight="bold",
            va="center", ha="center", zorder=6)
    bg.text(cx + 14, HY + 60, "{}".format(sc_a), color=ORANGE_HX, fontsize=34, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=6)
    bg.add_patch(Rectangle((cx + 143, HY + 44), 7, 32, fc=ORANGE_HX, ec="none", zorder=6))
    bg.text(cx + 158, HY + 60, ALG.upper(), color=TXT_HI, fontsize=23, family=SANS, fontweight="bold",
            va="center", ha="left", zorder=6)

    def hmetric(xr, label, value, vcolor=TXT_HI):
        bg.text(xr, HY + 32, label, color=TXT_MUT, fontsize=7.6, family=MONO, fontweight="bold",
                va="center", ha="left", zorder=6)
        bg.text(xr, HY + 56, value, color=vcolor, fontsize=15, family=MONO, fontweight="bold",
                va="center", ha="left", zorder=6)
    rx0 = HX + HW - 360
    bg.plot([rx0 - 24, rx0 - 24], [HY + 16, HY + HH - 16], color=HAIR_HI, lw=1.1, zorder=4)
    hmetric(rx0, "XG  ARG", "{:.2f}".format(float(m["xg_h"][ti])), CYAN_HX)
    hmetric(rx0 + 92, "XG  ALG", "{:.2f}".format(float(m["xg_a"][ti])), ORANGE_HX)
    hmetric(rx0 + 184, "CV CONF", "{}%".format(int(d.get("conf", 0) * 100)), ACCENT)
    hmetric(rx0 + 276, "TRACKED", "{}".format(len(d.get("tracks", []))), TXT_HI)


def _draw_winprob(bg, ax, m, ti):
    _card(bg, *WP_BOX, title="WIN PROBABILITY", kicker="LIVE MODEL  ·  SCORE-AWARE", tag="OUTRIGHT", tag_color=ACCENT)
    _reset_inset(ax)
    ARG, ALG = m["home"], m["away"]
    ph, pd_, pa = float(m["wp_home"][ti]), float(m["wp_draw"][ti]), float(m["wp_away"][ti])
    pxg = float(m["wp_xg"][ti])
    bar_y, bar_h = 0.30, 0.30
    disp = [round(v * 100) for v in (ph, pd_, pa)]
    widths = [max(v, 0.012) for v in (ph, pd_, pa)]
    widths = [w / sum(widths) for w in widths]
    cols = [CYAN_HX, "#7a8694", ORANGE_HX]
    x0 = 0.0
    for w, col in zip(widths, cols):
        ax.add_patch(Rectangle((x0, bar_y), w, bar_h, fc=col, ec=INK_BG, lw=2.0, zorder=3)); x0 += w
    ax.text(0.0, bar_y + bar_h + 0.34, "{}%".format(disp[0]), color=CYAN_HX, fontsize=19, family=MONO,
            fontweight="bold", va="center", ha="left")
    ax.text(1.0, bar_y + bar_h + 0.34, "{}%".format(disp[2]), color=ORANGE_HX, fontsize=14, family=MONO,
            fontweight="bold", va="center", ha="right")
    ax.text(1.0, bar_y - 0.30, "DRAW {}%".format(disp[1]), color=TXT_MUT, fontsize=8, family=MONO,
            fontweight="bold", va="center", ha="right")
    ax.text(0.0, bar_y - 0.30, ARG.upper(), color=TXT, fontsize=9, family=MONO, fontweight="bold",
            va="center", ha="left")
    ax.text(0.46, bar_y - 0.30, ALG.upper(), color=TXT, fontsize=9, family=MONO, fontweight="bold",
            va="center", ha="center")
    mx = pxg
    ax.plot([mx, mx], [bar_y - 0.10, bar_y + bar_h + 0.12], color=TXT_HI, lw=2.4, zorder=6, solid_capstyle="round")
    ax.add_patch(RegularPolygon((mx, bar_y + bar_h + 0.14), 3, radius=0.016, orientation=np.pi, fc=TXT_HI, ec="none", zorder=6))
    ax.text(mx - 0.012 if mx > 0.5 else mx + 0.012, bar_y + bar_h + 0.42, "xG-DESERVED  {:.0f}%".format(pxg * 100),
            color=TXT_HI, fontsize=8.5, family=MONO, fontweight="bold", va="center", ha="right" if mx > 0.5 else "left")
    ax.set_xlim(-0.004, 1.004); ax.set_ylim(0, 1)


def _draw_broadcast(bg, ax, d):
    _card(bg, *BC_BOX, title="BROADCAST VISION", kicker="PLAYER DETECTION  ·  WIDE CAM", tag="LIVE CV", tag_color=ACCENT)
    _reset_inset(ax)
    img = cv2.imread(d["fp"]) if d.get("fp") else None
    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto", zorder=1, extent=[0, 1, 0, 1])
    else:
        ax.set_facecolor(CARD_HI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def bracket(x, y, dx, dy):
        ax.plot([x, x + dx], [y, y], color=ACCENT, lw=1.6, zorder=5)
        ax.plot([x, x], [y, y + dy], color=ACCENT, lw=1.6, zorder=5)
    for (cx_, cy_, sx, sy) in [(0.012, 0.04, 0.05, 0.10), (0.988, 0.04, -0.05, 0.10),
                               (0.012, 0.96, 0.05, -0.10), (0.988, 0.96, -0.05, -0.10)]:
        bracket(cx_, cy_, sx, sy)
    tracks = d.get("tracks", []) or []
    n_home = sum(1 for t in tracks if t[2] == 0); n_away = sum(1 for t in tracks if t[2] == 1)
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 0.14, fc="#000000", ec="none", alpha=0.55, zorder=4, transform=ax.transAxes))
    ax.text(0.025, 0.07, "{} PLAYERS DETECTED".format(len(tracks)), color=TXT_HI, fontsize=9, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=6, transform=ax.transAxes)
    ax.text(0.975, 0.07, "ARG {}   ALG {}".format(n_home, n_away), color=ACCENT, fontsize=9, family=MONO,
            fontweight="bold", va="center", ha="right", zorder=6, transform=ax.transAxes)


def _draw_call(bg, ax, m):
    pm = m["pre_match"]
    call_home = pm["p_h"] >= pm["p_a"]
    correct = m["final_h"] != m["final_a"] and (m["final_h"] > m["final_a"]) == call_home
    _card(bg, *CL_BOX, title="PRE-MATCH CALL", kicker="WORLD-FOOTBALL ELO  ·  RESULT",
          tag="CORRECT" if correct else "LIVE", tag_color=GOOD if correct else ACCENT)
    _reset_inset(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.92, "MODEL CALLED", color=TXT_MUT, fontsize=7.6, family=MONO, fontweight="bold", va="top", ha="left")
    for i, (lab, val, col) in enumerate([("ARG", pm["p_h"], CYAN_HX), ("DRAW", pm["p_d"], "#8893a1"), ("ALG", pm["p_a"], ORANGE_HX)]):
        xc = 0.06 + i * 0.135
        ax.text(xc, 0.60, "{:.0f}".format(val * 100), color=col, fontsize=20, family=MONO, fontweight="bold", va="center", ha="center")
        ax.text(xc, 0.60, "%", color=col, fontsize=9, family=MONO, va="center", ha="left")
        ax.text(xc, 0.20, lab, color=TXT, fontsize=8, family=MONO, fontweight="bold", va="center", ha="center")
    ax.add_patch(Rectangle((0.50, 0.10), 0.495, 0.80, fc=CARD_HI, ec=HAIR_HI, lw=1.0, zorder=2))
    ax.text(0.525, 0.74, "OUR CALL", color=TXT_MUT, fontsize=7.6, family=MONO, fontweight="bold", va="center", ha="left")
    ax.text(0.525, 0.50, (m["home"] if call_home else m["away"]).upper(), color=CYAN_HX if call_home else ORANGE_HX,
            fontsize=15, family=SANS, fontweight="bold", va="center", ha="left")
    ax.text(0.525, 0.22, "FINAL  {}-{}".format(m["final_h"], m["final_a"]), color=TXT, fontsize=10, family=MONO,
            fontweight="bold", va="center", ha="left")
    if correct:
        ax.add_patch(Circle((0.93, 0.50), 0.075, fc="none", ec=GOOD, lw=2.0, zorder=4))
        ax.plot([0.905, 0.925, 0.965], [0.50, 0.42, 0.60], color=GOOD, lw=2.4, zorder=5,
                solid_capstyle="round", solid_joinstyle="round")


def _draw_shape(bg, ax, d):
    _card(bg, *TD_BOX, title="FORMATION SHAPE", kicker="TOP-DOWN  ·  CONVEX TEAM BLOCK", tag="5M ZONE", tag_color=ACCENT)
    _reset_inset(ax)
    blank = d.get("note") == "no pitch view"
    T.mpl_pitch(ax, line="#2f3e4d", face="#0d1217", lw=1.2, line_zorder=1)
    if blank:
        lbl = "NO PITCH VIEW · graphic/replay" if d.get("cam") == "other" else "NO PITCH VIEW"
        ax.text(PL / 2, PW / 2, lbl, ha="center", va="center", color=AMBER, fontsize=12, family=SANS, fontweight="bold")
        return
    tracks = d.get("tracks", []) or []
    arg_pts = [(t[0], PW - t[1]) for t in tracks if t[2] == 0]
    alg_pts = [(t[0], PW - t[1]) for t in tracks if t[2] == 1]
    T.hull(ax, arg_pts, CYAN_HX, alpha=0.10, lw=1.8)
    T.hull(ax, alg_pts, ORANGE_HX, alpha=0.10, lw=1.8)
    for t in tracks:
        col = CYAN_HX if t[2] == 0 else ORANGE_HX
        ax.add_patch(Circle((t[0], PW - t[1]), 1.5, fc=col, ec="#0d1217", lw=1.0, alpha=float(t[3]), zorder=5))
    ball = d.get("ball")
    if ball is not None:
        ax.add_patch(Circle((ball[0], PW - ball[1]), 1.5, fc="#ffffff", ec="#101010", lw=1.0, zorder=7))
        ax.add_patch(Circle((ball[0], PW - ball[1]), 3.0, fc="none", ec="#ffffff", lw=1.0, alpha=0.6, zorder=6))
    ax.annotate("", xy=(95, PW + 1.5), xytext=(80, PW + 1.5),
                arrowprops=dict(arrowstyle="-|>", color=TXT_MUT, lw=1.4), annotation_clip=False)
    if d.get("est"):
        ax.text(PL / 2, 2, "HOLDING LAST SHAPE (ESTIMATED)", ha="center", va="bottom", color=AMBER,
                fontsize=8, family=MONO, fontweight="bold")


def _draw_control(bg, ax, d):
    _card(bg, *PC_BOX, title="PITCH CONTROL", kicker="VORONOI OWNERSHIP  ·  mplsoccer", tag="LIVE", tag_color=ACCENT)
    _reset_inset(ax)
    blank = d.get("note") == "no pitch view"
    tracks = d.get("tracks", []) or []
    T.mpl_pitch(ax, line="#9fb4c6", face="#0d1217", lw=1.2, line_zorder=3)
    if blank or len(tracks) < 2:
        if blank:
            lbl = "NO PITCH VIEW · graphic/replay" if d.get("cam") == "other" else "NO PITCH VIEW"
            ax.text(PL / 2, PW / 2, lbl, ha="center", va="center", color=AMBER, fontsize=11, family=SANS, fontweight="bold")
        return
    th, ta, share = T.voronoi_regions(tracks)
    for poly in th:
        ax.fill(poly[:, 0], poly[:, 1], color=CYAN_HX, alpha=0.40, ec="#0d1217", lw=0.6, zorder=2)
    for poly in ta:
        ax.fill(poly[:, 0], poly[:, 1], color=ORANGE_HX, alpha=0.40, ec="#0d1217", lw=0.6, zorder=2)
    ax.text(2, PW - 3, "ARG {:.0f}%".format(share * 100), color=CYAN_HX, fontsize=9, family=MONO, fontweight="bold",
            va="center", ha="left", zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="#0a0d12")])
    ax.text(PL - 2, PW - 3, "{:.0f}% ALG".format((1 - share) * 100), color=ORANGE_HX, fontsize=9, family=MONO,
            fontweight="bold", va="center", ha="right", zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="#0a0d12")])


def _draw_xg(bg, ax, m, ti):
    _card(bg, *XG_BOX, title="EXPECTED GOALS RACE", kicker="CUMULATIVE xG  ·  0-{}'".format(ti), tag="xG", tag_color=ACCENT)
    _reset_inset(ax)
    mins = m["wp_mins"]; mask = mins <= ti
    xh = m["xg_h"][mask]; xa = m["xg_a"][mask]; mm = mins[mask]
    ymax = max(float(xh.max()), float(xa.max()), 0.6) * 1.18
    ax.set_xlim(0, ti); ax.set_ylim(0, ymax)
    for yv in np.arange(0, ymax, 0.5):
        ax.plot([0, ti], [yv, yv], color=GRID, lw=0.8, zorder=1)
        ax.text(-1.0, yv, "{:.1f}".format(yv), color=TXT_FAINT, fontsize=7, family=MONO, va="center", ha="right", zorder=2)
    for xv in [0, 15, 30, 45, 60, 75, 90]:
        if xv <= ti:
            ax.plot([xv, xv], [0, ymax], color=GRID, lw=0.7, zorder=1)
            ax.text(xv, -ymax * 0.05, "{}'".format(xv), color=TXT_FAINT, fontsize=7, family=MONO, va="top", ha="center", zorder=2)
    ax.fill_between(mm, xh, step="post", color=CYAN_HX, alpha=0.14, zorder=2)
    ax.step(mm, xh, where="post", color=CYAN_HX, lw=2.4, zorder=4)
    ax.step(mm, xa, where="post", color=ORANGE_HX, lw=2.4, zorder=4)
    ax.fill_between(mm, xa, step="post", color=ORANGE_HX, alpha=0.10, zorder=2)
    for s in m["shots"]:
        if s["goal"] and s["min"] <= ti:
            yv = float(m["xg_h"][s["min"]]) if s["is_home"] else float(m["xg_a"][s["min"]])
            col = CYAN_HX if s["is_home"] else ORANGE_HX
            ax.scatter([s["min"]], [yv], marker="*", s=190, color=col, edgecolors=INK_BG, linewidths=1.2, zorder=6)
    ax.text(ti + 0.5, float(xh[-1]), "{:.2f}".format(float(xh[-1])), color=CYAN_HX, fontsize=9.5, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=6)
    ax.text(ti + 0.5, float(xa[-1]), "{:.2f}".format(float(xa[-1])), color=ORANGE_HX, fontsize=9.5, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=6)
    ax.set_xlim(-3.5, ti + 6); ax.set_ylim(-ymax * 0.10, ymax)


def _draw_ticker(bg, ax, m, ti):
    _card(bg, *EV_BOX, title="MATCH EVENTS", kicker="TIMELINE  ·  LATEST FIRST", tag="FEED", tag_color=ACCENT)
    _reset_inset(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    evs = [e for e in m["events"] if e["min"] <= ti and not (e["type"] == "Substitution" and not e["player"])][::-1]
    EV_COL = {"Card": AMBER, "Substitution": GOOD, "VAR": AMBER}
    EV_ABBR = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}
    n = len(evs)
    if n == 0:
        return
    row_h = 1.0 / max(n, 1)
    for i, e in enumerate(evs):
        yc = 1.0 - (i + 0.5) * row_h
        newest = (i == 0)
        team_col = CYAN_HX if e["is_home"] else ORANGE_HX
        chip_col = EV_COL.get(e["type"]) or team_col
        if newest:
            ax.add_patch(Rectangle((0.0, 1.0 - row_h * (i + 1) + 0.01), 1.0, row_h - 0.02, fc="#18222c", ec="none", zorder=1))
        ax.add_patch(Rectangle((0.0, 1.0 - row_h * (i + 1) + 0.02), 0.006, row_h - 0.04, fc=team_col, ec="none", zorder=3))
        ax.text(0.035, yc, "{:>2}'".format(e["min"]), color=TXT_HI if newest else TXT, fontsize=11, family=MONO,
                fontweight="bold", va="center", ha="left", zorder=4)
        ab = EV_ABBR.get(e["type"], e["type"]); cw = 0.085
        ax.add_patch(FancyBboxPatch((0.105, yc - row_h * 0.26), cw, row_h * 0.52, boxstyle="round,pad=0,rounding_size=0.02",
                     fc="none", ec=chip_col, lw=1.1, zorder=3))
        ax.text(0.105 + cw / 2, yc, ab, color=chip_col, fontsize=7.6, family=MONO, fontweight="bold",
                va="center", ha="center", zorder=4)
        ax.text(0.215, yc, e["player"], color=TXT_HI if newest else TXT, fontsize=10.5, family=SANS,
                fontweight="bold" if newest else "normal", va="center", ha="left", zorder=4)
        if e["type"] == "Goal" and e.get("score"):
            det = "{}-{}".format(e["score"][0], e["score"][1])
        elif e["type"] == "VAR":
            det = "OFFSIDE"
        elif e["type"] == "Card":
            det = "YELLOW"
        else:
            det = "ON"
        ax.text(0.985, yc, det, color=team_col if e["type"] == "Goal" else chip_col, fontsize=9, family=MONO,
                fontweight="bold", va="center", ha="right", zorder=4)
        if i < n - 1:
            yr = 1.0 - row_h * (i + 1)
            ax.plot([0.02, 0.98], [yr, yr], color=RULE, lw=0.8, zorder=2)


def _draw_ratings(bg, ax, m):
    _card(bg, *RT_BOX, title="TOP PERFORMERS", kicker="PLAYER RATING  ·  0-10", tag="POTM", tag_color=AMBER)
    _reset_inset(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rs = m["ratings"][:5]
    n = len(rs)
    if n == 0:
        return
    row_h = 1.0 / n
    RT_LO, RT_HI = 6.0, 10.0
    bx0, bx1 = 0.0, 0.74
    for i, r in enumerate(rs):
        yc = 1.0 - (i + 0.5) * row_h
        col = CYAN_HX if r["is_home"] else ORANGE_HX
        ax.text(0.0, yc + row_h * 0.22, r["name"], color=TXT_HI, fontsize=10.5, family=SANS, fontweight="bold",
                va="center", ha="left", zorder=4)
        if r["potm"]:
            badge_x = 0.012 * len(r["name"]) + 0.135; bw = 0.115
            ax.add_patch(FancyBboxPatch((badge_x, yc + row_h * 0.10), bw, row_h * 0.24, boxstyle="round,pad=0,rounding_size=0.015",
                         fc=AMBER, ec="none", zorder=5))
            ax.text(badge_x + bw / 2, yc + row_h * 0.22, "POTM", color="#1a1205", fontsize=7.0, family=MONO,
                    fontweight="bold", va="center", ha="center", zorder=6)
        by = yc - row_h * 0.34; bh = row_h * 0.18
        ax.add_patch(Rectangle((bx0, by), bx1, bh, fc=CARD_HI, ec=HAIR, lw=0.8, zorder=2))
        frac = float(np.clip((r["rating"] - RT_LO) / (RT_HI - RT_LO), 0.04, 1.0))
        ax.add_patch(Rectangle((bx0, by), bx1 * frac, bh, fc=col, ec="none", zorder=3))
        ax.text(0.995, yc, "{:.2f}".format(r["rating"]), color=TXT_HI, fontsize=14, family=MONO,
                fontweight="bold", va="center", ha="right", zorder=4)


def _draw_footer(bg):
    bg.plot([28, W - 28], [H - 14 - 6, H - 14 - 6], color=RULE, lw=1.0, zorder=3)
    bg.text(28, H - 8, "PITCHWALL", color=ACCENT, fontsize=8.5, family=SANS, fontweight="bold", va="bottom", ha="left", zorder=5)
    foot = ("Broadcast CV ~5 m zone-grade  ·  win-prob OOS log-loss 0.82  ·  "
            "World-Football-Elo predictor  ·  pitch-control proximity softmax  ·  futbol_tech")
    bg.text(W - 28, H - 8, foot, color=TXT_MUT, fontsize=8, family=MONO, va="bottom", ha="right", zorder=5)


def draw_frame(axes, d, m, ti, team_rgb, label=""):
    bg = axes["bg"]
    _backdrop(bg)
    _draw_header(bg, d, m, ti)
    _draw_winprob(bg, axes["wp"], m, ti)
    _draw_broadcast(bg, axes["bc"], d)
    _draw_call(bg, axes["cl"], m)
    _draw_shape(bg, axes["td"], d)
    _draw_control(bg, axes["pc"], d)
    _draw_xg(bg, axes["xg"], m, ti)
    _draw_ticker(bg, axes["ev"], m, ti)
    _draw_ratings(bg, axes["rt"], m)
    _draw_footer(bg)
