"""THEME: broadcast  (PITCHVISION) -- TV-graphics look for fans.
Ported from _variant_a.py into the visual_ai theme interface:
    BG, make_axes(fig) -> axes dict, draw_frame(axes, d, m, ti, team_rgb, label).
Bold team-colour blocks, scoreboard lockup, lower-third header strips, navy night base.
Handles the real animation's live / no-pitch / estimated frames.
"""
import numpy as np
import cv2
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, RegularPolygon
from matplotlib.colors import LinearSegmentedColormap

import dashboard_themes as T

PL, PW = T.PL, T.PW

# ---- palette ----
BASE = "#0a0f1c"; PANEL = "#121a2e"; PANEL_HI = "#18223b"; STRIP = "#1d2942"
HAIR = "#2c3a5a"; HAIR_HI = "#3a4c72"; INK = "#f4f7ff"; INK2 = "#aab6d4"; INK3 = "#6e7da0"
GOLD = "#ffcf4d"; AMBER = "#ffb020"; GREEN = "#3ddc84"; RED = "#ff5a5a"
CYAN_HEX = "#4ec7ff"; ORANGE_HEX = "#ff7333"
F_COND = "Bahnschrift"; F_NUM = "DejaVu Sans"; F_MONO = "DejaVu Sans Mono"
BG = BASE

# ---- geometry (figure fractions) ----
M = 0.012
HDR_H = 0.135
HX, HY, HW, HH = M, 1 - M - HDR_H, 1 - 2 * M, HDR_H
BODY_TOP = HY - M
BODY_BOT = M + 0.034
COL_W = (1 - 4 * M) / 3
C1 = M
C2 = M + COL_W + M
C3 = M + 2 * (COL_W + M)
WPH = 0.140
WPX, WPY, WPW = M, BODY_TOP - WPH, 1 - 2 * M
B2_TOP = WPY - M
GH2 = B2_TOP - BODY_BOT
ROW1_H = GH2 * 0.52
ROW2_H = GH2 - ROW1_H - M
R1Y = B2_TOP - ROW1_H
R2Y = BODY_BOT
INNER = (0.034, 0.03, 0.932, 0.815)   # _card inner box for strip_h=0.135


def _fy(y):
    return PW - y


def _pct_label(f):
    p = f * 100.0
    if p >= 99.95: return ">99.9%"
    if p < 0.05: return "<0.1%"
    if p < 1: return "{:.1f}%".format(p)
    return "{:.0f}%".format(p)


def _AX(fig, x, y, w, h, z=1):
    a = fig.add_axes([x, y, w, h], zorder=z)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_axis_off()
    return a


def _reset(ax):
    ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()


def _card(ax, kicker, accent=GOLD, strip_h=0.135, fill=PANEL, value_right=None, value_color=INK):
    x = y = 0.0; w = h = 1.0
    ax.add_patch(FancyBboxPatch((x + 0.006, y + 0.006), w - 0.012, h - 0.012,
                 boxstyle="round,pad=0,rounding_size=0.022", linewidth=1.2,
                 edgecolor=HAIR, facecolor=fill, mutation_aspect=h / max(w, 1e-6),
                 transform=ax.transAxes, zorder=2))
    sh = strip_h
    sx, sy, sw = x + 0.006, y + h - 0.006 - sh, w - 0.012
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.022",
                 linewidth=0, facecolor=STRIP, mutation_aspect=sh / max(sw, 1e-6),
                 transform=ax.transAxes, zorder=3))
    ax.add_patch(Rectangle((sx, sy), sw, sh * 0.5, facecolor=STRIP, lw=0, transform=ax.transAxes, zorder=3))
    ax.add_patch(Rectangle((sx, sy + sh * 0.12), 0.012, sh * 0.76, facecolor=accent, lw=0,
                 transform=ax.transAxes, zorder=4))
    ax.text(sx + 0.028, sy + sh * 0.5, kicker.upper(), transform=ax.transAxes, ha="left",
            va="center", color=INK, fontsize=12.5, fontfamily=F_COND, fontweight="bold", zorder=5)
    if value_right is not None:
        ax.text(sx + sw - 0.022, sy + sh * 0.5, value_right, transform=ax.transAxes, ha="right",
                va="center", color=value_color, fontsize=12.5, fontfamily=F_COND, fontweight="bold", zorder=5)
    ax.plot([sx, sx + sw], [sy, sy], color=HAIR_HI, lw=0.8, transform=ax.transAxes, zorder=4)


def _crest(ax, cxp, cyp, half, col):
    hw_ = half; hh_ = half * (HW / HH)
    ax.add_patch(FancyBboxPatch((cxp - hw_, cyp - hh_), 2 * hw_, 2 * hh_,
                 boxstyle="round,pad=0,rounding_size=0.012", facecolor=col, ec="#ffffff",
                 lw=2.0, mutation_aspect=(HH / HW), transform=ax.transAxes, zorder=5))
    ax.add_patch(Rectangle((cxp - hw_ * 0.36, cyp - hh_), hw_ * 0.72, 2 * hh_,
                 facecolor="#ffffff", alpha=0.45, lw=0, transform=ax.transAxes, zorder=6))


def make_axes(fig):
    # static vignette background
    gy, gx = np.mgrid[0:108, 0:192]
    r = np.sqrt(((gx - 96) / 120) ** 2 + ((gy - 50) / 78) ** 2)
    vig = np.clip(1 - r * 0.85, 0, 1)
    bg = np.zeros((108, 192, 3))
    top = np.array([0.09, 0.13, 0.22]); bot = np.array([0.035, 0.05, 0.10])
    for i in range(3):
        bg[:, :, i] = bot[i] + (top[i] - bot[i]) * vig
    axbg = fig.add_axes([0, 0, 1, 1]); axbg.set_axis_off()
    axbg.imshow(bg, extent=[0, 1, 0, 1], aspect="auto", zorder=-10, interpolation="bilinear")
    axbg.set_xlim(0, 1); axbg.set_ylim(0, 1)

    ix, iy, iw, ih = INNER
    A = {
        "bg": axbg,
        "hdr": _AX(fig, HX, HY, HW, HH, z=5),
        "wp": _AX(fig, WPX, WPY, WPW, WPH, z=4),
        "bcard": _AX(fig, C1, R1Y, COL_W, ROW1_H, z=3),
        "tcard": _AX(fig, C1, R2Y, COL_W, ROW2_H, z=3),
        "pcard": _AX(fig, C2, R1Y, COL_W, ROW1_H, z=3),
        "xcard": _AX(fig, C2, R2Y, COL_W, ROW2_H, z=3),
        "ecard": _AX(fig, C3, R1Y, COL_W, ROW1_H, z=3),
        "rcard": _AX(fig, C3, R2Y, COL_W, ROW2_H, z=3),
        "foot": _AX(fig, M, 0.004, 1 - 2 * M, 0.028, z=5),
        # sub-axes inside the four media cards
        "bsub": fig.add_axes([C1 + COL_W * ix, R1Y + ROW1_H * iy, COL_W * iw, ROW1_H * ih], zorder=4),
        "tsub": fig.add_axes([C1 + COL_W * ix, R2Y + ROW2_H * iy, COL_W * iw, ROW2_H * ih], zorder=4),
        "psub": fig.add_axes([C2 + COL_W * ix, R1Y + ROW1_H * iy, COL_W * iw, ROW1_H * ih], zorder=4),
        "xsub": fig.add_axes([C2 + COL_W * ix, R2Y + ROW2_H * iy, COL_W * iw, ROW2_H * ih], zorder=4),
    }
    return A


# ============================================================================ #
def _draw_header(ax, d, m, ti, team_rgb):
    _reset(ax)
    sc_h, sc_a = int(m["sc_h"][ti]), int(m["sc_a"][ti])
    ARG, ALG = m["home"], m["away"]
    ax.add_patch(FancyBboxPatch((0.004, 0.04), 0.992, 0.92, boxstyle="round,pad=0,rounding_size=0.05",
                 linewidth=1.4, edgecolor=HAIR_HI, facecolor=PANEL_HI, mutation_aspect=HH / HW,
                 transform=ax.transAxes, zorder=2))
    ax.add_patch(Rectangle((0.004, 0.04), 0.012, 0.92, facecolor=CYAN_HEX, lw=0, transform=ax.transAxes, zorder=4))
    ax.add_patch(Rectangle((0.984, 0.04), 0.012, 0.92, facecolor=ORANGE_HEX, lw=0, transform=ax.transAxes, zorder=4))
    ax.text(0.028, 0.66, "PITCH", transform=ax.transAxes, ha="left", va="center", color=INK,
            fontsize=27, fontfamily=F_COND, fontweight="bold")
    ax.text(0.118, 0.66, "VISION", transform=ax.transAxes, ha="left", va="center", color=GOLD,
            fontsize=27, fontfamily=F_COND, fontweight="bold")
    ax.text(0.028, 0.30, "LIVE MATCH INTELLIGENCE", transform=ax.transAxes, ha="left", va="center",
            color=INK2, fontsize=11, fontfamily=F_COND, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.214, 0.20), 0.052, 0.24, boxstyle="round,pad=0,rounding_size=0.08",
                 linewidth=0, facecolor=RED, mutation_aspect=HH / HW * (0.052 / 0.24),
                 transform=ax.transAxes, zorder=4))
    ax.add_patch(Circle((0.226, 0.32), 0.010, facecolor=INK, lw=0, transform=ax.transAxes, zorder=5))
    ax.text(0.243, 0.32, "LIVE", transform=ax.transAxes, ha="center", va="center", color=INK,
            fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=5)
    ax.text(0.393, 0.50, ARG.upper(), transform=ax.transAxes, ha="right", va="center", color=INK,
            fontsize=23, fontfamily=F_COND, fontweight="bold")
    _crest(ax, 0.418, 0.50, 0.013, CYAN_HEX)
    ax.add_patch(FancyBboxPatch((0.452, 0.16), 0.096, 0.68, boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=1.2, edgecolor=HAIR_HI, facecolor=BASE, mutation_aspect=HH / HW * (0.096 / 0.68),
                 transform=ax.transAxes, zorder=4))
    ax.text(0.500, 0.50, "{} - {}".format(sc_h, sc_a), transform=ax.transAxes, ha="center", va="center",
            color=INK, fontsize=34, fontfamily=F_NUM, fontweight="bold", zorder=6)
    _crest(ax, 0.582, 0.50, 0.013, ORANGE_HEX)
    ax.text(0.607, 0.50, ALG.upper(), transform=ax.transAxes, ha="left", va="center", color=INK,
            fontsize=23, fontfamily=F_COND, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.70, 0.24), 0.075, 0.52, boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=0, facecolor=GREEN, mutation_aspect=HH / HW * (0.075 / 0.52),
                 transform=ax.transAxes, zorder=4))
    ax.text(0.7375, 0.52, "{}'".format(ti), transform=ax.transAxes, ha="center", va="center",
            color="#06210f", fontsize=24, fontfamily=F_NUM, fontweight="bold", zorder=5)
    ax.text(0.7375, 0.18, "SECOND HALF" if ti > 45 else "FIRST HALF", transform=ax.transAxes,
            ha="center", va="center", color=INK2, fontsize=8.5, fontfamily=F_COND, fontweight="bold", zorder=5)
    pm = m["pre_match"]
    ax.text(0.815, 0.66, "PRE-MATCH CALL", transform=ax.transAxes, ha="left", va="center",
            color=INK2, fontsize=9.5, fontfamily=F_COND, fontweight="bold")
    call_home = pm["p_h"] >= pm["p_a"]
    ax.text(0.815, 0.40, "{} WIN".format((ARG if call_home else ALG).upper()), transform=ax.transAxes,
            ha="left", va="center", color=CYAN_HEX if call_home else ORANGE_HEX, fontsize=15,
            fontfamily=F_COND, fontweight="bold")
    correct = (m["final_h"] > m["final_a"]) == call_home and m["final_h"] != m["final_a"]
    badge_col = GREEN if correct else INK2
    ax.add_patch(FancyBboxPatch((0.905, 0.30), 0.066, 0.40, boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=0, facecolor=badge_col, mutation_aspect=HH / HW * (0.066 / 0.40),
                 transform=ax.transAxes, zorder=4))
    ax.text(0.938, 0.50, "CORRECT" if correct else "LIVE", transform=ax.transAxes, ha="center",
            va="center", color="#06210f", fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=5)


def _draw_winprob(ax, d, m, ti):
    _reset(ax)
    _card(ax, "Live Win Probability", accent=GOLD, strip_h=0.26,
          value_right="MODEL: ELO + LIVE SCORE", value_color=INK2)
    ARG, ALG = m["home"], m["away"]
    ix, iy, iw, ih = (0.034, 0.03, 0.932, 0.69)
    ph, pdr, pa = float(m["wp_home"][ti]), float(m["wp_draw"][ti]), float(m["wp_away"][ti])
    pxg = float(m["wp_xg"][ti])
    glab = iw * 0.16; bx = ix + glab; bw = iw - 2 * glab
    by = iy + ih * 0.34; bh = ih * 0.40
    MINW = bw * 0.045
    raw = [ph, pdr, pa]
    ws = [max(r * bw, MINW) if r > 0 else 0 for r in raw]
    ws = [w * (bw / sum(ws)) for w in ws]
    cols = [CYAN_HEX, "#7e8ab0", ORANGE_HEX]
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=0, facecolor="#0c1322", mutation_aspect=bh / bw, transform=ax.transAxes, zorder=3))
    xacc = bx; centers = []
    for i in range(3):
        w = ws[i]
        ax.add_patch(Rectangle((xacc, by + 0.015), w, bh - 0.03, facecolor=cols[i], lw=0,
                     transform=ax.transAxes, zorder=4))
        centers.append(xacc + w / 2); xacc += w
        if i < 2:
            ax.plot([xacc, xacc], [by + 0.02, by + bh - 0.02], color=BASE, lw=2.0,
                    transform=ax.transAxes, zorder=5)
    ax.text(bx + ws[0] * 0.5, by + bh * 0.52, _pct_label(ph), transform=ax.transAxes, ha="center",
            va="center", color="#06121f", fontsize=19, fontfamily=F_NUM, fontweight="bold", zorder=6)
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=1.4, edgecolor=HAIR_HI, facecolor="none", mutation_aspect=bh / bw,
                 transform=ax.transAxes, zorder=7))
    ax.text(bx - iw * 0.012, by + bh / 2, ARG.upper(), transform=ax.transAxes, ha="right", va="center",
            color=CYAN_HEX, fontsize=13, fontfamily=F_COND, fontweight="bold")
    ax.text(bx + bw + iw * 0.012, by + bh / 2, ALG.upper(), transform=ax.transAxes, ha="left", va="center",
            color=ORANGE_HEX, fontsize=13, fontfamily=F_COND, fontweight="bold")
    if abs(centers[2] - centers[1]) < iw * 0.10:
        cmid = (centers[1] + centers[2]) / 2
        ax.text(cmid, by - 0.16, "DRAW {}   ·   {} {}".format(_pct_label(pdr), ALG.upper(), _pct_label(pa)),
                transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=9.5,
                fontfamily=F_COND, fontweight="bold")
    else:
        ax.text(centers[1], by - 0.16, "DRAW {}".format(_pct_label(pdr)), transform=ax.transAxes,
                ha="center", va="top", color=INK2, fontsize=9.5, fontfamily=F_COND, fontweight="bold")
        ax.text(centers[2], by - 0.16, "{} {}".format(ALG.upper(), _pct_label(pa)), transform=ax.transAxes,
                ha="center", va="top", color=ORANGE_HEX, fontsize=9.5, fontfamily=F_COND, fontweight="bold")
    xgx = bx + pxg * bw
    ax.plot([xgx, xgx], [by + bh + 0.02, by + bh + 0.30], color=GOLD, lw=2.6, transform=ax.transAxes, zorder=8)
    ax.add_patch(RegularPolygon((xgx, by + bh + 0.04), numVertices=3, radius=0.05, orientation=0.0,
                 facecolor=GOLD, lw=0, transform=ax.transAxes, zorder=9))
    ax.text(xgx, by + bh + 0.36, "xG-DESERVED  {:.0f}%".format(pxg * 100), transform=ax.transAxes,
            ha="center", va="bottom", color=GOLD, fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=9)


def _draw_broadcast(card_ax, sub, d):
    _reset(card_ax)
    _card(card_ax, "Broadcast Vision", accent=CYAN_HEX,
          value_right="{} PLAYERS  ·  {}% CONF".format(len(d.get("tracks", [])), int(d.get("conf", 0) * 100)),
          value_color=CYAN_HEX)
    sub.clear(); sub.set_xticks([]); sub.set_yticks([])
    img = cv2.imread(d["fp"]) if d.get("fp") else None
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sub.imshow(img, aspect="auto", interpolation="bilinear")
        H, W = img.shape[:2]; bl = 0.06
        for (cxf, cyf, dx, dy) in [(0.02, 0.04, 1, 1), (0.98, 0.04, -1, 1), (0.02, 0.96, 1, -1), (0.98, 0.96, -1, -1)]:
            sub.plot([cxf * W, (cxf + dx * bl) * W], [cyf * H, cyf * H], color=GOLD, lw=2.2, transform=sub.transAxes)
            sub.plot([cxf * W, cxf * W], [cyf * H, (cyf + dy * bl) * H], color=GOLD, lw=2.2, transform=sub.transAxes)
    else:
        sub.set_facecolor("#0c1322")
    for s in sub.spines.values():
        s.set_color(CYAN_HEX); s.set_linewidth(1.6)


def _draw_topdown(card_ax, sp, d, team_rgb):
    _reset(card_ax)
    _card(card_ax, "Team Shape", accent=GOLD, value_right="TOP-DOWN  ·  HULL", value_color=INK2)
    sp.clear()
    blank = d.get("note") == "no pitch view"
    T.mpl_pitch(sp, line="#cfe0ff", face="#0e2a18", lw=1.2, line_zorder=1)
    if blank:
        lbl = "NO PITCH VIEW · graphic/replay" if d.get("cam") == "other" else "NO PITCH VIEW"
        sp.text(PL / 2, PW / 2, lbl, ha="center", va="center", color=AMBER, fontsize=12,
                fontfamily=F_COND, fontweight="bold")
        return
    tracks = d.get("tracks", []) or []
    c_home, c_away = team_rgb[0], team_rgb[1]
    T.hull(sp, [(t[0], _fy(t[1])) for t in tracks if t[2] == 0], c_home, alpha=0.16, lw=2.0)
    T.hull(sp, [(t[0], _fy(t[1])) for t in tracks if t[2] == 1], c_away, alpha=0.16, lw=2.0)
    for t in tracks:
        col = c_home if t[2] == 0 else c_away
        sp.add_patch(Circle((t[0], _fy(t[1])), 1.6, facecolor=col, ec="white", lw=0.9,
                     alpha=float(t[3]), zorder=6))
    ball = d.get("ball")
    if ball is not None:
        sp.add_patch(Circle((ball[0], _fy(ball[1])), 1.3, facecolor=GOLD, ec="#06210f", lw=1.0, zorder=7))
    if d.get("est"):
        sp.text(PL / 2, 2, "CLOSE-UP · holding last shape (estimated)", ha="center", va="bottom",
                color=AMBER, fontsize=8.5, fontfamily=F_COND, fontweight="bold")


def _draw_control(card_ax, pc, d, team_rgb):
    _reset(card_ax)
    _card(card_ax, "Pitch Control", accent=ORANGE_HEX, value_right="VORONOI OWNERSHIP", value_color=INK2)
    pc.clear()
    blank = d.get("note") == "no pitch view"
    tracks = d.get("tracks", []) or []
    c_home, c_away = team_rgb[0], team_rgb[1]
    T.mpl_pitch(pc, line="#eaf2ff", face="#0e1626", lw=1.3, line_zorder=3)
    if blank or len(tracks) < 2:
        if blank:
            lbl = "NO PITCH VIEW · graphic/replay" if d.get("cam") == "other" else "NO PITCH VIEW"
            pc.text(PL / 2, PW / 2, lbl, ha="center", va="center", color=AMBER, fontsize=11,
                    fontfamily=F_COND, fontweight="bold")
        return
    th, ta, share = T.voronoi_regions(tracks)
    for poly in th:
        pc.fill(poly[:, 0], poly[:, 1], color=c_home, alpha=0.42, ec="#0e1626", lw=0.6, zorder=2)
    for poly in ta:
        pc.fill(poly[:, 0], poly[:, 1], color=c_away, alpha=0.42, ec="#0e1626", lw=0.6, zorder=2)
    ball = d.get("ball")
    if ball is not None:
        pc.add_patch(Circle((ball[0], _fy(ball[1])), 1.2, facecolor=GOLD, ec="#06210f", lw=1.0, zorder=7))
    pc.text(18, PW - 3, "ARG {:.0f}%".format(share * 100), ha="center", va="top", color="#06121f",
            fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=8,
            path_effects=[pe.withStroke(linewidth=2.4, foreground=CYAN_HEX)])
    pc.text(PL - 18, PW - 3, "{:.0f}% ALG".format((1 - share) * 100), ha="center", va="top", color="#1a0a02",
            fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=8,
            path_effects=[pe.withStroke(linewidth=2.4, foreground=ORANGE_HEX)])


def _draw_xg(card_ax, xr, d, m, ti):
    _reset(card_ax)
    xg_h_end, xg_a_end = float(m["xg_h"][ti]), float(m["xg_a"][ti])
    _card(card_ax, "Expected Goals Race", accent=GOLD,
          value_right="{:.1f}  -  {:.1f}".format(xg_h_end, xg_a_end), value_color=INK)
    xr.clear(); xr.set_facecolor("none")
    mins = m["wp_mins"][:ti + 1]; xgh = m["xg_h"][:ti + 1]; xga = m["xg_a"][:ti + 1]
    ymax = max(float(xgh.max()), float(xga.max()), 1.0) * 1.18
    for gyv in np.arange(0.5, ymax, 0.5):
        xr.plot([0, ti], [gyv, gyv], color=HAIR, lw=0.7, zorder=1)
        xr.text(-1.5, gyv, "{:.1f}".format(gyv), ha="right", va="center", color=INK3, fontsize=8,
                fontfamily=F_MONO, zorder=2)
    xr.fill_between(mins, 0, xgh, color=CYAN_HEX, alpha=0.18, step="post", zorder=2)
    xr.fill_between(mins, 0, xga, color=ORANGE_HEX, alpha=0.16, step="post", zorder=2)
    xr.step(mins, xgh, where="post", color=CYAN_HEX, lw=2.6, zorder=4)
    xr.step(mins, xga, where="post", color=ORANGE_HEX, lw=2.6, zorder=4)
    for s in m["shots"]:
        if s["goal"] and s["min"] <= ti:
            yv = float(m["xg_h"][s["min"]]) if s["is_home"] else float(m["xg_a"][s["min"]])
            xr.scatter([s["min"]], [yv], marker="*", s=320, color=GOLD, edgecolor="#06210f",
                       linewidth=1.0, zorder=6)
    xr.set_xlim(-1.5, ti + 1); xr.set_ylim(0, ymax)
    xr.set_xticks([]); xr.set_yticks([])
    for s in xr.spines.values():
        s.set_visible(False)
    xr.plot([0, ti], [0, 0], color=HAIR_HI, lw=1.2, zorder=3)
    for xm in [0, 15, 30, 45, 60, 75, 90]:
        if xm <= ti:
            xr.text(xm, -ymax * 0.06, "{}'".format(xm), ha="center", va="top", color=INK3,
                    fontsize=8, fontfamily=F_MONO)
    xr.text(ti, float(xgh[-1]), " {:.1f}".format(xg_h_end), ha="left", va="center", color=CYAN_HEX,
            fontsize=11, fontfamily=F_NUM, fontweight="bold", zorder=7)
    xr.text(ti, float(xga[-1]), " {:.1f}".format(xg_a_end), ha="left", va="center", color=ORANGE_HEX,
            fontsize=11, fontfamily=F_NUM, fontweight="bold", zorder=7)


def _draw_ticker(ax, d, m, ti, team_rgb):
    _reset(ax)
    _card(ax, "Match Ticker", accent=AMBER, value_right="LATEST", value_color=INK2)
    ix, iy, iw, ih = INNER
    c_home, c_away = team_rgb[0], team_rgb[1]
    evs = [e for e in m["events"] if e["min"] <= ti and not (e["type"] == "Substitution" and not e["player"])][::-1]
    TYPE_LAB = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}
    TYPE_COL = {"Card": GOLD, "Substitution": GREEN, "VAR": AMBER}
    n = min(len(evs), 6)
    if n == 0:
        return
    rh = ih / n
    for i in range(n):
        e = evs[i]; ry = iy + ih - (i + 1) * rh; rx = ix; newest = (i == 0)
        if newest:
            ax.add_patch(FancyBboxPatch((rx, ry + rh * 0.06), iw, rh * 0.88,
                         boxstyle="round,pad=0,rounding_size=0.03", linewidth=1.2, edgecolor=AMBER,
                         facecolor=PANEL_HI, mutation_aspect=(rh * 0.88) / iw, transform=ax.transAxes, zorder=4))
        tcol = TYPE_COL.get(e["type"]) or (c_home if e["is_home"] else c_away)
        ax.text(rx + 0.03, ry + rh / 2, "{}'".format(e["min"]), transform=ax.transAxes, ha="left",
                va="center", color=INK, fontsize=13, fontfamily=F_NUM, fontweight="bold", zorder=6)
        ax.add_patch(FancyBboxPatch((rx + 0.11, ry + rh * 0.30), 0.135, rh * 0.40,
                     boxstyle="round,pad=0,rounding_size=0.03", linewidth=0, facecolor=tcol,
                     mutation_aspect=(rh * 0.40) / 0.135, transform=ax.transAxes, zorder=6))
        ax.text(rx + 0.1775, ry + rh / 2, TYPE_LAB.get(e["type"], e["type"]), transform=ax.transAxes,
                ha="center", va="center", color="#06121f", fontsize=9, fontfamily=F_COND, fontweight="bold", zorder=7)
        detail = e["player"]
        if e["type"] == "Goal" and e.get("score"):
            detail = "{}  ({}-{})".format(e["player"], e["score"][0], e["score"][1])
        elif e["type"] == "VAR" and e.get("note"):
            detail = "{} - {}".format(e["player"], e["note"])
        ax.text(rx + 0.27, ry + rh / 2, detail, transform=ax.transAxes, ha="left", va="center",
                color=INK if newest else INK2, fontsize=10.5, fontfamily=F_COND,
                fontweight="bold" if newest else "normal", zorder=6)
        if i < n - 1 and not newest:
            ax.plot([rx + 0.02, rx + iw - 0.02], [ry, ry], color=HAIR, lw=0.7, transform=ax.transAxes, zorder=3)


def _draw_ratings(ax, d, m, team_rgb):
    _reset(ax)
    _card(ax, "Player Ratings", accent=GOLD, value_right="PLAYER OF THE MATCH", value_color=GOLD)
    ix, iy, iw, ih = INNER
    c_home, c_away = team_rgb[0], team_rgb[1]
    rats = m["ratings"][:5]
    n = len(rats)
    if n == 0:
        return
    rh = ih / n
    for i, r in enumerate(rats):
        ry = iy + ih - (i + 1) * rh; rx = ix
        col = c_home if r["is_home"] else c_away
        ax.text(rx, ry + rh * 0.62, r["name"].upper(), transform=ax.transAxes, ha="left", va="center",
                color=INK, fontsize=11, fontfamily=F_COND, fontweight="bold", zorder=6)
        btw = iw - 0.16; bty = ry + rh * 0.18; bbh = rh * 0.22
        ax.add_patch(FancyBboxPatch((rx, bty), btw, bbh, boxstyle="round,pad=0,rounding_size=0.02",
                     linewidth=0, facecolor="#0c1322", mutation_aspect=bbh / btw, transform=ax.transAxes, zorder=4))
        frac = r["rating"] / 10.0
        ax.add_patch(FancyBboxPatch((rx, bty), btw * frac, bbh, boxstyle="round,pad=0,rounding_size=0.02",
                     linewidth=0, facecolor=col, mutation_aspect=bbh / (btw * frac + 1e-6),
                     transform=ax.transAxes, zorder=5))
        ax.text(rx + iw, ry + rh * 0.55, "{:.2f}".format(r["rating"]), transform=ax.transAxes, ha="right",
                va="center", color=INK, fontsize=15, fontfamily=F_NUM, fontweight="bold", zorder=6)
        if r["potm"]:
            ax.add_patch(FancyBboxPatch((rx + iw - 0.30, ry + rh * 0.55), 0.12, rh * 0.30,
                         boxstyle="round,pad=0,rounding_size=0.04", linewidth=0, facecolor=GOLD,
                         mutation_aspect=(rh * 0.30) / 0.12, transform=ax.transAxes, zorder=6))
            ax.text(rx + iw - 0.24, ry + rh * 0.70, "POTM", transform=ax.transAxes, ha="center",
                    va="center", color="#2a1d00", fontsize=8.5, fontfamily=F_COND, fontweight="bold", zorder=7)


def _draw_footer(ax):
    _reset(ax)
    foot = ("Broadcast CV ~5 m zone-grade   ·   win-prob OOS log-loss 0.82   "
            "·   World-Football-Elo predictor   ·   futbol_tech")
    ax.text(0.004, 0.5, foot, transform=ax.transAxes, ha="left", va="center", color=INK3,
            fontsize=9.5, fontfamily=F_COND)
    ax.text(0.996, 0.5, "PITCHVISION · LIVE MATCH INTELLIGENCE", transform=ax.transAxes, ha="right",
            va="center", color=INK3, fontsize=9.5, fontfamily=F_COND, fontweight="bold")


def draw_frame(axes, d, m, ti, team_rgb, label=""):
    _draw_header(axes["hdr"], d, m, ti, team_rgb)
    _draw_winprob(axes["wp"], d, m, ti)
    _draw_broadcast(axes["bcard"], axes["bsub"], d)
    _draw_topdown(axes["tcard"], axes["tsub"], d, team_rgb)
    _draw_control(axes["pcard"], axes["psub"], d, team_rgb)
    _draw_xg(axes["xcard"], axes["xsub"], d, m, ti)
    _draw_ticker(axes["ecard"], d, m, ti, team_rgb)
    _draw_ratings(axes["rcard"], d, m, team_rgb)
    _draw_footer(axes["foot"])
