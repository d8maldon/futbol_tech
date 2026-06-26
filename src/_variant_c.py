"""VARIANT C - PREMIUM SPORTS-TECH / TELEMETRY DASHBOARD
A Second-Spectrum / F1-pit-wall styled broadcast-analytics console.
Dark charcoal grid, monospaced numerals, restrained neon (team cyan/orange).
Renders 1920x1080 PNG. No ffmpeg / GPU / network needed.
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, RegularPolygon, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

import _preview_data as P

# ----------------------------------------------------------------------------- data
m = P.build_m()
d = P.build_state()
ti = P.PREVIEW_MIN

CYAN   = P.CYAN
ORANGE = P.ORANGE
ARG    = m["home"]
ALG    = m["away"]
SC_H, SC_A = int(m["sc_h"][ti]), int(m["sc_a"][ti])

# ----------------------------------------------------------------------------- palette / type
INK_BG     = "#0a0d12"   # page background, near-black blue charcoal
CARD       = "#11161d"   # card fill
CARD_HI    = "#161d26"   # inner / raised fill
HAIR       = "#222c38"   # hairline border
HAIR_HI    = "#2c3a49"
RULE       = "#1a232d"
GRID       = "#1c2530"
TXT_HI     = "#f2f6fb"   # primary text
TXT        = "#aeb9c6"   # secondary text
TXT_MUT    = "#6a7787"   # muted captions
TXT_FAINT  = "#48535f"
ACCENT     = "#39d0c8"   # restrained teal system accent (non-team)
AMBER      = "#f4b740"   # VAR / caution
GOOD       = "#5fcf80"

CYAN_HX   = "#4ec8ff"
ORANGE_HX = "#ff7333"

MONO   = "DejaVu Sans Mono"
SANS   = "Bahnschrift"
SANS2  = "DejaVu Sans"

plt.rcParams["font.family"] = SANS2
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor(INK_BG)

# global background coordinate axis (0..1920 / 0..1080)
W, H = 1920.0, 1080.0
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H)
bg.invert_yaxis(); bg.axis("off"); bg.set_facecolor(INK_BG)

# faint engineered backdrop grid
for gx in range(0, int(W) + 1, 60):
    bg.plot([gx, gx], [0, H], color="#0e131a", lw=0.6, zorder=0)
for gy in range(0, int(H) + 1, 60):
    bg.plot([0, W], [gy, gy], color="#0e131a", lw=0.6, zorder=0)


def card(x, y, w, h, title=None, kicker=None, tag=None, tag_color=ACCENT,
         pad=18, fill=CARD, header_h=34):
    """Draw a telemetry card: subtle fill, hairline border, header strip with
    a kicker title and an optional right-aligned tag chip. Returns the inner
    content rectangle (x, y, w, h) below the header."""
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7",
                       fc=fill, ec=HAIR, lw=1.2, zorder=2,
                       mutation_aspect=1)
    bg.add_patch(r)
    # accent tick on the top-left corner
    bg.add_patch(Rectangle((x, y), 3.2, header_h, fc=tag_color, ec="none",
                           zorder=4))
    inner_y = y
    if title is not None:
        # header strip baseline rule
        bg.plot([x + 14, x + w - 14], [y + header_h, y + header_h],
                color=RULE, lw=1.0, zorder=3)
        if kicker:
            bg.text(x + 16, y + 12.5, kicker, color=TXT_MUT, fontsize=7.6,
                    family=MONO, fontweight="bold", va="center", ha="left",
                    zorder=5)
        bg.text(x + 16, y + (24 if kicker else header_h / 2), title,
                color=TXT_HI, fontsize=11.5, family=SANS, fontweight="bold",
                va="center", ha="left", zorder=5)
        if tag:
            tw = 11 + len(tag) * 6.4
            bg.add_patch(FancyBboxPatch((x + w - 14 - tw, y + 8), tw, header_h - 16,
                         boxstyle="round,pad=0,rounding_size=3",
                         fc="none", ec=tag_color, lw=1.1, zorder=4))
            bg.text(x + w - 14 - tw / 2, y + header_h / 2, tag, color=tag_color,
                    fontsize=7.8, family=MONO, fontweight="bold",
                    va="center", ha="center", zorder=5)
        inner_y = y + header_h
    return (x + pad, inner_y + 12, w - 2 * pad, h - (inner_y - y) - 24)


def axes_in(rect):
    """matplotlib axes positioned over a bg pixel rect (x,y,w,h) with y-down."""
    x, y, w, h = rect
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


# =============================================================================
# HEADER LOCKUP
# =============================================================================
HX, HY, HW, HH = 28, 22, W - 56, 96
bg.add_patch(FancyBboxPatch((HX, HY), HW, HH, boxstyle="round,pad=0,rounding_size=8",
             fc=CARD, ec=HAIR, lw=1.2, zorder=2))
bg.add_patch(Rectangle((HX, HY), 3.2, HH, fc=ACCENT, ec="none", zorder=4))

# wordmark
bg.text(HX + 26, HY + 30, "PITCHWALL", color=TXT_HI, fontsize=20,
        family=SANS, fontweight="bold", va="center", ha="left", zorder=5)
bg.text(HX + 26, HY + 30, "PITCHWALL", color=ACCENT, fontsize=20,
        family=SANS, fontweight="bold", va="center", ha="left", zorder=4,
        alpha=0.0)
bg.text(HX + 178, HY + 28, "LIVE  MATCH  TELEMETRY", color=ACCENT, fontsize=8.5,
        family=MONO, fontweight="bold", va="center", ha="left", zorder=5)
bg.text(HX + 26, HY + 64, "BROADCAST COMPUTER-VISION  /  WIN-PROBABILITY ENGINE",
        color=TXT_MUT, fontsize=8.2, family=MONO, va="center", ha="left", zorder=5)
# vertical divider
bg.plot([HX + 470, HX + 470], [HY + 16, HY + HH - 16], color=HAIR_HI, lw=1.1, zorder=4)

# center scoreboard
cx = HX + HW / 2
# LIVE pill
bg.add_patch(Circle((cx - 86, HY + 24), 4.2, fc="#ff4040", ec="none", zorder=6))
bg.text(cx - 76, HY + 24, "LIVE", color="#ff5c5c", fontsize=9, family=MONO,
        fontweight="bold", va="center", ha="left", zorder=6)
bg.add_patch(FancyBboxPatch((cx + 36, HY + 16), 56, 18,
             boxstyle="round,pad=0,rounding_size=3", fc=CARD_HI, ec=HAIR_HI,
             lw=1.0, zorder=5))
bg.text(cx + 64, HY + 25, f"{ti}'", color=TXT_HI, fontsize=10, family=MONO,
        fontweight="bold", va="center", ha="center", zorder=6)

# team names + score
bg.text(cx - 158, HY + 60, ARG.upper(), color=TXT_HI, fontsize=23, family=SANS,
        fontweight="bold", va="center", ha="right", zorder=6)
bg.add_patch(Rectangle((cx - 150, HY + 44), 7, 32, fc=CYAN_HX, ec="none", zorder=6))
bg.text(cx - 14, HY + 60, f"{SC_H}", color=CYAN_HX, fontsize=34, family=MONO,
        fontweight="bold", va="center", ha="right", zorder=6)
bg.text(cx, HY + 59, "–", color=TXT, fontsize=22, family=MONO,
        fontweight="bold", va="center", ha="center", zorder=6)
bg.text(cx + 14, HY + 60, f"{SC_A}", color=ORANGE_HX, fontsize=34, family=MONO,
        fontweight="bold", va="center", ha="left", zorder=6)
bg.add_patch(Rectangle((cx + 143, HY + 44), 7, 32, fc=ORANGE_HX, ec="none", zorder=6))
bg.text(cx + 158, HY + 60, ALG.upper(), color=TXT_HI, fontsize=23, family=SANS,
        fontweight="bold", va="center", ha="left", zorder=6)

# right metrics strip
def hmetric(xr, label, value, vcolor=TXT_HI):
    bg.text(xr, HY + 32, label, color=TXT_MUT, fontsize=7.6, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=6)
    bg.text(xr, HY + 56, value, color=vcolor, fontsize=15, family=MONO,
            fontweight="bold", va="center", ha="left", zorder=6)

rx0 = HX + HW - 360
bg.plot([rx0 - 24, rx0 - 24], [HY + 16, HY + HH - 16], color=HAIR_HI, lw=1.1, zorder=4)
hmetric(rx0,        "XG  ARG", f"{m['xg_h'][ti]:.2f}", CYAN_HX)
hmetric(rx0 + 92,   "XG  ALG", f"{m['xg_a'][ti]:.2f}", ORANGE_HX)
hmetric(rx0 + 184,  "CV CONF", f"{int(d['conf']*100)}%", ACCENT)
hmetric(rx0 + 276,  "TRACKED", f"{len(d['tracks'])}", TXT_HI)

# =============================================================================
# GRID LAYOUT (3 columns)
# =============================================================================
TOP = 134
GAP = 14
COLL_X = 28
COLL_W = 560
COLM_X = COLL_X + COLL_W + GAP
COLM_W = 660
COLR_X = COLM_X + COLM_W + GAP
COLR_W = W - 28 - COLR_X
BOT = H - 22

# ---------------------------------------------------------------- WIN PROBABILITY (top, spans L+M)
wpb = card(COLL_X, TOP, COLL_W + GAP + COLM_W, 118,
           title="WIN PROBABILITY", kicker="LIVE MODEL  ·  SCORE-AWARE",
           tag="OUTRIGHT", tag_color=ACCENT)
ax = axes_in(wpb)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ph = float(m["wp_home"][ti]); pd_ = float(m["wp_draw"][ti]); pa = float(m["wp_away"][ti])
pxg = float(m["wp_xg"][ti])
bar_y, bar_h = 0.30, 0.30
# percentages (rounded to whole, honest), with a minimum *visual* sliver so the
# bar never collapses to one flat block in a blowout.
pcts = [ph, pd_, pa]
disp = [round(v * 100) for v in pcts]
MINSLIVER = 0.012
widths = [max(v, MINSLIVER) for v in pcts]
widths = [w / sum(widths) for w in widths]
cols = [CYAN_HX, "#7a8694", ORANGE_HX]
labs = [ARG.upper(), "DRAW", ALG.upper()]
x0 = 0.0
xcenters = []
for w, col in zip(widths, cols):
    ax.add_patch(Rectangle((x0, bar_y), w, bar_h, fc=col, ec=INK_BG, lw=2.0,
                 zorder=3))
    xcenters.append(x0 + w / 2)
    x0 += w
# chips: ARG % on the left inside, ALG/DRAW small on the right
ax.text(0.0, bar_y + bar_h + 0.34, f"{disp[0]}%", color=CYAN_HX, fontsize=19,
        family=MONO, fontweight="bold", va="center", ha="left")
ax.text(1.0, bar_y + bar_h + 0.34, f"{disp[2]}%", color=ORANGE_HX, fontsize=14,
        family=MONO, fontweight="bold", va="center", ha="right")
# small draw read-out near right, left of ALG
ax.text(1.0, bar_y - 0.30, f"DRAW {disp[1]}%", color=TXT_MUT, fontsize=8,
        family=MONO, fontweight="bold", va="center", ha="right")
# team labels under bar ends
ax.text(0.0, bar_y - 0.30, ARG.upper(), color=TXT, fontsize=9, family=MONO,
        fontweight="bold", va="center", ha="left")
ax.text(0.46, bar_y - 0.30, ALG.upper(), color=TXT, fontsize=9, family=MONO,
        fontweight="bold", va="center", ha="center")
# xG-deserved marker (clamp label so it never clips the card edge)
mx = pxg
ax.plot([mx, mx], [bar_y - 0.10, bar_y + bar_h + 0.12], color=TXT_HI, lw=2.4,
        zorder=6, solid_capstyle="round")
ax.add_patch(RegularPolygon((mx, bar_y + bar_h + 0.14), 3, radius=0.016,
             orientation=np.pi, fc=TXT_HI, ec="none", zorder=6))
lab_align = "right" if mx > 0.5 else "left"
lab_x = mx - 0.012 if mx > 0.5 else mx + 0.012
ax.text(lab_x, bar_y + bar_h + 0.42, f"xG-DESERVED  {pxg*100:.0f}%",
        color=TXT_HI, fontsize=8.5, family=MONO, fontweight="bold",
        va="center", ha=lab_align)
ax.set_xlim(-0.004, 1.004); ax.set_ylim(0, 1)

# =============================================================================
# LEFT COLUMN
# =============================================================================
LY = TOP + 118 + GAP
# ---- broadcast panel
bcast_h = 250
bcb = card(COLL_X, LY, COLL_W, bcast_h, title="BROADCAST VISION",
           kicker="PLAYER DETECTION  ·  WIDE CAM", tag="LIVE CV", tag_color=ACCENT)
ax = axes_in(bcb)
img = cv2.cvtColor(cv2.imread(d["fp"]), cv2.COLOR_BGR2RGB)
ax.imshow(img, aspect="auto", zorder=1, extent=[0, 1, 0, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
# corner framing brackets
def bracket(x, y, dx, dy):
    ax.plot([x, x+dx], [y, y], color=ACCENT, lw=1.6, zorder=5)
    ax.plot([x, x], [y, y+dy], color=ACCENT, lw=1.6, zorder=5)
for (cx_, cy_, sx, sy) in [(0.012,0.04,0.05,0.10),(0.988,0.04,-0.05,0.10),
                            (0.012,0.96,0.05,-0.10),(0.988,0.96,-0.05,-0.10)]:
    bracket(cx_, cy_, sx, sy)
# count label
n_home = sum(1 for t in d["tracks"] if t[2] == 0)
n_away = sum(1 for t in d["tracks"] if t[2] == 1)
ax.add_patch(Rectangle((0.0, 0.0), 1.0, 0.14, fc="#000000", ec="none",
             alpha=0.55, zorder=4, transform=ax.transAxes))
ax.text(0.025, 0.07, f"{len(d['tracks'])} PLAYERS DETECTED", color=TXT_HI,
        fontsize=9, family=MONO, fontweight="bold", va="center", ha="left",
        zorder=6, transform=ax.transAxes)
ax.text(0.975, 0.07, f"ARG {n_home}   ALG {n_away}", color=ACCENT,
        fontsize=9, family=MONO, fontweight="bold", va="center", ha="right",
        zorder=6, transform=ax.transAxes)

# ---- pre-match call + result
LY2 = LY + bcast_h + GAP
callh = 116
clb = card(COLL_X, LY2, COLL_W, callh, title="PRE-MATCH CALL",
           kicker="WORLD-FOOTBALL ELO  ·  RESULT", tag="CORRECT", tag_color=GOOD)
ax = axes_in(clb)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
pm = m["pre_match"]
# three probability columns
cols = [("ARG", pm["p_h"], CYAN_HX), ("DRAW", pm["p_d"], "#8893a1"),
        ("ALG", pm["p_a"], ORANGE_HX)]
bx = 0.0
ax.text(0.0, 0.92, "MODEL CALLED", color=TXT_MUT, fontsize=7.6, family=MONO,
        fontweight="bold", va="top", ha="left")
for i, (lab, val, col) in enumerate(cols):
    xc = 0.06 + i * 0.135
    ax.text(xc, 0.60, f"{val*100:.0f}", color=col, fontsize=20, family=MONO,
            fontweight="bold", va="center", ha="center")
    ax.text(xc, 0.60, "%", color=col, fontsize=9, family=MONO, va="center",
            ha="left")
    ax.text(xc, 0.20, lab, color=TXT, fontsize=8, family=MONO, fontweight="bold",
            va="center", ha="center")
# verdict block on the right
ax.add_patch(Rectangle((0.50, 0.10), 0.495, 0.80, fc=CARD_HI, ec=HAIR_HI,
             lw=1.0, zorder=2))
ax.text(0.525, 0.74, "OUR CALL", color=TXT_MUT, fontsize=7.6, family=MONO,
        fontweight="bold", va="center", ha="left")
ax.text(0.525, 0.50, "ARGENTINA", color=CYAN_HX, fontsize=15, family=SANS,
        fontweight="bold", va="center", ha="left")
ax.text(0.525, 0.22, f"FINAL  {m['final_h']}–{m['final_a']}", color=TXT,
        fontsize=10, family=MONO, fontweight="bold", va="center", ha="left")
# tick badge
ax.add_patch(Circle((0.93, 0.50), 0.075, fc="none", ec=GOOD, lw=2.0, zorder=4))
ax.plot([0.905, 0.925, 0.965], [0.50, 0.42, 0.60], color=GOOD, lw=2.4,
        zorder=5, solid_capstyle="round", solid_joinstyle="round")

# =============================================================================
# MIDDLE COLUMN  (top-down shapes + pitch control)
# =============================================================================
MY = TOP + 118 + GAP
# ---- top-down team shapes
tdh = 250
tdb = card(COLM_X, MY, COLM_W, tdh, title="FORMATION SHAPE",
           kicker="TOP-DOWN  ·  CONVEX TEAM BLOCK", tag="5M ZONE", tag_color=ACCENT)
ax = axes_in(tdb)
P.draw_pitch(ax, line="#2f3e4d", lw=1.2, face="#0d1217", alpha=0.9)
arg_pts, alg_pts = [], []
for t in d["tracks"]:
    px, py, team = t[0], t[1], int(t[2])
    yy = P.PW - py
    if team == 0: arg_pts.append((px, yy))
    else:         alg_pts.append((px, yy))
P.hull(ax, arg_pts, CYAN_HX, alpha=0.10, lw=1.8)
P.hull(ax, alg_pts, ORANGE_HX, alpha=0.10, lw=1.8)
for (px, py) in arg_pts:
    ax.add_patch(Circle((px, py), 1.5, fc=CYAN_HX, ec="#0d1217", lw=1.0, zorder=5))
for (px, py) in alg_pts:
    ax.add_patch(Circle((px, py), 1.5, fc=ORANGE_HX, ec="#0d1217", lw=1.0, zorder=5))
# ball
bxp, byp = d["ball"][0], P.PW - d["ball"][1]
ax.add_patch(Circle((bxp, byp), 1.5, fc="#ffffff", ec="#101010", lw=1.0, zorder=7))
ax.add_patch(Circle((bxp, byp), 3.0, fc="none", ec="#ffffff", lw=1.0,
             alpha=0.6, zorder=6))
# attack direction arrow + legend baked top
ax.annotate("", xy=(95, P.PW + 1.5), xytext=(80, P.PW + 1.5),
            arrowprops=dict(arrowstyle="-|>", color=TXT_MUT, lw=1.4),
            annotation_clip=False)

# ---- pitch control map
MY2 = MY + tdh + GAP
pch = BOT - MY2
pcb = card(COLM_X, MY2, COLM_W, pch, title="PITCH CONTROL",
           kicker="OWNERSHIP SURFACE  ·  PROXIMITY", tag="LIVE", tag_color=ACCENT)
ax = axes_in(pcb)
# compute proximity softmax control grid
gx = np.linspace(0, P.PL, 160)
gy = np.linspace(0, P.PW, 104)
GX, GY = np.meshgrid(gx, gy)
ctrl_h = np.zeros_like(GX)
ctrl_a = np.zeros_like(GX)
for t in d["tracks"]:
    px, py, team = t[0], t[1], int(t[2])
    yy = P.PW - py
    dist = np.sqrt((GX - px) ** 2 + (GY - yy) ** 2)
    infl = np.exp(-dist / 6.0)
    if team == 0: ctrl_h += infl
    else:         ctrl_a += infl
own = ctrl_h / (ctrl_h + ctrl_a + 1e-9)   # 1 = ARG, 0 = ALG
from matplotlib.colors import LinearSegmentedColormap
# muted telemetry tints (blended toward charcoal) so the surface reads as an
# overlay, not solid neon blocks
cmap = LinearSegmentedColormap.from_list("ctrl",
        ["#9c4d2a", "#5c3322", "#241813", "#0d1116", "#13212c", "#1d4257", "#2f8fc4"])
ax.imshow(own, extent=[0, P.PL, 0, P.PW], origin="lower", cmap=cmap,
          vmin=0.10, vmax=0.90, aspect="auto", zorder=1, interpolation="bilinear")
P.draw_pitch(ax, line="#3a4b5c", lw=1.1, face="none", alpha=0.85)
ax.set_facecolor("none")
# small footer share readout inside
sh_h = float(own.mean())
ax.text(2, P.PW - 3, f"ARG {sh_h*100:.0f}%", color=CYAN_HX, fontsize=9,
        family=MONO, fontweight="bold", va="center", ha="left", zorder=6,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#0a0d12")])
ax.text(P.PL - 2, P.PW - 3, f"{(1-sh_h)*100:.0f}% ALG", color=ORANGE_HX,
        fontsize=9, family=MONO, fontweight="bold", va="center", ha="right",
        zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="#0a0d12")])

# =============================================================================
# RIGHT COLUMN (xG race + events + ratings)
# =============================================================================
RY = TOP + 118 + GAP
# ---- xG race
xgh = 232
xgb = card(COLR_X, RY, COLR_W, xgh, title="EXPECTED GOALS RACE",
           kicker="CUMULATIVE xG  ·  0–{}'".format(ti), tag="xG", tag_color=ACCENT)
ax = axes_in(xgb)
mins = m["wp_mins"]
mask = mins <= ti
xh = m["xg_h"][mask]; xa = m["xg_a"][mask]; mm = mins[mask]
ymax = max(xh.max(), xa.max(), 0.6) * 1.18
# axis frame
ax.set_xlim(0, ti); ax.set_ylim(0, ymax)
# horizontal grid
for yv in np.arange(0, ymax, 0.5):
    ax.plot([0, ti], [yv, yv], color=GRID, lw=0.8, zorder=1)
    ax.text(-1.0, yv, f"{yv:.1f}", color=TXT_FAINT, fontsize=7, family=MONO,
            va="center", ha="right", zorder=2)
for xv in [0, 15, 30, 45, 60]:
    if xv <= ti:
        ax.plot([xv, xv], [0, ymax], color=GRID, lw=0.7, zorder=1)
        ax.text(xv, -ymax*0.05, f"{xv}'", color=TXT_FAINT, fontsize=7,
                family=MONO, va="top", ha="center", zorder=2)
# step area fills
ax.fill_between(mm, xh, step="post", color=CYAN_HX, alpha=0.14, zorder=2)
ax.step(mm, xh, where="post", color=CYAN_HX, lw=2.4, zorder=4)
ax.step(mm, xa, where="post", color=ORANGE_HX, lw=2.4, zorder=4)
ax.fill_between(mm, xa, step="post", color=ORANGE_HX, alpha=0.10, zorder=2)
# goal stars
for s in m["shots"]:
    if s["goal"] and s["min"] <= ti:
        yv = m["xg_h"][s["min"]] if s["is_home"] else m["xg_a"][s["min"]]
        col = CYAN_HX if s["is_home"] else ORANGE_HX
        ax.scatter([s["min"]], [yv], marker="*", s=190, color=col,
                   edgecolors=INK_BG, linewidths=1.2, zorder=6)
# end value labels
ax.text(ti + 0.5, xh[-1], f"{xh[-1]:.2f}", color=CYAN_HX, fontsize=9.5,
        family=MONO, fontweight="bold", va="center", ha="left", zorder=6)
ax.text(ti + 0.5, xa[-1], f"{xa[-1]:.2f}", color=ORANGE_HX, fontsize=9.5,
        family=MONO, fontweight="bold", va="center", ha="left", zorder=6)
ax.set_xlim(-3.5, ti + 6); ax.set_ylim(-ymax*0.10, ymax)

# ---- events ticker
RY2 = RY + xgh + GAP
evh = 250
evb = card(COLR_X, RY2, COLR_W, evh, title="MATCH EVENTS",
           kicker="TIMELINE  ·  LATEST FIRST", tag="FEED", tag_color=ACCENT)
ax = axes_in(evb)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
evs = [e for e in m["events"] if e["min"] <= ti][::-1]
EV_COL = {"Goal": None, "Card": AMBER, "Substitution": GOOD, "VAR": AMBER}
EV_ABBR = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}
n = len(evs)
row_h = 1.0 / max(n, 1)
for i, e in enumerate(evs):
    yc = 1.0 - (i + 0.5) * row_h
    newest = (i == 0)
    team_col = CYAN_HX if e["is_home"] else ORANGE_HX
    chip_col = EV_COL[e["type"]] if EV_COL[e["type"]] else team_col
    if newest:
        ax.add_patch(Rectangle((0.0, 1.0 - row_h*(i+1)+0.01), 1.0, row_h-0.02,
                     fc="#18222c", ec="none", zorder=1))
    # team accent bar
    ax.add_patch(Rectangle((0.0, 1.0 - row_h*(i+1)+0.02), 0.006, row_h-0.04,
                 fc=team_col, ec="none", zorder=3))
    # minute
    ax.text(0.035, yc, f"{e['min']:>2}'", color=TXT_HI if newest else TXT,
            fontsize=11, family=MONO, fontweight="bold", va="center", ha="left",
            zorder=4)
    # type chip
    ab = EV_ABBR[e["type"]]
    cw = 0.085
    ax.add_patch(FancyBboxPatch((0.105, yc - row_h*0.26), cw, row_h*0.52,
                 boxstyle="round,pad=0,rounding_size=0.02", fc="none",
                 ec=chip_col, lw=1.1, zorder=3))
    ax.text(0.105 + cw/2, yc, ab, color=chip_col, fontsize=7.6, family=MONO,
            fontweight="bold", va="center", ha="center", zorder=4)
    # player
    ax.text(0.215, yc, e["player"], color=TXT_HI if newest else TXT,
            fontsize=10.5, family=SANS, fontweight="bold" if newest else "normal",
            va="center", ha="left", zorder=4)
    # detail right
    if e["type"] == "Goal":
        det = f"{e['score'][0]}–{e['score'][1]}"
    elif e["type"] == "VAR":
        det = "OFFSIDE"
    elif e["type"] == "Card":
        det = "YELLOW"
    else:
        det = "ON"
    ax.text(0.985, yc, det, color=chip_col if e["type"] != "Goal" else team_col,
            fontsize=9, family=MONO, fontweight="bold", va="center", ha="right",
            zorder=4)
    if i < n - 1:
        yr = 1.0 - row_h * (i + 1)
        ax.plot([0.02, 0.98], [yr, yr], color=RULE, lw=0.8, zorder=2)

# ---- ratings
RY3 = RY2 + evh + GAP
rth = BOT - RY3
rtb = card(COLR_X, RY3, COLR_W, rth, title="TOP PERFORMERS",
           kicker="PLAYER RATING  ·  0–10", tag="POTM", tag_color=AMBER)
ax = axes_in(rtb)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
rs = m["ratings"]
n = len(rs)
row_h = 1.0 / n
RT_LO, RT_HI = 6.0, 10.0   # baseline-zoomed scale so 7..9 reads clearly
bx0, bx1 = 0.0, 0.74
for i, r in enumerate(rs):
    yc = 1.0 - (i + 0.5) * row_h
    col = CYAN_HX if r["is_home"] else ORANGE_HX
    # name (top of row)
    ax.text(0.0, yc + row_h*0.22, r["name"], color=TXT_HI, fontsize=10.5,
            family=SANS, fontweight="bold", va="center", ha="left", zorder=4)
    # POTM badge inline after the name
    if r["potm"]:
        badge_x = 0.012 * len(r["name"]) + 0.135
        bw = 0.115
        ax.add_patch(FancyBboxPatch((badge_x, yc + row_h*0.10), bw, row_h*0.24,
                     boxstyle="round,pad=0,rounding_size=0.015", fc=AMBER,
                     ec="none", zorder=5))
        ax.text(badge_x + bw/2, yc + row_h*0.22, "POTM", color="#1a1205",
                fontsize=7.0, family=MONO, fontweight="bold", va="center",
                ha="center", zorder=6)
    # rating bar track (lower in the row)
    by = yc - row_h*0.34
    bh = row_h * 0.18
    ax.add_patch(Rectangle((bx0, by), bx1, bh, fc=CARD_HI, ec=HAIR,
                 lw=0.8, zorder=2))
    frac = np.clip((r["rating"] - RT_LO) / (RT_HI - RT_LO), 0.04, 1.0)
    ax.add_patch(Rectangle((bx0, by), bx1*frac, bh, fc=col, ec="none", zorder=3))
    # value
    ax.text(0.995, yc, f"{r['rating']:.2f}", color=TXT_HI, fontsize=14,
            family=MONO, fontweight="bold", va="center", ha="right", zorder=4)

# =============================================================================
# FOOTER  (single method/credit line carrying all jargon)
# =============================================================================
FY = H - 16
bg.plot([28, W - 28], [H - 14 - 6, H - 14 - 6], color=RULE, lw=1.0, zorder=3)
bg.text(28, H - 8, "PITCHWALL", color=ACCENT, fontsize=8.5, family=SANS,
        fontweight="bold", va="bottom", ha="left", zorder=5)
foot = ("Broadcast CV ~5 m zone-grade  ·  win-prob OOS log-loss 0.82  ·  "
        "World-Football-Elo predictor  ·  pitch-control proximity softmax  ·  futbol_tech")
bg.text(W - 28, H - 8, foot, color=TXT_MUT, fontsize=8, family=MONO,
        va="bottom", ha="right", zorder=5)

# ----------------------------------------------------------------------------- save
OUT = os.path.join(P.ROOT, "_frames_review", "variant_c.png")
fig.savefig(OUT, dpi=100, facecolor=INK_BG)
print("saved", OUT)
