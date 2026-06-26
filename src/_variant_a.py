"""VARIANT A - BROADCAST TV GRAPHICS dashboard redesign.
Think Amazon Prime Bundesliga / TNT Sports / Sky MNF. Bold team-colour blocks,
chunky scoreboard lockup, lower-third header strips, big numerals, rich navy night base.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, RegularPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import cv2

import _preview_data as P

m = P.build_m()
d = P.build_state()
ti = P.PREVIEW_MIN

# ----------------------------------------------------------------------------
# PALETTE — deep stadium-night navy, intentional, with team colours as accents
# ----------------------------------------------------------------------------
BASE      = "#0a0f1c"      # deepest base (vignette outer)
PANEL     = "#121a2e"      # card fill
PANEL_HI  = "#18223b"      # card fill lighter
STRIP     = "#1d2942"      # lower-third header strip
HAIR      = "#2c3a5a"      # hairline border
HAIR_HI   = "#3a4c72"
INK       = "#f4f7ff"      # primary text
INK2      = "#aab6d4"      # secondary text
INK3      = "#6e7 da0"[:-1] if False else "#6e7da0"  # muted
GOLD      = "#ffcf4d"      # accent / POTM / wordmark
AMBER     = "#ffb020"      # VAR
GREEN     = "#3ddc84"      # correct / positive
RED       = "#ff5a5a"

CYAN   = P.CYAN     # ARG
ORANGE = P.ORANGE   # ALG
CYAN_HEX   = "#4ec7ff"
ORANGE_HEX = "#ff7333"

F_COND = "Bahnschrift"
F_NUM  = "DejaVu Sans"
F_MONO = "DejaVu Sans Mono"

ARG = m["home"]; ALG = m["away"]
sc_h = int(m["sc_h"][ti]); sc_a = int(m["sc_a"][ti])

# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor(BASE)

# Background vignette via radial gradient image
gy, gx = np.mgrid[0:108, 0:192]
cx, cy = 96, 50
r = np.sqrt(((gx - cx) / 120) ** 2 + ((gy - cy) / 78) ** 2)
vig = np.clip(1 - r * 0.85, 0, 1)
bg = np.zeros((108, 192, 3))
top = np.array([0.09, 0.13, 0.22])      # subtle navy glow centre
bot = np.array([0.035, 0.05, 0.10])     # darker edge
for i in range(3):
    bg[:, :, i] = bot[i] + (top[i] - bot[i]) * vig
ax_bg = fig.add_axes([0, 0, 1, 1]); ax_bg.set_axis_off()
ax_bg.imshow(bg, extent=[0, 1, 0, 1], aspect="auto", zorder=-10, interpolation="bilinear")
ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1)

# Coordinate helper: everything in figure fraction (0..1)
def AX(x, y, w, h, z=1):
    a = fig.add_axes([x, y, w, h], zorder=z)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_axis_off()
    return a

def card(ax, kicker, x=0.0, y=0.0, w=1.0, h=1.0, accent=GOLD, strip_h=0.135,
         fill=PANEL, value_right=None, value_color=INK):
    """Draw a TV-graphics card: rounded panel + lower-third header strip."""
    # panel body
    ax.add_patch(FancyBboxPatch((x + 0.006, y + 0.006), w - 0.012, h - 0.012,
                 boxstyle="round,pad=0,rounding_size=0.022", linewidth=1.2,
                 edgecolor=HAIR, facecolor=fill, mutation_aspect=h / max(w, 1e-6),
                 transform=ax.transAxes, zorder=2))
    # header strip
    sh = strip_h
    sx, sy, sw = x + 0.006, y + h - 0.006 - sh, w - 0.012
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh,
                 boxstyle="round,pad=0,rounding_size=0.022", linewidth=0,
                 facecolor=STRIP, mutation_aspect=sh / max(sw, 1e-6),
                 transform=ax.transAxes, zorder=3))
    # mask bottom corners of strip so it sits flush
    ax.add_patch(Rectangle((sx, sy), sw, sh * 0.5, facecolor=STRIP, lw=0,
                 transform=ax.transAxes, zorder=3))
    # accent tab on left of strip
    ax.add_patch(Rectangle((sx, sy + sh * 0.12), 0.012, sh * 0.76, facecolor=accent, lw=0,
                 transform=ax.transAxes, zorder=4))
    ax.text(sx + 0.028, sy + sh * 0.5, kicker.upper(), transform=ax.transAxes,
            ha="left", va="center", color=INK, fontsize=12.5, fontfamily=F_COND,
            fontweight="bold", zorder=5,
            path_effects=[pe.withStroke(linewidth=0, foreground=BASE)])
    if value_right is not None:
        ax.text(sx + sw - 0.022, sy + sh * 0.5, value_right, transform=ax.transAxes,
                ha="right", va="center", color=value_color, fontsize=12.5,
                fontfamily=F_COND, fontweight="bold", zorder=5)
    # hairline under strip
    ax.plot([sx, sx + sw], [sy, sy], color=HAIR_HI, lw=0.8, transform=ax.transAxes, zorder=4)
    return (x + 0.034, y + 0.03, w - 0.068, h - sh - 0.05)  # inner content box

# ============================================================================
# LAYOUT GRID (figure fractions). Margins.
# ============================================================================
M = 0.012
HDR_H = 0.135
# Header band across the very top
hx, hy, hw, hh = M, 1 - M - HDR_H, 1 - 2 * M, HDR_H

# ===== HEADER LOCKUP =========================================================
axh = AX(hx, hy, hw, hh, z=5)
# header background block
axh.add_patch(FancyBboxPatch((0.004, 0.04), 0.992, 0.92,
              boxstyle="round,pad=0,rounding_size=0.05", linewidth=1.4,
              edgecolor=HAIR_HI, facecolor=PANEL_HI, mutation_aspect=hh / hw,
              transform=axh.transAxes, zorder=2))
# Left team colour block (ARG)
axh.add_patch(Rectangle((0.004, 0.04), 0.30, 0.92, facecolor=CYAN_HEX, alpha=0.0,
              transform=axh.transAxes, zorder=2))
# team colour bars left/right
axh.add_patch(Rectangle((0.004, 0.04), 0.012, 0.92, facecolor=CYAN_HEX, lw=0,
              transform=axh.transAxes, zorder=4))
axh.add_patch(Rectangle((0.984, 0.04), 0.012, 0.92, facecolor=ORANGE_HEX, lw=0,
              transform=axh.transAxes, zorder=4))

# Wordmark (left)
axh.text(0.028, 0.66, "PITCH", transform=axh.transAxes, ha="left", va="center",
         color=INK, fontsize=27, fontfamily=F_COND, fontweight="bold")
axh.text(0.028, 0.66, "PITCH", transform=axh.transAxes, ha="left", va="center",
         color=GOLD, fontsize=27, fontfamily=F_COND, fontweight="bold", alpha=0.0)
axh.text(0.118, 0.66, "VISION", transform=axh.transAxes, ha="left", va="center",
         color=GOLD, fontsize=27, fontfamily=F_COND, fontweight="bold")
axh.text(0.028, 0.30, "LIVE MATCH INTELLIGENCE", transform=axh.transAxes, ha="left",
         va="center", color=INK2, fontsize=11, fontfamily=F_COND, fontweight="bold")
# LIVE pill
axh.add_patch(FancyBboxPatch((0.214, 0.20), 0.052, 0.24,
              boxstyle="round,pad=0,rounding_size=0.08", linewidth=0,
              facecolor=RED, mutation_aspect=hh / hw * (0.052 / 0.24),
              transform=axh.transAxes, zorder=4))
axh.add_patch(Circle((0.226, 0.32), 0.010, facecolor=INK, lw=0, transform=axh.transAxes, zorder=5))
axh.text(0.243, 0.32, "LIVE", transform=axh.transAxes, ha="center", va="center",
         color=INK, fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=5)

# ----- Centre scoreboard lockup -----
def crest(ax, cxp, cyp, half, col):
    # aspect-corrected so the kit tile reads square, not squashed
    hw_ = half
    hh_ = half * (hw / hh)
    ax.add_patch(FancyBboxPatch((cxp - hw_, cyp - hh_), 2 * hw_, 2 * hh_,
                 boxstyle="round,pad=0,rounding_size=0.012", facecolor=col, ec="#ffffff",
                 lw=2.0, mutation_aspect=(hh / hw), transform=ax.transAxes, zorder=5))
    ax.add_patch(Rectangle((cxp - hw_ * 0.36, cyp - hh_), hw_ * 0.72, 2 * hh_,
                 facecolor="#ffffff", alpha=0.45, lw=0, transform=ax.transAxes, zorder=6))

# Team names + crests flanking the score
axh.text(0.393, 0.50, ARG.upper(), transform=axh.transAxes, ha="right", va="center",
         color=INK, fontsize=23, fontfamily=F_COND, fontweight="bold")
crest(axh, 0.418, 0.50, 0.013, CYAN_HEX)
# score block
axh.add_patch(FancyBboxPatch((0.452, 0.16), 0.096, 0.68,
              boxstyle="round,pad=0,rounding_size=0.06", linewidth=1.2,
              edgecolor=HAIR_HI, facecolor=BASE, mutation_aspect=hh / hw * (0.096 / 0.68),
              transform=axh.transAxes, zorder=4))
axh.text(0.500, 0.50, f"{sc_h} - {sc_a}", transform=axh.transAxes,
         ha="center", va="center", color=INK, fontsize=34, fontfamily=F_NUM,
         fontweight="bold", zorder=6)
crest(axh, 0.582, 0.50, 0.013, ORANGE_HEX)
axh.text(0.607, 0.50, ALG.upper(), transform=axh.transAxes, ha="left", va="center",
         color=INK, fontsize=23, fontfamily=F_COND, fontweight="bold")

# Minute clock (right of score)
axh.add_patch(FancyBboxPatch((0.70, 0.24), 0.075, 0.52,
              boxstyle="round,pad=0,rounding_size=0.06", linewidth=0,
              facecolor=GREEN, mutation_aspect=hh / hw * (0.075 / 0.52),
              transform=axh.transAxes, zorder=4))
axh.text(0.7375, 0.52, f"{ti}'", transform=axh.transAxes, ha="center", va="center",
         color="#06210f", fontsize=24, fontfamily=F_NUM, fontweight="bold", zorder=5)
axh.text(0.7375, 0.18, "SECOND HALF", transform=axh.transAxes, ha="center", va="center",
         color=INK2, fontsize=8.5, fontfamily=F_COND, fontweight="bold", zorder=5)

# Pre-match call result (far right of header)
axh.text(0.815, 0.66, "PRE-MATCH CALL", transform=axh.transAxes, ha="left", va="center",
         color=INK2, fontsize=9.5, fontfamily=F_COND, fontweight="bold")
axh.text(0.815, 0.40, f"{ARG.upper()} WIN", transform=axh.transAxes, ha="left", va="center",
         color=CYAN_HEX, fontsize=15, fontfamily=F_COND, fontweight="bold")
axh.add_patch(FancyBboxPatch((0.905, 0.30), 0.066, 0.40,
              boxstyle="round,pad=0,rounding_size=0.06", linewidth=0,
              facecolor=GREEN, mutation_aspect=hh / hw * (0.066 / 0.40),
              transform=axh.transAxes, zorder=4))
axh.text(0.938, 0.50, "CORRECT", transform=axh.transAxes, ha="center", va="center",
         color="#06210f", fontsize=10.5, fontfamily=F_COND, fontweight="bold", zorder=5)

# ============================================================================
# BODY GRID
# ============================================================================
body_top = hy - M
body_bot = M + 0.034            # leave footer room
GH = body_top - body_bot
# 3 columns
col_w = (1 - 2 * M - 2 * M) / 3
c1 = M
c2 = M + col_w + M
c3 = M + 2 * (col_w + M)

# ---- WIN PROBABILITY (full width, just under header) ----
WPH = 0.140
wpx, wpy = M, body_top - WPH
wpw = 1 - 2 * M
axw = AX(wpx, wpy, wpw, WPH, z=4)
inner = card(axw, "Live Win Probability", 0, 0, 1, 1, accent=GOLD, strip_h=0.26,
             value_right=f"MODEL: ELO + LIVE SCORE", value_color=INK2)
ix, iy, iw, ih = inner
ph = float(m["wp_home"][ti]); pd = float(m["wp_draw"][ti]); pa = float(m["wp_away"][ti])
pxg = float(m["wp_xg"][ti])

def pct_label(f):
    p = f * 100.0
    if p >= 99.95: return ">99.9%"
    if p < 0.05:   return "<0.1%"
    if p < 1:      return f"{p:.1f}%"
    return f"{p:.0f}%"

# bar geometry — leave gutters for ARG / ALG end labels
glab = iw * 0.16
bx = ix + glab
bw = iw - 2 * glab
by = iy + ih * 0.34
bh = ih * 0.40
MINW = bw * 0.045   # minimum visible sliver so tiny segments still render
raw = [ph, pd, pa]
ws = [max(r * bw, MINW) if r > 0 else 0 for r in raw]
scale = bw / sum(ws)
ws = [w * scale for w in ws]
cols = [CYAN_HEX, "#7e8ab0", ORANGE_HEX]
# track
axw.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
              linewidth=0, facecolor="#0c1322", mutation_aspect=bh / bw,
              transform=axw.transAxes, zorder=3))
xacc = bx
centers = []
for i in range(3):
    w = ws[i]
    axw.add_patch(Rectangle((xacc, by + 0.015), w, bh - 0.03, facecolor=cols[i],
                  lw=0, transform=axw.transAxes, zorder=4))
    centers.append(xacc + w / 2)
    if i != 1:  # thin separators between cyan/orange handled by gaps; draw white hairline
        pass
    xacc += w
    if i < 2:
        axw.plot([xacc, xacc], [by + 0.02, by + bh - 0.02], color=BASE, lw=2.0,
                 transform=axw.transAxes, zorder=5)
# big % for ARG inside its (dominant) segment
axw.text(bx + ws[0] * 0.5, by + bh * 0.52, pct_label(ph), transform=axw.transAxes,
         ha="center", va="center", color="#06121f", fontsize=19, fontfamily=F_NUM,
         fontweight="bold", zorder=6)
# rounded outline
axw.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
              linewidth=1.4, edgecolor=HAIR_HI, facecolor="none", mutation_aspect=bh / bw,
              transform=axw.transAxes, zorder=7))
# end labels (ARG / ALG) in gutters
axw.text(bx - iw * 0.012, by + bh / 2, ARG.upper(), transform=axw.transAxes, ha="right",
         va="center", color=CYAN_HEX, fontsize=13, fontfamily=F_COND, fontweight="bold")
axw.text(bx + bw + iw * 0.012, by + bh / 2, ALG.upper(), transform=axw.transAxes, ha="left",
         va="center", color=ORANGE_HEX, fontsize=13, fontfamily=F_COND, fontweight="bold")
# small caption below — draw + away values (placed apart to avoid collision when both tiny)
sep = abs(centers[2] - centers[1])
if sep < iw * 0.10:   # segments collide -> single combined caption under them
    cmid = (centers[1] + centers[2]) / 2
    axw.text(cmid, by - 0.16, f"DRAW {pct_label(pd)}   ·   {ALG.upper()} {pct_label(pa)}",
             transform=axw.transAxes, ha="right", va="top", color=INK2, fontsize=9.5,
             fontfamily=F_COND, fontweight="bold")
else:
    axw.text(centers[1], by - 0.16, f"DRAW {pct_label(pd)}", transform=axw.transAxes,
             ha="center", va="top", color=INK2, fontsize=9.5, fontfamily=F_COND, fontweight="bold")
    axw.text(centers[2], by - 0.16, f"{ALG.upper()} {pct_label(pa)}", transform=axw.transAxes,
             ha="center", va="top", color=ORANGE_HEX, fontsize=9.5, fontfamily=F_COND,
             fontweight="bold")
# xG-deserved marker — anchored on a SECOND mini-scale (0..100% across the bar)
xgx = bx + pxg * bw
axw.plot([xgx, xgx], [by + bh + 0.02, by + bh + 0.30], color=GOLD, lw=2.6,
         transform=axw.transAxes, zorder=8)
axw.add_patch(RegularPolygon((xgx, by + bh + 0.04), numVertices=3, radius=0.05,
              orientation=0.0, facecolor=GOLD, lw=0, transform=axw.transAxes, zorder=9))
axw.text(xgx, by + bh + 0.36, f"xG-DESERVED  {pxg*100:.0f}%", transform=axw.transAxes,
         ha="center", va="bottom", color=GOLD, fontsize=10.5, fontfamily=F_COND,
         fontweight="bold", zorder=9)

# ---- Remaining body area under win prob ----
b2_top = wpy - M
GH2 = b2_top - body_bot

# COLUMN 1: broadcast (top) + top-down shapes (bottom)
# COLUMN 2: pitch control (top) + xG race (bottom)
# COLUMN 3: events ticker (top) + ratings (bottom)

# heights within a column
gap = M
row1_h = GH2 * 0.52
row2_h = GH2 - row1_h - gap
r1y = b2_top - row1_h
r2y = body_bot

# ---------------- BROADCAST PANEL (c1, row1) ----------------
axb = AX(c1, r1y, col_w, row1_h, z=3)
inner = card(axb, "Broadcast Vision", 0, 0, 1, 1, accent=CYAN_HEX,
             value_right=f"{len(d['tracks'])} PLAYERS  ·  {int(d['conf']*100)}% CONF",
             value_color=CYAN_HEX)
ix, iy, iw, ih = inner
sub = fig.add_axes([c1 + col_w * ix, r1y + row1_h * iy, col_w * iw, row1_h * ih], zorder=4)
img = cv2.cvtColor(cv2.imread(d["fp"]), cv2.COLOR_BGR2RGB)
sub.imshow(img, aspect="auto", interpolation="bilinear")
sub.set_xticks([]); sub.set_yticks([])
for s in sub.spines.values():
    s.set_color(CYAN_HEX); s.set_linewidth(1.6)
# corner brackets overlay (framing)
H, W = img.shape[:2]
bl = 0.06
for (cxf, cyf, dx, dy) in [(0.02, 0.04, 1, 1), (0.98, 0.04, -1, 1), (0.02, 0.96, 1, -1), (0.98, 0.96, -1, -1)]:
    sub.plot([cxf*W, (cxf+dx*bl)*W], [cyf*H, cyf*H], color=GOLD, lw=2.2,
             transform=sub.transData if False else sub.transAxes)
    sub.plot([cxf*W, cxf*W], [cyf*H, (cyf+dy*bl)*H], color=GOLD, lw=2.2, transform=sub.transAxes)

# ---------------- TOP-DOWN SHAPES (c1, row2) ----------------
axt = AX(c1, r2y, col_w, row2_h, z=3)
inner = card(axt, "Team Shape", 0, 0, 1, 1, accent=GOLD,
             value_right="TOP-DOWN  ·  HULL", value_color=INK2)
ix, iy, iw, ih = inner
sp = fig.add_axes([c1 + col_w * ix, r2y + row2_h * iy, col_w * iw, row2_h * ih], zorder=4)
P.draw_pitch(sp, line="#cfe0ff", lw=1.2, face="#0e2a18", alpha=0.55)
def fy(y): return P.PW - y
arg_pts = [(t[0], fy(t[1])) for t in d["tracks"] if t[2] == 0]
alg_pts = [(t[0], fy(t[1])) for t in d["tracks"] if t[2] == 1]
P.hull(sp, arg_pts, CYAN, alpha=0.16, lw=2.0)
P.hull(sp, alg_pts, ORANGE, alpha=0.16, lw=2.0)
for t in d["tracks"]:
    col = CYAN if t[2] == 0 else ORANGE
    sp.add_patch(Circle((t[0], fy(t[1])), 1.6, facecolor=col, ec="white", lw=0.9, zorder=6))
bx_, by_ = d["ball"]
sp.add_patch(Circle((bx_, fy(by_)), 1.3, facecolor=GOLD, ec="#06210f", lw=1.0, zorder=7))

# ---------------- PITCH CONTROL (c2, row1) ----------------
axp = AX(c2, r1y, col_w, row1_h, z=3)
inner = card(axp, "Pitch Control", 0, 0, 1, 1, accent=ORANGE_HEX,
             value_right="ZONE OWNERSHIP", value_color=INK2)
ix, iy, iw, ih = inner
pc = fig.add_axes([c2 + col_w * ix, r1y + row1_h * iy, col_w * iw, row1_h * ih], zorder=4)
# build proximity softmax field
nx, ny = 140, 92
gx2 = np.linspace(0, P.PL, nx); gy2 = np.linspace(0, P.PW, ny)
GX, GY = np.meshgrid(gx2, gy2)
inf_h = np.zeros_like(GX); inf_a = np.zeros_like(GX)
for t in d["tracks"]:
    dist = np.sqrt((GX - t[0])**2 + (GY - fy(t[1]))**2)
    w = np.exp(-dist / 6.0)
    if t[2] == 0: inf_h += w
    else: inf_a += w
own = inf_h / (inf_h + inf_a + 1e-9)   # 1 = ARG, 0 = ALG
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("pc", [ORANGE_HEX, "#0e1626", CYAN_HEX])
pc.imshow(own, extent=[0, P.PL, 0, P.PW], origin="lower", cmap=cmap, vmin=0, vmax=1,
          aspect="auto", interpolation="bilinear", zorder=1, alpha=0.92)
P.draw_pitch(pc, line="#dfe9ff", lw=1.2, face="none", alpha=0.45)
pc.set_facecolor("none")
bx_, by_ = d["ball"]
pc.add_patch(Circle((bx_, fy(by_)), 1.2, facecolor=GOLD, ec="#06210f", lw=1.0, zorder=7))
# team ownership labels on the surface
pc.text(18, P.PW - 3, ARG.upper(), ha="center", va="top", color="#06121f", fontsize=11,
        fontfamily=F_COND, fontweight="bold", zorder=8,
        path_effects=[pe.withStroke(linewidth=2.4, foreground=CYAN_HEX)])
pc.text(P.PL - 18, P.PW - 3, ALG.upper(), ha="center", va="top", color="#1a0a02", fontsize=11,
        fontfamily=F_COND, fontweight="bold", zorder=8,
        path_effects=[pe.withStroke(linewidth=2.4, foreground=ORANGE_HEX)])

# ---------------- xG RACE (c2, row2) ----------------
axx = AX(c2, r2y, col_w, row2_h, z=3)
xg_h_end = float(m["xg_h"][ti]); xg_a_end = float(m["xg_a"][ti])
inner = card(axx, "Expected Goals Race", 0, 0, 1, 1, accent=GOLD,
             value_right=f"{xg_h_end:.1f}  –  {xg_a_end:.1f}", value_color=INK)
ix, iy, iw, ih = inner
xr = fig.add_axes([c2 + col_w * ix, r2y + row2_h * iy, col_w * iw, row2_h * ih], zorder=4)
xr.set_facecolor("none")
mins = m["wp_mins"][:ti+1]
xgh = m["xg_h"][:ti+1]; xga = m["xg_a"][:ti+1]
ymax = max(xgh.max(), xga.max(), 1.0) * 1.18
# minimal axis: baseline + soft gridlines
for gyv in np.arange(0.5, ymax, 0.5):
    xr.plot([0, ti], [gyv, gyv], color=HAIR, lw=0.7, zorder=1)
    xr.text(-1.5, gyv, f"{gyv:.1f}", ha="right", va="center", color=INK3, fontsize=8,
            fontfamily=F_MONO, zorder=2)
xr.fill_between(mins, 0, xgh, color=CYAN_HEX, alpha=0.18, step="post", zorder=2)
xr.fill_between(mins, 0, xga, color=ORANGE_HEX, alpha=0.16, step="post", zorder=2)
xr.step(mins, xgh, where="post", color=CYAN_HEX, lw=2.6, zorder=4)
xr.step(mins, xga, where="post", color=ORANGE_HEX, lw=2.6, zorder=4)
# goal stars (home goals only here)
for s in m["shots"]:
    if s["goal"] and s["min"] <= ti:
        yv = float(m["xg_h"][s["min"]]) if s["is_home"] else float(m["xg_a"][s["min"]])
        col = CYAN_HEX if s["is_home"] else ORANGE_HEX
        xr.scatter([s["min"]], [yv], marker="*", s=320, color=GOLD, edgecolor="#06210f",
                   linewidth=1.0, zorder=6)
xr.set_xlim(-1.5, ti + 1); xr.set_ylim(0, ymax)
xr.set_xticks([]); xr.set_yticks([])
for s in xr.spines.values(): s.set_visible(False)
xr.plot([0, ti], [0, 0], color=HAIR_HI, lw=1.2, zorder=3)
for xm in [0, 15, 30, 45, 60]:
    if xm <= ti:
        xr.text(xm, -ymax*0.06, f"{xm}'", ha="center", va="top", color=INK3, fontsize=8,
                fontfamily=F_MONO)
# endpoint value chips
xr.text(ti, xgh[-1], f" {xg_h_end:.1f}", ha="left", va="center", color=CYAN_HEX,
        fontsize=11, fontfamily=F_NUM, fontweight="bold", zorder=7)
xr.text(ti, xga[-1], f" {xg_a_end:.1f}", ha="left", va="center", color=ORANGE_HEX,
        fontsize=11, fontfamily=F_NUM, fontweight="bold", zorder=7)

# ---------------- EVENTS TICKER (c3, row1) ----------------
axe = AX(c3, r1y, col_w, row1_h, z=3)
inner = card(axe, "Match Ticker", 0, 0, 1, 1, accent=AMBER,
             value_right="LATEST", value_color=INK2)
ix, iy, iw, ih = inner
evs = [e for e in m["events"] if e["min"] <= ti][::-1]   # newest first
TYPE_COL = {"Goal": None, "Card": GOLD, "Substitution": GREEN, "VAR": AMBER}
TYPE_LAB = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}
n = min(len(evs), 6)
rh = ih / n
for i in range(n):
    e = evs[i]
    ry = iy + ih - (i + 1) * rh
    rx = ix
    newest = (i == 0)
    fillc = PANEL_HI if newest else "none"
    if newest:
        axe.add_patch(FancyBboxPatch((rx, ry + rh*0.06), iw, rh*0.88,
                      boxstyle="round,pad=0,rounding_size=0.03", linewidth=1.2,
                      edgecolor=AMBER, facecolor=fillc, mutation_aspect=(rh*0.88)/iw,
                      transform=axe.transAxes, zorder=4))
    tcol = TYPE_COL[e["type"]]
    if tcol is None:
        tcol = CYAN_HEX if e["is_home"] else ORANGE_HEX
    # minute chip
    axe.text(rx + 0.03, ry + rh/2, f"{e['min']}'", transform=axe.transAxes, ha="left",
             va="center", color=INK, fontsize=13, fontfamily=F_NUM, fontweight="bold", zorder=6)
    # type chip
    axe.add_patch(FancyBboxPatch((rx + 0.11, ry + rh*0.30), 0.135, rh*0.40,
                  boxstyle="round,pad=0,rounding_size=0.03", linewidth=0, facecolor=tcol,
                  mutation_aspect=(rh*0.40)/0.135, transform=axe.transAxes, zorder=6))
    axe.text(rx + 0.1775, ry + rh/2, TYPE_LAB[e["type"]], transform=axe.transAxes,
             ha="center", va="center", color="#06121f", fontsize=9, fontfamily=F_COND,
             fontweight="bold", zorder=7)
    # player + detail
    detail = e["player"]
    if e["type"] == "Goal":
        detail = f"{e['player']}  ({e['score'][0]}-{e['score'][1]})"
    elif e["type"] == "VAR":
        detail = f"{e['player']} — {e['note']}"
    axe.text(rx + 0.27, ry + rh/2, detail, transform=axe.transAxes, ha="left", va="center",
             color=INK if newest else INK2, fontsize=10.5, fontfamily=F_COND,
             fontweight="bold" if newest else "normal", zorder=6)
    if i < n - 1 and not newest:
        axe.plot([rx + 0.02, rx + iw - 0.02], [ry, ry], color=HAIR, lw=0.7,
                 transform=axe.transAxes, zorder=3)

# ---------------- RATINGS (c3, row2) ----------------
axr = AX(c3, r2y, col_w, row2_h, z=3)
inner = card(axr, "Player Ratings", 0, 0, 1, 1, accent=GOLD,
             value_right="PLAYER OF THE MATCH", value_color=GOLD)
ix, iy, iw, ih = inner
rats = m["ratings"][:5]
n = len(rats)
rh = ih / n
maxr = 10.0
for i, r in enumerate(rats):
    ry = iy + ih - (i + 1) * rh
    rx = ix
    col = CYAN_HEX if r["is_home"] else ORANGE_HEX
    # name
    nm = r["name"]
    axr.text(rx, ry + rh*0.62, nm.upper(), transform=axr.transAxes, ha="left", va="center",
             color=INK, fontsize=11, fontfamily=F_COND, fontweight="bold", zorder=6)
    # bar track
    btw = iw - 0.16
    bty = ry + rh*0.18
    bbh = rh * 0.22
    axr.add_patch(FancyBboxPatch((rx, bty), btw, bbh, boxstyle="round,pad=0,rounding_size=0.02",
                  linewidth=0, facecolor="#0c1322", mutation_aspect=bbh/btw,
                  transform=axr.transAxes, zorder=4))
    frac = r["rating"] / maxr
    axr.add_patch(FancyBboxPatch((rx, bty), btw*frac, bbh, boxstyle="round,pad=0,rounding_size=0.02",
                  linewidth=0, facecolor=col, mutation_aspect=bbh/(btw*frac+1e-6),
                  transform=axr.transAxes, zorder=5))
    # rating value
    axr.text(rx + iw, ry + rh*0.55, f"{r['rating']:.2f}", transform=axr.transAxes, ha="right",
             va="center", color=INK, fontsize=15, fontfamily=F_NUM, fontweight="bold", zorder=6)
    if r["potm"]:
        # POTM badge
        axr.add_patch(FancyBboxPatch((rx + iw - 0.30, ry + rh*0.55), 0.12, rh*0.30,
                      boxstyle="round,pad=0,rounding_size=0.04", linewidth=0, facecolor=GOLD,
                      mutation_aspect=(rh*0.30)/0.12, transform=axr.transAxes, zorder=6))
        axr.text(rx + iw - 0.24, ry + rh*0.70, "POTM", transform=axr.transAxes, ha="center",
                 va="center", color="#2a1d00", fontsize=8.5, fontfamily=F_COND,
                 fontweight="bold", zorder=7)

# ---------------- FOOTER (single method/credit line) ----------------
axf = AX(M, 0.004, 1 - 2*M, 0.028, z=5)
foot = ("Broadcast CV ~5 m zone-grade   ·   win-prob OOS log-loss 0.82   "
        "·   World-Football-Elo predictor   ·   futbol_tech")
axf.text(0.004, 0.5, foot, transform=axf.transAxes, ha="left", va="center",
         color=INK3, fontsize=9.5, fontfamily=F_COND)
axf.text(0.996, 0.5, "PITCHVISION · LIVE MATCH INTELLIGENCE", transform=axf.transAxes,
         ha="right", va="center", color=INK3, fontsize=9.5, fontfamily=F_COND, fontweight="bold")

out = os.path.join(P.ROOT, "_frames_review", "variant_a.png")
fig.savefig(out, dpi=100, facecolor=BASE)
print("SAVED", out)
