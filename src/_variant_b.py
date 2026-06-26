"""DIRECTION B - Editorial data journalism dashboard (The Athletic / FT graphics).
Renders a single 1920x1080 PNG from the preview harness."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, RegularPolygon
from matplotlib.lines import Line2D
from matplotlib.path import Path
import matplotlib.patches as mpatches
import cv2

import _preview_data as P

# ----------------------------------------------------------------------------- data
m = P.build_m()
d = P.build_state()
ti = P.PREVIEW_MIN

HOME, AWAY = m["home"], m["away"]
HCODE, ACODE = "ARG", "ALG"
sc_h, sc_a = int(m["sc_h"][ti]), int(m["sc_a"][ti])
ARG = P.CYAN
ALG = P.ORANGE

# ----------------------------------------------------------------------------- palette
# Editorial "warm paper / ink" scheme. Calm, authoritative, limited.
PAPER   = "#f4efe6"   # warm newsprint
PAPER2  = "#ece5d8"   # card fill (very subtle)
INK     = "#1c1a17"   # near-black ink (primary text)
INK2    = "#4a4640"   # secondary text
INK3    = "#8a857a"   # tertiary / captions
RULE    = "#c9c1b1"   # hairline rules
RULE2   = "#ddd6c8"   # lighter hairline
ARGc    = "#0e7c9b"   # deepened cyan -> teal ink (editorial-safe)
ALGc    = "#c4561e"   # deepened orange -> rust
ARG_FILL= "#cfe2e6"
ALG_FILL= "#ecd6c4"
AMBER   = "#b8860b"   # VAR / accent
CARD    = "#faf6ee"   # card surface (slightly lighter than paper)

SERIF = "Georgia"
SERIF2 = "Constantia"
SANS  = "Bahnschrift"
MONO  = "DejaVu Sans Mono"

plt.rcParams["font.family"] = SERIF
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor(PAPER)

# Use a free-form axes covering the whole canvas for drawing in 0..1920 / 0..1080
W, H = 1920.0, 1080.0
root = fig.add_axes([0, 0, 1, 1])
root.set_xlim(0, W); root.set_ylim(0, H)
root.invert_yaxis()  # y grows downward -> easier layout
root.axis("off")


def kicker(ax_or_root, x, y, text, color=INK3, size=11, weight="bold"):
    ax_or_root.text(x, y, text.upper(), color=color, fontsize=size, family=SANS,
                    weight=weight, va="center", ha="left", zorder=20)


def smallcaps(s):
    return s.upper()


# ---------- card helper (subtle fill + hairline border + header strip) ----------
def card(x, y, w, h, kick=None, title=None, accent=None, pad=18):
    """x,y top-left in pixel space (y downward). returns (cx,cy,cw,ch) inner box for content."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0,rounding_size=8",
                         linewidth=1.1, edgecolor=RULE, facecolor=CARD, zorder=2)
    root.add_patch(box)
    # subtle top shadow line for lift
    inner_top = y + pad
    if kick is not None:
        # small accent tick
        if accent is not None:
            root.add_patch(Rectangle((x + pad, y + pad - 5), 16, 3.2, color=accent, zorder=6))
            kx = x + pad + 24
        else:
            kx = x + pad
        kicker(root, kx, y + pad + 1, kick, color=INK3, size=10.5)
        inner_top = y + pad + 18
    if title is not None:
        root.text(x + pad, inner_top + 12, title, color=INK, fontsize=18,
                  family=SERIF, weight="bold", va="center", ha="left", zorder=6)
        # hairline rule under header
        ry = inner_top + 30
        root.add_patch(Rectangle((x + pad, ry), w - 2 * pad, 1.0, color=RULE2, zorder=5))
        inner_top = ry + 10
    return (x + pad, inner_top, w - 2 * pad, (y + h) - inner_top - pad)


def add_inset(x, y, w, h):
    """Return a matplotlib axes positioned in pixel coords (y downward)."""
    fx = x / W
    fy = 1.0 - (y + h) / H
    return fig.add_axes([fx, fy, w / W, h / H])


# =============================================================================
# HEADER LOCKUP
# =============================================================================
MARGIN = 40
root.add_patch(Rectangle((0, 0), W, 118, color=INK, zorder=1))
# wordmark
root.text(MARGIN, 40, "THE", color=PAPER, fontsize=15, family=SANS, weight="bold",
          va="center", ha="left", zorder=10)
root.text(MARGIN + 44, 40, "PRESSING", color="#e8a23a", fontsize=22, family=SERIF,
          weight="bold", va="center", ha="left", zorder=10, style="italic")
root.text(MARGIN + 192, 41, "LINES", color=PAPER, fontsize=22, family=SERIF,
          weight="bold", va="center", ha="left", zorder=10)
root.text(MARGIN, 78, "A LIVE MATCH EXPLAINER  ·  BROADCAST COMPUTER VISION + WIN-PROBABILITY MODEL",
          color="#b7b0a2", fontsize=10.5, family=SANS, weight="bold", va="center", ha="left", zorder=10)

# centre: competition tag
root.text(W/2, 34, "FIFA WORLD CUP FINAL", color="#b7b0a2", fontsize=10.5, family=SANS,
          weight="bold", va="center", ha="center", zorder=10)
root.text(W/2, 64, "Argentina  vs  Algeria", color=PAPER, fontsize=19, family=SERIF,
          weight="bold", va="center", ha="center", zorder=10, style="italic")

# right: live score + minute
sx = W - MARGIN
root.text(sx, 36, "LIVE", color="#e85a4a", fontsize=11, family=SANS, weight="bold",
          va="center", ha="right", zorder=10)
root.add_patch(Circle((sx - 56, 33), 4.5, color="#e85a4a", zorder=10))
root.text(sx, 80, f"{HCODE}  {sc_h} – {sc_a}  {ACODE}", color=PAPER, fontsize=23,
          family=SERIF, weight="bold", va="center", ha="right", zorder=10)
# minute on its own, with a hairline divider, well clear of the score
mdx = sx - 318
root.add_patch(Rectangle((mdx + 10, 62), 1.2, 34, color="#4a463f", zorder=10))
root.text(mdx, 80, f"{ti}'", color="#e8a23a", fontsize=23, family=SERIF,
          weight="bold", va="center", ha="right", zorder=10)
root.text(mdx, 56, "MINUTE", color="#b7b0a2", fontsize=8.5, family=SANS,
          weight="bold", va="center", ha="right", zorder=10)

# =============================================================================
# GRID LAYOUT (pixel coords, y downward)
# =============================================================================
Gx = MARGIN
GTOP = 140
GAP = 18
GW = W - 2 * MARGIN

# Three columns
colW = (GW - 2 * GAP) / 3.0
c0 = Gx
c1 = Gx + colW + GAP
c2 = Gx + 2 * (colW + GAP)
bottomY = H - 36   # leave footer band

# Row heights
# Left column: win prob (top), pre-match call (mid), ratings (bottom)
# Middle column: broadcast (top), xg race (bottom)
# Right column: top-down shapes (top), pitch control (mid)... we'll tune

# ---- Left column ----
L_winH = 196
L_callH = 150
L_rateH = 0  # fill remainder
lx = c0
ly = GTOP
win_box = (lx, ly, colW, L_winH); ly += L_winH + GAP
call_box = (lx, ly, colW, L_callH); ly += L_callH + GAP
rate_box = (lx, ly, colW, bottomY - ly)

# ---- Middle column ----
M_bcastH = 300
mx = c1
my = GTOP
bcast_box = (mx, my, colW, M_bcastH); my += M_bcastH + GAP
xg_box = (mx, my, colW, bottomY - my)

# ---- Right column ----
rx = c2
ry = GTOP
R_shapeH = 300
shape_box = (rx, ry, colW, R_shapeH); ry += R_shapeH + GAP
# pitch control + events ticker share remaining; ticker on far right top? Keep editorial:
ctrl_box = (rx, ry, colW, 196); ry += 196 + GAP
tick_box = (rx, ry, colW, bottomY - ry)

# =============================================================================
# 1. LIVE WIN PROBABILITY  (dual-sided bar with chips + xG marker)
# =============================================================================
ix, iy, iw, ih = card(*win_box, kick="The model", title="Live win probability", accent=AMBER)
ph = float(m["wp_home"][ti]); pd = float(m["wp_draw"][ti]); pa = float(m["wp_away"][ti])
pxg = float(m["wp_xg"][ti])

bar_y = iy + 46
bar_h = 34
# Display fractions with a tiny floor so draw/away read as ">0" not invisible.
def pct_label(v):
    p = v * 100
    if p >= 99.5:
        return "99%+"
    if p < 0.5:
        return "<1%"
    return f"{round(p)}%"
seg = [(max(ph, 0.0), ARGc, f"{HCODE}", pct_label(ph)),
       (max(pd, 0.0), INK3, "DRAW", pct_label(pd)),
       (max(pa, 0.0), ALGc, f"{ACODE}", pct_label(pa))]
# legend row above bar
root.text(ix, iy + 12, "WHO WINS FROM HERE", color=INK3, fontsize=10, family=SANS,
          weight="bold", va="center", ha="left")
total_w = iw
# Draw segments; give the tiny segments a visible minimum sliver
draw_fracs = []
floor = 0.012
remaining = 1.0
sizes = []
for frac, col, lab, pl in seg:
    sizes.append(max(frac, floor if frac > 0 else 0))
ssum = sum(sizes)
sizes = [s / ssum for s in sizes]
xrun = ix
for (frac, col, lab, pl), sz in zip(seg, sizes):
    wseg = sz * total_w
    root.add_patch(Rectangle((xrun, bar_y), wseg, bar_h, color=col, zorder=6))
    xrun += wseg
# crisp separators
xrun = ix
for sz in sizes[:-1]:
    xrun += sz * total_w
    root.add_patch(Rectangle((xrun - 1, bar_y), 2, bar_h, color=CARD, zorder=7))

# Primary chip: ARG (hero), inside the bar
root.text(ix + 14, bar_y + bar_h/2, pct_label(ph), color=PAPER, fontsize=18,
          family=SERIF, weight="bold", va="center", ha="left", zorder=8)
root.text(ix + 14, bar_y + bar_h/2 + 0, "", color=PAPER)
# label row beneath the bar
lbl_y = bar_y + bar_h + 16
root.text(ix, lbl_y, f"{HCODE} WIN", color=ARGc, fontsize=10.5, family=SANS,
          weight="bold", va="center", ha="left")
root.text(ix + iw, lbl_y,
          f"DRAW {pct_label(pd)}    ·    {ACODE} {pct_label(pa)}",
          color=INK3, fontsize=10.5, family=SANS, weight="bold", va="center", ha="right")

# xG-deserved marker: a thin vertical pointer onto the bar (the real story)
xg_x = ix + pxg * total_w
root.add_patch(Polygon([[xg_x, bar_y - 8], [xg_x - 7, bar_y - 19], [xg_x + 7, bar_y - 19]],
               closed=True, color=AMBER, zorder=9))
root.add_patch(Rectangle((xg_x - 1.0, bar_y - 8), 2.0, bar_h + 8, color=AMBER, zorder=9))
root.text(xg_x - 6, bar_y - 24, f"xG-DESERVED  {round(pxg*100)}%", color=AMBER, fontsize=10,
          family=SANS, weight="bold", va="bottom", ha="right")

# =============================================================================
# 8. PRE-MATCH CALL + RESULT
# =============================================================================
ix, iy, iw, ih = card(*call_box, kick="Before kickoff", title="Our pre-match call", accent=ARGc)
pm = m["pre_match"]
# Probability strip (mini)
root.text(ix, iy + 6, "ELO FORECAST", color=INK3, fontsize=9.5, family=SANS, weight="bold",
          va="center", ha="left")
mini_y = iy + 18
mini_h = 12
segs = [(pm["p_h"], ARGc), (pm["p_d"], INK3), (pm["p_a"], ALGc)]
xr = ix
for f, c in segs:
    root.add_patch(Rectangle((xr, mini_y), f * iw, mini_h, color=c, zorder=6))
    xr += f * iw
root.text(ix, mini_y + mini_h + 14, f"Argentina {round(pm['p_h']*100)}%",
          color=ARGc, fontsize=12.5, family=SERIF, weight="bold", va="center", ha="left")
root.text(ix + iw, mini_y + mini_h + 14, f"Algeria {round(pm['p_a']*100)}%",
          color=ALGc, fontsize=12.5, family=SERIF, weight="bold", va="center", ha="right")

# The verdict line
vy = mini_y + mini_h + 40
root.text(ix, vy, "Called", color=INK3, fontsize=10.5, family=SANS, weight="bold",
          va="center", ha="left")
root.text(ix + 64, vy, "ARGENTINA", color=INK, fontsize=15, family=SERIF, weight="bold",
          va="center", ha="left", style="italic")
# check badge
bx = ix + iw - 88
root.add_patch(FancyBboxPatch((bx, vy - 13), 88, 26, boxstyle="round,pad=0,rounding_size=13",
               facecolor="#2f7d32", edgecolor="none", zorder=6))
root.text(bx + 44, vy, "CORRECT", color="#f4efe6", fontsize=10.5, family=SANS,
          weight="bold", va="center", ha="center", zorder=7)
root.text(ix, vy + 24, f"Final result  3 – 0   ·   forecast held", color=INK2,
          fontsize=11.5, family=SERIF, va="center", ha="left", style="italic")

# =============================================================================
# 9. TOP PLAYER RATINGS  (clean bars + POTM)
# =============================================================================
ix, iy, iw, ih = card(*rate_box, kick="Player index", title="Top performers", accent=ARGc)
ratings = m["ratings"]
n = len(ratings)
row_h = ih / n
maxr = 10.0
bar_x0 = ix + 150
bar_w = iw - 150 - 46
for i, r in enumerate(ratings):
    ry0 = iy + i * row_h + row_h / 2
    col = ARGc if r["is_home"] else ALGc
    # name
    nm = r["name"]
    root.text(ix, ry0, nm, color=INK, fontsize=12.5, family=SERIF, weight="bold",
              va="center", ha="left")
    # bar track
    root.add_patch(Rectangle((bar_x0, ry0 - 6), bar_w, 12, color=RULE2, zorder=4))
    fillw = bar_w * (r["rating"] / maxr)
    root.add_patch(Rectangle((bar_x0, ry0 - 6), fillw, 12, color=col, zorder=5))
    # rating value
    root.text(ix + iw, ry0, f"{r['rating']:.2f}", color=INK, fontsize=13.5, family=MONO,
              weight="bold", va="center", ha="right")
    if r.get("potm"):
        # POTM badge after the name
        bw = 58
        root.add_patch(FancyBboxPatch((ix + 96, ry0 - 9), bw, 18,
                       boxstyle="round,pad=0,rounding_size=9",
                       facecolor=AMBER, edgecolor="none", zorder=6))
        root.text(ix + 96 + bw/2, ry0, "POTM", color=PAPER, fontsize=9, family=SANS,
                  weight="bold", va="center", ha="center", zorder=7)

# =============================================================================
# 3. BROADCAST PANEL
# =============================================================================
ix, iy, iw, ih = card(*bcast_box, kick="Computer vision", title="Players detected on the broadcast feed",
                      accent=ALGc)
ax = add_inset(ix, iy, iw, ih)
img = cv2.cvtColor(cv2.imread(d["fp"]), cv2.COLOR_BGR2RGB)
ax.imshow(img, aspect="auto")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color(INK); s.set_linewidth(1.2)
# overlay caption strip
bw606, bh332 = d["wh"]
ax.text(0.015, 0.045, f"WIDE CAMERA", transform=ax.transAxes, color="#f4efe6",
        fontsize=9, family=SANS, weight="bold", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc=(0,0,0,0.55), ec="none"))
n_tracks = len(d["tracks"])
ax.text(0.985, 0.045, f"{n_tracks} PLAYERS TRACKED  ·  CONF {d['conf']:.2f}", transform=ax.transAxes,
        color="#f4efe6", fontsize=9, family=SANS, weight="bold", va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc=(0,0,0,0.55), ec="none"))

# =============================================================================
# 6. xG RACE  (filled step + goal stars + minimal axis)
# =============================================================================
ix, iy, iw, ih = card(*xg_box, kick="Chance quality", title="Expected goals, minute by minute",
                      accent=ARGc, pad=18)
ax = add_inset(ix, iy + 6, iw, ih - 10)
mins = m["wp_mins"]
xg_h = m["xg_h"]; xg_a = m["xg_a"]
xmax = 78
ymax = max(xg_h.max(), xg_a.max()) * 1.18 + 0.05
ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
ax.set_facecolor(CARD)
# minimal axis: only baseline + a couple horizontal gridlines
for yv in np.linspace(0, ymax, 4)[1:]:
    ax.axhline(yv, color=RULE2, lw=0.8, zorder=1)
    ax.text(-0.6, yv, f"{yv:.1f}", color=INK3, fontsize=9, family=MONO, va="center", ha="right")
# step area fills
ax.fill_between(mins[:ti+1], xg_h[:ti+1], step="post", color=ARG_FILL, alpha=0.9, zorder=2)
ax.step(mins[:ti+1], xg_h[:ti+1], where="post", color=ARGc, lw=2.4, zorder=4)
ax.step(mins[:ti+1], xg_a[:ti+1], where="post", color=ALGc, lw=2.0, zorder=4)
ax.fill_between(mins[:ti+1], xg_a[:ti+1], step="post", color=ALG_FILL, alpha=0.55, zorder=2)
# goals as stars
for s in m["shots"]:
    if s["goal"] and s["min"] <= ti:
        yv = xg_h[s["min"]] if s["is_home"] else xg_a[s["min"]]
        col = ARGc if s["is_home"] else ALGc
        ax.scatter([s["min"]], [yv], marker="*", s=240, color=AMBER,
                   edgecolor=INK, linewidth=0.8, zorder=6)
# now marker (clean, not dashed)
ax.axvline(ti, color=INK, lw=1.0, alpha=0.5, zorder=3)
ax.text(ti, ymax*0.985, f"{ti}'", color=INK, fontsize=9.5, family=SANS, weight="bold",
        va="top", ha="right")
# end labels
ax.text(ti+1.5, xg_h[ti], f"{HCODE} {xg_h[ti]:.2f}", color=ARGc, fontsize=11.5,
        family=SERIF, weight="bold", va="center", ha="left")
ax.text(ti+1.5, xg_a[ti], f"{ACODE} {xg_a[ti]:.2f}", color=ALGc, fontsize=11.5,
        family=SERIF, weight="bold", va="center", ha="left")
# x ticks minimal
for xv in [0, 15, 30, 45, 60]:
    ax.text(xv, -ymax*0.05, f"{xv}'", color=INK3, fontsize=9, family=SANS,
            va="top", ha="center")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(0, color=INK, lw=1.2, zorder=3)

# =============================================================================
# 4. TOP-DOWN TEAM SHAPES
# =============================================================================
ix, iy, iw, ih = card(*shape_box, kick="Shape & space", title="Team shape on the pitch",
                      accent=ARGc)
ax = add_inset(ix, iy + 4, iw, ih - 8)
# editorial pitch: paper-toned, ink lines
P.draw_pitch(ax, line=INK3, lw=1.0, face="#e9e2d2", alpha=0.7)
tracks = d["tracks"]
arg_pts = [[t[0], P.PW - t[1]] for t in tracks if t[2] == 0]
alg_pts = [[t[0], P.PW - t[1]] for t in tracks if t[2] == 1]
P.hull(ax, arg_pts, ARGc, alpha=0.14, lw=1.6)
P.hull(ax, alg_pts, ALGc, alpha=0.14, lw=1.6)
for t in tracks:
    x, y = t[0], P.PW - t[1]
    col = ARGc if t[2] == 0 else ALGc
    ax.add_patch(Circle((x, y), 1.6, facecolor=col, edgecolor=PAPER, lw=1.0, zorder=6))
bx, by = d["ball"]
ax.add_patch(Circle((bx, P.PW - by), 1.4, facecolor=INK, edgecolor=PAPER, lw=1.0, zorder=7))
# direction-of-play annotation
ax.annotate("", xy=(0.72, 1.04), xytext=(0.28, 1.04), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=INK3, lw=1.3))
ax.text(0.5, 1.10, "ARGENTINA ATTACKING", transform=ax.transAxes, color=INK3,
        fontsize=8.5, family=SANS, weight="bold", va="bottom", ha="center")

# =============================================================================
# 5. PITCH CONTROL MAP
# =============================================================================
ix, iy, iw, ih = card(*ctrl_box, kick="Territory", title="Live pitch control", accent=ALGc)
ax = add_inset(ix, iy + 4, iw, ih - 26)
# compute proximity softmax ownership on a grid
gx = np.linspace(0, P.PL, 90)
gy = np.linspace(0, P.PW, 60)
GX, GY = np.meshgrid(gx, gy)
home_w = np.zeros_like(GX)
away_w = np.zeros_like(GX)
for t in tracks:
    px, py = t[0], P.PW - t[1]
    dd = np.sqrt((GX - px)**2 + (GY - py)**2)
    w = np.exp(-dd / 6.0)
    if t[2] == 0:
        home_w += w
    else:
        away_w += w
own = home_w / (home_w + away_w + 1e-9)   # 1 = ARG, 0 = ALG
# editorial diverging colormap teal <-> paper <-> rust
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("ctrl", [ALGc, "#efe8da", ARGc])
ax.imshow(own, extent=[0, P.PL, 0, P.PW], origin="lower", cmap=cmap,
          vmin=0, vmax=1, alpha=0.85, aspect="auto", zorder=1)
# overlay pitch lines without resetting aspect
def pitch_lines(ax, col, lw, alpha):
    ax.plot([0, 0, P.PL, P.PL, 0], [0, P.PW, P.PW, 0, 0], color=col, lw=lw, alpha=alpha)
    ax.plot([P.PL/2, P.PL/2], [0, P.PW], color=col, lw=lw*0.85, alpha=alpha*0.8)
    th = np.linspace(0, 2*np.pi, 80)
    ax.plot(P.PL/2 + 9.15*np.cos(th), P.PW/2 + 9.15*np.sin(th), color=col, lw=lw*0.85, alpha=alpha*0.8)
    for x0 in (0, P.PL-16.5):
        ax.plot([x0, x0+16.5, x0+16.5, x0], [13.84, 13.84, 54.16, 54.16], color=col, lw=lw*0.85, alpha=alpha*0.8)
pitch_lines(ax, "#3a352c", 1.0, 0.5)
ax.set_xlim(0, P.PL); ax.set_ylim(0, P.PW)
ax.set_aspect("auto")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
# control split readout (placed in card under the inset)
home_share = own.mean()
ry_lab = iy + ih - 12
root.text(ix, ry_lab, f"{HCODE} {round(home_share*100)}% TERRITORY", color=ARGc,
          fontsize=9.5, family=SANS, weight="bold", va="center", ha="left")
root.text(ix + iw, ry_lab, f"{round((1-home_share)*100)}% {ACODE}", color=ALGc,
          fontsize=9.5, family=SANS, weight="bold", va="center", ha="right")

# =============================================================================
# 7. MATCH EVENTS TICKER
# =============================================================================
ix, iy, iw, ih = card(*tick_box, kick="Timeline", title="Match events", accent=AMBER)
evs = [e for e in m["events"] if e["min"] <= ti]
evs = evs[::-1]  # newest first
maxshow = 6
evs = evs[:maxshow]
row_h = ih / maxshow
type_col = {"Goal": None, "Card": "#c9a227", "Substitution": INK3, "VAR": AMBER}
for i, e in enumerate(evs):
    ey = iy + i * row_h + row_h / 2
    newest = (i == 0)
    tcol = ARGc if e["is_home"] else ALGc
    chipcol = tcol if e["type"] == "Goal" else type_col.get(e["type"], INK3)
    if e["type"] == "VAR":
        chipcol = AMBER
    # newest highlight band
    if newest:
        root.add_patch(FancyBboxPatch((ix - 6, ey - row_h/2 + 4), iw + 12, row_h - 8,
                       boxstyle="round,pad=0,rounding_size=6", facecolor=PAPER2,
                       edgecolor="none", zorder=3))
    # minute
    root.text(ix + 6, ey, f"{e['min']}'", color=INK, fontsize=13, family=MONO,
              weight="bold", va="center", ha="left", zorder=6)
    # type chip
    chip_x = ix + 52
    chip_label = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}[e["type"]]
    cw = 52
    root.add_patch(FancyBboxPatch((chip_x, ey - 9), cw, 18, boxstyle="round,pad=0,rounding_size=4",
                   facecolor=chipcol, edgecolor="none", zorder=6))
    root.text(chip_x + cw/2, ey, chip_label, color=PAPER, fontsize=8.5, family=SANS,
              weight="bold", va="center", ha="center", zorder=7)
    # description
    desc = e["player"]
    if e["type"] == "Goal":
        h, a = e["score"]
        desc = f"{e['player']}  ({h}–{a})"
    elif e["type"] == "VAR":
        desc = f"{e['player']} — {e['note']}"
    root.text(chip_x + cw + 12, ey, desc, color=INK if newest else INK2,
              fontsize=11.5, family=SERIF, weight="bold" if newest else "normal",
              va="center", ha="left", zorder=6, style="italic" if not newest else "normal")
    # hairline between rows
    if i < len(evs) - 1:
        root.add_patch(Rectangle((ix, ey + row_h/2 - 0.5), iw, 0.8, color=RULE2, zorder=4))

# =============================================================================
# FOOTER METHOD LINE
# =============================================================================
root.add_patch(Rectangle((0, H - 30), W, 30, color=INK, zorder=8))
root.text(MARGIN, H - 15,
          "Broadcast CV ~5 m zone-grade  ·  win-prob OOS log-loss 0.82  ·  World-Football-Elo predictor",
          color="#b7b0a2", fontsize=10, family=SANS, weight="bold", va="center", ha="left", zorder=9)
root.text(W - MARGIN, H - 15, "THE PRESSING LINES  ·  futbol_tech", color="#e8a23a",
          fontsize=10, family=SANS, weight="bold", va="center", ha="right", zorder=9)

# =============================================================================
OUT = os.path.join(P.ROOT, "_frames_review", "variant_b.png")
fig.savefig(OUT, dpi=100, facecolor=PAPER)
print("saved", OUT)
