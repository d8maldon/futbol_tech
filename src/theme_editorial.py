"""THEME: editorial  (THE PRESSING LINES) -- data-journalism look for analysts.
Ported from _variant_b.py into the visual_ai theme interface. Warm-paper / ink,
serif headlines, hairline rules, viewer-facing titles. Uses its own editorial
palette (deepened team colours) as a house style. Handles live/no-pitch/estimated.
"""
import numpy as np
import cv2
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon

import dashboard_themes as T

PL, PW = T.PL, T.PW
W, H = 1920.0, 1080.0

# ---- editorial palette ----
PAPER = "#f4efe6"; PAPER2 = "#ece5d8"; INK = "#1c1a17"; INK2 = "#4a4640"; INK3 = "#8a857a"
RULE = "#c9c1b1"; RULE2 = "#ddd6c8"; CARD = "#faf6ee"
ARGc = "#0e7c9b"; ALGc = "#c4561e"; ARG_FILL = "#cfe2e6"; ALG_FILL = "#ecd6c4"; AMBER = "#b8860b"
SERIF = "Georgia"; SANS = "Bahnschrift"; MONO = "DejaVu Sans Mono"
BG = PAPER

MARGIN = 40
GTOP = 140
GAP = 18
GW = W - 2 * MARGIN
COLW = (GW - 2 * GAP) / 3.0
C0 = MARGIN
C1 = MARGIN + COLW + GAP
C2 = MARGIN + 2 * (COLW + GAP)
BOTTOMY = H - 36

WIN_BOX = (C0, GTOP, COLW, 196)
CALL_BOX = (C0, GTOP + 196 + GAP, COLW, 150)
RATE_BOX = (C0, GTOP + 196 + GAP + 150 + GAP, COLW, BOTTOMY - (GTOP + 196 + GAP + 150 + GAP))
BCAST_BOX = (C1, GTOP, COLW, 300)
XG_BOX = (C1, GTOP + 300 + GAP, COLW, BOTTOMY - (GTOP + 300 + GAP))
SHAPE_BOX = (C2, GTOP, COLW, 300)
CTRL_BOX = (C2, GTOP + 300 + GAP, COLW, 196)
TICK_BOX = (C2, GTOP + 300 + GAP + 196 + GAP, COLW, BOTTOMY - (GTOP + 300 + GAP + 196 + GAP))


def _card_inner(x, y, w, h, pad=18):
    return (x + pad, y + pad + 58, w - 2 * pad, h - (pad + 58) - pad)


def _inset(fig, x, y, w, h):
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def _pct_label(v):
    p = v * 100
    if p >= 99.5: return "99%+"
    if p < 0.5: return "<1%"
    return "{}%".format(round(p))


def _kicker(root, x, y, text, color=INK3, size=11, weight="bold"):
    root.text(x, y, text.upper(), color=color, fontsize=size, family=SANS, weight=weight,
              va="center", ha="left", zorder=20)


def _card(root, x, y, w, h, kick=None, title=None, accent=None, pad=18):
    root.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                   linewidth=1.1, edgecolor=RULE, facecolor=CARD, zorder=2))
    inner_top = y + pad
    if kick is not None:
        if accent is not None:
            root.add_patch(Rectangle((x + pad, y + pad - 5), 16, 3.2, color=accent, zorder=6))
            kx = x + pad + 24
        else:
            kx = x + pad
        _kicker(root, kx, y + pad + 1, kick, color=INK3, size=10.5)
        inner_top = y + pad + 18
    if title is not None:
        root.text(x + pad, inner_top + 12, title, color=INK, fontsize=18, family=SERIF,
                  weight="bold", va="center", ha="left", zorder=6)
        ry = inner_top + 30
        root.add_patch(Rectangle((x + pad, ry), w - 2 * pad, 1.0, color=RULE2, zorder=5))
        inner_top = ry + 10
    return (x + pad, inner_top, w - 2 * pad, (y + h) - inner_top - pad)


def make_axes(fig):
    import matplotlib as mpl
    mpl.rcParams["axes.unicode_minus"] = False
    root = fig.add_axes([0, 0, 1, 1])
    root.set_xlim(0, W); root.set_ylim(0, H); root.invert_yaxis(); root.axis("off")
    bi = _card_inner(*BCAST_BOX)
    xi = _card_inner(*XG_BOX)
    si = _card_inner(*SHAPE_BOX)
    ci = _card_inner(*CTRL_BOX)
    return {
        "root": root,
        "bcast": _inset(fig, bi[0], bi[1], bi[2], bi[3]),
        "xg": _inset(fig, xi[0], xi[1] + 6, xi[2], xi[3] - 10),
        "shape": _inset(fig, si[0], si[1] + 4, si[2], si[3] - 8),
        "ctrl": _inset(fig, ci[0], ci[1] + 4, ci[2], ci[3] - 26),
    }


def _draw_header(root, m, ti):
    sc_h, sc_a = int(m["sc_h"][ti]), int(m["sc_a"][ti])
    root.add_patch(Rectangle((0, 0), W, 118, color=INK, zorder=1))
    root.text(MARGIN, 40, "THE", color=PAPER, fontsize=15, family=SANS, weight="bold", va="center", ha="left", zorder=10)
    root.text(MARGIN + 44, 40, "PRESSING", color="#e8a23a", fontsize=22, family=SERIF, weight="bold",
              va="center", ha="left", zorder=10, style="italic")
    root.text(MARGIN + 192, 41, "LINES", color=PAPER, fontsize=22, family=SERIF, weight="bold",
              va="center", ha="left", zorder=10)
    root.text(MARGIN, 78, "A LIVE MATCH EXPLAINER  ·  BROADCAST COMPUTER VISION + WIN-PROBABILITY MODEL",
              color="#b7b0a2", fontsize=10.5, family=SANS, weight="bold", va="center", ha="left", zorder=10)
    root.text(W / 2, 34, "FIFA WORLD CUP", color="#b7b0a2", fontsize=10.5, family=SANS, weight="bold",
              va="center", ha="center", zorder=10)
    root.text(W / 2, 64, "{}  vs  {}".format(m["home"], m["away"]), color=PAPER, fontsize=19, family=SERIF,
              weight="bold", va="center", ha="center", zorder=10, style="italic")
    sx = W - MARGIN
    root.text(sx, 36, "LIVE", color="#e85a4a", fontsize=11, family=SANS, weight="bold", va="center", ha="right", zorder=10)
    root.add_patch(Circle((sx - 56, 33), 4.5, color="#e85a4a", zorder=10))
    root.text(sx, 80, "{}  {} - {}  {}".format(m["home"][:3].upper(), sc_h, sc_a, m["away"][:3].upper()),
              color=PAPER, fontsize=23, family=SERIF, weight="bold", va="center", ha="right", zorder=10)
    mdx = sx - 318
    root.add_patch(Rectangle((mdx + 10, 62), 1.2, 34, color="#4a463f", zorder=10))
    root.text(mdx, 80, "{}'".format(ti), color="#e8a23a", fontsize=23, family=SERIF, weight="bold",
              va="center", ha="right", zorder=10)
    root.text(mdx, 56, "MINUTE", color="#b7b0a2", fontsize=8.5, family=SANS, weight="bold", va="center", ha="right", zorder=10)


def _draw_winprob(root, m, ti):
    ix, iy, iw, ih = _card(root, *WIN_BOX, kick="The model", title="Live win probability", accent=AMBER)
    ph, pd, pa = float(m["wp_home"][ti]), float(m["wp_draw"][ti]), float(m["wp_away"][ti])
    pxg = float(m["wp_xg"][ti])
    bar_y = iy + 46; bar_h = 34
    seg = [(ph, ARGc), (pd, INK3), (pa, ALGc)]
    root.text(ix, iy + 12, "WHO WINS FROM HERE", color=INK3, fontsize=10, family=SANS, weight="bold", va="center", ha="left")
    floor = 0.006   # tiny visible sliver only; small enough not to distort the labelled %
    sizes = [max(f, floor if f > 0 else 0) for f, _ in seg]
    ssum = sum(sizes); sizes = [s / ssum for s in sizes]
    xrun = ix
    for (f, col), sz in zip(seg, sizes):
        root.add_patch(Rectangle((xrun, bar_y), sz * iw, bar_h, color=col, zorder=6)); xrun += sz * iw
    xrun = ix
    for sz in sizes[:-1]:
        xrun += sz * iw
        root.add_patch(Rectangle((xrun - 1, bar_y), 2, bar_h, color=CARD, zorder=7))
    root.text(ix + 14, bar_y + bar_h / 2, _pct_label(ph), color=PAPER, fontsize=18, family=SERIF,
              weight="bold", va="center", ha="left", zorder=8)
    lbl_y = bar_y + bar_h + 16
    root.text(ix, lbl_y, "ARG WIN", color=ARGc, fontsize=10.5, family=SANS, weight="bold", va="center", ha="left")
    root.text(ix + iw, lbl_y, "DRAW {}    ·    ALG {}".format(_pct_label(pd), _pct_label(pa)),
              color=INK3, fontsize=10.5, family=SANS, weight="bold", va="center", ha="right")
    xg_x = ix + pxg * iw
    root.add_patch(Polygon([[xg_x, bar_y - 8], [xg_x - 7, bar_y - 19], [xg_x + 7, bar_y - 19]], closed=True, color=AMBER, zorder=9))
    root.add_patch(Rectangle((xg_x - 1.0, bar_y - 8), 2.0, bar_h + 8, color=AMBER, zorder=9))
    root.text(xg_x - 6, bar_y - 24, "xG-DESERVED  {}%".format(round(pxg * 100)), color=AMBER, fontsize=10,
              family=SANS, weight="bold", va="bottom", ha="right")


def _draw_call(root, m):
    ix, iy, iw, ih = _card(root, *CALL_BOX, kick="Before kickoff", title="Our pre-match call", accent=ARGc)
    pm = m["pre_match"]
    root.text(ix, iy + 6, "ELO FORECAST", color=INK3, fontsize=9.5, family=SANS, weight="bold", va="center", ha="left")
    mini_y = iy + 18; mini_h = 12
    xr = ix
    for f, c in [(pm["p_h"], ARGc), (pm["p_d"], INK3), (pm["p_a"], ALGc)]:
        root.add_patch(Rectangle((xr, mini_y), f * iw, mini_h, color=c, zorder=6)); xr += f * iw
    root.text(ix, mini_y + mini_h + 14, "{} {}%".format(m["home"], round(pm["p_h"] * 100)), color=ARGc,
              fontsize=12.5, family=SERIF, weight="bold", va="center", ha="left")
    root.text(ix + iw, mini_y + mini_h + 14, "{} {}%".format(m["away"], round(pm["p_a"] * 100)), color=ALGc,
              fontsize=12.5, family=SERIF, weight="bold", va="center", ha="right")
    call_home = pm["p_h"] >= pm["p_a"]
    vy = mini_y + mini_h + 40
    root.text(ix, vy, "Called", color=INK3, fontsize=10.5, family=SANS, weight="bold", va="center", ha="left")
    root.text(ix + 64, vy, (m["home"] if call_home else m["away"]).upper(), color=INK, fontsize=15,
              family=SERIF, weight="bold", va="center", ha="left", style="italic")
    correct = m["final_h"] != m["final_a"] and (m["final_h"] > m["final_a"]) == call_home
    bx = ix + iw - 88
    root.add_patch(FancyBboxPatch((bx, vy - 13), 88, 26, boxstyle="round,pad=0,rounding_size=13",
                   facecolor="#2f7d32" if correct else INK3, edgecolor="none", zorder=6))
    root.text(bx + 44, vy, "CORRECT" if correct else "LIVE", color="#f4efe6", fontsize=10.5, family=SANS,
              weight="bold", va="center", ha="center", zorder=7)
    root.text(ix, vy + 24, "Final result  {} - {}   ·   forecast held".format(m["final_h"], m["final_a"]),
              color=INK2, fontsize=11.5, family=SERIF, va="center", ha="left", style="italic")


def _draw_ratings(root, m):
    ix, iy, iw, ih = _card(root, *RATE_BOX, kick="Player index", title="Top performers", accent=ARGc)
    ratings = m["ratings"]
    n = len(ratings)
    if n == 0:
        return
    row_h = ih / n
    bar_x0 = ix + 150; bar_w = iw - 150 - 46
    for i, r in enumerate(ratings):
        ry0 = iy + i * row_h + row_h / 2
        col = ARGc if r["is_home"] else ALGc
        root.text(ix, ry0, r["name"], color=INK, fontsize=12.5, family=SERIF, weight="bold", va="center", ha="left")
        root.add_patch(Rectangle((bar_x0, ry0 - 6), bar_w, 12, color=RULE2, zorder=4))
        root.add_patch(Rectangle((bar_x0, ry0 - 6), bar_w * (r["rating"] / 10.0), 12, color=col, zorder=5))
        root.text(ix + iw, ry0, "{:.2f}".format(r["rating"]), color=INK, fontsize=13.5, family=MONO,
                  weight="bold", va="center", ha="right")
        if r.get("potm"):
            root.add_patch(FancyBboxPatch((ix + 96, ry0 - 9), 58, 18, boxstyle="round,pad=0,rounding_size=9",
                           facecolor=AMBER, edgecolor="none", zorder=6))
            root.text(ix + 96 + 29, ry0, "POTM", color=PAPER, fontsize=9, family=SANS, weight="bold",
                      va="center", ha="center", zorder=7)


def _draw_broadcast(root, ax, d):
    _card(root, *BCAST_BOX, kick="Computer vision", title="Players detected on the broadcast feed", accent=ALGc)
    ax.clear(); ax.set_xticks([]); ax.set_yticks([])
    img = cv2.imread(d["fp"]) if d.get("fp") else None
    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
    else:
        ax.set_facecolor("#e9e2d2")
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(1.2)
    ax.text(0.015, 0.045, "WIDE CAMERA", transform=ax.transAxes, color="#f4efe6", fontsize=9, family=SANS,
            weight="bold", va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.55), ec="none"))
    ax.text(0.985, 0.045, "{} PLAYERS TRACKED  ·  CONF {:.2f}".format(len(d.get("tracks", [])), d.get("conf", 0.0)),
            transform=ax.transAxes, color="#f4efe6", fontsize=9, family=SANS, weight="bold", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.55), ec="none"))


def _draw_xg(root, ax, m, ti):
    _card(root, *XG_BOX, kick="Chance quality", title="Expected goals, minute by minute", accent=ARGc)
    ax.clear()
    mins = m["wp_mins"]; xg_h = m["xg_h"]; xg_a = m["xg_a"]
    xmax = max(78, ti + 4)
    ymax = max(float(xg_h.max()), float(xg_a.max())) * 1.18 + 0.05
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_facecolor(CARD)
    for yv in np.linspace(0, ymax, 4)[1:]:
        ax.axhline(yv, color=RULE2, lw=0.8, zorder=1)
        ax.text(-0.6, yv, "{:.1f}".format(yv), color=INK3, fontsize=9, family=MONO, va="center", ha="right")
    ax.fill_between(mins[:ti + 1], xg_h[:ti + 1], step="post", color=ARG_FILL, alpha=0.9, zorder=2)
    ax.step(mins[:ti + 1], xg_h[:ti + 1], where="post", color=ARGc, lw=2.4, zorder=4)
    ax.step(mins[:ti + 1], xg_a[:ti + 1], where="post", color=ALGc, lw=2.0, zorder=4)
    ax.fill_between(mins[:ti + 1], xg_a[:ti + 1], step="post", color=ALG_FILL, alpha=0.55, zorder=2)
    for s in m["shots"]:
        if s["goal"] and s["min"] <= ti:
            yv = xg_h[s["min"]] if s["is_home"] else xg_a[s["min"]]
            ax.scatter([s["min"]], [yv], marker="*", s=240, color=AMBER, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.axvline(ti, color=INK, lw=1.0, alpha=0.5, zorder=3)
    ax.text(ti, ymax * 0.985, "{}'".format(ti), color=INK, fontsize=9.5, family=SANS, weight="bold", va="top", ha="right")
    ax.text(ti + 1.5, float(xg_h[ti]), "ARG {:.2f}".format(float(xg_h[ti])), color=ARGc, fontsize=11.5,
            family=SERIF, weight="bold", va="center", ha="left")
    ax.text(ti + 1.5, float(xg_a[ti]), "ALG {:.2f}".format(float(xg_a[ti])), color=ALGc, fontsize=11.5,
            family=SERIF, weight="bold", va="center", ha="left")
    for xv in [0, 15, 30, 45, 60, 75]:
        if xv <= xmax:
            ax.text(xv, -ymax * 0.05, "{}'".format(xv), color=INK3, fontsize=9, family=SANS, va="top", ha="center")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=INK, lw=1.2, zorder=3)


def _draw_shape(root, ax, d, m):
    _card(root, *SHAPE_BOX, kick="Shape & space", title="Team shape on the pitch", accent=ARGc)
    ax.clear()
    blank = d.get("note") == "no pitch view"
    T.mpl_pitch(ax, line=INK3, face="#e9e2d2", lw=1.1, line_zorder=1)
    if blank:
        lbl = "NO PITCH VIEW · graphic/replay" if d.get("cam") == "other" else "NO PITCH VIEW"
        ax.text(PL / 2, PW / 2, lbl, ha="center", va="center", color=ALGc, fontsize=12, family=SANS, weight="bold")
        return
    tracks = d.get("tracks", []) or []
    T.hull(ax, [[t[0], PW - t[1]] for t in tracks if t[2] == 0], ARGc, alpha=0.14, lw=1.6)
    T.hull(ax, [[t[0], PW - t[1]] for t in tracks if t[2] == 1], ALGc, alpha=0.14, lw=1.6)
    for t in tracks:
        col = ARGc if t[2] == 0 else ALGc
        ax.add_patch(Circle((t[0], PW - t[1]), 1.6, facecolor=col, edgecolor=PAPER, lw=1.0, alpha=float(t[3]), zorder=6))
    ball = d.get("ball")
    if ball is not None:
        ax.add_patch(Circle((ball[0], PW - ball[1]), 1.4, facecolor=INK, edgecolor=PAPER, lw=1.0, zorder=7))
    ax.annotate("", xy=(0.72, 1.04), xytext=(0.28, 1.04), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK3, lw=1.3))
    msg = "{} ATTACKING".format(m["home"].upper()) if not d.get("est") else "HOLDING LAST SHAPE (ESTIMATED)"
    ax.text(0.5, 1.10, msg, transform=ax.transAxes, color=INK3, fontsize=8.5, family=SANS,
            weight="bold", va="bottom", ha="center")


def _draw_control(root, ax, d):
    ix, iy, iw, ih = _card(root, *CTRL_BOX, kick="Territory", title="Live pitch control", accent=ALGc)
    ax.clear()
    blank = d.get("note") == "no pitch view"
    tracks = d.get("tracks", []) or []
    T.draw_pitch(ax, line="#3a352c", lw=1.0, face="#e9e2d2", alpha=0.75, equal=False)
    if blank or len(tracks) < 2:
        if blank:
            lbl = "NO PITCH VIEW" if d.get("cam") != "other" else "NO PITCH VIEW · graphic/replay"
            ax.text(PL / 2, PW / 2, lbl, ha="center", va="center", color=ALGc, fontsize=11, family=SANS, weight="bold")
        return
    th, ta, share = T.voronoi_regions(tracks)
    for poly in th:
        ax.fill(poly[:, 0], poly[:, 1], color=ARGc, alpha=0.32, ec="#efe8da", lw=0.5, zorder=2)
    for poly in ta:
        ax.fill(poly[:, 0], poly[:, 1], color=ALGc, alpha=0.32, ec="#efe8da", lw=0.5, zorder=2)
    home_share = share
    ry_lab = iy + ih - 12
    root.text(ix, ry_lab, "ARG {}% TERRITORY".format(round(home_share * 100)), color=ARGc, fontsize=9.5,
              family=SANS, weight="bold", va="center", ha="left")
    root.text(ix + iw, ry_lab, "{}% ALG".format(round((1 - home_share) * 100)), color=ALGc, fontsize=9.5,
              family=SANS, weight="bold", va="center", ha="right")


def _draw_ticker(root, m, ti):
    ix, iy, iw, ih = _card(root, *TICK_BOX, kick="Timeline", title="Match events", accent=AMBER)
    evs = [e for e in m["events"] if e["min"] <= ti and not (e["type"] == "Substitution" and not e["player"])][::-1][:6]
    row_h = ih / 6
    type_col = {"Card": "#c9a227", "Substitution": INK3, "VAR": AMBER}
    for i, e in enumerate(evs):
        ey = iy + i * row_h + row_h / 2
        newest = (i == 0)
        chipcol = (ARGc if e["is_home"] else ALGc) if e["type"] == "Goal" else type_col.get(e["type"], INK3)
        if newest:
            root.add_patch(FancyBboxPatch((ix - 6, ey - row_h / 2 + 4), iw + 12, row_h - 8,
                           boxstyle="round,pad=0,rounding_size=6", facecolor=PAPER2, edgecolor="none", zorder=3))
        root.text(ix + 6, ey, "{}'".format(e["min"]), color=INK, fontsize=13, family=MONO, weight="bold",
                  va="center", ha="left", zorder=6)
        chip_x = ix + 52; cw = 52
        chip_label = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}.get(e["type"], e["type"])
        root.add_patch(FancyBboxPatch((chip_x, ey - 9), cw, 18, boxstyle="round,pad=0,rounding_size=4",
                       facecolor=chipcol, edgecolor="none", zorder=6))
        root.text(chip_x + cw / 2, ey, chip_label, color=PAPER, fontsize=8.5, family=SANS, weight="bold",
                  va="center", ha="center", zorder=7)
        desc = e["player"]
        if e["type"] == "Goal" and e.get("score"):
            desc = "{}  ({}-{})".format(e["player"], e["score"][0], e["score"][1])
        elif e["type"] == "VAR" and e.get("note"):
            desc = "{} - {}".format(e["player"], e["note"])
        root.text(chip_x + cw + 12, ey, desc, color=INK if newest else INK2, fontsize=11.5, family=SERIF,
                  weight="bold" if newest else "normal", va="center", ha="left", zorder=6,
                  style="italic" if not newest else "normal")
        if i < len(evs) - 1:
            root.add_patch(Rectangle((ix, ey + row_h / 2 - 0.5), iw, 0.8, color=RULE2, zorder=4))


def _draw_footer(root):
    root.add_patch(Rectangle((0, H - 30), W, 30, color=INK, zorder=8))
    root.text(MARGIN, H - 15, "Broadcast CV ~5 m zone-grade (12 m gate)  ·  World-Football-Elo predictor OOS 0.86  ·  in-game win-prob: score-anchored Skellam",
              color="#b7b0a2", fontsize=10, family=SANS, weight="bold", va="center", ha="left", zorder=9)
    root.text(W - MARGIN, H - 15, "THE PRESSING LINES  ·  futbol_tech", color="#e8a23a", fontsize=10,
              family=SANS, weight="bold", va="center", ha="right", zorder=9)


def draw_frame(axes, d, m, ti, team_rgb, label=""):
    root = axes["root"]
    root.clear(); root.set_xlim(0, W); root.set_ylim(0, H); root.invert_yaxis(); root.axis("off")
    _draw_header(root, m, ti)
    _draw_winprob(root, m, ti)
    _draw_call(root, m)
    _draw_ratings(root, m)
    _draw_broadcast(root, axes["bcast"], d)
    _draw_xg(root, axes["xg"], m, ti)
    _draw_shape(root, axes["shape"], d, m)
    _draw_control(root, axes["ctrl"], d)
    _draw_ticker(root, m, ti)
    _draw_footer(root)
