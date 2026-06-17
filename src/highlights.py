"""World Cup 2026 highlight board: the chess-engine eval bar applied to the
tournament's most dramatic real finishes so far. One shareable image -- every
panel is the live win-probability model reacting to the actual goals of THIS
World Cup, ranked by how violently the match swung.

    python src/highlights.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from live_eval import (ACCENT, AWAY_C, BG, FIG, HOME_C, INK, MUT, OUT,
                       biggest_swing, build_timeline, font, list_matches,
                       win_prob_curve)

PANEL = "#131a23"


def panel(ax, c):
    tl, grid, edge, sw = c["tl"], c["grid"], c["edge"], c["sw"]
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.fill_between(grid, 0, edge, where=edge >= 0, color=HOME_C, alpha=0.9, interpolate=True, lw=0)
    ax.fill_between(grid, 0, edge, where=edge <= 0, color=AWAY_C, alpha=0.9, interpolate=True, lw=0)
    ax.plot(grid, edge, color=INK, lw=0.8, alpha=0.5)
    ax.axhline(0, color=INK, lw=1.0)
    ax.axvline(45.5, color=MUT, lw=0.6, ls=(0, (4, 3)), alpha=0.5)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlim(0, 96)
    ax.set_xticks([]); ax.set_yticks([])
    # goal markers, on the eval line
    for g in tl["goals"]:
        e = np.interp(g["m"], grid, edge)
        ax.plot([g["m"]], [e], "o", ms=6, mfc=ACCENT, mec=BG, mew=1.2, zorder=6)
    # turning-point flag
    if abs(sw.get("delta", 0)) > 0.15:
        ax.annotate("turned {:.0f}'".format(sw["at"]),
                    xy=(sw["at"], np.interp(sw["at"], grid, edge)),
                    xytext=(sw["at"] + (8 if sw["at"] < 60 else -8),
                            0.7 if edge.mean() < 0 else -0.7),
                    ha="left" if sw["at"] < 60 else "right", va="center",
                    color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0),
                    **font(8, True))
    # scoreline header + swing magnitude
    hs, as_ = tl["score"].split("-")
    ax.text(1.5, 0.93, "{} {}".format(tl["home"].upper()[:14], hs), color=HOME_C, va="top", **font(9.5, True))
    ax.text(1.5, -0.93, "{} {}".format(tl["away"].upper()[:14], as_), color=AWAY_C, va="bottom", **font(9.5, True))
    ax.text(94, 0.95, "{:+.0f} pts".format(100 * sw["delta"]), ha="right", va="top",
            color=ACCENT, **font(9, True))


def main():
    with open(os.path.join(OUT, "winprob_model.json")) as f:
        model = json.load(f)
    cards = []
    for mid in list_matches():
        tl = build_timeline(mid)
        grid, edge, ph, _ = win_prob_curve(model, tl)
        sw = biggest_swing(grid, edge, ph, tl)
        cards.append({"tl": tl, "grid": grid, "edge": edge, "sw": sw,
                      "drama": abs(sw.get("delta", 0.0))})
    cards.sort(key=lambda c: -c["drama"])
    top = cards[:6]
    ngoals = sum(len(c["tl"]["goals"]) for c in cards)

    fig = plt.figure(figsize=(12, 8.2), dpi=170)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.62, 2.4, 2.4], hspace=0.42, wspace=0.1,
                          left=0.04, right=0.96, top=0.97, bottom=0.07)
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    axt.text(0, 0.78, "THE WORLD CUP THROUGH A CHESS ENGINE", color=INK, **font(26, True))
    axt.text(0, 0.30, "A win-probability eval bar -- up = home favoured, down = away -- reacting live to every goal of WC 2026.",
             color=MUT, **font(11))
    axt.text(0, 0.02, "The {} biggest swings so far. Blue/orange = who the match favours; amber dots = goals; the engine flags the turning point.".format(len(top)),
             color=MUT, **font(9.5))
    cells = [gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0], gs[2, 1], gs[2, 2]]
    for cell, c in zip(cells, top):
        panel(fig.add_subplot(cell), c)
    fig.text(0.5, 0.018,
             "model: self-adjusting Elo seeded on 49k internationals + an in-game win-prob model (OOS log loss 0.82, calibrated) | live data: FotMob | {} goals tracked | github.com/d8maldon/hidden-timeout".format(ngoals),
             ha="center", color=MUT, **font(8))
    out = os.path.join(FIG, "wc2026_highlights.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("wrote", out)
    print("top swings:")
    for c in top:
        t = c["tl"]
        print("  {} {} {}  {:+.0f} pts at {:.0f}'".format(
            t["home"], t["score"], t["away"], 100 * c["sw"]["delta"], c["sw"]["at"]))


if __name__ == "__main__":
    main()
