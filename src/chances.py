"""WC2026 chance board: every shot placed on the pitch and rated by its
goal-likelihood (xG) -- the goals that were always coming and the big chances
that should have scored. Real FotMob shot data; xG is the established model for
"how likely was this a goal".

    python src/chances.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "wc2026")
FIG = os.path.join(ROOT, "figures")
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"
GOAL = "#ffd23f"; SAVE = "#5e9bff"; MISS = "#7d8590"
PL, PW = 120.0, 80.0


def draw_pitch(ax):
    ax.set_facecolor("#143d2a")
    for c in ([0, 120, 0, 0], [0, 120, 80, 80], [0, 0, 0, 80], [120, 120, 0, 80],
              [60, 60, 0, 80], [102, 120, 18, 18], [102, 102, 18, 62], [102, 120, 62, 62],
              [114, 120, 30, 30], [114, 114, 30, 50], [114, 120, 50, 50],
              [0, 18, 18, 18], [18, 18, 18, 62], [0, 18, 62, 62]):
        ax.plot(c[:2], c[2:], color="#ffffff", lw=1.0, alpha=0.5)
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, color="#fff", lw=1.0, alpha=0.5))
    ax.set_xlim(-2, 122); ax.set_ylim(-2, 82); ax.set_aspect("equal"); ax.axis("off")


def main():
    shots = []
    for f in glob.glob(os.path.join(CACHE, "fm_match_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for s in (((d.get("content") or {}).get("shotmap") or {}).get("shots", []) or []):
            xg = float(s.get("expectedGoals") or 0)
            shots.append({"x": float(s.get("x", 0)) * 1.2, "y": float(s.get("y", 0)) * 0.8,
                          "xg": xg, "ev": s.get("eventType"), "pl": s.get("playerName", ""),
                          "min": s.get("min")})
    goals = [s for s in shots if s["ev"] == "Goal"]
    other = [s for s in shots if s["ev"] != "Goal"]
    big_miss = sorted([s for s in other if s["xg"] >= 0.3], key=lambda s: -s["xg"])

    fig = plt.figure(figsize=(12, 8), dpi=170); fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.04, left=0.03, right=0.97, top=0.86, bottom=0.05)
    ax = fig.add_subplot(gs[0]); draw_pitch(ax)
    for s in other:
        col = SAVE if s["ev"] in ("AttemptSaved", "Post") else MISS
        ax.scatter(s["x"], s["y"], s=40 + 900 * s["xg"], facecolor=col, edgecolor="none",
                   alpha=0.35, zorder=2)
    for s in goals:
        ax.scatter(s["x"], s["y"], s=60 + 900 * s["xg"], marker="*", facecolor=GOAL,
                   edgecolor=BG, lw=0.8, alpha=0.95, zorder=4)
    # annotate the biggest chances (highest xG, goals + misses)
    for s in sorted(shots, key=lambda s: -s["xg"])[:6]:
        ax.annotate("{} {:.2f}{}".format(s["pl"].split()[-1] if s["pl"] else "?", s["xg"],
                    "" if s["ev"] == "Goal" else " (" + s["ev"].replace("Attempt", "") + ")"),
                    (s["x"], s["y"]), xytext=(s["x"] - 22, s["y"] + (6 if s["y"] < 40 else -6)),
                    color=INK, fontsize=8, fontfamily="Bahnschrift",
                    arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
    ax.scatter([], [], marker="*", s=180, facecolor=GOAL, edgecolor=BG, label="goal")
    ax.scatter([], [], s=180, facecolor=SAVE, alpha=0.5, label="saved / off the woodwork")
    ax.scatter([], [], s=180, facecolor=MISS, alpha=0.5, label="off target / blocked")
    ax.legend(loc="lower left", frameon=False, labelcolor=INK, prop={"family": "Bahnschrift", "size": 9})

    axr = fig.add_subplot(gs[1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    axr.text(0, 0.97, "BIGGEST CHANCES", color=INK, fontfamily="Bahnschrift", fontsize=13, fontweight="bold")
    axr.text(0, 0.93, "by goal-likelihood (xG)", color=MUT, fontfamily="Bahnschrift", fontsize=9)
    for i, s in enumerate(sorted(shots, key=lambda s: -s["xg"])[:14]):
        y = 0.87 - i * 0.058
        mark = "GOAL" if s["ev"] == "Goal" else s["ev"].replace("Attempt", "")
        col = GOAL if s["ev"] == "Goal" else (SAVE if mark in ("Saved", "Post") else MUT)
        axr.text(0, y, "{:.2f}".format(s["xg"]), color=col, fontfamily="Bahnschrift", fontsize=10, fontweight="bold")
        axr.text(0.16, y, "{} {}'".format((s["pl"][:16] or "?"), s["min"]), color=INK, fontfamily="Bahnschrift", fontsize=9)
        axr.text(0.99, y, mark, color=col, fontfamily="Bahnschrift", fontsize=8, ha="right")

    fig.text(0.03, 0.93, "WORLD CUP 2026: EVERY CHANCE, RATED BY GOAL PROBABILITY", color=INK,
             fontfamily="Bahnschrift", fontsize=20, fontweight="bold")
    fig.text(0.03, 0.895, "{} shots placed by location and sized by xG. {} goals (stars); {} 'big chances' (xG>=0.30) went begging -- including Spain's 0.53 off the post.".format(
        len(shots), len(goals), len(big_miss)), color=MUT, fontfamily="Bahnschrift", fontsize=10)
    fig.text(0.5, 0.015, "xG from FotMob shot data | the higher the xG, the more certain a goal | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontfamily="Bahnschrift", fontsize=8)
    out = os.path.join(FIG, "wc2026_chances.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("wrote", out, "|", len(shots), "shots,", len(goals), "goals,", len(big_miss), "big chances missed")


if __name__ == "__main__":
    main()
