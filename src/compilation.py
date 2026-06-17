"""WC2026 goals-and-chances compilation: the goals scored against the longest
odds (lowest xG -- the screamers and improbable finishes) next to the biggest
chances that did NOT go in (highest xG, no goal). One board, two stories, all
from real FotMob shot data.

    python src/compilation.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "wc2026")
FIG = os.path.join(ROOT, "figures")
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; GOAL = "#ffd23f"; MISS = "#f0506a"


def draw_pitch(ax):
    ax.set_facecolor("#143d2a")
    for c in ([0, 120, 0, 0], [0, 120, 80, 80], [0, 0, 0, 80], [120, 120, 0, 80],
              [60, 60, 0, 80], [102, 120, 18, 18], [102, 102, 18, 62], [102, 120, 62, 62],
              [114, 120, 30, 30], [114, 114, 30, 50], [114, 120, 50, 50],
              [0, 18, 18, 18], [18, 18, 18, 62], [0, 18, 62, 62]):
        ax.plot(c[:2], c[2:], color="#fff", lw=1.0, alpha=0.45)
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, color="#fff", lw=1.0, alpha=0.45))
    ax.set_xlim(40, 122); ax.set_ylim(-2, 82); ax.set_aspect("equal"); ax.axis("off")


def last(name):
    return name.split()[-1] if name else "?"


def main():
    shots = []
    for f in glob.glob(os.path.join(CACHE, "fm_match_*.json")):
        d = json.load(open(f, encoding="utf-8")); g = d.get("general", {})
        h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
        for s in (((d.get("content") or {}).get("shotmap") or {}).get("shots", []) or []):
            shots.append({"xg": float(s.get("expectedGoals") or 0), "ev": s.get("eventType"),
                          "pl": s.get("playerName", ""), "min": s.get("min"),
                          "x": float(s.get("x", 0)) * 1.2, "y": float(s.get("y", 0)) * 0.8,
                          "team": h.get("name") if s.get("teamId") == h.get("id") else a.get("name"),
                          "match": "{} v {}".format(h.get("name", ""), a.get("name", ""))})
    goals = sorted([s for s in shots if s["ev"] == "Goal"], key=lambda s: s["xg"])
    miss = sorted([s for s in shots if s["ev"] != "Goal"], key=lambda s: -s["xg"])
    odds = goals[:8]          # lowest-xG goals: against the odds
    want = miss[:8]           # highest-xG misses: should have scored

    fig = plt.figure(figsize=(13.5, 7.6), dpi=170); fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.03, left=0.02, right=0.98, top=0.84, bottom=0.06)
    ax = fig.add_subplot(gs[0]); draw_pitch(ax)
    # all goals as gold stars; the lowest-xG ones bigger (more improbable)
    for s in goals:
        ax.scatter(s["x"], s["y"], marker="*", s=120 + 600 * (1 - s["xg"]) if s in odds else 90,
                   facecolor=GOAL, edgecolor=BG, lw=0.8, alpha=0.95, zorder=4)
    # biggest misses as open red rings
    for s in want:
        ax.scatter(s["x"], s["y"], s=120 + 700 * s["xg"], facecolor="none", edgecolor=MISS,
                   lw=2.0, alpha=0.9, zorder=3)
    for s in odds[:5]:
        ax.annotate("{} {:.2f}".format(last(s["pl"]), s["xg"]), (s["x"], s["y"]),
                    xytext=(s["x"] - 16, s["y"] + (5 if s["y"] < 40 else -5)), color=GOAL,
                    fontsize=8.5, fontfamily="Bahnschrift", arrowprops=dict(arrowstyle="-", color=GOAL, lw=0.6))
    for s in want[:2]:
        ax.annotate("{} {:.2f}".format(last(s["pl"]), s["xg"]), (s["x"], s["y"]),
                    xytext=(s["x"] - 16, s["y"] + (5 if s["y"] < 40 else -5)), color=MISS,
                    fontsize=8.5, fontfamily="Bahnschrift", arrowprops=dict(arrowstyle="-", color=MISS, lw=0.6))
    ax.scatter([], [], marker="*", s=160, facecolor=GOAL, edgecolor=BG, label="goal (star size = improbability)")
    ax.scatter([], [], s=160, facecolor="none", edgecolor=MISS, lw=2, label="big chance missed (ring size = xG)")
    ax.legend(loc="lower left", frameon=False, labelcolor=INK, prop={"family": "Bahnschrift", "size": 9})

    axr = fig.add_subplot(gs[1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    axr.text(0, 0.99, "GOALS AGAINST THE ODDS", color=GOAL, fontfamily="Bahnschrift", fontsize=12, fontweight="bold")
    axr.text(0, 0.955, "lowest xG -- scored from where goals almost never come", color=MUT, fontfamily="Bahnschrift", fontsize=8)
    for i, s in enumerate(odds):
        y = 0.91 - i * 0.045
        axr.text(0, y, "{:.3f}".format(s["xg"]), color=GOAL, fontfamily="Bahnschrift", fontsize=9.5, fontweight="bold")
        axr.text(0.13, y, "{} {}'  ".format(last(s["pl"]), s["min"]), color=INK, fontfamily="Bahnschrift", fontsize=9)
        axr.text(0.99, y, s["match"][:22], color=MUT, fontfamily="Bahnschrift", fontsize=7.5, ha="right")
    axr.text(0, 0.5, "THE ONES THEY'LL WANT BACK", color=MISS, fontfamily="Bahnschrift", fontsize=12, fontweight="bold")
    axr.text(0, 0.465, "highest xG that did NOT go in", color=MUT, fontfamily="Bahnschrift", fontsize=8)
    for i, s in enumerate(want):
        y = 0.42 - i * 0.045
        axr.text(0, y, "{:.2f}".format(s["xg"]), color=MISS, fontfamily="Bahnschrift", fontsize=9.5, fontweight="bold")
        axr.text(0.13, y, "{} {}'  ".format(last(s["pl"]), s["min"]), color=INK, fontfamily="Bahnschrift", fontsize=9)
        axr.text(0.99, y, s["match"][:22], color=MUT, fontfamily="Bahnschrift", fontsize=7.5, ha="right")

    fig.text(0.02, 0.92, "WORLD CUP 2026: THE GOALS THAT SHOULDN'T HAVE GONE IN, AND THE CHANCES THAT SHOULD HAVE",
             color=INK, fontfamily="Bahnschrift", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.885, "{} goals and {} shots rated by xG. Mbappe's 91' winner today vs Senegal: xG 0.04 -- a goal against the odds.".format(
        len(goals), len(shots)), color=MUT, fontfamily="Bahnschrift", fontsize=10)
    fig.text(0.5, 0.02, "xG from FotMob shot data | lower xG goal = more improbable finish | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontfamily="Bahnschrift", fontsize=8)
    out = os.path.join(FIG, "wc2026_compilation.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("wrote", out)
    print("against-odds goals:", [(last(s["pl"]), round(s["xg"], 3)) for s in odds[:4]])
    print("biggest misses:", [(last(s["pl"]), round(s["xg"], 2)) for s in want[:4]])


if __name__ == "__main__":
    main()
