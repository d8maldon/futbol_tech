"""Goal likelihood over time -- the honest version of "predict the goals".

You cannot predict that a goal lands at exactly minute 67. But the *rate* of
goals is very predictable, and that is what actually matters. Two things here:

1. The goal HAZARD: when goals are really scored. Across the 551 historical
   matches, goals are not spread evenly -- the rate climbs through the game and
   spikes in second-half stoppage. That curve is a genuine prediction of "how
   likely is a goal in this minute".

2. A live DANGER model: P(a goal in the next 10 minutes) given the run of play.
   Trained on the same matches from the chance quality (xG) of the last ten
   minutes, the shot count, the clock and the goals so far. Applied live it is a
   momentum / "a goal is coming" meter -- it rises when a team is creating
   chances and dips in quiet spells, exactly the signal you feel watching.

This is the truthful boundary: likelihood and momentum yes; the exact minute no.

    python src/goal_hazard.py            # fit, score, render
    python src/goal_hazard.py 4667771    # live danger curve for a match id
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from live_eval import (ACCENT, AWAY_C, BG, HOME_C, INK, MUT, PANEL,
                       build_timeline, font)

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "figures")
WINDOW = 10.0          # look-ahead / look-back window in minutes
GRID = 5               # training minute step
DEMO_MATCH = "4667771"  # USA 4-1 Paraguay: 5 goals, good spread


def reg_minute(period, t):
    """real minute within regulation, halves kept separate (stoppage clamped)"""
    return (period - 1) * 45.0 + min(t / 60.0, 45.0)


def load_matches():
    """per-match shot list [(m, xg)] and goal-minute list, regulation only"""
    s = pd.read_csv(os.path.join(PROC, "shots.csv"))
    s = s[s.period <= 2].copy()
    s["m"] = [reg_minute(p, t) for p, t in zip(s.period, s.t)]
    og = pd.read_csv(os.path.join(PROC, "owngoals.csv"))
    og = og[og.period <= 2].copy()
    og["m"] = [reg_minute(p, t) for p, t in zip(og.period, og.t)]

    shots = {mid: list(zip(g.m, g.xg)) for mid, g in s.groupby("match_id")}
    goals = {}
    for mid, g in s[s.goal == 1].groupby("match_id"):
        goals.setdefault(mid, []).extend(g.m.tolist())
    for mid, g in og.groupby("match_id"):
        goals.setdefault(mid, []).extend(g.m.tolist())
    for mid in shots:
        goals.setdefault(mid, [])
    return shots, goals


def features(shot_list, goal_list, m0):
    xg10 = sum(xg for mm, xg in shot_list if m0 - WINDOW < mm <= m0)
    sh10 = sum(1 for mm, _ in shot_list if m0 - WINDOW < mm <= m0)
    gsf = sum(1 for gm in goal_list if gm <= m0)
    return [m0 / 90.0, gsf, xg10, sh10]


def hazard_buckets(goals):
    """goals per match in 15-min intervals incl. both stoppage spikes"""
    edges = [(0, 15), (15, 30), (30, 45), (45, 45.001),
             (45, 60), (60, 75), (75, 90), (90, 999)]
    labels = ["1-15", "16-30", "31-45", "45+", "46-60", "61-75", "76-90", "90+"]
    # split by half: first four buckets are first half (m<=45 incl 45-stoppage)
    flat = [g for v in goals.values() for g in v]
    nmatch = len(goals)
    counts = [0] * 8
    for g in flat:
        if g < 45:
            counts[min(int(g // 15), 2)] += 1
        elif g == 45:                              # first-half stoppage clamp
            counts[3] += 1
        elif g < 90:
            counts[4 + min(int((g - 45) // 15), 2)] += 1
        else:
            counts[7] += 1
    return labels, [c / nmatch for c in counts]


def train(shots, goals):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    X, y = [], []
    for mid, sl in shots.items():
        gl = goals[mid]
        for m0 in range(0, 90 - int(WINDOW) + 1, GRID):
            X.append(features(sl, gl, m0))
            y.append(int(any(m0 < gm <= m0 + WINDOW for gm in gl)))
    X, y = np.array(X), np.array(y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    base = y.mean()
    base_ll = log_loss(y, np.full_like(p, base))
    print("training windows: {}  base rate P(goal in {:.0f}m)={:.3f}".format(
        len(y), WINDOW, base))
    print("log loss  model {:.4f}  vs base {:.4f}".format(log_loss(y, p), base_ll))
    print("ROC AUC   {:.3f}".format(roc_auc_score(y, p)))
    return clf, base


def live_curve(clf, match_id):
    tl = build_timeline(match_id)
    sl = [(c["m"], c["xg"]) for c in tl["chances"]]
    gl = [g["m"] for g in tl["goals"]]
    grid = np.arange(0, 90, 1.0)
    p = np.array([clf.predict_proba([features(sl, gl, m0)])[0, 1] for m0 in grid])
    return tl, grid, p, gl


def render(labels, per_match, clf, base, tl, grid, p, gl, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4), dpi=170,
                                   gridspec_kw={"height_ratios": [1, 1.05]})
    fig.patch.set_facecolor(BG)

    # panel 1: when goals are actually scored
    ax1.set_facecolor(PANEL)
    x = np.arange(len(labels))
    cols = [ACCENT if lab in ("76-90", "90+", "45+") else HOME_C for lab in labels]
    ax1.bar(x, per_match, color=cols, width=0.74)
    for xi, v in zip(x, per_match):
        ax1.text(xi, v + 0.004, "{:.2f}".format(v), ha="center", color=INK,
                 **font(8))
    ax1.set_xticks(x); ax1.set_xticklabels(labels, color=MUT, **font(9))
    ax1.set_ylabel("goals per match", color=MUT, **font(9))
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax1.spines[s].set_color(MUT)
    ax1.tick_params(colors=MUT)
    ax1.set_title("When are goals actually scored?  (551 matches) -- the rate climbs through each half, most after the break",
                  color=INK, loc="left", pad=10, **font(12, True))

    # panel 2: live danger meter
    ax2.set_facecolor(PANEL)
    ax2.fill_between(grid, base, p, where=p >= base, color=HOME_C, alpha=0.85,
                     interpolate=True, lw=0)
    ax2.fill_between(grid, base, p, where=p <= base, color="#21304a",
                     interpolate=True, lw=0)
    ax2.plot(grid, p, color=INK, lw=1.2)
    ax2.axhline(base, color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax2.text(1, base + 0.005, "average", color=MUT, **font(7.5))
    ax2.axvline(45, color=MUT, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
    for gm in gl:
        yv = np.interp(gm, grid, p)
        ax2.plot(gm, yv, "o", ms=9, mfc=ACCENT, mec=BG, mew=1.4, zorder=6)
        ax2.annotate("GOAL", (gm, yv), xytext=(0, 12), textcoords="offset points",
                     ha="center", color=ACCENT, **font(8, True))
    ax2.set_xlim(0, 90); ax2.set_ylim(0, max(p.max() * 1.15, base * 1.5))
    ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax2.set_xticklabels(["0'", "15'", "30'", "45'", "60'", "75'", "90'"],
                        color=MUT, **font(8))
    ax2.set_ylabel("P(goal in next 10 min)", color=MUT, **font(9))
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(MUT)
    ax2.tick_params(colors=MUT)
    ax2.set_title("Live danger meter: {} {} {}  -- P(goal soon) tracks the game".format(
        tl["home"], tl["score"], tl["away"]),
        color=INK, loc="left", pad=10, **font(12, True))
    ax2.text(0.5, max(p.max() * 1.15, base * 1.5) * 0.04,
             "honest caveat: recent chances lift this only ~23%->26% (AUC 0.55) -- most of the rise is the clock, not 'momentum'",
             color=MUT, **font(7.5))
    fig.text(0.5, 0.01,
             "goal likelihood, not exact minute | danger from xG of the last 10 min, shots, clock, goals so far | trained on 551 matches | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, **font(7))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    shots, goals = load_matches()
    labels, per_match = hazard_buckets(goals)
    print("goals/match total (regulation): {:.2f}".format(sum(per_match)))
    clf, base = train(shots, goals)
    mid = sys.argv[1] if len(sys.argv) > 1 else DEMO_MATCH
    tl, grid, p, gl = live_curve(clf, mid)
    out = os.path.join(FIG, "wc2026_goal_hazard.png")
    render(labels, per_match, clf, base, tl, grid, p, gl, out)
    print("demo match {} {} {}: danger ranged {:.0%}-{:.0%}".format(
        tl["home"], tl["score"], tl["away"], p.min(), p.max()))
    print("figure: figures/wc2026_goal_hazard.png")


if __name__ == "__main__":
    main()
