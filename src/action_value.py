"""Action value: what every pass and carry is worth (soccer EPV on event data).

This is the soccer translation of the NBA Expected Possession Value model
(Cervone, D'Amour, Bornn, Goldsberry -- POINTWISE, MIT Sloan 2014 / JASA 2016),
the one whose Figure 1 shows Kawhi Leonard at 0.88 expected points and a pass to
Danny Green lifting it to 1.08. The value of a decision is value_after minus
value_before.

We do it on event data using our existing Expected Threat grid (xt_model.py):
the value of a position is xT(cell), and the value an on-ball action ADDS is

    delta_xT = xT(end cell) - xT(start cell)

That is the literal "Change in EPV" column, in soccer. Two outputs:
  1. a player leaderboard -- who moves the ball into danger the most (the soccer
     cousin of EPV-added); and
  2. a possession "stock ticker" -- one real goal buildup with every pass/carry
     annotated by the threat it added, the threat climbing to the finish.

Honest scope: this values ON-BALL actions by location only. It is offence-only
and risk-blind (a failed pass is not punished), and -- having no optical
tracking -- it cannot credit off-ball runs, space creation or defending. The
full tracking-based soccer-EPV (SoccerMap) is out of reach without that data.

    python src/action_value.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw", "events")
OUT = os.path.join(ROOT, "wc2026")
FIG = os.path.join(ROOT, "figures")
NX, NY = 16, 12
MEN = {"WC 2018", "WC 2022", "Euro 2020", "Euro 2024", "Copa America 2024",
       "AFCON 2023"}
MIN_ACTIONS = 150

BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; ACCENT = "#ffb347"
BAR = "#5e9bff"; GRN = "#3fb950"; PANEL = "#131a23"

XT = np.load(os.path.join(PROC, "xt_grid.npy")).ravel()


def cell(x, y):
    cx = int(np.clip(int(x / 120.0 * NX), 0, NX - 1))
    cy = int(np.clip(int(y / 80.0 * NY), 0, NY - 1))
    return cx * NY + cy


def val(x, y):
    return float(XT[cell(x, y)])


def scan():
    matches = pd.read_csv(os.path.join(PROC, "matches.csv"))
    men_ids = set(matches[matches.tournament.isin(MEN)].match_id.astype(str))
    players = {}
    best = None     # best goal buildup: (score, info)
    for fp in glob.glob(os.path.join(RAW, "*.json")):
        mid = os.path.splitext(os.path.basename(fp))[0]
        if mid not in men_ids:
            continue
        with open(fp, encoding="utf-8") as f:
            evs = json.load(f)
        poss = {}
        for e in evs:
            t = e.get("type", {}).get("name")
            loc = e.get("location")
            name = (e.get("player") or {}).get("name")
            team = (e.get("team") or {}).get("name")
            end = None
            if t == "Pass":
                p = e["pass"]
                if "outcome" in p:                 # incomplete: risk-blind, skip
                    continue
                end = p.get("end_location")
            elif t == "Carry":
                end = e["carry"].get("end_location")
            if end and loc and name:
                d = val(end[0], end[1]) - val(loc[0], loc[1])
                a = players.setdefault(name, {"team": team, "xt": 0.0,
                                              "pos": 0.0, "n": 0})
                a["xt"] += d
                a["pos"] += max(d, 0.0)
                a["n"] += 1
            # collect the possession sequence for buildup detection
            pid = e.get("possession")
            if pid is not None and t in ("Pass", "Carry", "Shot"):
                poss.setdefault(pid, []).append(e)

        for pid, seq in poss.items():
            shot = next((e for e in seq if e.get("type", {}).get("name") == "Shot"
                         and (e.get("shot") or {}).get("outcome", {}).get("name") == "Goal"), None)
            if not shot:
                continue
            chain = _chain(seq, shot)
            if chain is None or not (5 <= len(chain) <= 11):
                continue
            gain = chain[-1]["xt_end"] - chain[0]["xt_start"]
            if best is None or gain > best[0]:
                team = (shot.get("team") or {}).get("name")
                tour = matches.loc[matches.match_id.astype(str) == mid,
                                   "tournament"].iloc[0]
                best = (gain, {"chain": chain, "team": team, "tour": tour,
                               "scorer": (shot.get("player") or {}).get("name", "")})
    return players, best


def _chain(seq, shot):
    """ordered pass/carry actions up to (and incl.) the goal, with xT deltas"""
    out = []
    for e in seq:
        t = e.get("type", {}).get("name")
        loc = e.get("location")
        if not loc:
            continue
        if t == "Pass":
            p = e["pass"]
            if "outcome" in p:
                return None                        # a turnover broke the buildup
            end = p.get("end_location")
            kind = "pass"
        elif t == "Carry":
            end = e["carry"].get("end_location")
            kind = "carry"
        elif t == "Shot" and e is shot:
            end = loc
            kind = "GOAL"
        else:
            continue
        if not end:
            continue
        out.append({"kind": kind, "player": (e.get("player") or {}).get("name", ""),
                    "sx": loc[0], "sy": loc[1], "ex": end[0], "ey": end[1],
                    "xt_start": val(loc[0], loc[1]), "xt_end": val(end[0], end[1])})
        if kind == "GOAL":
            break
    return out if out and out[-1]["kind"] == "GOAL" else None


def pitch(ax):
    ax.set_facecolor("#16341f")
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color="#ffffff", lw=1.2, alpha=0.5)
    ax.plot([60, 60], [0, 80], color="#ffffff", lw=1.0, alpha=0.4)
    for x0 in (0, 102):
        ax.plot([x0, x0 + 18, x0 + 18, x0], [18, 18, 62, 62], color="#ffffff",
                lw=1.0, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(60 + 10 * np.cos(th), 40 + 10 * np.sin(th), color="#ffffff", lw=1.0, alpha=0.4)
    ax.set_xlim(-3, 123); ax.set_ylim(-3, 83); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_leaderboard(players, out):
    rows = [{"player": k, "team": v["team"], "xt": v["xt"], "n": v["n"]}
            for k, v in players.items() if v["n"] >= MIN_ACTIONS]
    df = pd.DataFrame(rows).sort_values("xt", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "action_value.csv"), index=False)
    top = df.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7.4), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    y = np.arange(len(top))
    ax.barh(y, top.xt, color=BAR, height=0.66, zorder=3)
    for yi, (_, r) in zip(y, top.iterrows()):
        ax.text(-0.03, yi, r.player, ha="right", va="center", color=INK,
                fontfamily="Bahnschrift", fontsize=9)
        ax.text(r.xt + 0.01, yi, "+{:.2f}".format(r.xt), va="center", color=MUT,
                fontsize=8.5, fontfamily="Bahnschrift")
    ax.set_xlim(0, top.xt.max() * 1.16)
    ax.set_ylim(-0.7, len(top) - 0.3); ax.set_yticks([])
    ax.set_xlabel("total Expected Threat added by passes + carries (goals of threat created)",
                  color=MUT, fontsize=9, fontfamily="Bahnschrift")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUT); ax.tick_params(colors=MUT)
    ax.set_title("Who moves the ball into danger?  (soccer EPV-added, on-ball)",
                 color=INK, loc="left", pad=18, fontsize=14,
                 fontfamily="Bahnschrift", fontweight="bold")
    ax.text(0, 1.02, "value of an action = xT(end) - xT(start), summed over a player's passes & carries | 314 men's national-team matches",
            transform=ax.transAxes, color=MUT, fontsize=8.3, fontfamily="Bahnschrift")
    fig.text(0.5, 0.01, "the soccer translation of NBA EPV (Cervone-D'Amour-Bornn-Goldsberry) | on-ball, location-only | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return df


def render_ticker(best, out):
    gain, info = best
    chain = info["chain"]
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=170,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(BG)
    pitch(axp)
    for a in chain:
        col = GRN if (a["xt_end"] - a["xt_start"]) > 0 else "#8b949e"
        if a["kind"] == "GOAL":
            axp.plot(a["sx"], a["sy"], "*", ms=22, color=ACCENT, mec=BG, mew=1, zorder=6)
        else:
            axp.annotate("", xy=(a["ex"], a["ey"]), xytext=(a["sx"], a["sy"]),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, alpha=0.9),
                         zorder=5)
    axp.set_title("{}'s goal vs the build-up  ({})".format(info["scorer"], info["tour"]),
                  color=INK, loc="left", pad=10, fontfamily="Bahnschrift",
                  fontsize=12, fontweight="bold")
    axp.text(0, -0.04, "green = the pass/carry raised the threat | star = the finish",
             transform=axp.transAxes, color=MUT, fontsize=8, fontfamily="Bahnschrift")

    # ticker: xT at the ball after each action
    axt.set_facecolor(PANEL)
    xs = np.arange(len(chain))
    ys = [a["xt_end"] for a in chain]
    axt.plot(xs, ys, "-o", color=BAR, mfc=INK, mec=BAR, lw=1.8, ms=5, zorder=4)
    axt.fill_between(xs, 0, ys, color=BAR, alpha=0.18)
    for i, a in enumerate(chain):
        d = a["xt_end"] - a["xt_start"]
        lab = "GOAL" if a["kind"] == "GOAL" else "{}{:.02f}".format("+" if d >= 0 else "", d)
        axt.annotate(lab, (i, ys[i]), xytext=(0, 8), textcoords="offset points",
                     ha="center", color=ACCENT if a["kind"] == "GOAL" else INK,
                     fontsize=8, fontfamily="Bahnschrift",
                     fontweight="bold" if a["kind"] == "GOAL" else "normal")
    axt.set_xticks(xs)
    axt.set_xticklabels([a["kind"][:1].upper() for a in chain], color=MUT, fontsize=8)
    axt.set_ylim(0, max(ys) * 1.28)
    axt.set_ylabel("Expected Threat at the ball", color=MUT, fontsize=9,
                   fontfamily="Bahnschrift")
    for s in ("top", "right"):
        axt.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axt.spines[s].set_color(MUT)
    axt.tick_params(colors=MUT)
    axt.set_title("the possession's 'stock ticker': each action's value added",
                  color=INK, loc="left", pad=10, fontfamily="Bahnschrift",
                  fontsize=12, fontweight="bold")
    fig.suptitle("Valuing every decision: a goal build-up, pass by pass  (soccer EPV)",
                 color=INK, x=0.5, fontsize=14, fontfamily="Bahnschrift",
                 fontweight="bold")
    fig.text(0.5, 0.01, "P (pass)  C (carry)  -> finish | value = xT(end)-xT(start), the soccer 'Change in EPV' | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(FIG, exist_ok=True)
    players, best = scan()
    df = render_leaderboard(players, os.path.join(FIG, "wc2026_action_value.png"))
    print("qualified players (>= {} actions): {}".format(MIN_ACTIONS, len(df)))
    print("\ntop 10 by Expected Threat added (passes + carries):")
    for _, r in df.head(10).iterrows():
        print("  {:26s} {:>12}  +{:.2f} xT  ({} actions)".format(
            r.player, r.team, r.xt, r.n))
    if best:
        render_ticker(best, os.path.join(FIG, "wc2026_possession_ticker.png"))
        g, info = best
        print("\nbest goal build-up: {} ({}), {} on-ball actions, +{:.3f} xT gained".format(
            info["scorer"], info["tour"], len(info["chain"]), g))
        print("figures: wc2026_action_value.png + wc2026_possession_ticker.png")


if __name__ == "__main__":
    main()
