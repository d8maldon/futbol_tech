"""Anytime-goalscorer model: P(a player scores at least once in a match).

The finishing counterpart to the goal-hazard model. A player's chances are
summarised by their expected goals (xG) per appearance -- a rate, not a count --
and a Poisson turns that into the probability they score at least once:

    P(>=1 goal) = 1 - exp(-lambda),  lambda = xG per appearance

xG per appearance is shrunk toward the league average (empirical-Bayes style) so
a striker with two lucky games does not top the list on noise. Rates come from
the raw StatsBomb events (player name + statsbomb_xg) over the men's national-team
pool, and we check them against who has actually scored at WC2026.

Honest limits: we have no per-player MINUTES, so a rate is xG per match-with-a-
shot (a sub who plays 20 minutes looks quieter than they are); no opponent
adjustment; penalties are included (a goal is a goal) but flagged.

    python src/goalscorer.py
"""
import glob
import json
import os
import unicodedata

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
MEN = {"WC 2018", "WC 2022", "Euro 2020", "Euro 2024", "Copa America 2024",
       "AFCON 2023"}
MIN_APPS = 4          # qualification: at least this many shooting appearances
SHRINK = 3.0          # pseudo-appearances pulling a rate toward league mean

BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; ACCENT = "#ffb347"
BAR = "#5e9bff"; SCORED = "#3fb950"; PANEL = "#131a23"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def name_key(s):
    """order-insensitive key so 'Hwang Hee-chan' ~ 'Hee-Chan Hwang' match"""
    return " ".join(sorted(fold(s).split()))


def build_player_table():
    """aggregate per-player shooting from raw events over the men's pool"""
    cached = os.path.join(OUT, "goal_threat.csv")
    matches = pd.read_csv(os.path.join(PROC, "matches.csv"))
    men_ids = set(matches[matches.tournament.isin(MEN)].match_id.astype(str))
    agg = {}   # player -> dict
    for fp in glob.glob(os.path.join(RAW, "*.json")):
        mid = os.path.splitext(os.path.basename(fp))[0]
        if mid not in men_ids:
            continue
        with open(fp, encoding="utf-8") as f:
            evs = json.load(f)
        seen = set()
        for e in evs:
            if e.get("type", {}).get("name") != "Shot":
                continue
            sh = e.get("shot", {})
            if sh.get("type", {}).get("name") == "Penalty Shootout":
                continue                            # exclude shootouts
            name = (e.get("player") or {}).get("name")
            team = (e.get("team") or {}).get("name")
            if not name:
                continue
            a = agg.setdefault(name, {"team": team, "xg": 0.0, "shots": 0,
                                      "goals": 0, "pens": 0, "apps": set()})
            a["xg"] += float(sh.get("statsbomb_xg") or 0.0)
            a["shots"] += 1
            a["apps"].add(mid)
            if sh.get("outcome", {}).get("name") == "Goal":
                a["goals"] += 1
            if sh.get("type", {}).get("name") == "Penalty":
                a["pens"] += 1
            seen.add(name)
    rows = []
    for name, a in agg.items():
        apps = len(a["apps"])
        if apps < MIN_APPS:
            continue
        rows.append({"player": name, "team": a["team"], "apps": apps,
                     "xg": round(a["xg"], 2), "shots": a["shots"],
                     "goals": a["goals"], "pens": a["pens"]})
    df = pd.DataFrame(rows)
    mu = df.xg.sum() / df.apps.sum()                # league mean xG per app
    df["xg_per_app"] = df.xg / df.apps
    df["lambda"] = (df.xg + SHRINK * mu) / (df.apps + SHRINK)
    df["p_score"] = 1.0 - np.exp(-df["lambda"])
    df = df.sort_values("p_score", ascending=False).reset_index(drop=True)
    df.to_csv(cached, index=False)
    return df, mu


def wc2026_scorers():
    """name-keys of players who have scored at WC2026 (live feed)"""
    keys = {}
    for fp in glob.glob(os.path.join(ROOT, "data", "wc2026", "fm_match_*.json")):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        evs = ((d.get("content", {}).get("matchFacts") or {})
               .get("events") or {}).get("events", [])
        for e in evs:
            if e.get("type") == "Goal" and not e.get("ownGoal"):
                nm = (e.get("player") or {}).get("name")
                if nm:
                    keys[name_key(nm)] = nm
    return keys


def render(df, note, out):
    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    y = np.arange(len(top))
    ax.barh(y, top.p_score * 100, color=BAR, height=0.66, zorder=3)
    for yi, (_, r) in zip(y, top.iterrows()):
        ax.text(-1.2, yi, "{}".format(r.player), ha="right", va="center",
                color=INK, fontfamily="Bahnschrift", fontsize=9)
        ax.text(r.p_score * 100 + 0.6, yi, "{:.0f}%".format(r.p_score * 100),
                va="center", color=MUT, fontsize=8.5, fontfamily="Bahnschrift")
    ax.set_xlim(0, df.head(20).p_score.max() * 100 * 1.28)
    ax.set_ylim(-0.7, len(top) - 0.3); ax.set_yticks([])
    ax.set_xlabel("P(scores at least once in a match)  %", color=MUT,
                  fontsize=9, fontfamily="Bahnschrift")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUT); ax.tick_params(colors=MUT)
    ax.text(0.97, 0.06, note, transform=ax.transAxes, ha="right", va="bottom",
            color=ACCENT, fontsize=8.7, fontfamily="Bahnschrift",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL,
                      edgecolor="#2a3340", lw=0.8))
    ax.set_title("Anytime goalscorer: who's most likely to score?",
                 color=INK, loc="left", pad=18, fontsize=15,
                 fontfamily="Bahnschrift", fontweight="bold")
    ax.text(0, 1.02, "P(>=1 goal) = 1 - exp(-xG per appearance), shrunk to the mean | 2018-2024 rates, filtered to actual WC2026 squads | penalties included",
            transform=ax.transAxes, color=MUT, fontsize=8.3,
            fontfamily="Bahnschrift")
    fig.text(0.5, 0.01, "no per-player minutes (rate = xG per match-with-a-shot) | even the top name is a coin-flip -- finishing is a long tail | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df, mu = build_player_table()
    n_all = len(df)
    # restrict to players actually in a WC2026 squad (team-scoped: drops retired
    # players AND non-qualified nations like Venezuela)
    from squads import current_squads_by_team, in_wc2026
    by_team = current_squads_by_team()
    if by_team:
        df = df[df.apply(lambda r: in_wc2026(r.player, r.team, by_team), axis=1)].reset_index(drop=True)
    else:
        print("WARNING: no wc2026_squads.json -> run src/squads.py; board UNFILTERED")
    print("WC2026-squad goalscorers: {} (from {} qualified)".format(len(df), n_all))
    print("\ntop 12 anytime-goalscorer probabilities:")
    for _, r in df.head(12).iterrows():
        print("  {:24s} {:>14}  {:.0%}  ({} g, {:.1f} xG in {} apps)".format(
            r.player, r.team, r.p_score, r.goals, r.xg, r.apps))

    scored = wc2026_scorers()
    # FotMob uses short names, StatsBomb full legal names -> match when the
    # scorer's tokens are a subset of a historical player's tokens.
    hist_tokens = [(i, set(fold(p).split())) for i, p in enumerate(df.player)]
    matched_idx = set()

    def match(scorer_name):
        st = set(fold(scorer_name).split())
        cands = [i for i, ht in hist_tokens if st and st <= ht]
        if not cands:
            return None
        return max(cands, key=lambda i: df.iloc[i].apps)   # most-established

    print("\nvalidation -- WC2026 scorers' pre-tournament rank (of {}):".format(len(df)))
    best = None
    for _, nm in sorted(scored.items()):
        i = match(nm)
        if i is not None:
            matched_idx.add(i)
            print("  {:22s} ranked #{:>3}  P {:.0%}".format(
                nm, i + 1, df.iloc[i].p_score))
            if best is None or i < best[1]:
                best = (nm, i, df.iloc[i].p_score)
        else:
            print("  {:22s} debut/no pool history".format(nm))
    hits = len(matched_idx)
    print("matched {}/{} WC2026 scorers to a pre-tournament rate".format(
        hits, len(scored)))
    if best is not None:
        note = ("WC2026 scorers so far ranked LOW:\nbest pre-ranked = {} (#{}, {:.0%})\n"
                "{}/{} even had pool history -- goals come from a long tail").format(
            best[0], best[1] + 1, best[2], hits, len(scored))
    else:
        note = "WC2026 scorers so far: none in our historical pool yet"
    render(df, note, os.path.join(FIG, "wc2026_goalscorer.png"))
    print("figure: figures/wc2026_goalscorer.png")


if __name__ == "__main__":
    main()
