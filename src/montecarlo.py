"""Monte-Carlo the rest of the World Cup to a champion probability.

Takes the self-adjusting Elo ratings (already updated for every game played) and
simulates the remainder of WC2026 ten thousand times: the unplayed group games,
the group tables, who advances, and the knockout bracket to the final. Counting
how often each nation lifts the trophy gives its title probability -- refreshed
after every match-day as the ratings move.

What is rigorous here: the group stage. Real group assignments and pinned
results from the FIFA calendar; remaining games sampled as independent Poisson
scorelines with an Elo-derived supremacy (so goal difference and goals-for, the
decisive tiebreakers, are simulated, not faked); standings resolved with the
2026 tiebreaker order -- HEAD-TO-HEAD FIRST, then overall GD/GF (FIFA reversed
the old GD-first rule for 2026), then rating as the final separator.

What is approximate, and labelled as such: the knockout BRACKET. The exact FIFA
slot map (which third-placed team meets which group winner) is a conditional
495-row table not exposed by the calendar API, so we use a valid, same-group-
avoiding single-elimination reconstruction. Champion probabilities are dominated
by group advancement and team strength across five rounds and are insensitive to
the precise slotting; the per-team "reach round of 16" number is the fully
rigorous one.

    python src/montecarlo.py        # N sims -> title odds + a figure

AUDIT-DRAFT #13 (accuracy; opt-in; validate on the data box): the default
knockout resolver is a RAW Elo coin-flip (`expected(dr)`), which silently buries
the draw and is NOT the Davidson/multinomial draw model that ratings.py fits and
validates out-of-sample (the README calls this "the very engine validated
out-of-sample"). The `--draw-model` flag below resolves knockouts from those
validated P(H/D/A) probabilities instead, splitting the draw mass to a decisive
winner the way an actual knockout would (extra time / penalties favour the side
already more likely to win in regulation). It DEFAULTS OFF so the committed
sim output and the 29 passing tests are unchanged. The Poisson score sampler is
deliberately retained for the group stage, where GD/GF are real tiebreakers.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fixtures import load_fixtures, played
from ratings import (BASE, HOME_ADV, HOSTS_2026, K_WORLD_CUP, PROVISIONAL,
                     build_normalizer, elo_update, expected, load_prematch_model,
                     prematch_proba, seed_history)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "wc2026")
FIG = os.path.join(ROOT, "figures")

N_SIMS = 10000
GOAL_TOTAL = 2.6       # WC mean goals per game
SUPREMACY = 1.3        # expected goal margin per 400 Elo points
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"


def current_ratings():
    """seed from the full international history, then self-adjust on every
    finished WC2026 game"""
    seed, _ = seed_history()
    norm = build_normalizer(seed)
    df = load_fixtures(norm=norm)
    rt = dict(seed)
    for r in played(df).itertuples():
        ha = HOME_ADV if r.home in HOSTS_2026 else 0.0
        rt[r.home], rt[r.away] = elo_update(
            rt.get(r.home, PROVISIONAL), rt.get(r.away, PROVISIONAL),
            r.home_score, r.away_score, K_WORLD_CUP, ha)
    return rt, df, norm


def group_blocks(df):
    """{group_letter: {'teams':[...], 'games':[(h,a,finished,hs,as)]}}"""
    gs = df[df.group.notna() & (df.home != "") & (df.away != "")]
    blocks = {}
    for r in gs.itertuples():
        g = r.group.replace("Group ", "")
        b = blocks.setdefault(g, {"teams": set(), "games": []})
        b["teams"].update([r.home, r.away])
        fin = bool(r.finished)
        b["games"].append((r.home, r.away,
                           fin,
                           int(r.home_score) if fin else None,
                           int(r.away_score) if fin else None))
    for g in blocks:
        blocks[g]["teams"] = sorted(blocks[g]["teams"])
    return blocks


def sample_score(rh, ra, ha, rng):
    sup = ((rh + ha) - ra) / 400.0 * SUPREMACY
    lh = max((GOAL_TOTAL + sup) / 2.0, 0.15)
    la = max((GOAL_TOTAL - sup) / 2.0, 0.15)
    return int(rng.poisson(lh)), int(rng.poisson(la))


def rank_group(teams, results, rt):
    """2026 order: points, then H2H(pts,GD,GF) among tied, then overall GD/GF,
    then Elo. results = list of (home, away, hs, as) for the full group."""
    ov = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for h, a, hs, as_ in results:
        ov[h]["gf"] += hs; ov[h]["ga"] += as_
        ov[a]["gf"] += as_; ov[a]["ga"] += hs
        if hs > as_:
            ov[h]["pts"] += 3
        elif as_ > hs:
            ov[a]["pts"] += 3
        else:
            ov[h]["pts"] += 1; ov[a]["pts"] += 1

    def overall_gd(t):
        return ov[t]["gf"] - ov[t]["ga"]

    def break_tie(tied):
        S = set(tied)
        h2h = {t: {"pts": 0, "gf": 0, "ga": 0} for t in tied}
        for hh, aa, hs, as_ in results:
            if hh in S and aa in S:
                h2h[hh]["gf"] += hs; h2h[hh]["ga"] += as_
                h2h[aa]["gf"] += as_; h2h[aa]["ga"] += hs
                if hs > as_:
                    h2h[hh]["pts"] += 3
                elif as_ > hs:
                    h2h[aa]["pts"] += 3
                else:
                    h2h[hh]["pts"] += 1; h2h[aa]["pts"] += 1
        return sorted(tied, reverse=True, key=lambda t: (
            h2h[t]["pts"], h2h[t]["gf"] - h2h[t]["ga"], h2h[t]["gf"],
            overall_gd(t), ov[t]["gf"], rt.get(t, BASE)))

    # group teams by points, break ties within each points-tier by H2H-first
    order = []
    for pts in sorted({ov[t]["pts"] for t in teams}, reverse=True):
        tier = [t for t in teams if ov[t]["pts"] == pts]
        order += tier if len(tier) == 1 else break_tie(tier)
    standings = [(t, ov[t]["pts"], overall_gd(t), ov[t]["gf"]) for t in order]
    return order, standings


def sim_once(blocks, rt, rng, draw_model=None):
    # AUDIT-DRAFT #13: `draw_model` is forwarded ONLY to the knockout resolver;
    # the group stage keeps the Poisson sampler (GD/GF tiebreakers are real).
    winners, runners, thirds = {}, {}, []   # thirds: (team, group, pts, gd, gf)
    for g, b in blocks.items():
        results = []
        for h, a, fin, hs, as_ in b["games"]:
            if fin:
                results.append((h, a, hs, as_))
            else:
                ha = HOME_ADV if h in HOSTS_2026 else 0.0
                sh, sa = sample_score(rt.get(h, PROVISIONAL),
                                      rt.get(a, PROVISIONAL), ha, rng)
                results.append((h, a, sh, sa))
        order, st = rank_group(b["teams"], results, rt)
        winners[g], runners[g] = order[0], order[1]
        t = order[2]
        thirds.append((t, g, st[2][1], st[2][2], st[2][3]))

    # best 8 of 12 third-placed teams: points, GD, GF, then Elo
    thirds.sort(reverse=True, key=lambda x: (x[2], x[3], x[4], rt.get(x[0], BASE)))
    top_thirds = thirds[:8]
    advance_r16 = set(winners.values()) | set(runners.values()) \
        | {t[0] for t in top_thirds}

    bracket = build_bracket(winners, runners, top_thirds)
    champion = sim_knockout(bracket, rt, rng, draw_model)
    return advance_r16, champion


def build_bracket(winners, runners, top_thirds):
    """valid same-group-avoiding R32 (representative reconstruction).

    12 winners + 12 runners + 8 thirds = 32. Winners A-H each take a third (not
    from their own group where possible); winners I-L play runners A-D; runners
    E-L pair off. Returns a flat list of 32 teams in bracket order."""
    groups = sorted(winners)                       # ['A'..'L']
    third_by = [(t[1], t[0]) for t in top_thirds]  # (group, team)
    used = set()

    def take_third(avoid):
        for i, (g, t) in enumerate(third_by):
            if i not in used and g != avoid:
                used.add(i); return t
        for i, (g, t) in enumerate(third_by):       # fallback: allow same group
            if i not in used:
                used.add(i); return t
        return None

    slots = []
    for g in groups[:8]:                            # A..H winners vs thirds
        slots += [winners[g], take_third(g)]
    for i, g in enumerate(groups[8:]):              # I..L winners vs runners A..D
        slots += [winners[g], runners[groups[i]]]
    rem = [runners[g] for g in groups[4:]]          # runners E..L pair off
    for i in range(0, len(rem), 2):
        slots += [rem[i], rem[i + 1]]
    return slots


def advance_prob(dr, draw_model):
    """P(the home/first side advances a knockout tie) for rating gap `dr`.

    Default (draw_model is None): the raw Elo win-expectancy `expected(dr)` --
    the current behaviour, unchanged.

    AUDIT-DRAFT #13 (opt-in): when `draw_model` is supplied, use the VALIDATED
    pre-match draw model from ratings.py to get P(H/D/A), then resolve the tie to
    a decisive winner by splitting the draw mass in proportion to the two sides'
    win probabilities -- i.e. P(advance) = pH + pD * pH/(pH+pA). This is the
    standard "conditional-on-a-result" reduction: extra time / penalties land in
    favour of whoever was already more likely to win in regulation, rather than
    a 50/50 on every draw. `draw_model` is the dict from load_prematch_model()
    (the multinomial logit); if it is the sentinel "davidson", the closed-form
    Davidson fallback is used.
    """
    if draw_model is None:
        return expected(dr)
    model = None if draw_model == "davidson" else draw_model
    p = prematch_proba(dr, model)
    ph, pa = p["H"], p["A"]
    denom = ph + pa
    if denom <= 0.0:
        return 0.5
    return ph + p["D"] * (ph / denom)


def sim_knockout(slots, rt, rng, draw_model=None):
    # AUDIT-DRAFT #13: `draw_model` defaults None -> identical raw-Elo coin flip.
    teams = slots
    while len(teams) > 1:
        nxt = []
        for i in range(0, len(teams), 2):
            h, a = teams[i], teams[i + 1]
            dr = rt.get(h, PROVISIONAL) - rt.get(a, PROVISIONAL)
            dr += HOME_ADV if h in HOSTS_2026 else 0.0
            dr -= HOME_ADV if a in HOSTS_2026 else 0.0
            nxt.append(h if rng.random() < advance_prob(dr, draw_model) else a)
        teams = nxt
    return teams[0]


def render(table, n, out):
    top = table.head(16)
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=160)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["champion"], color="#5e9bff", height=0.62, zorder=3)
    ax.barh(y, top["reach_r16"], color="#21304a", height=0.62, zorder=2)
    for yi, (_, r) in zip(y, top.iterrows()):
        ax.text(-0.6, yi, r["team"], ha="right", va="center", color=INK,
                fontsize=10, fontfamily="Bahnschrift")
        ax.text(r["champion"] + 0.4, yi, "{:.1f}%".format(r["champion"]),
                va="center", color="#5e9bff", fontsize=9,
                fontfamily="Bahnschrift", fontweight="bold")
    ax.set_xlim(0, max(top["reach_r16"].max(), top["champion"].max()) * 1.18)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("probability (%)", color=MUT, fontsize=9,
                  fontfamily="Bahnschrift")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUT); ax.tick_params(colors=MUT)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#5e9bff"),
        plt.Rectangle((0, 0), 1, 1, color="#21304a")],
        labels=["win the World Cup", "reach the round of 16 (rigorous)"],
        loc="lower right", frameon=False, labelcolor=INK,
        prop={"family": "Bahnschrift", "size": 9})
    ax.set_title("Who wins the World Cup?  Self-adjusting Elo, {:,} simulations".format(n),
                 color=INK, loc="left", pad=18, fontsize=15,
                 fontfamily="Bahnschrift", fontweight="bold")
    ax.text(0, 1.02, "re-run after every match-day | group stage simulated to the 2026 H2H-first tiebreakers | knockout bracket is a representative reconstruction",
            transform=ax.transAxes, color=MUT, fontsize=8.5,
            fontfamily="Bahnschrift")
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def resolve_draw_model(mode):
    """Map a --draw-model CLI value to the `draw_model` argument sim_once wants.

    AUDIT-DRAFT #13 (opt-in): 'off' (default) -> None -> raw-Elo coin flip, the
    committed behaviour. 'logit' -> the validated multinomial draw model
    (prematch_model.json); falls back to Davidson if that file is absent.
    'davidson' -> force the closed-form Davidson fallback.
    """
    if mode == "off":
        return None, "raw-Elo coin flip (expected(dr)) [default]"
    if mode == "davidson":
        return "davidson", "validated Davidson closed-form draw model"
    # mode == "logit"
    model = load_prematch_model()
    if model is None:
        return "davidson", ("prematch_model.json absent -> Davidson fallback "
                            "(run `python src/ratings.py` to fit the logit)")
    return model, "validated multinomial-logit draw model (prematch_model.json)"


def main(draw_mode="off"):
    os.makedirs(FIG, exist_ok=True)
    rt, df, _ = current_ratings()
    blocks = group_blocks(df)
    draw_model, draw_desc = resolve_draw_model(draw_mode)
    print("groups: {}  teams: {}".format(
        len(blocks), sum(len(b["teams"]) for b in blocks.values())))
    print("knockout resolver: {}".format(draw_desc))

    rng = np.random.default_rng(7)
    champ = {}
    r16 = {}
    for _ in range(N_SIMS):
        adv, c = sim_once(blocks, rt, rng, draw_model)
        champ[c] = champ.get(c, 0) + 1
        for t in adv:
            r16[t] = r16.get(t, 0) + 1

    teams = sorted(set(champ) | set(r16))
    table = pd.DataFrame({
        "team": teams,
        "champion": [100.0 * champ.get(t, 0) / N_SIMS for t in teams],
        "reach_r16": [100.0 * r16.get(t, 0) / N_SIMS for t in teams],
        "rating": [round(rt.get(t, PROVISIONAL)) for t in teams],
    }).sort_values("champion", ascending=False).reset_index(drop=True)
    # AUDIT-DRAFT #13: default ('off') still writes the committed sim_probs.csv;
    # the opt-in modes write a SEPARATE file so the maintainer can diff them and
    # nothing committed is clobbered while validating.
    out_csv = "sim_probs.csv" if draw_mode == "off" \
        else "sim_probs_drawmodel_{}.csv".format(draw_mode)
    table.to_csv(os.path.join(OUT, out_csv), index=False)

    print("\ntitle odds (top 12):")
    for _, r in table.head(12).iterrows():
        print("  {:24s} win {:5.1f}%   R16 {:5.1f}%   (Elo {})".format(
            r["team"], r["champion"], r["reach_r16"], int(r["rating"])))
    # default mode keeps writing the committed figure; opt-in modes write their
    # own figure file so the canonical PNG is not overwritten during validation.
    fig_name = "wc2026_champion.png" if draw_mode == "off" \
        else "wc2026_champion_drawmodel_{}.png".format(draw_mode)
    render(table, N_SIMS, os.path.join(FIG, fig_name))
    print("\nwrote wc2026/{} + figures/{}".format(out_csv, fig_name))


if __name__ == "__main__":
    # AUDIT-DRAFT #13: --draw-model is opt-in and defaults to 'off' (the exact
    # current behaviour). 'logit'/'davidson' resolve knockouts from the validated
    # draw model instead of the raw Elo coin flip.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draw-model", dest="draw_mode", default="off",
                    choices=["off", "logit", "davidson"],
                    help="knockout resolver: 'off' (default, raw-Elo coin flip, "
                         "unchanged) or the validated 'logit'/'davidson' draw "
                         "model (AUDIT-DRAFT #13, opt-in)")
    args = ap.parse_args()
    main(draw_mode=args.draw_mode)
