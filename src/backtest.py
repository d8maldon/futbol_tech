"""Leakage-free walk-forward backtest of the pre-match predictor.

The honest test of "can it predict the games?": walk the WC2026 matches in
kickoff order and, for each one, predict P(H/D/A) using ONLY ratings built from
strictly-earlier results -- then reveal the result and roll the ratings forward.
A prediction can never see its own outcome, and the verify_no_leak() check
proves it (recomputing each snapshot from scratch on the earlier matches only).

Scored on Ranked Probability Score (the right metric for an ordered 1X2
outcome) and log loss, against three baselines through the identical loop:
uniform, the historical base rate (climatology), and a FROZEN-seed model that
never updates -- the control for whether self-adjustment is doing anything.

With only a handful of games played this is illustrative, not significant; every
figure leads with n and we make no significance claims yet.

    python src/backtest.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fixtures import load_fixtures, played
from ratings import (BASE, HOME_ADV, HOSTS_2026, K_WORLD_CUP, PROVISIONAL,
                     build_normalizer, elo_update, fit_kappa,
                     load_prematch_model, prematch_proba, seed_history)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "wc2026")
FIG = os.path.join(ROOT, "figures")
ORDER = ["A", "D", "H"]   # ordinal axis: away win < draw < home win

BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"
CH = "#3fb950"; CW = "#f85149"            # correct / wrong
CLR = {"H": "#5e9bff", "D": "#8b949e", "A": "#ff7a1a"}


def outcome_of(hs, as_):
    return "H" if hs > as_ else ("A" if as_ > hs else "D")


def rps(p, o):
    """ranked probability score on the [A,D,H] ordinal axis (0 = perfect)"""
    cp = co = s = 0.0
    for k in ORDER[:-1]:
        cp += p[k]; co += 1.0 if o == k else 0.0
        s += (cp - co) ** 2
    return s / (len(ORDER) - 1)


def logloss(p, o):
    return -np.log(max(p[o], 1e-12))


def brier(p, o):
    return sum((p[k] - (1.0 if o == k else 0.0)) ** 2 for k in ORDER)


def dr_of(rt, h, a):
    dr = rt.get(h, PROVISIONAL) - rt.get(a, PROVISIONAL)
    return dr + (HOME_ADV if h in HOSTS_2026 else 0.0)


def walk_forward(seed, pl, model):
    """predict each played match from strictly-earlier ratings, then update"""
    rt = dict(seed)
    rows = []
    for r in pl.itertuples():
        h, a, hs, as_ = r.home, r.away, r.home_score, r.away_score
        o = outcome_of(hs, as_)
        rows.append({
            "date": r.kickoff[:10], "home": h, "away": a,
            "score": "{}-{}".format(hs, as_), "outcome": o,
            "dr_live": dr_of(rt, h, a), "dr_frozen": dr_of(seed, h, a),
            "p_live": prematch_proba(dr_of(rt, h, a), model),
            "p_frozen": prematch_proba(dr_of(seed, h, a), model),
        })
        nh, na = elo_update(rt.get(h, PROVISIONAL), rt.get(a, PROVISIONAL),
                            hs, as_, K_WORLD_CUP,
                            HOME_ADV if h in HOSTS_2026 else 0.0)
        rt[h], rt[a] = nh, na
    return rows, rt


def verify_no_leak(seed, pl, rows):
    """rebuild each prediction's rating gap from scratch on matches[:t] only and
    assert it equals the incremental walk-forward value. Because the from-scratch
    pass applies ONLY strictly-earlier matches, it cannot depend on match t's own
    result or any later one -- so equality proves the live loop never leaked."""
    pl = pl.reset_index(drop=True)
    for t in range(len(pl)):
        rt = dict(seed)
        for j in range(t):                       # apply only strictly-earlier
            rj = pl.iloc[j]
            nh, na = elo_update(rt.get(rj.home, PROVISIONAL),
                                rt.get(rj.away, PROVISIONAL),
                                int(rj.home_score), int(rj.away_score),
                                K_WORLD_CUP,
                                HOME_ADV if rj.home in HOSTS_2026 else 0.0)
            rt[rj.home], rt[rj.away] = nh, na
        row = pl.iloc[t]
        scratch = dr_of(rt, row.home, row.away)
        assert abs(scratch - rows[t]["dr_live"]) < 1e-9, \
            "leak at match {}: {} vs {}".format(t, scratch, rows[t]["dr_live"])
    return True


def score(rows, key, base_rate):
    """mean RPS / log loss / accuracy for a model column, plus baselines."""
    def agg(getter):
        rp = [rps(getter(r), r["outcome"]) for r in rows]
        ll = [logloss(getter(r), r["outcome"]) for r in rows]
        acc = [1.0 if max(getter(r), key=getter(r).get) == r["outcome"] else 0.0
               for r in rows]
        return np.mean(rp), np.mean(ll), np.mean(acc)
    uni = {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}
    return {
        key: agg(lambda r: r["p_live"]),
        "frozen": agg(lambda r: r["p_frozen"]),
        "climatology": agg(lambda r: base_rate),
        "uniform": agg(lambda r: uni),
    }


def render(rows, metrics, n, out):
    fig = plt.figure(figsize=(12, 0.95 * len(rows) + 3.4), dpi=160)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(rows), 2.4], hspace=0.32)
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG)

    for i, r in enumerate(reversed(rows)):
        y = i
        p = r["p_live"]; pick = max(p, key=p.get)
        ok = pick == r["outcome"]
        ax.text(0.0, y, "{}".format(r["date"]), color=MUT, fontsize=8,
                va="center", fontfamily="Bahnschrift")
        ax.text(0.10, y, "{:>18}  {}  {:<18}".format(
            r["home"], r["score"], r["away"]), color=INK, fontsize=10,
            va="center", fontfamily="Consolas")
        # stacked probability bar A|D|H
        x0 = 0.56; w = 0.30
        for k in ["H", "D", "A"]:
            seg = p[k] * w
            ax.add_patch(plt.Rectangle((x0, y - 0.28), seg, 0.56,
                         color=CLR[k], alpha=0.92))
            if p[k] > 0.12:
                ax.text(x0 + seg / 2, y, "{:.0%}".format(p[k]), ha="center",
                        va="center", color="#0d1117", fontsize=8,
                        fontweight="bold", fontfamily="Bahnschrift")
            x0 += seg
        ax.text(0.89, y, "pick {}".format(pick), color=INK, fontsize=9,
                va="center", fontfamily="Bahnschrift")
        # a drawn game is not a "miss" -- a single argmax pick can never be a draw
        if r["outcome"] == "D" and not ok:
            mark, mc, fs = "draw", MUT, 9
        else:
            mark, mc, fs = ("OK", CH, 11) if ok else ("X", CW, 11)
        ax.text(0.97, y, mark, color=mc, fontsize=fs, va="center",
                fontweight="bold", fontfamily="Bahnschrift")
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axis("off")
    ax.set_title("Did it call the games already played?  Leakage-free walk-forward, n={}".format(n),
                 color=INK, loc="left", pad=14, fontsize=15,
                 fontfamily="Bahnschrift", fontweight="bold")
    ax.text(0, len(rows) - 0.45,
            "each prediction used ONLY results before kickoff   |   bar = P(home / draw / away)   |   RPS / log loss (below) are the proper scores, not pick-accuracy",
            color=MUT, fontsize=8.5, fontfamily="Bahnschrift")

    # metrics panel
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor(BG)
    names = ["model", "frozen", "climatology", "uniform"]
    rpsv = [metrics[n_][0] for n_ in names]
    xs = np.arange(len(names))
    bars = ax2.bar(xs, rpsv, color=["#5e9bff", "#6e7681", "#8b949e", "#30363d"],
                   width=0.6)
    for x, v in zip(xs, rpsv):
        ax2.text(x, v + 0.004, "{:.3f}".format(v), ha="center", color=INK,
                 fontsize=9, fontfamily="Bahnschrift")
    ax2.set_xticks(xs); ax2.set_xticklabels(
        ["model\n(self-adjusting)", "frozen\nseed", "base rate", "uniform"],
        color=INK, fontsize=9, fontfamily="Bahnschrift")
    ax2.set_ylabel("mean RPS (lower better)", color=MUT, fontsize=9,
                   fontfamily="Bahnschrift")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(MUT)
    ax2.tick_params(colors=MUT)
    ax2.set_title("Ranked Probability Score vs baselines  (n={}, illustrative -- too few games for significance)".format(n),
                  color=INK, loc="left", fontsize=11, pad=8,
                  fontfamily="Bahnschrift", fontweight="bold")
    fig.text(0.5, 0.01, "WC2026 self-adjusting Elo predictor | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontsize=8, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    seed, pairs = seed_history()
    norm = build_normalizer(seed)
    model = load_prematch_model()

    # base rate (climatology) from the pre-2026 international outcomes
    out = np.array([o for _dr, o, _y in pairs])
    base_rate = {k: float((out == k).mean()) for k in ORDER}

    df = load_fixtures(norm=norm)
    pl = played(df)
    print("played matches: {}".format(len(pl)))

    rows, final_rt = walk_forward(seed, pl, model)
    assert verify_no_leak(seed, pl, rows)
    print("leakage guard passed: every prediction used only pre-kickoff results")
    metrics = score(rows, "model", base_rate)
    n = len(rows)
    print("\nproper scores (lower better) -- RPS and log loss are the honest metrics:")
    for k in ("model", "frozen", "climatology", "uniform"):
        rp, ll, acc = metrics[k]
        print("  {:12s} RPS {:.3f}  logloss {:.3f}  pick-acc {:.0%}".format(
            k, rp, ll, acc))
    # pick-accuracy is a poor metric here: a single argmax pick can never be a
    # draw, so drawn games are unwinnable by construction. Report the decisive
    # subset separately.
    dec = [r for r in rows if r["outcome"] != "D"]
    ndraw = n - len(dec)
    dec_hits = sum(1 for r in dec if max(r["p_live"], key=r["p_live"].get) == r["outcome"])
    print("  decisive games: {}/{} called right ({:.0%})  |  {} of {} ended level "
          "(a single pick cannot call a draw)".format(
              dec_hits, len(dec), dec_hits / max(len(dec), 1), ndraw, n))

    # write per-match predictions for played + all known-team upcoming fixtures
    csv_rows = []
    for r in rows:
        p = r["p_live"]
        csv_rows.append({"date": r["date"], "home": r["home"], "away": r["away"],
                         "status": "played", "p_H": round(p["H"], 3),
                         "p_D": round(p["D"], 3), "p_A": round(p["A"], 3),
                         "pick": max(p, key=p.get), "score": r["score"],
                         "outcome": r["outcome"],
                         "correct": int(max(p, key=p.get) == r["outcome"]),
                         "rps": round(rps(p, r["outcome"]), 3)})
    upcoming = df[(~df.finished) & (df.home != "") & (df.away != "")
                  & df.home_id.notna() & df.away_id.notna()]
    for r in upcoming.itertuples():
        if r.home not in final_rt and r.away not in final_rt:
            continue
        p = prematch_proba(dr_of(final_rt, r.home, r.away), model)
        csv_rows.append({"date": r.kickoff[:10], "home": r.home, "away": r.away,
                         "status": "upcoming", "p_H": round(p["H"], 3),
                         "p_D": round(p["D"], 3), "p_A": round(p["A"], 3),
                         "pick": max(p, key=p.get), "score": "", "outcome": "",
                         "correct": ""})
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(OUT, "backtest_predictions.csv"), index=False)
    print("wrote wc2026/backtest_predictions.csv ({} rows)".format(len(csv_rows)))

    render(rows, metrics, n, os.path.join(FIG, "wc2026_backtest.png"))
    print("figure: figures/wc2026_backtest.png")


if __name__ == "__main__":
    main()
