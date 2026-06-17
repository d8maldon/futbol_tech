"""Large-sample out-of-sample validation of the Elo predictor (the real test).

The WC2026 backtest is honest but tiny (~9 games). This replays ~49,000 men's
internationals (1872-2026, the public martj42 dataset) through the SAME Elo engine
in date order, fits the draw model on a mature train window, and scores the
HELD-OUT last five years (2021-2026) out-of-sample -- so "well-calibrated" becomes
a claim with thousands of matches behind it, not nine.

Leadership metric is LOG LOSS (proper + local); Brier secondary; RPS reported but
not headlined (it is non-local and noisy). Benchmarked against a uniform baseline
and the historical base rate (climatology). No bookmaker odds in this dataset, so
we do NOT claim to beat the market -- only that the model is properly validated
and calibrated. A reliability curve shows calibration directly.

    python src/backtest_history.py
"""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import urllib3

import ratings as rt
from backtest import ORDER, brier, logloss, rps

urllib3.disable_warnings()
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
FIG = os.path.join(ROOT, "figures")
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TRAIN_FROM, TEST_FROM = 2005, 2021      # fit draw model 2005-2020; test 2021-2026
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; HOME_C = "#5e9bff"; ACC = "#ffb347"


def k_for(t):
    t = str(t).lower()
    if "world cup" in t and "qualif" not in t:
        return 60
    if any(x in t for x in ("euro", "copa", "african cup", "asian cup", "gold cup")) and "qualif" not in t:
        return 50
    if "qualif" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def load():
    path = os.path.join(RAW, "intl_results.csv")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        os.makedirs(RAW, exist_ok=True)
        r = requests.get(URL, verify=False, timeout=60)
        r.raise_for_status()
        open(path, "w", encoding="utf-8").write(r.text)
    df = pd.read_csv(path)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["year"] = df.date.str.slice(0, 4).astype(int)
    return df.sort_values("date", kind="mergesort")


def walk_forward(df):
    """date-order Elo; record (dr, outcome, year) BEFORE each update (leak-free)"""
    elo, pairs = {}, []
    for r in df.itertuples():
        rh = elo.get(r.home_team, rt.BASE)
        ra = elo.get(r.away_team, rt.BASE)
        ha = 0.0 if r.neutral else rt.HOME_ADV
        dr = (rh + ha) - ra
        hs, as_ = int(r.home_score), int(r.away_score)
        o = "H" if hs > as_ else ("A" if as_ > hs else "D")
        pairs.append((dr, o, r.year))
        elo[r.home_team], elo[r.away_team] = rt.elo_update(rh, ra, hs, as_, k_for(r.tournament), ha)
    return pairs


def score(test, predict):
    ll = [logloss(predict(dr), o) for dr, o in test]
    br = [brier(predict(dr), o) for dr, o in test]
    rp = [rps(predict(dr), o) for dr, o in test]
    acc = [1.0 if max(predict(dr), key=predict(dr).get) == o else 0.0 for dr, o in test]
    return np.array(ll), np.array(br), np.array(rp), np.array(acc)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load()
    pairs = walk_forward(df)
    train = [(dr, o) for dr, o, y in pairs if TRAIN_FROM <= y < TEST_FROM]
    test = [(dr, o) for dr, o, y in pairs if y >= TEST_FROM]
    print("replayed {} internationals 1872-2026 | train {} (2005-20) | test OOS {} (2021-26)".format(
        len(pairs), len(train), len(test)))

    model, train_ll = rt.fit_prematch_logit(train, len(train))   # fit draw model on TRAIN only
    base = {k: float(np.mean([o == k for _, o in train])) for k in ORDER}
    uni = {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}

    def p_model(dr):
        return rt.prematch_proba(dr, model)
    ll, br, rp, acc = score(test, p_model)
    bll, _, _, _ = score(test, lambda dr: base)
    ull, _, _, _ = score(test, lambda dr: uni)

    rng = np.random.default_rng(0)
    boot = [ll[rng.integers(0, len(ll), len(ll))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\nOUT-OF-SAMPLE (2021-2026, n={}):".format(len(test)))
    print("  LOG LOSS  model {:.4f}  [95% CI {:.4f}-{:.4f}]  | climatology {:.4f} | uniform {:.4f}".format(
        ll.mean(), lo, hi, bll.mean(), ull.mean()))
    print("  Brier {:.4f} | RPS {:.4f} | accuracy {:.1%}".format(br.mean(), rp.mean(), acc.mean()))
    print("  (lower log loss = better; model should beat climatology < uniform = {:.4f})".format(np.log(3)))

    # reliability curve on P(home win), 10 bins, with observed rate + Wilson CI
    ph = np.array([p_model(dr)["H"] for dr, _ in test])
    yh = np.array([o == "H" for _, o in test], float)
    edges = np.linspace(0, 1, 11)
    xs, ys, los, his, ece = [], [], [], [], 0.0
    for i in range(10):
        sel = (ph >= edges[i]) & (ph < edges[i + 1]) if i < 9 else (ph >= edges[i]) & (ph <= 1.0)
        n = int(sel.sum())
        if n < 20:
            continue
        p_, obs = ph[sel].mean(), yh[sel].mean()
        z = 1.96; ph_ = obs
        denom = 1 + z * z / n
        centre = (ph_ + z * z / (2 * n)) / denom
        half = z * np.sqrt(ph_ * (1 - ph_) / n + z * z / (4 * n * n)) / denom
        xs.append(p_); ys.append(obs); los.append(centre - half); his.append(centre + half)
        ece += n / len(test) * abs(p_ - obs)
    print("  P(home-win) calibration ECE: {:.3f}  (KDD2021 benchmark ~0.011)".format(ece))

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160,
                                   gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(BG)
    axc.set_facecolor(BG)
    axc.plot([0, 1], [0, 1], color=MUT, ls=(0, (4, 3)), lw=1, label="perfect")
    xs = np.array(xs); ys = np.array(ys)
    axc.errorbar(xs, ys, yerr=[ys - np.array(los), np.array(his) - ys], fmt="o-",
                 color=HOME_C, ecolor=MUT, ms=6, lw=1.4, capsize=3, label="model")
    axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.set_aspect("equal")
    axc.set_xlabel("predicted P(home win)", color=MUT, fontsize=9, fontfamily="Bahnschrift")
    axc.set_ylabel("observed home-win rate", color=MUT, fontsize=9, fontfamily="Bahnschrift")
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axc.spines[s].set_color(MUT)
    axc.tick_params(colors=MUT, labelsize=8)
    axc.legend(frameon=False, labelcolor=INK, loc="upper left", prop={"family": "Bahnschrift", "size": 9})
    axc.set_title("Calibration on held-out 2021-26  (ECE {:.3f})".format(ece),
                  color=INK, loc="left", **{"fontfamily": "Bahnschrift", "fontsize": 12, "fontweight": "bold"})

    axb.set_facecolor(BG)
    names = ["model", "base rate", "uniform"]
    vals = [ll.mean(), bll.mean(), ull.mean()]
    xb = np.arange(3)
    axb.bar(xb, vals, color=[HOME_C, "#8b949e", "#30363d"], width=0.6)
    axb.errorbar([0], [ll.mean()], yerr=[[ll.mean() - lo], [hi - ll.mean()]], fmt="none", ecolor=ACC, capsize=4, lw=1.5)
    for x, v in zip(xb, vals):
        axb.text(x, v + 0.004, "{:.3f}".format(v), ha="center", color=INK, fontsize=9, fontfamily="Bahnschrift")
    axb.set_xticks(xb); axb.set_xticklabels(names, color=MUT, fontsize=9, fontfamily="Bahnschrift")
    axb.set_ylabel("out-of-sample log loss (lower better)", color=MUT, fontsize=9, fontfamily="Bahnschrift")
    axb.set_ylim(0, max(vals) * 1.15)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axb.spines[s].set_color(MUT)
    axb.tick_params(colors=MUT)
    axb.set_title("Log loss vs baselines  (n={:,} held-out matches)".format(len(test)),
                  color=INK, loc="left", **{"fontfamily": "Bahnschrift", "fontsize": 12, "fontweight": "bold"})
    fig.suptitle("Predictor validated out-of-sample on ~49k internationals (1872-2026)",
                 color=INK, x=0.5, fontsize=14, fontfamily="Bahnschrift", fontweight="bold")
    fig.text(0.5, 0.01, "fit on 2005-2020, tested on held-out 2021-2026 | log-loss-first | no bookmaker odds -> not a market-beating claim | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(os.path.join(FIG, "wc2026_predictor_validation.png"), facecolor=BG)
    plt.close(fig)
    print("figure: figures/wc2026_predictor_validation.png")


if __name__ == "__main__":
    main()
