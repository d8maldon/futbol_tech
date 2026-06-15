"""Out-of-sample hyperparameter search for the Elo predictor -- the honest way
to improve it.

The cardinal rule: a change earns its place only by improving the HELD-OUT
2021-2026 log loss, never the games we have already seen. And we do not even
tune to that test set -- hyperparameters are selected on a validation slice of
the TRAIN era (2018-2020), and the chosen config is scored on 2021-2026 exactly
once. Anything else is overfitting one level up.

Knobs, all defensible football-Elo choices:
  - home advantage (rating bump for the non-neutral home side)
  - a global multiplier on the per-competition K (learning rate)
  - optional annual regression-to-mean (down-weighting stale form)

    python src/model_search.py
"""
import numpy as np

import ratings as rt
from backtest import logloss
from backtest_history import k_for, load

FIT_FROM, VAL_FROM, TEST_FROM = 2005, 2018, 2021   # fit 2005-17 | select 2018-20 | report 2021-26
BASELINE = (65, 1.0, 0.0)                          # current production: HA=65, K x1, no regression


def walk_forward(df, home_adv, k_scale, regress):
    """date-order Elo with tunable home advantage, K scale and optional annual
    regression-to-mean; records (dr, outcome, year) BEFORE each update (leak-free)"""
    elo, pairs, cur_year = {}, [], None
    for r in df.itertuples():
        if regress and r.year != cur_year:
            if cur_year is not None:
                for t in elo:
                    elo[t] = rt.BASE + (1.0 - regress) * (elo[t] - rt.BASE)
            cur_year = r.year
        rh = elo.get(r.home_team, rt.BASE)
        ra = elo.get(r.away_team, rt.BASE)
        ha = 0.0 if r.neutral else home_adv
        hs, as_ = int(r.home_score), int(r.away_score)
        o = "H" if hs > as_ else ("A" if as_ > hs else "D")
        pairs.append(((rh + ha) - ra, o, r.year))
        elo[r.home_team], elo[r.away_team] = rt.elo_update(
            rh, ra, hs, as_, k_for(r.tournament) * k_scale, ha)
    return pairs


def fit_then_score(pairs, fit_lo, fit_hi, score_lo, score_hi):
    fit = [(dr, o) for dr, o, y in pairs if fit_lo <= y < fit_hi]
    scr = [(dr, o) for dr, o, y in pairs if score_lo <= y < score_hi]
    model, _ = rt.fit_prematch_logit(fit, len(fit))
    lls = np.array([logloss(rt.prematch_proba(dr, model), o) for dr, o in scr])
    return lls, model, scr


def report_test(df, params):
    """fit on the full train (2005-2020), score ONCE on the held-out 2021-2026"""
    pairs = walk_forward(df, *params)
    lls, _, scr = fit_then_score(pairs, FIT_FROM, TEST_FROM, TEST_FROM, 2100)
    rng = np.random.default_rng(0)
    boot = [lls[rng.integers(0, len(lls), len(lls))].mean() for _ in range(2000)]
    return lls.mean(), np.percentile(boot, [2.5, 97.5]), len(scr)


def main():
    df = load()
    print("loaded {} internationals (1872-2026)".format(len(df)))

    grid = [(ha, ks, rg)
            for ha in (40, 55, 65, 80, 100)
            for ks in (0.7, 1.0, 1.3)
            for rg in (0.0, 0.03, 0.06)]
    scored = []
    for params in grid:
        pairs = walk_forward(df, *params)
        val, _, _ = fit_then_score(pairs, FIT_FROM, VAL_FROM, VAL_FROM, TEST_FROM)
        scored.append((params, float(val.mean())))
    scored.sort(key=lambda x: x[1])

    print("\nselection -- top 8 configs by VALIDATION log loss (2018-2020):")
    for (ha, ks, rg), v in scored[:8]:
        flag = "  <- baseline" if (ha, ks, rg) == BASELINE else ""
        print("  HA={:<3} K x{:<4} regress={:.2f}   val {:.4f}{}".format(ha, ks, rg, v, flag))
    best = scored[0][0]
    base_val = next(v for p, v in scored if p == BASELINE)
    print("  (baseline val {:.4f})".format(base_val))

    print("\nFINAL report -- fit 2005-2020, scored ONCE on held-out 2021-2026:")
    for label, params in (("baseline", BASELINE), ("best-on-val", best)):
        ll, ci, n = report_test(df, params)
        print("  {:<12} HA={:<3} Kx{:<4} reg={:.2f}:  OOS log loss {:.4f}  [95% CI {:.4f}-{:.4f}]  (n={})".format(
            label, params[0], params[1], params[2], ll, ci[0], ci[1], n))
    if best == BASELINE:
        print("\nverdict: the current settings already win on validation -- no change is justified.")


if __name__ == "__main__":
    main()
