"""Self-adjusting team-strength ratings for the World Cup predictor.

This is the pre-match counterpart to the in-game winprob.py. Where winprob
evaluates a match in progress, this estimates who should win BEFORE kickoff,
from a single number per team: a World-Football-Elo rating that updates itself
after every result. Feed it the games already played and it learns; feed it a
fixture and it returns P(home win / draw / away win).

Elo only emits a scalar win-expectancy, which silently buries the draw. Football
draws ~27% of the time, so we split the expectancy into three calibrated
probabilities. Two routes, both fit ONCE on pre-2026 men's matches and frozen
(never refit while scoring live games, to keep the backtest honest):
  - a closed-form Davidson draw model (no fitting, the fallback), and
  - a 3-class multinomial logit on the rating gap, reusing winprob.py's exact
    softmax + JSON contract so the live scorer needs no sklearn.

The live predictor seeds from the FULL international record (~49k matches, the
martj42 dataset) via seed_history() -- the same engine validated out-of-sample
at 0.86 log loss -- so every nation, debutants included, carries an earned
rating. (seed_ratings() below is the older narrow seed of ~314 elite-tournament
matches; kept for reference, it mis-rated low-data teams like debutants.)

    python src/ratings.py        # seed, fit the draw models, rank, predict

AUDIT-DRAFT #12 (opt-in neutral-venue handling; validate on the data box):
the pooled pre-match logit is fit on a train window that is ~72% real home
games, so its intercept absorbs the average home edge and at dr=0 returns a
~9pp tilt to whichever team FIFA lists first -- spurious for the neutral-venue
WC2026 fixtures. prematch_proba(..., neutral=True) cancels that tilt by
mirroring (+dr vs H<->A-swapped -dr), and fit_prematch_logit(...,
neutral_indicator=True) can instead carry the venue effect in an explicit
feature. Both default OFF -> current output is byte-identical. To validate:

    NEUTRAL_AUDIT=1 python src/backtest_history.py

    (PowerShell:  $env:NEUTRAL_AUDIT=1; python src/backtest_history.py)

Expected signal: the [NEUTRAL_AUDIT] block prints dr=0 asymmetry |H-A| ~0.09
for the current scorer and ~0.0000 for neutral=True (symmetric P(home)=P(away)
at equal ratings), and on the genuinely-neutral OOS subset the neutral scorer's
log-loss / ECE(home) is <= the current scorer's.
"""
import json
import os
import unicodedata

import numpy as np
import pandas as pd

from winprob import softmax  # one source of truth for the scoring contract

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "wc2026")

# men's national-team finals tournaments only; exclude women's + club comps
MEN_TOURNAMENTS = {"WC 2018", "WC 2022", "Euro 2020", "Euro 2024",
                   "Copa America 2024", "AFCON 2023"}
# Elo K by competition importance (eloratings.net scale)
K_WORLD_CUP = 60
K_CONTINENTAL = 50
HOME_ADV = 65          # rating bump, applied ONLY to a host on home soil
PROVISIONAL = 1450.0   # prior for teams with no match history (self-corrects)
BASE = 1500.0

# host nations get the home bump when nominally "home"
HOSTS_2026 = {"United States", "Canada", "Mexico"}

# normalise FIFA / FotMob / ESPN spellings onto the StatsBomb names used as keys
ALIASES = {
    "czechia": "Czech Republic",
    "korea republic": "South Korea", "south korea": "South Korea",
    "ir iran": "Iran", "iran islamic republic": "Iran",
    "usa": "United States", "united states of america": "United States",
    "cote divoire": "Côte d'Ivoire", "ivory coast": "Côte d'Ivoire",
    "turkiye": "Turkey", "türkiye": "Turkey",
    "cabo verde": "Cape Verde Islands", "cape verde": "Cape Verde Islands",
    "dr congo": "Congo DR", "congo dr": "Congo DR",
    "democratic republic of the congo": "Congo DR",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
    "bosnia herzegovina": "Bosnia-Herzegovina",
    "china pr": "China", "north macedonia": "North Macedonia",
}


def fold(name):
    """ascii-fold + lowercase + strip punctuation, for robust name matching"""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def build_normalizer(known):
    """returns norm(name) -> canonical key, given the set of known rating keys"""
    by_fold = {fold(k): k for k in known}
    alias_fold = {fold(k): v for k, v in ALIASES.items()}

    def norm(name):
        f = fold(name)
        if f in alias_fold:
            return alias_fold[f]
        if f in by_fold:
            return by_fold[f]
        return str(name).strip()
    return norm


# ----------------------------------------------------------------- elo engine
def expected(dr):
    """Elo win-expectancy from rating gap dr (incl. any home bump)"""
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def g_mult(goal_diff):
    """goal-difference multiplier: bigger wins move ratings more"""
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def result_w(hs, as_):
    return 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)


def elo_update(rh, ra, hs, as_, k, ha=0.0):
    """one self-adjusting step; returns (new_home, new_away)"""
    dr = (rh + ha) - ra
    delta = k * g_mult(hs - as_) * (result_w(hs, as_) - expected(dr))
    return rh + delta, ra - delta


def seed_ratings(matches):
    """replay the men's pool in date order -> ratings dict + walk-forward pairs.

    Returns (ratings, pairs) where pairs is a list of (pre_match_dr, outcome)
    recorded BEFORE each update -- the leakage-free data to fit the draw models.
    """
    men = matches[matches.tournament.isin(MEN_TOURNAMENTS)].copy()
    men = men.dropna(subset=["home_score", "away_score"])
    men = men.sort_values(["date", "kick_off"], kind="mergesort")
    ratings, pairs = {}, []
    for r in men.itertuples():
        rh = ratings.get(r.home, BASE)
        ra = ratings.get(r.away, BASE)
        hs, as_ = int(r.home_score), int(r.away_score)
        dr = rh - ra  # HA=0: tournament games on neutral/host ground
        pairs.append((dr, "H" if hs > as_ else ("A" if as_ > hs else "D")))
        k = K_WORLD_CUP if r.tournament.startswith("WC") else K_CONTINENTAL
        ratings[r.home], ratings[r.away] = elo_update(rh, ra, hs, as_, k, ha=0.0)
    return ratings, pairs


# ----------------------------------------------------- broad-history seed ----
# The seed pool above is only ~314 elite-tournament matches, so debutants get a
# flat provisional and even established sides can be mis-rated off one bad finals
# (Scotland landed dead last). The broad seed below replays the FULL martj42
# international record -- the SAME engine validated out-of-sample at 0.86 log
# loss -- so the tournament predictor IS the validated model and every nation
# carries an earned rating. (A recency half-life was tested and rejected: it
# degrades out-of-sample; see model_search.py.)
INTL_RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "intl_results.csv")
MARTJ42_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WC2026_START = "2026-06-11"
# martj42 spellings that differ from our canonical keys (verified: only these 4)
MARTJ42_TO_KEY = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
}


def _k_international(tournament):
    """Elo K by match importance across ALL internationals, not just finals."""
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return K_WORLD_CUP            # 60
    if any(x in t for x in ("euro", "copa", "african cup", "asian cup", "gold cup")) \
            and "qualif" not in t:
        return K_CONTINENTAL          # 50
    if "qualif" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def seed_history(cutoff=WC2026_START):
    """Replay every international up to the World Cup -> (ratings, pairs).

    ratings are keyed by our canonical names; pairs is the leak-free list of
    (pre_match_dr, outcome, year) used to fit the draw model on a mature window.
    """
    if not (os.path.exists(INTL_RESULTS) and os.path.getsize(INTL_RESULTS) > 0):
        import requests
        import urllib3
        urllib3.disable_warnings()
        os.makedirs(os.path.dirname(INTL_RESULTS), exist_ok=True)
        resp = requests.get(MARTJ42_URL, verify=False, timeout=60)
        resp.raise_for_status()
        with open(INTL_RESULTS, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
    df = pd.read_csv(INTL_RESULTS).dropna(subset=["home_score", "away_score"])
    df = df[df.date < cutoff].sort_values("date", kind="mergesort")
    df["year"] = df.date.str.slice(0, 4).astype(int)
    elo, pairs = {}, []
    for r in df.itertuples():
        rh = elo.get(r.home_team, BASE)
        ra = elo.get(r.away_team, BASE)
        ha = 0.0 if r.neutral else HOME_ADV
        hs, as_ = int(r.home_score), int(r.away_score)
        dr = (rh + ha) - ra
        pairs.append((dr, "H" if hs > as_ else ("A" if as_ > hs else "D"), r.year))
        elo[r.home_team], elo[r.away_team] = elo_update(
            rh, ra, hs, as_, _k_international(r.tournament), ha)
    ratings = {MARTJ42_TO_KEY.get(k, k): round(v, 1) for k, v in elo.items()}
    return ratings, pairs


# ------------------------------------------------------- draw-aware 3-way map
def davidson_proba(dr, kappa):
    """closed-form Elo->P(H/D/A); kappa controls draw mass (no fitting)"""
    s = 10.0 ** ((dr / 400.0) / 2.0)
    z = s + 1.0 / s + kappa
    return {"H": s / z, "D": kappa / z, "A": (1.0 / s) / z}


def fit_kappa(pairs):
    """1-D search for the kappa minimising Davidson log loss on pre-2026 data"""
    drs = np.array([d for d, _ in pairs])
    out = np.array([o for _, o in pairs])
    best_k, best_ll = 1.0, 1e9
    for kappa in np.linspace(0.3, 2.0, 171):
        ll = 0.0
        for dr, o in zip(drs, out):
            p = davidson_proba(dr, kappa)[o]
            ll -= np.log(max(p, 1e-12))
        ll /= len(out)
        if ll < best_ll:
            best_ll, best_k = ll, kappa
    return round(float(best_k), 3), round(float(best_ll), 4)


def fit_prematch_logit(pairs, n_matches, neutral_indicator=False):
    """3-class multinomial logit on x=[dr/400], in winprob.py's JSON schema.

    AUDIT-DRAFT #12 (opt-in; validate on the data box): with
    neutral_indicator=True the model is fit on TWO features,
    x=[dr/400, is_neutral], so the venue effect is carried by an explicit
    indicator coefficient instead of leaking into the intercept (AUDIT fix
    option (a)). This requires 3-tuple pairs (dr, outcome, neutral) where neutral
    is 0/1; build them in walk_forward / seed_history by appending r.neutral.
    The resulting model carries features=["dr_over_400","is_neutral"]; to score a
    neutral fixture, prematch_proba's mirror path (neutral=True) still gives the
    venue-symmetric answer, and a true-host fixture is scored with is_neutral=0.

    Defaults False -> single-feature model byte-identical to the committed
    prematch_model.json (intercept of shape used by the live scorer). Accepts the
    existing 2-tuple pairs unchanged in that default path.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    if neutral_indicator:
        # pairs are (dr, outcome, neutral 0/1); venue gets its own coefficient
        X = np.array([[d / 400.0, float(nv)] for d, _, nv in pairs])
        features = ["dr_over_400", "is_neutral"]
    else:
        X = np.array([[d / 400.0] for d, _, *_ in pairs])
        features = ["dr_over_400"]
    y = np.array([p[1] for p in pairs])
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(X, y)
    ll = round(float(log_loss(y, clf.predict_proba(X))), 4)
    return {
        "classes": list(clf.classes_),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "features": features,
        "n_matches": int(n_matches),
        "n_samples": int(X.shape[0]),
    }, ll


def load_prematch_model():
    p = os.path.join(OUT, "prematch_model.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def prematch_proba(dr, model=None, kappa=1.0, neutral=False):
    """P(H/D/A) for a rating gap dr; logit if a model is given, else Davidson.

    AUDIT-DRAFT #12: pass neutral=True for a true neutral-venue fixture (every
    WC2026 group game except a true host on home soil). The pooled logit is fit
    on a train window that is ~72% real home games, so its intercept absorbs the
    average home edge: at dr=0 the production model returns H~0.404 / A~0.311, a
    ~9pp "phantom" tilt to whichever team FIFA happens to list first. On a
    neutral pitch there is no home edge, so that listing-order bias is spurious.

    neutral=True removes it WITHOUT re-fitting, by symmetrising the venue term:
    we average the model's prediction at +dr with its H<->A-swapped prediction
    at -dr. This cancels the intercept's home/away asymmetry while preserving the
    rating-gap signal and the draw mass, so at equal ratings (dr=0) it yields
    P(home) == P(away) exactly. For the Davidson fallback (already symmetric at
    dr=0) and the only-thing-that-matters production logit, this is the venue
    fix. Defaults False -> byte-identical to current behaviour. (opt-in; the
    maintainer should validate OOS log-loss / ECE on the data box -- see the
    neutral_only re-fit path in fit_prematch_logit and the command at the bottom
    of this module's docstring section in backtest_history.py.)
    """
    if model is not None:
        def _logit(d):
            x = np.array([d / 400.0])
            z = np.array(model["coef"]) @ x + np.array(model["intercept"])
            return dict(zip(model["classes"], softmax(z)))
        p = _logit(dr)
        if neutral:
            # mirror: swap the listed teams (dr -> -dr) so "home" becomes "away",
            # then read that mirrored game's A as this game's H, and average. The
            # venue-driven intercept asymmetry cancels; the gap signal survives.
            m = _logit(-dr)
            sym = {"H": 0.5 * (p["H"] + m["A"]),
                   "D": 0.5 * (p["D"] + m["D"]),
                   "A": 0.5 * (p["A"] + m["H"])}
            tot = sym["H"] + sym["D"] + sym["A"]
            return {k: v / tot for k, v in sym.items()}
        return p
    # Davidson is already venue-symmetric at dr=0; neutral is a no-op here.
    return davidson_proba(dr, kappa)


def main():
    os.makedirs(OUT, exist_ok=True)
    ratings, pairs = seed_history()
    norm = build_normalizer(ratings)
    print("broad seed: {} internationals -> {} teams".format(len(pairs), len(ratings)))

    # fit the draw models on a mature in-sample window (2005-2020) -- the exact
    # protocol the out-of-sample 0.86 validation uses, so the live predictor is
    # the validated model
    train = [(dr, o) for dr, o, y in pairs if 2005 <= y <= 2020]
    kappa, kll = fit_kappa(train)
    model, mll = fit_prematch_logit(train, len(train))
    print("draw model on {} matches (2005-2020): Davidson kappa {} (ll {}); logit ll {}".format(
        len(train), kappa, kll, mll))
    with open(os.path.join(OUT, "prematch_model.json"), "w") as f:
        json.dump(model, f, indent=2)

    # apply already-played WC2026 results so the saved state is self-adjusted.
    # Source = the FIFA calendar via fixtures.py (the same schedule montecarlo
    # and backtest use), not the legacy wc2026/matches.csv that only the old
    # wc2026.py wrote and which lagged the live feed. Lazy import: fixtures
    # imports this module at top, so importing it here avoids a cycle.
    applied, as_of = 0, "pre-tournament"
    try:
        import fixtures
        fx = fixtures.played(fixtures.load_fixtures(norm=norm))
        for r in fx.sort_values("kickoff", kind="mergesort").itertuples():
            h, a = r.home, r.away
            hs, as_ = int(r.home_score), int(r.away_score)
            rh, ra = ratings.get(h, PROVISIONAL), ratings.get(a, PROVISIONAL)
            ha = HOME_ADV if h in HOSTS_2026 else 0.0
            ratings[h], ratings[a] = elo_update(rh, ra, hs, as_, K_WORLD_CUP, ha)
            applied, as_of = applied + 1, str(r.kickoff)[:10]
    except Exception as e:
        print("  (no live results applied -- fixtures unavailable: {})".format(e))
    print("applied {} played WC2026 results (as of {})".format(applied, as_of))

    with open(os.path.join(OUT, "elo_ratings.json"), "w") as f:
        json.dump({"ratings": {k: round(v, 1) for k, v in ratings.items()},
                   "meta": {"as_of": as_of, "n_seed_matches": len(pairs),
                            "n_wc_applied": applied, "kappa": kappa,
                            "pool": "all-internationals", "base": BASE}}, f, indent=2)

    rank = sorted(ratings.items(), key=lambda kv: -kv[1])
    print("\ntop 20 by rating:")
    for i, (t, r) in enumerate(rank[:20], 1):
        print("  {:2d}. {:24s} {:6.0f}".format(i, t, r))

    print("\nsample fixtures (P home / draw / away):")
    for h, a in (("Brazil", "South Korea"), ("Argentina", "Mexico"),
                 ("United States", "Wales"), ("France", "England")):
        dr = ratings.get(h, PROVISIONAL) - ratings.get(a, PROVISIONAL)
        if h in HOSTS_2026:
            dr += HOME_ADV
        p = prematch_proba(dr, model)
        print("  {:14s} vs {:14s}  H {:.2f}  D {:.2f}  A {:.2f}".format(
            h, a, p["H"], p["D"], p["A"]))


if __name__ == "__main__":
    main()
