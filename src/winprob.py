"""In-game win probability model, chess-engine style.

Evaluates the match state the way a chess engine evaluates a position: from
the scoreboard, the accumulated chance quality, the man count and the clock,
with no knowledge of who is playing. Trained on the 551 historical matches
in this repo: P(home win / draw / away win) at the 90-minute mark, given the
state at minute m.

Features: goal difference (raw and clock-amplified), cumulative xG
difference (raw and clock-decayed), man advantage from red cards, and time
remaining. Multinomial logistic regression; coefficients are saved as plain
JSON so the live tracker can score matches without sklearn.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "wc2026")

FEATURES = ["gd", "gd_clock", "xgd", "xgd_early", "mad", "mad_clock", "rem"]


def feature_row(gd, xgd, mad, minute):
    rem = max(90.0 - minute, 0.0) / 90.0
    return [gd, gd / (rem + 0.1), xgd, xgd * rem, mad, mad * (1 - rem), rem]


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict(model, gd, xgd, mad, minute):
    x = np.array(feature_row(gd, xgd, mad, minute))
    z = np.array(model["coef"]) @ x + np.array(model["intercept"])
    p = softmax(z)
    return dict(zip(model["classes"], p))


def eff_min(period, t):
    return 45.0 * (period - 1) + min(t / 60.0, 44.99)


def main():
    matches = pd.read_csv(os.path.join(ROOT, "matches.csv")).set_index("match_id")
    shots = pd.read_csv(os.path.join(ROOT, "shots.csv"))
    shots = shots[shots.period <= 2].copy()
    shots["m"] = [eff_min(p, t) for p, t in zip(shots.period, shots.t)]
    og = pd.read_csv(os.path.join(ROOT, "owngoals.csv"))
    og = og[og.period <= 2].copy()
    og["m"] = [eff_min(p, t) for p, t in zip(og.period, og.t)]
    cards = pd.read_csv(os.path.join(ROOT, "cards.csv"))
    cards = cards[(cards.period <= 2) & cards.card.isin(["Red Card", "Second Yellow"])].copy()
    cards["m"] = [eff_min(p, t) for p, t in zip(cards.period, cards.t)]

    sg = {k: v for k, v in shots.groupby("match_id")}
    ogg = {k: v for k, v in og.groupby("match_id")}
    cg = {k: v for k, v in cards.groupby("match_id")}

    X, y = [], []
    for mid, m in matches.iterrows():
        s = sg.get(mid)
        if s is None:
            continue
        o = ogg.get(mid)
        c = cg.get(mid)
        hid, aid = m.home_id, m.away_id

        def goals_at(minute):
            hg = int(((s.team_id == hid) & (s.goal == 1) & (s.m <= minute)).sum())
            ag = int(((s.team_id == aid) & (s.goal == 1) & (s.m <= minute)).sum())
            if o is not None:
                hg += int(((o.team_id == hid) & (o.m <= minute)).sum())
                ag += int(((o.team_id == aid) & (o.m <= minute)).sum())
            return hg, ag

        hg90, ag90 = goals_at(90)
        outcome = "H" if hg90 > ag90 else ("A" if ag90 > hg90 else "D")
        for minute in range(0, 90):
            hg, ag = goals_at(minute)
            hxg = float(s[(s.team_id == hid) & (s.m <= minute)].xg.sum())
            axg = float(s[(s.team_id == aid) & (s.m <= minute)].xg.sum())
            mad = 0
            if c is not None:
                mad = int(((c.team_id == aid) & (c.m <= minute)).sum()) - \
                    int(((c.team_id == hid) & (c.m <= minute)).sum())
            X.append(feature_row(hg - ag, hxg - axg, mad, minute))
            y.append(outcome)

    X = np.array(X)
    y = np.array(y)
    print("training samples:", X.shape, "outcome mix:",
          {k: int((y == k).sum()) for k in ("H", "D", "A")})

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(X, y)
    p = clf.predict_proba(X)
    print("log loss:", round(log_loss(y, p), 4))

    # calibration: predicted home-win probability decile vs observed rate
    ph = p[:, list(clf.classes_).index("H")]
    obs = (y == "H").astype(float)
    print("calibration (P(home win) decile -> predicted / observed):")
    qs = np.quantile(ph, np.linspace(0, 1, 11))
    for i in range(10):
        sel = (ph >= qs[i]) & (ph <= qs[i + 1])
        if sel.sum():
            print("  {:.2f}-{:.2f}: {:.3f} / {:.3f}  (n={})".format(
                qs[i], qs[i + 1], ph[sel].mean(), obs[sel].mean(), int(sel.sum())))

    model = {
        "classes": list(clf.classes_),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "features": FEATURES,
        "n_matches": int(matches.shape[0]),
        "n_samples": int(X.shape[0]),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "winprob_model.json"), "w") as f:
        json.dump(model, f, indent=2)

    print("\nsanity checks:")
    for desc, args in (
            ("0-0 at kickoff", (0, 0.0, 0, 0)),
            ("0-0 at 85'", (0, 0.0, 0, 85)),
            ("home +1 at 85'", (1, 0.3, 0, 85)),
            ("home +1 at 45'", (1, 0.3, 0, 45)),
            ("level but home xG +1.5 at 70'", (0, 1.5, 0, 70)),
            ("away up a man (red) at 60', level", (0, 0.0, -1, 60))):
        pr = predict(model, *args)
        print("  {:38s} H {:.2f} D {:.2f} A {:.2f}".format(
            desc, pr["H"], pr["D"], pr["A"]))


if __name__ == "__main__":
    main()
