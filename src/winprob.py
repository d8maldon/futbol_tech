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
    z = z / model.get("temperature", 1.0)   # calibration: tame over-confidence
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

    X, y, groups = [], [], []
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
            groups.append(mid)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    print("training samples:", X.shape, "outcome mix:",
          {k: int((y == k).sum()) for k in ("H", "D", "A")})

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    # Match-grouped split: each match is 90 autocorrelated rows sharing one
    # label, so we hold out whole MATCHES. Three-way (fit / calibrate / test):
    # the raw logistic model is over-confident, so we fit ONE temperature on the
    # calib fold and report log loss on a separate test fold -- no leakage, and
    # the temperature is chosen without ever seeing the test fold.
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    k = max(len(uniq) // 5, 1)
    test_ids, calib_ids = set(uniq[:k]), set(uniq[k:2 * k])
    te = np.array([g in test_ids for g in groups])
    ca = np.array([g in calib_ids for g in groups])
    tr = ~(te | ca)
    base = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
    cls = list(base.classes_)

    def logits(mask):
        return X[mask] @ base.coef_.T + base.intercept_

    # temperature: 1-D search for the T minimising calib log loss (T>1 means the
    # raw model was over-confident; dividing the logits by T softens it)
    zc = logits(ca)
    grid = np.linspace(0.5, 3.0, 51)
    T = float(min(grid, key=lambda t: log_loss(y[ca], softmax(zc / t), labels=cls)))
    zt = logits(te)
    raw = log_loss(y[te], softmax(zt), labels=cls)
    cal = log_loss(y[te], softmax(zt / T), labels=cls)
    print("OUT-OF-SAMPLE log loss ({} test matches): raw {:.4f} -> temperature-scaled {:.4f}  (T={:.2f})".format(
        len(test_ids), raw, cal, T))

    # P(home-win) calibration on the held-out test, after temperature scaling
    pte = softmax(zt / T)
    ph = pte[:, cls.index("H")]
    obs = (y[te] == "H").astype(float)
    qs = np.quantile(ph, np.linspace(0, 1, 11))
    print("calibration after scaling (P(home win) decile -> predicted / observed):")
    for i in range(10):
        sel = (ph >= qs[i]) & (ph <= qs[i + 1])
        if sel.sum():
            print("  {:.2f}-{:.2f}: {:.3f} / {:.3f}  (n={})".format(
                qs[i], qs[i + 1], ph[sel].mean(), obs[sel].mean(), int(sel.sum())))

    # production model: refit on ALL data, keep the calibrated temperature
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(X, y)

    model = {
        "classes": list(clf.classes_),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "temperature": round(T, 3),
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
