"""Match data layer for the visual-AI dashboard: everything the always-on panels
need, pulled from the real FotMob match JSON + our own pre-match prediction.

These panels stay alive every second of the clip (even on close-ups where the
top-down can't track), because they are driven by data valid for the whole match:
  - events     : goals / cards / subs with the running score and the minute
  - shots      : shot map with xG (for the xG race and shot markers)
  - ratings    : FotMob player ratings + player of the match
  - pre_match  : OUR model's pre-match P(H/D/A) from backtest_predictions.csv
  - wp_curve   : in-game win-probability over the match, a Skellam remaining-goals
                 model anchored so minute 0 == our pre-match number

No public WC2026 tracking telemetry exists, so this event/xG data + our models
ARE the "telemetry" we have; it is honest and complete for the whole 90+.

    python src/match_data.py 4667812
"""
import csv
import json
import os

import numpy as np
from scipy.stats import skellam

ROOT = os.path.join(os.path.dirname(__file__), "..")
FM = os.path.join(ROOT, "data", "wc2026")
PRED = os.path.join(ROOT, "wc2026", "backtest_predictions.csv")


def _rating(p):
    for grp in p.get("stats", []) or []:
        for k, v in (grp.get("stats", {}) or {}).items():
            if "ating" in k and isinstance(v, dict):
                s = (v.get("stat") or {}).get("value")
                if s is not None:
                    return float(s)
    return None


def pre_match(home, away):
    """our model's pre-match P(H/D/A) for this fixture, from the backtest CSV"""
    if not os.path.exists(PRED):
        return None
    for r in csv.DictReader(open(PRED, encoding="utf-8")):
        if r["home"] == home and r["away"] == away:
            return {"p_h": float(r["p_H"]), "p_d": float(r["p_D"]), "p_a": float(r["p_A"])}
    return None


def calibrate(p_h, p_d):
    """pick full-match goal rates (mu_h, mu_a) whose Skellam win/draw probs match
    our pre-match P(home win) / P(draw) -- so the live curve starts at our number"""
    best, bk = None, None
    for mh in np.arange(0.8, 3.21, 0.05):
        for ma in np.arange(0.3, 1.61, 0.05):
            win = float(skellam.sf(0, mh, ma))          # P(diff >= 1)
            draw = float(skellam.pmf(0, mh, ma))         # P(diff == 0)
            err = (win - p_h) ** 2 + (draw - p_d) ** 2
            if best is None or err < best:
                best, bk = err, (float(mh), float(ma))
    return bk


def wp(d, t, mh, ma):
    """(P home win, P draw, P away win) given goal diff d at minute t (Skellam)"""
    f = max(0.0, (90.0 - t) / 90.0)
    lh, la = mh * f, ma * f
    if lh <= 1e-6 and la <= 1e-6:
        return (1.0, 0.0, 0.0) if d > 0 else ((0.0, 1.0, 0.0) if d == 0 else (0.0, 0.0, 1.0))
    win = float(skellam.sf(-d, lh, la))                 # P(remaining diff > -d)
    draw = float(skellam.pmf(-d, lh, la))               # P(remaining diff == -d)
    return win, draw, max(0.0, 1.0 - win - draw)


def load(mid):
    d = json.load(open(os.path.join(FM, "fm_match_{}.json".format(mid)), encoding="utf-8"))
    g = d.get("general", {})
    home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
    hn, an = home.get("name"), away.get("name")
    hid, aid = home.get("id"), away.get("id")
    c = d["content"]

    # --- events: goals / cards / subs ---
    raw = (c["matchFacts"].get("events", {}) or {}).get("events", []) or []
    events = []
    for e in raw:
        typ = e.get("type")
        if typ not in ("Goal", "Card", "Substitution"):
            continue
        events.append({"min": int(e.get("time") or 0), "type": typ,
                        "player": (e.get("player") or {}).get("name") or e.get("nameStr") or "",
                        "h": e.get("homeScore"), "a": e.get("awayScore"),
                        "is_home": bool(e.get("isHome")), "card": e.get("card")})

    # --- shots (xG) ---
    shots = []
    for s in (((c.get("shotmap") or {}).get("shots")) or []):
        shots.append({"min": int(s.get("min") or 0), "xg": float(s.get("expectedGoals") or 0),
                      "goal": s.get("eventType") == "Goal", "is_home": s.get("teamId") == hid,
                      "player": s.get("playerName", ""),
                      "x": float(s.get("x", 0)) * 1.2, "y": float(s.get("y", 0)) * 0.8})

    # --- player ratings (+ POTM) ---
    ratings = []
    for pid, p in (c.get("playerStats") or {}).items():
        r = _rating(p)
        if r is None:
            continue
        ratings.append({"name": p.get("name", ""), "is_home": p.get("teamId") == hid,
                        "rating": r, "potm": bool(p.get("isPotm"))})
    ratings.sort(key=lambda x: -x["rating"])

    # --- score timeline from goal events (FotMob's homeScore/awayScore on a goal
    #     is the score BEFORE that goal, so we COUNT goals to get the running score) ---
    goals = sorted([e for e in events if e["type"] == "Goal"], key=lambda e: e["min"])
    hh = aa = 0
    for gl in goals:
        if gl["is_home"]:
            hh += 1
        else:
            aa += 1
        gl["score"] = (hh, aa)
    final_h, final_a = hh, aa

    # --- our pre-match number + calibrated in-game win-prob curve ---
    pm = pre_match(hn, an) or {"p_h": 0.5, "p_d": 0.27, "p_a": 0.23}
    mh, ma = calibrate(pm["p_h"], pm["p_d"])
    mins = np.arange(0, 99)
    hs = as_ = 0
    ph, pd, pa, sc_h, sc_a = [], [], [], [], []
    gi = 0
    for t in mins:
        while gi < len(goals) and goals[gi]["min"] <= t:
            if goals[gi]["is_home"]:
                hs += 1
            else:
                as_ += 1
            gi += 1
        w, dr, l = wp(hs - as_, t, mh, ma)
        ph.append(w); pd.append(dr); pa.append(l); sc_h.append(hs); sc_a.append(as_)

    # xG race: cumulative xG per team per minute
    cum_h = np.array([sum(s["xg"] for s in shots if s["is_home"] and s["min"] <= t) for t in mins])
    cum_a = np.array([sum(s["xg"] for s in shots if not s["is_home"] and s["min"] <= t) for t in mins])

    return {"home": hn, "away": an, "events": events, "shots": shots, "ratings": ratings,
            "goals": goals, "pre_match": pm, "wp_mins": mins,
            "wp_home": np.array(ph), "wp_draw": np.array(pd), "wp_away": np.array(pa),
            "sc_h": np.array(sc_h), "sc_a": np.array(sc_a),
            "xg_h": cum_h, "xg_a": cum_a,
            "final_h": final_h, "final_a": final_a, "mu": (mh, ma)}


def main():
    import sys
    mid = sys.argv[1] if len(sys.argv) > 1 else "4667812"
    m = load(mid)
    print("{} {}-{} {}".format(m["home"], m["final_h"], m["final_a"], m["away"]))
    print("pre-match (our model): H {p_h:.3f} D {p_d:.3f} A {p_a:.3f}".format(**m["pre_match"]))
    print("calibrated goal rates mu:", tuple(round(x, 2) for x in m["mu"]))
    print("win-prob (home) at 0/15/20/45/60/77/90:",
          [round(float(m["wp_home"][t]), 3) for t in (0, 15, 20, 45, 60, 77, 90)])
    print("goals:", [(g["min"], g["player"]) for g in m["goals"]])
    print("shots:", len(m["shots"]), "| home xG {:.2f}  away xG {:.2f}".format(
        sum(s["xg"] for s in m["shots"] if s["is_home"]),
        sum(s["xg"] for s in m["shots"] if not s["is_home"])))
    print("top ratings:", [(r["name"], r["rating"], "POTM" if r["potm"] else "")
                           for r in m["ratings"][:5]])


if __name__ == "__main__":
    main()
