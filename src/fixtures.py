"""WC2026 fixtures + results loader (the data layer for the predictor).

One place to get the whole tournament: every match, played or upcoming, from
FIFA's open calendar API. The existing wc2026.py only kept finished matches
(it does `if MatchStatus != 0: continue`), which throws away the entire
remaining schedule the simulator needs -- so this returns ALL matches, tagged
by status, and normalises team names onto the rating keys.

Source of truth = FIFA calendar (idCompetition=17, idSeason=285023), which on
this machine returns the full 104-match schedule with scores, penalty scores,
group/stage labels and ISO kickoff times. FotMob and openfootball are kept as
fallbacks. Every fetch uses the repo's TLS skip-verify form.

    python src/fixtures.py        # summarise the schedule + played results
"""
import json
import os

import pandas as pd
import requests
import urllib3

from ratings import build_normalizer, seed_ratings

urllib3.disable_warnings()

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "wc2026")
FIFA_CAL = ("https://api.fifa.com/api/v3/calendar/matches"
            "?idCompetition=17&idSeason=285023&count=500&language=en")
UA = {"User-Agent": "Mozilla/5.0"}

# FIFA MatchStatus: 0 = finished (repo-proven; 5 matches with scores confirm it),
# 1 = upcoming/scheduled (the bulk of the schedule). Others are rare; treat ONLY
# 0-with-scores as a final result, everything else as not-yet-final.
# observed live: 0 finished, 1 upcoming, 3 in-progress (a running score, e.g.
# Brazil 1-1 Morocco mid-match) -- only 0-with-scores counts as a final result.
STATUS = {0: "finished", 1: "upcoming", 2: "scheduled", 3: "live"}


def _desc(val):
    """FIFA fields are often a list of localised {Description}; pull the text"""
    if isinstance(val, list):
        return val[0].get("Description") if val else None
    return val


def _team(side):
    """(name, id) from a FIFA Home/Away block; ('', None) for a TBD slot"""
    if not side:
        return "", None
    name = ""
    tn = side.get("TeamName")
    if isinstance(tn, list) and tn:
        name = tn[0].get("Description", "")
    return name, side.get("IdTeam")


def fetch_calendar(force=False):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, "calendar.json")
    if force or not (os.path.exists(path) and os.path.getsize(path) > 0):
        r = requests.get(FIFA_CAL, headers=UA, verify=False, timeout=60)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r.json(), f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_fixtures(force=False, norm=None):
    """returns a DataFrame of all 104 matches, names normalised to rating keys"""
    j = fetch_calendar(force=force)
    rows = []
    for m in j.get("Results", []):
        hn, hid = _team(m.get("Home"))
        an, aid = _team(m.get("Away"))
        st = m.get("MatchStatus")
        hs, as_ = m.get("HomeTeamScore"), m.get("AwayTeamScore")
        rows.append({
            "match_id": m.get("IdMatch"),
            "stage": _desc(m.get("StageName")),
            "group": _desc(m.get("GroupName")),
            "matchday": m.get("MatchDay"),
            "kickoff": m.get("Date"),
            "status": STATUS.get(st, str(st)),
            "home": hn, "away": an, "home_id": hid, "away_id": aid,
            "home_score": hs, "away_score": as_,
            "home_pens": m.get("HomeTeamPenaltyScore"),
            "away_pens": m.get("AwayTeamPenaltyScore"),
        })
    df = pd.DataFrame(rows)
    if norm is not None:
        df["home"] = df["home"].map(lambda s: norm(s) if s else s)
        df["away"] = df["away"].map(lambda s: norm(s) if s else s)
    df["finished"] = (df.status == "finished") & df.home_score.notna() \
        & df.away_score.notna()
    return df.sort_values("kickoff", kind="mergesort").reset_index(drop=True)


def played(df):
    """finished matches with both real teams known, chronological"""
    p = df[df.finished & (df.home != "") & (df.away != "")].copy()
    p["home_score"] = p.home_score.astype(int)
    p["away_score"] = p.away_score.astype(int)
    return p


def main():
    matches = pd.read_csv(os.path.join(ROOT, "data", "processed", "matches.csv"))
    ratings, _ = seed_ratings(matches)
    norm = build_normalizer(ratings)
    df = load_fixtures(force=True, norm=norm)
    print("total matches:", len(df))
    print("by status:", df.status.value_counts().to_dict())
    print("group-stage rows:", int(df.group.notna().sum()),
          "| knockout rows:", int(df.group.isna().sum()))
    pl = played(df)
    print("\nplayed so far ({}):".format(len(pl)))
    for r in pl.itertuples():
        seen = "" if (r.home in ratings and r.away in ratings) else "  [new team]"
        print("  {}  {:>16} {}-{} {:<16}  ({}){}".format(
            r.kickoff[:10], r.home, r.home_score, r.away_score, r.away,
            r.group or r.stage, seen))
    # which played teams are missing from the seed pool (need provisional prior)?
    miss = sorted({t for r in pl.itertuples() for t in (r.home, r.away)
                   if t not in ratings})
    print("\nplayed teams NOT in seed pool:", miss or "none")


if __name__ == "__main__":
    main()
