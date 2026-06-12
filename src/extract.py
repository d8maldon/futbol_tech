"""Single pass over all raw event files.

Produces compact CSVs in data/processed:
  matches.csv    match metadata (teams, tournament, kickoff, stadium)
  stoppages.csv  every dead-ball gap >= 60s with context for classification
  admin.csv      substitutions and tactical shifts with timing
  moves.csv      completed passes and carries (for the xT model)
  shots.csv      shots with xG and outcome
"""
import csv
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "processed")

COMPETITIONS = [
    (43, 3, "WC 2018"),
    (43, 106, "WC 2022"),
    (72, 30, "WWC 2019"),
    (72, 107, "WWC 2023"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
    (55, 282, "Euro 2024"),
    (55, 43, "Euro 2020"),
    (44, 107, "MLS 2023"),
    (1238, 108, "ISL 2021-22"),
]

# Event types that do not involve the ball being in play. Time between two
# consecutive in-play events is a dead-ball gap; these can occur inside it.
NON_PLAY = {
    "Starting XI", "Half Start", "Half End", "Substitution", "Tactical Shift",
    "Injury Stoppage", "Player Off", "Player On", "Bad Behaviour",
    "Referee Ball-Drop", "Camera On", "Camera off", "Camera Off",
}


def tsec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def match_minute(period, t):
    if period == 1:
        return t / 60.0
    if period == 2:
        return 45.0 + t / 60.0
    if period == 3:
        return 90.0 + t / 60.0
    return 105.0 + t / 60.0


def process_match(match_id, tournament, w):
    path = os.path.join(RAW, "events", "{}.json".format(match_id))
    with open(path, encoding="utf-8") as f:
        events = json.load(f)

    goals = []      # (period, t)
    cards = []      # (period, t)
    play = []       # (period, t_start, t_end, type, idx)
    nonplay = []    # (period, t, type)

    for ev in events:
        period = ev["period"]
        t = tsec(ev["timestamp"])
        etype = ev["type"]["name"]
        team_id = ev["team"]["id"] if "team" in ev else None

        if etype in NON_PLAY:
            nonplay.append((period, t, etype))
            if etype in ("Substitution", "Tactical Shift"):
                w["admin"].writerow([
                    match_id, tournament, period, round(t, 1),
                    round(match_minute(period, t), 2), etype, team_id,
                ])
            continue

        dur = ev.get("duration") or 0.0
        play.append((period, t, t + dur, etype))

        if etype == "Shot":
            shot = ev["shot"]
            goal = 1 if shot["outcome"]["name"] == "Goal" else 0
            if goal:
                goals.append((period, t))
            loc = ev.get("location") or [None, None]
            pen = 1 if shot["type"]["name"] == "Penalty" else 0
            w["shots"].writerow([
                match_id, tournament, period, round(t, 1), team_id,
                round(shot.get("statsbomb_xg") or 0.0, 4), goal,
                loc[0], loc[1], pen,
            ])
        elif etype == "Own Goal For":
            goals.append((period, t))
        elif etype == "Pass":
            p = ev["pass"]
            ok = 0 if "outcome" in p else 1
            loc, end = ev.get("location"), p.get("end_location")
            if loc and end:
                w["moves"].writerow([
                    match_id, period, round(t, 1), team_id,
                    loc[0], loc[1], end[0], end[1], ok,
                ])
        elif etype == "Carry":
            loc, end = ev.get("location"), ev["carry"].get("end_location")
            if loc and end:
                w["moves"].writerow([
                    match_id, period, round(t, 1), team_id,
                    loc[0], loc[1], end[0], end[1], 1,
                ])
        elif etype == "Foul Committed":
            if "card" in (ev.get("foul_committed") or {}):
                cards.append((period, t))

    # dead-ball gaps between consecutive in-play events
    play.sort(key=lambda x: (x[0], x[1]))
    for a, b in zip(play, play[1:]):
        if a[0] != b[0] or a[0] > 2:
            continue
        period = a[0]
        start, end = a[2], b[1]
        gap = end - start
        if gap < 60:
            continue
        inside = [x for x in nonplay if x[0] == period and start - 1 <= x[1] <= end + 1]
        subs_in = sum(1 for x in inside if x[2] == "Substitution")
        shifts_in = sum(1 for x in inside if x[2] == "Tactical Shift")
        injury = any(x[2] in ("Injury Stoppage", "Player Off") for x in inside)
        goal_before = any(p == period and start - 75 <= t <= start + 1 for p, t in goals)
        card_near = any(p == period and start - 45 <= t <= end + 45 for p, t in cards)
        w["stoppages"].writerow([
            match_id, tournament, period, round(start, 1), round(gap, 1),
            round(match_minute(period, start), 2), round(start / 60.0, 2),
            subs_in, shifts_in, int(injury), int(goal_before), int(card_near),
            a[3], b[3],
        ])


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {
        "matches": ["match_id", "tournament", "date", "kick_off", "home", "away",
                    "home_score", "away_score", "home_id", "away_id", "stadium", "stage"],
        "stoppages": ["match_id", "tournament", "period", "t_start", "gap_sec",
                      "match_min", "half_min", "subs_in", "shifts_in", "injury",
                      "goal_before", "card_near", "prev_type", "next_type"],
        "admin": ["match_id", "tournament", "period", "t", "match_min", "type", "team_id"],
        "moves": ["match_id", "period", "t", "team_id", "sx", "sy", "ex", "ey", "ok"],
        "shots": ["match_id", "tournament", "period", "t", "team_id", "xg", "goal", "x", "y", "pen"],
    }
    handles, writers = {}, {}
    for name, header in files.items():
        fh = open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8")
        handles[name] = fh
        writers[name] = csv.writer(fh)
        writers[name].writerow(header)

    n = 0
    for cid, sid, tournament in COMPETITIONS:
        with open(os.path.join(RAW, "matches", "{}_{}.json".format(cid, sid)), encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            writers["matches"].writerow([
                m["match_id"], tournament, m["match_date"], m.get("kick_off"),
                m["home_team"]["home_team_name"], m["away_team"]["away_team_name"],
                m["home_score"], m["away_score"],
                m["home_team"]["home_team_id"], m["away_team"]["away_team_id"],
                (m.get("stadium") or {}).get("name"),
                (m.get("competition_stage") or {}).get("name"),
            ])
            process_match(m["match_id"], tournament, writers)
            n += 1
            if n % 50 == 0:
                print("processed", n)

    for fh in handles.values():
        fh.close()
    print("done:", n, "matches")


if __name__ == "__main__":
    main()
