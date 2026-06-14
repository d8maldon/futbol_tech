"""Who is actually at WC2026: fetch the 48 squads, build a current-player set.

The action-value and goalscorer models rank players from 2018-2024 history, so
they include people who are not at this World Cup (the user rightly flagged Di
Maria, Kroos, James Rodriguez -- all retired/absent). This builds the set of
players who are ACTUALLY in a WC2026 squad, from FotMob, so those models can be
filtered to the real tournament field.

Source: the FotMob WC2026 league (id 77) lists all 48 teams with their FotMob
ids; each team's page (data/teams?id=) carries the squad. We key players by a
sorted-token name fold so the FotMob short names join to StatsBomb legal names.
Every fetch uses the repo's TLS skip-verify path (curl.exe -k via live_eval).

    python src/squads.py        # fetch 48 squads -> wc2026/wc2026_squads.json
"""
import json
import os
import unicodedata

from live_eval import FOTMOB, fetch

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "wc2026")
WC_LEAGUE = 77


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def name_key(s):
    return " ".join(sorted(fold(s).split()))


def _players(node, out, in_coach=False):
    """walk a FotMob squad structure, collecting player names (skip coaches)"""
    if isinstance(node, dict):
        title = str(node.get("title", "")).lower()
        coach = in_coach or "coach" in title
        # a player member: has a name and a player-ish id, not a coach group
        if not coach and isinstance(node.get("name"), str) and \
                ("id" in node) and ("positionId" in node or "role" in node
                                    or "ccode" in node or "shirtNumber" in node):
            out.add(name_key(node["name"]))
        for v in node.values():
            _players(v, out, coach)
    elif isinstance(node, list):
        for v in node:
            _players(v, out, in_coach)


def team_ids():
    d = fetch("{}/data/leagues?id={}".format(FOTMOB, WC_LEAGUE), "fm_league_77")
    teams = (((d.get("fixtures") or {}).get("fixtureInfo") or {}).get("teams")
             or ((d.get("overview") or {}).get("matches") or {})
             .get("fixtureInfo", {}).get("teams") or [])
    return [(t["id"], t["name"]) for t in teams if t.get("id")]


def build():
    os.makedirs(OUT, exist_ok=True)
    teams = team_ids()
    squads = {}
    for tid, name in teams:
        try:
            d = fetch("{}/data/teams?id={}".format(FOTMOB, tid),
                      "fm_team_{}".format(tid))
        except Exception as e:
            print("  {:20s} fetch failed: {}".format(name, str(e)[:50]))
            continue
        keys = set()
        _players(d.get("squad"), keys)
        squads[name] = sorted(keys)
        print("  {:22s} {} players".format(name, len(keys)))
    allkeys = sorted({k for v in squads.values() for k in v})
    with open(os.path.join(OUT, "wc2026_squads.json"), "w", encoding="utf-8") as f:
        json.dump({"teams": squads, "n_teams": len(squads),
                   "n_players": len(allkeys)}, f, ensure_ascii=False, indent=2)
    return squads, allkeys


# team-name folds differ between StatsBomb history and FotMob squads
TEAM_ALIASES = {
    "czechia": "czech", "czech republic": "czech",
    "usa": "usa", "united states": "usa",
    "cape verde": "capeverde", "cape verde islands": "capeverde",
    "turkiye": "turkey", "turkey": "turkey",
    "ivory coast": "cotedivoire", "cote divoire": "cotedivoire",
    "dr congo": "drcongo", "congo dr": "drcongo",
    "south korea": "southkorea", "korea republic": "southkorea",
    "bosnia and herzegovina": "bosnia", "bosnia herzegovina": "bosnia",
}


def team_canon(name):
    f = fold(name)
    return TEAM_ALIASES.get(f, f.replace(" ", ""))


def current_squads_by_team():
    """{canonical_team: set(name_keys)}; empty dict if squads not built yet"""
    p = os.path.join(OUT, "wc2026_squads.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {team_canon(t): set(ks) for t, ks in d["teams"].items()}


def current_keys():
    """the set of name_keys in any WC2026 squad; empty set if not built yet"""
    by = current_squads_by_team()
    return {k for v in by.values() for k in v}


def in_wc2026(player_name, player_team, by_team):
    """current iff the player's NATION is at WC2026 AND a squad member's tokens
    are a subset of the player's legal name. Scoping to the team kills cross-team
    false positives (Venezuela's Rondon was matching Uruguay's Jose Gimenez)."""
    squad = by_team.get(team_canon(player_team))
    if not squad:
        return False                               # nation not at WC2026
    pt = set(fold(player_name).split())
    if not pt:
        return False
    for sk in squad:
        st = set(sk.split())
        if st and st <= pt:
            return True
    return False


def is_in_squad(player_name, squad_keys):
    """flat (team-agnostic) match -- kept for quick checks; prefer in_wc2026"""
    pt = set(fold(player_name).split())
    return bool(pt) and any(set(sk.split()) <= pt for sk in squad_keys if sk)


def main():
    squads, allkeys = build()
    print("\n{} teams, {} unique squad players".format(len(squads), len(allkeys)))
    for who in ("Ángel Fabián Di María Hernández", "Toni Kroos",
                "Lionel Andrés Messi Cuccittini", "Kylian Mbappé Lottin"):
        sk = {name_key(p) for v in squads.values() for p in v}
        print("  {:34s} in a WC2026 squad? {}".format(
            who, is_in_squad(who, sk)))


if __name__ == "__main__":
    main()
