"""Chess-engine-style win-probability eval graph for live World Cup matches.

This is the chess.com evaluation bar, translated to football. A single line
tracks who is winning the *match* (not the scoreboard): up means the home
team is favoured to win, down means the away team is, zero is a coin flip.
The line moves on goals (big steps), red cards (man advantage), and the
slow accumulation of chance quality (xG) between events. The biggest single
swing is annotated the way a chess engine flags the move that lost the game.

Data: FotMob's public match feed (shotmap with per-shot xG, goals, cards,
weather). The model is trained in winprob.py on this repo's 551 historical
matches and stored as plain JSON, so this module needs only numpy.

    python src/live_eval.py            # all finished WC 2026 matches so far
    python src/live_eval.py 4667757    # one FotMob match id
"""
import datetime
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from winprob import predict

FOTMOB = "https://www.fotmob.com/api"
WC_LEAGUE = 77
START = datetime.date(2026, 6, 11)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "wc2026")
OUT = os.path.join(ROOT, "wc2026")
FIG = os.path.join(ROOT, "figures")

BG = "#0d1117"
PANEL = "#131a23"
INK = "#e6edf3"
MUT = "#7d8590"
ACCENT = "#ffb347"
HOME_C = "#5e9bff"
AWAY_C = "#ff7a1a"
GRID = "#2a3340"


def font(size, bold=False):
    name = "Bahnschrift" if any("Bahnschrift" in f.name for f in font_manager.fontManager.ttflist) else "Segoe UI"
    return {"fontfamily": name, "fontsize": size,
            "fontweight": "bold" if bold else "normal"}


def fetch(url, key):
    """FotMob fingerprints plain HTTP clients; curl with browser headers and
    the cache path is what reliably returns 200 on this machine."""
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    subprocess.run(
        ["curl.exe", "-k", "-s", "-o", path, url,
         "-H", "User-Agent: " + UA, "-H", "Accept: */*",
         "-H", "Referer: https://www.fotmob.com/"],
        check=True, timeout=60)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_matches():
    """finished WC 2026 men's matches, FotMob ids, oldest first"""
    out = []
    today = datetime.date.today()
    day = START
    while day <= today:
        key = "fm_day_{:%Y%m%d}".format(day)
        # only the live (today) file should ever skip the cache
        if day == today:
            p = os.path.join(CACHE, key + ".json")
            if os.path.exists(p):
                os.remove(p)
        d = fetch("{}/data/matches?date={:%Y%m%d}".format(FOTMOB, day), key)
        for lg in d.get("leagues", []):
            name = lg.get("name", "")
            if "World Cup" in name and "Women" not in name:
                for m in lg.get("matches", []):
                    if (m.get("status") or {}).get("finished"):
                        out.append(m["id"])
        day += datetime.timedelta(days=1)
    return out


def eff_minute(min_, added, period):
    """collapse stoppage time so the two halves do not overlap on the axis"""
    if period == "FirstHalf" or (min_ is not None and min_ <= 45):
        return min(min_ + (added or 0) / 5.0, 45.5)
    return min(min_ + (added or 0) / 5.0, 96.0)


def build_timeline(match_id):
    d = fetch("{}/data/matchDetails?matchId={}".format(FOTMOB, match_id),
              "fm_match_{}".format(match_id))
    g = d["general"]
    content = d["content"]
    home, away = g["homeTeam"], g["awayTeam"]
    hid, aid = home["id"], away["id"]
    hscore = ascore = None
    for t in (d.get("header") or {}).get("teams", []):
        if t.get("id") == hid:
            hscore = t.get("score")
        elif t.get("id") == aid:
            ascore = t.get("score")

    shots = (content.get("shotmap") or {}).get("shots", []) or []
    chances = []
    for s in shots:
        m = eff_minute(s.get("min"), s.get("minAdded"),
                       s.get("period", "FirstHalf"))
        chances.append({"m": m, "team": s["teamId"],
                        "xg": float(s.get("expectedGoals") or 0.0)})

    goals, cards = [], []
    weather = content.get("weather") or {}
    events = ((content.get("matchFacts") or {}).get("events") or {}).get("events", [])
    for e in events:
        m = eff_minute(e.get("time"), e.get("overloadTime"),
                       "FirstHalf" if (e.get("time") or 0) <= 45 else "SecondHalf")
        if e.get("type") == "Goal":
            own = bool(e.get("ownGoal"))
            scoring = (aid if e.get("isHome") else hid) if own else (hid if e.get("isHome") else aid)
            goals.append({"m": m, "team": scoring,
                          "scorer": (e.get("player") or {}).get("name", ""), "own": own})
        elif e.get("type") == "Card" and e.get("card") in ("Red", "RedYellow"):
            cards.append({"m": m, "team": hid if e.get("isHome") else aid})

    return {
        "match_id": match_id,
        "home": home["name"], "away": away["name"], "hid": hid, "aid": aid,
        "score": "{}-{}".format(hscore, ascore),
        "date": (g.get("matchTimeUTC") or "")[:16],
        "temp_c": weather.get("temperature"), "humidity": weather.get("relativeHumidity"),
        "chances": sorted(chances, key=lambda c: c["m"]),
        "goals": sorted(goals, key=lambda c: c["m"]),
        "cards": sorted(cards, key=lambda c: c["m"]),
    }


def win_prob_curve(model, tl, step=0.5):
    grid = np.arange(0, 96 + step, step)
    edge, ph, pa = [], [], []
    for minute in grid:
        hg = sum(1 for x in tl["goals"] if x["team"] == tl["hid"] and x["m"] <= minute)
        ag = sum(1 for x in tl["goals"] if x["team"] == tl["aid"] and x["m"] <= minute)
        hxg = sum(c["xg"] for c in tl["chances"] if c["team"] == tl["hid"] and c["m"] <= minute)
        axg = sum(c["xg"] for c in tl["chances"] if c["team"] == tl["aid"] and c["m"] <= minute)
        mad = (sum(1 for c in tl["cards"] if c["team"] == tl["aid"] and c["m"] <= minute)
               - sum(1 for c in tl["cards"] if c["team"] == tl["hid"] and c["m"] <= minute))
        p = predict(model, hg - ag, hxg - axg, mad, min(minute, 90))
        ph.append(p["H"]); pa.append(p["A"])
        edge.append(p["H"] - p["A"])
    return grid, np.array(edge), np.array(ph), np.array(pa)


def biggest_swing(grid, edge, ph, tl):
    """Largest swing in the match eval over a ~3 min window: the chess
    'turning point'. The narrative is chosen from what actually happened so a
    comeback is never mislabelled as a collapse."""
    win = max(int(3 / (grid[1] - grid[0])), 1)
    best = {"delta": 0.0, "at": None}
    for i in range(len(grid) - win):
        d = edge[i + win] - edge[i]
        if abs(d) > abs(best["delta"]):
            best = {"delta": d, "at": grid[i + win],
                    "pre_edge": edge[i], "post_edge": edge[i + win],
                    "pre_home": ph[i]}
    if best["at"] is None:
        return best
    gainer = tl["home"] if best["delta"] > 0 else tl["away"]
    loser = tl["away"] if best["delta"] > 0 else tl["home"]
    # did the side that lost ground here come into the swing clearly ahead?
    loser_was_ahead = (best["pre_edge"] > 0.35 and best["delta"] < 0) or \
                      (best["pre_edge"] < -0.35 and best["delta"] > 0)
    # and did that same side fail to win in the end?
    hs, as_ = (int(x) for x in tl["score"].split("-"))
    loser_result = (hs - as_) if loser == tl["home"] else (as_ - hs)
    if loser_was_ahead and loser_result <= 0:
        best["headline"] = "the match turned here"
        best["sub"] = "{} were on top, then let it slip".format(loser)
    else:
        best["headline"] = "the match swung here"
        best["sub"] = "{} seized control".format(gainer)
    return best


def render(model, tl):
    grid, edge, ph, pa = win_prob_curve(model, tl)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.fill_between(grid, 0, edge, where=edge >= 0, color=HOME_C, alpha=0.9,
                    interpolate=True, lw=0)
    ax.fill_between(grid, 0, edge, where=edge <= 0, color=AWAY_C, alpha=0.9,
                    interpolate=True, lw=0)
    ax.plot(grid, edge, color=INK, lw=1.0, alpha=0.6)
    ax.axhline(0, color=INK, lw=1.2)
    ax.axvline(45.5, color=MUT, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, 96)

    sw = biggest_swing(grid, edge, ph, tl)
    swing_x = sw["at"] if abs(sw["delta"]) > 0.15 else None

    for g in tl["goals"]:
        x = g["m"]
        e = np.interp(x, grid, edge)
        ax.plot([x], [e], "o", ms=8, mfc=ACCENT, mec=BG, mew=1.5, zorder=6)
        lbl = "OG {:.0f}'".format(x) if g["own"] else "GOAL {:.0f}'".format(x)
        up = e >= 0
        ax.annotate(lbl, (x, e), xytext=(0, 15 if up else -15),
                    textcoords="offset points", ha="center",
                    va="bottom" if up else "top", color=INK,
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.7, pad=1),
                    zorder=7, **font(8, True))
    for c in tl["cards"]:
        x = c["m"]
        who = tl["home"] if c["team"] == tl["hid"] else tl["away"]
        ax.axvline(x, color="#e5484d", lw=1.4, alpha=0.7, zorder=2)
        ax.text(x - 0.6, 0.96, "RED {:.0f}' {}".format(x, who[:3].upper()),
                rotation=90, va="top", ha="right", color="#e5484d", **font(7.5, True))

    if swing_x is not None:
        e = np.interp(swing_x, grid, edge)
        # drop the callout into the emptiest quadrant: opposite the swing
        # horizontally, and on the side of the axis the match spent least time
        text_x = 19 if swing_x > 50 else 77
        text_y = 0.62 if edge.mean() < 0 else -0.62
        ax.annotate("{}\n{}".format(sw["headline"], sw["sub"]),
                    (swing_x, e), xytext=(text_x, text_y),
                    ha="left", va="center", color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1,
                                    connectionstyle="arc3,rad=0.15"),
                    bbox=dict(facecolor=BG, edgecolor=ACCENT, lw=0.8, alpha=0.9, pad=3),
                    zorder=8, **font(8.5, True))

    ax.text(1.5, 0.92, "{} {}".format(tl["home"].upper(), tl["score"].split("-")[0]),
            color=HOME_C, va="top", **font(12, True))
    ax.text(1.5, -0.92, "{} {}".format(tl["away"].upper(), tl["score"].split("-")[1]),
            color=AWAY_C, va="bottom", **font(12, True))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["away\nwin", "", "even", "", "home\nwin"], color=MUT, fontsize=7.5)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xticklabels(["0'", "15'", "30'", "45'", "60'", "75'", "90'"], color=MUT, fontsize=8)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    heat = ""
    if tl["temp_c"] is not None:
        heat = "  |  {}C".format(tl["temp_c"])
        if tl["humidity"] is not None:
            heat += ", {}% humidity".format(tl["humidity"])
    ax.set_title("Win-probability eval  |  {} {} {}{}".format(
        tl["home"], tl["score"], tl["away"], heat),
        color=INK, pad=24, loc="left", **font(13, True))
    ax.text(0, 1.04, "how the match favoured each side, minute by minute, from chance quality and the scoreboard",
            transform=ax.transAxes, color=MUT, va="bottom", **font(8.5))
    fig.text(0.5, 0.008,
             "model trained on 551 historical matches (log loss 0.79) | live data: FotMob | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, **font(7.5))

    slug = "".join(ch if ch.isalnum() else "_" for ch in
                   "{}_{}".format(tl["home"], tl["away"]).lower()).strip("_")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(FIG, "wc2026_eval_{}.png".format(slug))
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    sw["edge_end"] = float(edge[-1])
    return out, sw


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "winprob_model.json")) as f:
        model = json.load(f)

    ids = sys.argv[1:] or list_matches()
    print("rendering {} match(es)".format(len(ids)))
    rows = []
    for mid in ids:
        tl = build_timeline(mid)
        out, sw = render(model, tl)
        turn = "  turning point {:.0f}' ({:+.0f} pts)".format(
            sw["at"], 100 * sw["delta"]) if abs(sw.get("delta", 0)) > 0.15 else ""
        print("  {} {} {} -> {}{}".format(
            tl["home"], tl["score"], tl["away"], os.path.basename(out), turn))
        rows.append({
            "match_id": mid, "date": tl["date"][:10],
            "home": tl["home"], "away": tl["away"], "score": tl["score"],
            "temp_c": tl["temp_c"], "humidity": tl["humidity"],
            "swing_minute": round(sw["at"], 1) if sw.get("at") else "",
            "swing_pts": round(100 * sw["delta"], 1) if sw.get("at") else "",
            "narrative": sw.get("sub", "") if abs(sw.get("delta", 0)) > 0.15 else "",
        })
    import csv
    with open(os.path.join(OUT, "winprob_swings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("summary -> wc2026/winprob_swings.csv")


if __name__ == "__main__":
    main()
