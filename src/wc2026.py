"""World Cup 2026 live cooling-break tracker.

StatsBomb-grade event streams will not be public for years, but two live
sources cover the analysis:

- FIFA's own timeline API logs every hydration break officially (type 83
  "Match paused for a hydration break") and its resume (type 78), both with
  millisecond wall clocks, plus substitutions with wall clocks. That gives
  exact break durations and exact subs-during-the-pause counts.
- ESPN's commentary feed describes every shot with a location phrase
  ("centre of the box", "very close range"...). Those zones are calibrated
  against the 13.6k open-play shots in the historical StatsBomb data to give
  each chance an xG-like weight, which drives the momentum rivers.

Run any time; finished matches are cached, new ones are fetched.
"""
import datetime
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import urllib3
from matplotlib import font_manager

urllib3.disable_warnings()

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
FIFA = "https://api.fifa.com/api/v3"
COMP, SEASON = 17, 285023
START = datetime.date(2026, 6, 11)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "wc2026")
OUT = os.path.join(ROOT, "wc2026")
FIG = os.path.join(ROOT, "figures")

CLOCK_RE = re.compile(r"(\d+)'(?:\s*\+\s*(\d+)')?")
TEAM_RE = re.compile(r"\(([^)]+)\)")

# mean open-play xG by commentary zone, calibrated on the 13,597 historical
# StatsBomb shots in this repo (see README); in-game penalties fixed at 0.78
ZONE_XG = {
    "very close range": 0.292,
    "centre of the box": 0.134,
    "left side of the box": 0.057,
    "right side of the box": 0.055,
    "outside the box": 0.035,
    "more than 35 yards": 0.007,
    "penalty": 0.78,
}
CORNER_VAL = 0.025

BG = "#0d1117"
PANEL = "#131a23"
INK = "#e6edf3"
MUT = "#7d8590"
ACCENT = "#ffb347"
GREEN = "#27c98f"
HOME_C = "#5e9bff"
AWAY_C = "#ff7a1a"

VENUE_TZ = {
    "Banorte": -6, "Azteca": -6, "Akron": -6, "BBVA": -6,
    "BMO": -4, "BC Place": -7, "Lumen": -7, "Levi": -7, "SoFi": -7,
    "Arrowhead": -5, "GEHA": -5, "AT&T": -5, "NRG": -5,
    "Mercedes-Benz": -4, "Hard Rock": -4, "Gillette": -4,
    "Lincoln Financial": -4, "MetLife": -4,
}


def font(size, bold=False):
    name = "Bahnschrift" if any("Bahnschrift" in f.name for f in font_manager.fontManager.ttflist) else "Segoe UI"
    return {"fontfamily": name, "fontsize": size,
            "fontweight": "bold" if bold else "normal"}


def clock(display):
    m = CLOCK_RE.search(display or "")
    if not m:
        return None
    base, extra = int(m.group(1)), int(m.group(2) or 0)
    # compress stoppage time so halves do not overlap on the minute axis
    if base <= 45:
        return min(base + extra, 45.9)
    return min(base + extra, 95.9)


def wall(ts):
    return datetime.datetime.strptime(ts[:23].rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f")


def local_hour(date_utc, venue):
    off = next((tz for k, tz in VENUE_TZ.items() if k.lower() in (venue or "").lower()), 0)
    dt = datetime.datetime.strptime(date_utc, "%Y-%m-%dT%H:%MZ")
    return (dt + datetime.timedelta(hours=off)).hour


def cached_json(session, key, url):
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    d = session.get(url, headers=UA, verify=False, timeout=30).json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d


# ------------------------------------------------------------------- FIFA side
def fifa_matches(session):
    url = "{}/calendar/matches?idCompetition={}&idSeason={}&count=500&language=en".format(
        FIFA, COMP, SEASON)
    d = session.get(url, headers=UA, verify=False, timeout=30).json()
    out = []
    for m in d.get("Results", []):
        if m.get("MatchStatus") != 0:  # 0 = finished
            continue
        out.append({
            "fifa_id": m["IdMatch"], "stage": m["IdStage"],
            "kickoff": m.get("Date", "")[:16],
            "home": (m.get("Home") or {}).get("TeamName", [{}])[0].get("Description", ""),
            "away": (m.get("Away") or {}).get("TeamName", [{}])[0].get("Description", ""),
            "home_score": (m.get("Home") or {}).get("Score"),
            "away_score": (m.get("Away") or {}).get("Score"),
        })
    return out


def fifa_timeline(session, m):
    d = cached_json(session, "fifa_{}".format(m["fifa_id"]),
                    "{}/timelines/{}/{}/{}/{}?language=en".format(
                        FIFA, COMP, SEASON, m["stage"], m["fifa_id"]))
    evs = sorted(d.get("Event", []), key=lambda e: e.get("Timestamp", ""))
    breaks, subs = [], []
    for i, e in enumerate(evs):
        if e.get("Type") == 83:  # hydration break
            start = wall(e["Timestamp"])
            end = None
            for nxt in evs[i + 1:]:
                if nxt.get("Type") == 78:  # match resumed
                    end = wall(nxt["Timestamp"])
                    break
            breaks.append({
                "minute": clock(e.get("MatchMinute", "")),
                "start": start, "end": end,
                "duration_sec": (end - start).total_seconds() if end else None,
            })
        elif e.get("Type") == 5:  # substitution
            subs.append({"minute": clock(e.get("MatchMinute", "")),
                         "wall": wall(e["Timestamp"])})
    return breaks, subs


# ------------------------------------------------------------------- ESPN side
def espn_matches(session):
    today = datetime.date.today()
    url = "{}/scoreboard?dates={:%Y%m%d}-{:%Y%m%d}".format(ESPN, START, today)
    sb = session.get(url, headers=UA, verify=False, timeout=30).json()
    out = []
    for e in sb.get("events", []):
        comp = e["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        out.append({
            "espn_id": e["id"], "kickoff": e["date"][:16],
            "venue": (comp.get("venue") or {}).get("fullName", ""),
            "home": teams["home"]["team"]["displayName"],
            "away": teams["away"]["team"]["displayName"],
            "score": "{}-{}".format(teams["home"].get("score", ""), teams["away"].get("score", "")),
            "finished": comp["status"]["type"]["name"] == "STATUS_FULL_TIME",
            "date_utc": e["date"],
        })
    return out


def espn_threat(session, m):
    """commentary -> (team, minute, weight) chance events, zone-calibrated"""
    d = cached_json(session, "espn_{}".format(m["espn_id"]),
                    "{}/summary?event={}".format(ESPN, m["espn_id"]))
    events = []
    for c in d.get("commentary", []):
        txt = c.get("text", "")
        minute = clock((c.get("time") or {}).get("displayValue", ""))
        if minute is None:
            continue
        low = txt.lower()
        team = None
        if txt.startswith("Corner,"):
            team = txt.split("Corner,")[1].split(".")[0].strip()
            events.append((team, minute, CORNER_VAL))
            continue
        if txt.startswith(("Attempt", "Goal!")) or "converts the penalty" in low or "penalty saved" in low:
            tm = TEAM_RE.search(txt)
            if not tm:
                continue
            team = tm.group(1).strip()
            val = next((v for z, v in ZONE_XG.items() if z in low), None)
            if "penalty" in low:
                val = ZONE_XG["penalty"]
            events.append((team, minute, val if val is not None else 0.05))
    goals = []
    for k in d.get("keyEvents", []):
        if k.get("scoringPlay"):
            minute = clock((k.get("clock") or {}).get("displayValue", ""))
            if minute is not None:
                goals.append({"minute": minute,
                              "team": (k.get("team") or {}).get("displayName", "")})
    return events, goals


# ------------------------------------------------------------------- pipeline
def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    session = requests.Session()

    fifa = fifa_matches(session)
    espn = [m for m in espn_matches(session) if m["finished"]]
    by_kickoff = {m["kickoff"]: m for m in espn}

    rows, brows = [], []
    rivers = []
    for fm in fifa:
        em = by_kickoff.get(fm["kickoff"])
        if em is None:
            print("no ESPN match for", fm["home"], "vs", fm["away"], fm["kickoff"])
            continue
        breaks, subs = fifa_timeline(session, fm)
        threat, goals = espn_threat(session, em)
        n_during = n_after = 0
        for b in breaks:
            if b["end"] is None:
                continue
            n_during += sum(1 for s in subs if b["start"] <= s["wall"] <= b["end"])
            n_after += sum(1 for s in subs
                           if b["end"] < s["wall"] <= b["end"] + datetime.timedelta(minutes=5))
        row = {
            "fifa_id": fm["fifa_id"], "espn_id": em["espn_id"], "date": fm["kickoff"][:10],
            "home": em["home"], "away": em["away"], "score": em["score"],
            "venue": em["venue"], "local_hour": local_hour(em["date_utc"], em["venue"]),
            "n_breaks": len(breaks), "n_subs": len(subs),
            "subs_during_breaks": n_during, "subs_5min_after": n_after,
        }
        rows.append(row)
        for b in breaks:
            brows.append({
                "fifa_id": fm["fifa_id"], "match": "{} vs {}".format(em["home"], em["away"]),
                "date": row["date"], "venue": em["venue"], "local_hour": row["local_hour"],
                "minute": b["minute"], "duration_sec": b["duration_sec"],
            })
        rivers.append((row, breaks, threat, goals))

    import csv
    for name, data, fields in (
            ("matches", rows, list(rows[0].keys())),
            ("breaks", brows, list(brows[0].keys()))):
        with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    print("{} matches, {} official hydration breaks, {} subs during pauses, {} within 5 min after".format(
        len(rows), len(brows), sum(r["subs_during_breaks"] for r in rows),
        sum(r["subs_5min_after"] for r in rows)))
    for r in rows:
        print("  {} {} {} {} | {}:00 local | {} breaks | {} subs in pause".format(
            r["date"], r["home"], r["score"], r["away"], r["local_hour"],
            r["n_breaks"], r["subs_during_breaks"]))

    render_tracker(rows, brows)
    for row, breaks, threat, goals in rivers:
        render_river(row, breaks, threat, goals)
    print("figures updated")


# ------------------------------------------------------------------- figures
def render_river(row, breaks, threat, goals):
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    tg = np.arange(0, 96.5, 0.05)

    def series(team):
        ev = [(m, v) for t, m, v in threat if t == team]
        out = np.zeros_like(tg)
        for m, v in ev:
            out += v * np.exp(-0.5 * ((tg - m) / 2.0) ** 2)
        return out / (2.0 * np.sqrt(2 * np.pi))

    hr, ar = series(row["home"]), series(row["away"])
    ax.fill_between(tg, 0, hr, color=HOME_C, alpha=0.9, lw=0)
    ax.fill_between(tg, 0, -ar, color=AWAY_C, alpha=0.9, lw=0)
    top = max(hr.max(), ar.max(), 0.01) * 1.25
    ax.set_ylim(-top, top)
    ax.set_xlim(0, 96.5)
    ax.axhline(0, color=BG, lw=1.3)
    ax.axvline(45.95, color=MUT, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
    for b in breaks:
        if b["minute"] is None:
            continue
        w = (b["duration_sec"] or 120) / 60.0
        ax.axvspan(b["minute"], b["minute"] + w, color=ACCENT, alpha=0.32, zorder=1)
        ax.text(b["minute"] + w / 2, -top * 0.78,
                "BREAK {:.0f}'\n{:.0f}s".format(b["minute"], b["duration_sec"] or 0),
                ha="center", va="center", color=ACCENT,
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.75, pad=2), **font(8, True))
    for g in goals:
        side = 1 if g["team"] == row["home"] else -1
        r = np.interp(g["minute"], tg, hr if side > 0 else ar)
        ax.plot([g["minute"]], [side * r], "o", ms=7, mfc=ACCENT, mec=BG, mew=1.3, zorder=5)
        ax.text(g["minute"], side * (r + top * 0.12), "GOAL {:.0f}'".format(g["minute"]),
                ha="center", va="center", color=INK, **font(7.5, True))
    ax.text(1.5, top * 0.86, "{} {}".format(row["home"].upper(), row["score"].split("-")[0]),
            color=HOME_C, **font(11, True))
    ax.text(1.5, -top * 0.94, "{} {}".format(row["away"].upper(), row["score"].split("-")[1]),
            color=AWAY_C, **font(11, True))
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xticklabels(["0'", "15'", "30'", "45'", "60'", "75'", "90'"], color=MUT, fontsize=8)
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_title("{} {} {} | {} | hydration breaks from the official FIFA feed".format(
        row["home"], row["score"], row["away"], row["date"]),
        color=INK, pad=10, loc="left", **font(12, True))
    ax.text(1.0, -0.12, "chance quality per minute, commentary shots weighted by historical zone xG",
            ha="right", color=MUT, transform=ax.transAxes, **font(8))
    slug = re.sub(r"[^a-z0-9]+", "_", "{}_{}".format(row["home"], row["away"]).lower()).strip("_")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "wc2026_river_{}.png".format(slug)),
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def render_tracker(rows, brows):
    day = (datetime.date.today() - START).days + 1
    n = len(rows)
    nb = len(brows)
    with_b = sum(1 for r in rows if r["n_breaks"] > 0)
    n_during = sum(r["subs_during_breaks"] for r in rows)

    fig = plt.figure(figsize=(7.2, 9), dpi=200)
    fig.patch.set_facecolor(BG)
    nrows = max(len(rows), 1)
    gs = fig.add_gridspec(4, 4, height_ratios=[0.85, 0.55, 2.6, 2.0],
                          hspace=0.50, wspace=0.16,
                          left=0.075, right=0.94, top=0.97, bottom=0.075)

    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    axt.text(0, 0.90, "WORLD CUP 2026 | DAY {}".format(day), color=ACCENT, **font(10.5, True))
    axt.text(0, 0.46, "HIDDEN TIMEOUT TRACKER", color=INK, **font(30, True))
    axt.text(0, 0.12, "Every official hydration break of the tournament, from FIFA's live feed.",
             color=INK, **font(11))

    counters = [("{}".format(n), "matches\nplayed"), ("{}".format(nb), "hydration\nbreaks"),
                ("{:.0f}%".format(100 * with_b / max(n, 1)), "matches with\nat least one"),
                ("{}".format(n_during), "subs made\nduring pauses")]
    for i, (big, small) in enumerate(counters):
        axc = fig.add_subplot(gs[1, i])
        axc.set_facecolor(PANEL); axc.set_xticks([]); axc.set_yticks([])
        for s in axc.spines.values():
            s.set_visible(False)
        axc.text(0.5, 0.58, big, ha="center", color=GREEN, transform=axc.transAxes, **font(22, True))
        axc.text(0.5, 0.14, small, ha="center", color=MUT, transform=axc.transAxes, **font(8.5))

    axm = fig.add_subplot(gs[2, :])
    axm.set_facecolor(PANEL)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.set_xlim(-38, 100)
    axm.set_ylim(-0.5, nrows - 0.5)
    axm.set_yticks([])
    axm.set_xticks([0, 15, 30, 45, 60, 75, 90])
    axm.set_xticklabels(["0'", "15'", "30'", "45'", "60'", "75'", "90'"], color=MUT, fontsize=8)
    axm.tick_params(length=0)
    for w0, w1 in ((25, 32), (70, 77)):
        axm.axvspan(w0, w1, color=ACCENT, alpha=0.10, zorder=0)
    bymatch = {}
    for b in brows:
        bymatch.setdefault(b["fifa_id"], []).append(b)
    for i, r in enumerate(reversed(rows)):
        y = i
        axm.plot([0, 95], [y, y], color="#2a3340", lw=4, solid_capstyle="round", zorder=1)
        axm.text(-37, y, "{} {} {}".format(r["home"][:18], r["score"], r["away"][:18]),
                 ha="left", va="center", color=INK, **font(7))
        axm.text(99, y, "{}:00".format(r["local_hour"]), ha="right", va="center",
                 color=MUT, **font(7))
        for b in bymatch.get(r["fifa_id"], []):
            if b["minute"] is not None:
                axm.plot([b["minute"]], [y], "o", ms=7, mfc=ACCENT, mec=BG, mew=1.2, zorder=3)
    axm.set_title("Breaks by match (amber dots), historical windows shaded | local kickoff hour at right",
                  color=MUT, pad=8, loc="left", **font(9))
    axm.text(0, 1.10, "The tournament so far", color=INK, transform=axm.transAxes, **font(12, True))

    axh = fig.add_subplot(gs[3, :])
    axh.set_facecolor(PANEL)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.tick_params(colors=MUT, labelsize=8, length=0)
    half_minutes = [b["minute"] if b["minute"] <= 45 else b["minute"] - 45
                    for b in brows if b["minute"] is not None]
    bins = np.arange(0, 50, 1)
    h, _ = np.histogram(half_minutes, bins=bins)
    axh.bar(bins[:-1] + 0.5, h, width=0.9, color=GREEN)
    axh.axvspan(25, 32, color=ACCENT, alpha=0.14, zorder=0)
    axh.text(28.5, max(h.max(), 1) * 1.04, "the historical break window\n(minutes 25-31 of the half)",
             ha="center", va="bottom", color=ACCENT, **font(8.5, True))
    axh.set_xlabel("minute of the half when the break was called", color=MUT, labelpad=2, **font(9))
    axh.set_ylabel("breaks", color=MUT, **font(9))
    axh.set_ylim(0, max(h.max(), 1) * 1.40)
    axh.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axh.set_title("The fingerprint, accumulating live", color=INK, pad=10, loc="left", **font(12, True))

    fig.text(0.075, 0.008,
             "breaks and sub timing: FIFA live timeline | chance data: ESPN commentary | github.com/d8maldon/hidden-timeout",
             color=MUT, **font(8))
    fig.savefig(os.path.join(FIG, "wc2026_tracker.png"), facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
