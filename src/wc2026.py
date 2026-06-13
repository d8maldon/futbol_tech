"""World Cup 2026 live cooling-break tracker.

StatsBomb-grade event streams will not exist publicly for years, but ESPN's
public match feeds log the breaks directly ("Delay in match for a drinks
break.") with minute stamps, plus substitutions and goals. This module pulls
every completed match, extracts breaks and tactical reactions, and renders a
dashboard that accumulates through the tournament.

Run it any time; it re-fetches finished matches not yet cached.
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

API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
START = datetime.date(2026, 6, 11)
ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "wc2026")
OUT = os.path.join(ROOT, "wc2026")
FIG = os.path.join(ROOT, "figures")

BREAK_RE = re.compile(r"drinks break|cooling break|hydration break|water break", re.I)
CLOCK_RE = re.compile(r"(\d+)'(?:\s*\+\s*(\d+)')?")
SUB_RE = re.compile(r"Substitution,\s*([^.]+)\.")

# venue -> UTC offset in June (DST where applicable)
VENUE_TZ = {
    "Banorte": -6, "Azteca": -6, "Akron": -6, "BBVA": -6,
    "BMO": -4, "BC Place": -7, "Lumen": -7, "Levi": -7, "SoFi": -7,
    "Arrowhead": -5, "GEHA": -5, "AT&T": -5, "NRG": -5,
    "Mercedes-Benz": -4, "Hard Rock": -4, "Gillette": -4,
    "Lincoln Financial": -4, "MetLife": -4,
}

BG = "#0d1117"
PANEL = "#131a23"
INK = "#e6edf3"
MUT = "#7d8590"
ACCENT = "#ffb347"
GREEN = "#27c98f"
GRAY = "#5c6773"


def font(size, bold=False):
    name = "Bahnschrift" if any("Bahnschrift" in f.name for f in font_manager.fontManager.ttflist) else "Segoe UI"
    return {"fontfamily": name, "fontsize": size,
            "fontweight": "bold" if bold else "normal"}


def clock(display):
    m = CLOCK_RE.search(display or "")
    if not m:
        return None, 0
    return int(m.group(1)), int(m.group(2) or 0)


def local_hour(date_utc, venue):
    off = 0
    for key, tz in VENUE_TZ.items():
        if key.lower() in (venue or "").lower():
            off = tz
            break
    dt = datetime.datetime.strptime(date_utc, "%Y-%m-%dT%H:%MZ")
    return (dt + datetime.timedelta(hours=off)).hour


def fetch_matches(session):
    today = datetime.date.today()
    url = "{}/scoreboard?dates={:%Y%m%d}-{:%Y%m%d}".format(API, START, today)
    sb = session.get(url, verify=False, timeout=30).json()
    out = []
    for e in sb.get("events", []):
        comp = e["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        out.append({
            "id": e["id"],
            "date_utc": e["date"],
            "venue": (comp.get("venue") or {}).get("fullName", ""),
            "home": teams["home"]["team"]["displayName"],
            "away": teams["away"]["team"]["displayName"],
            "score": "{}-{}".format(teams["home"].get("score", ""), teams["away"].get("score", "")),
            "status": comp["status"]["type"]["name"],
        })
    return out


def fetch_summary(session, mid):
    path = os.path.join(CACHE, "{}.json".format(mid))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    d = session.get("{}/summary?event={}".format(API, mid), verify=False, timeout=30).json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d


def parse_match(summary):
    breaks, subs, goals = [], [], []
    for c in summary.get("commentary", []):
        txt = c.get("text", "")
        base, extra = clock((c.get("time") or {}).get("displayValue", ""))
        if base is None:
            continue
        if BREAK_RE.search(txt):
            breaks.append({"minute": base, "extra": extra, "text": txt.strip()})
        m = SUB_RE.search(txt)
        if m:
            subs.append({"minute": base, "extra": extra, "team": m.group(1).strip()})
    for k in summary.get("keyEvents", []):
        if k.get("scoringPlay"):
            base, extra = clock((k.get("clock") or {}).get("displayValue", ""))
            if base is not None:
                goals.append({"minute": base, "extra": extra,
                              "team": (k.get("team") or {}).get("displayName", "")})
    return breaks, subs, goals


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    session = requests.Session()
    matches = fetch_matches(session)

    rows, brows = [], []
    for m in matches:
        if m["status"] != "STATUS_FULL_TIME":
            print("skipping (not finished): {} vs {}".format(m["home"], m["away"]))
            continue
        breaks, subs, goals = parse_match(fetch_summary(session, m["id"]))
        m2 = dict(m)
        m2["local_hour"] = local_hour(m["date_utc"], m["venue"])
        m2["n_breaks"] = len(breaks)
        m2["n_subs"] = len(subs)
        # subs in the 5 minutes after a break vs elsewhere (minute resolution)
        after = sum(1 for s in subs for b in breaks
                    if 0 <= (s["minute"] + s["extra"]) - (b["minute"] + b["extra"]) <= 5)
        m2["subs_after_breaks"] = after
        rows.append(m2)
        for b in breaks:
            brows.append({"match_id": m["id"], "match": "{} vs {}".format(m["home"], m["away"]),
                          "date": m["date_utc"][:10], "venue": m["venue"],
                          "local_hour": m2["local_hour"], "minute": b["minute"],
                          "extra": b["extra"], "text": b["text"]})

    import csv
    with open(os.path.join(OUT, "matches.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "breaks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["match_id", "match", "date", "venue",
                                          "local_hour", "minute", "extra", "text"])
        w.writeheader()
        w.writerows(brows)

    n = len(rows)
    nb = len(brows)
    with_b = sum(1 for r in rows if r["n_breaks"] > 0)
    print("\n{} finished matches, {} drinks breaks, {} matches with at least one".format(n, nb, with_b))
    for r in rows:
        print("  {} {} {} ({}:00 local, {}): {} breaks".format(
            r["date_utc"][:10], "{} {} {}".format(r["home"], r["score"], r["away"]),
            "", r["local_hour"], r["venue"], r["n_breaks"]))

    render(rows, brows)
    print("figure: figures/wc2026_tracker.png")


def render(rows, brows):
    day = (datetime.date.today() - START).days + 1
    n = len(rows)
    nb = len(brows)
    with_b = sum(1 for r in rows if r["n_breaks"] > 0)

    fig = plt.figure(figsize=(7.2, 9), dpi=200)
    fig.patch.set_facecolor(BG)
    nrows = max(len(rows), 1)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.85, 0.55, 2.6, 2.0],
                          hspace=0.50, wspace=0.18,
                          left=0.075, right=0.94, top=0.97, bottom=0.075)

    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    axt.text(0, 0.90, "WORLD CUP 2026 | DAY {}".format(day), color=ACCENT, **font(10.5, True))
    axt.text(0, 0.46, "HIDDEN TIMEOUT TRACKER", color=INK, **font(30, True))
    axt.text(0, 0.12, "Every cooling break of the tournament, logged live from match feeds.",
             color=INK, **font(11))

    # counters
    counters = [("{}".format(n), "matches\nplayed"), ("{}".format(nb), "drinks\nbreaks"),
                ("{:.0f}%".format(100 * with_b / max(n, 1)), "matches with\nat least one")]
    for i, (big, small) in enumerate(counters):
        axc = fig.add_subplot(gs[1, i])
        axc.set_facecolor(PANEL); axc.set_xticks([]); axc.set_yticks([])
        for s in axc.spines.values():
            s.set_visible(False)
        axc.text(0.5, 0.58, big, ha="center", color=GREEN, transform=axc.transAxes, **font(24, True))
        axc.text(0.5, 0.16, small, ha="center", color=MUT, transform=axc.transAxes, **font(8.5))

    # match timeline
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
        bymatch.setdefault(b["match_id"], []).append(b)
    for i, r in enumerate(reversed(rows)):
        y = i
        axm.plot([0, 95], [y, y], color="#2a3340", lw=4, solid_capstyle="round", zorder=1)
        label = "{} {} {}".format(r["home"][:18], r["score"], r["away"][:18])
        axm.text(-37, y, label, ha="left", va="center", color=INK, **font(7))
        axm.text(99, y, "{}:00".format(r["local_hour"]), ha="right", va="center",
                 color=MUT, **font(7))
        for b in bymatch.get(r["id"], []):
            axm.plot([b["minute"] + b["extra"]], [y], "o", ms=7, mfc=ACCENT, mec=BG,
                     mew=1.2, zorder=3)
    axm.set_title("Breaks by match (amber dots), protocol windows shaded | local kickoff hour at right",
                  color=MUT, pad=8, loc="left", **font(9))
    axm.text(0, 1.10, "The tournament so far", color=INK, transform=axm.transAxes, **font(12, True))

    # live fingerprint vs history
    axh = fig.add_subplot(gs[3, :])
    axh.set_facecolor(PANEL)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.tick_params(colors=MUT, labelsize=8, length=0)
    half_minutes = []
    for b in brows:
        m = b["minute"]
        half_minutes.append(m if m <= 45 else m - 45)
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
             "breaks parsed from live match commentary | historical baseline: github.com/d8maldon/hidden-timeout",
             color=MUT, **font(8))
    fig.savefig(os.path.join(FIG, "wc2026_tracker.png"), facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
