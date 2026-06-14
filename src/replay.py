"""Replay a played match event-by-event: the real match-report timeline.

The predictor works in final scores; this is the opposite end -- the exact story
of a game that already happened, pulled from FotMob's event feed: every goal
with its minute, scorer, assist and how it was scored; every yellow and red
card; and every substitution with the correct player on/off direction (resolved
against the starting XI, not guessed from field order). Rendered as the vertical
match-report timeline you see on a match page.

Honest scope: injuries are NOT a reliable separate field in the feed -- they
appear only as substitutions, unlabelled -- so we do not invent injury events.

    python src/replay.py             # every played WC2026 match
    python src/replay.py 4667757     # one FotMob match id
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from live_eval import (BG, INK, MUT, FOTMOB, eff_minute, fetch, font,
                       list_matches)

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
HOME_C = "#5e9bff"; AWAY_C = "#ff7a1a"
YEL = "#f2cc0c"; RED = "#e5484d"; GRN = "#3fb950"; PANEL = "#131a23"


def starter_ids(team_block):
    return {str(p.get("id")) for p in (team_block or {}).get("starters", [])}


def build_replay(match_id):
    d = fetch("{}/data/matchDetails?matchId={}".format(FOTMOB, match_id),
              "fm_match_{}".format(match_id))
    g, c = d["general"], d["content"]
    home, away = g["homeTeam"], g["awayTeam"]
    hid, aid = home["id"], away["id"]
    hs = asc = None
    for t in (d.get("header") or {}).get("teams", []):
        if t.get("id") == hid:
            hs = t.get("score")
        elif t.get("id") == aid:
            asc = t.get("score")
    lu = c.get("lineup") or {}
    starters = starter_ids(lu.get("homeTeam")) | starter_ids(lu.get("awayTeam"))

    evs = ((c.get("matchFacts") or {}).get("events") or {}).get("events", [])
    out = []
    for e in evs:
        t = e.get("type")
        side = "home" if e.get("isHome") else "away"
        m = eff_minute(e.get("time"), e.get("overloadTime"),
                       "FirstHalf" if (e.get("time") or 0) <= 45 else "SecondHalf")
        label = "{:.0f}'".format(m)
        if t == "Goal":
            own = bool(e.get("ownGoal"))
            scorer = (e.get("player") or {}).get("name", "")
            desc = e.get("goalDescription") or ""
            extra = []
            if own:
                extra.append("o.g.")
            elif "pen" in desc.lower():
                extra.append("pen")
            elif desc:
                extra.append(desc.lower())
            assist = e.get("assistInput")
            if assist and not own:
                extra.append("assist " + assist)
            # FotMob's isHome already names the team CREDITED with the goal (for
            # an own goal that is the beneficiary, not the scorer's side), so use
            # it directly -- do not flip.
            out.append({"m": m, "label": label, "kind": "goal", "side": side,
                        "text": scorer, "note": ", ".join(extra)})
        elif t == "Card":
            card = e.get("card")
            kind = "red" if card in ("Red", "RedYellow") else "yellow"
            out.append({"m": m, "label": label, "kind": kind, "side": side,
                        "text": (e.get("player") or {}).get("name", ""),
                        "note": "second yellow" if card == "RedYellow" else ""})
        elif t == "Substitution":
            sw = e.get("swap") or []
            names = [s.get("name", "") for s in sw]
            ids = [str(s.get("id")) for s in sw]
            # the starter is the one going OFF; the other comes ON
            if len(sw) == 2 and ids[0] in starters and ids[1] not in starters:
                on, off = names[1], names[0]
            elif len(sw) == 2 and ids[1] in starters and ids[0] not in starters:
                on, off = names[0], names[1]
            else:
                on, off = (names + ["", ""])[:2]   # fallback: FotMob order
            out.append({"m": m, "label": label, "kind": "sub", "side": side,
                        "text": on, "note": "for " + off})
    out.sort(key=lambda x: x["m"])
    return {"home": home["name"], "away": away["name"],
            "score": "{}-{}".format(hs, asc),
            "date": (g.get("matchTimeUTC") or "")[:10], "events": out}


def render(tl, out_path):
    evs = tl["events"]
    n = max(len(evs), 1)
    fig, ax = plt.subplots(figsize=(9, 1.05 + 0.42 * n), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, n + 0.4)
    ax.axis("off")
    cx = 0.5
    ax.plot([cx, cx], [-0.4, n - 0.6], color="#2a3340", lw=1.4, zorder=1)

    KIND_C = {"goal": INK, "yellow": YEL, "red": RED, "sub": GRN}
    ht_drawn = False
    for i, e in enumerate(evs):
        y = n - 1 - i
        home = e["side"] == "home"
        # half-time divider
        if not ht_drawn and e["m"] > 45.5:
            ax.axhline(y + 0.5, xmin=0.30, xmax=0.70, color=MUT, lw=0.8,
                       ls=(0, (4, 3)), alpha=0.6)
            ax.text(cx, y + 0.5, " half-time ", ha="center", va="center",
                    color=MUT, **font(7), bbox=dict(facecolor=BG, edgecolor="none", pad=1))
            ht_drawn = True
        # minute pill
        ax.text(cx, y, e["label"], ha="center", va="center", color=MUT,
                **font(8), bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL,
                edgecolor="#2a3340", lw=0.8), zorder=4)
        # event marker just inside the centre line
        mx = cx - 0.045 if home else cx + 0.045
        col = KIND_C[e["kind"]]
        if e["kind"] == "goal":
            ax.plot(mx, y, "o", ms=11, mfc="#ffffff", mec=BG, mew=1.3, zorder=5)
            ax.plot(mx, y, "o", ms=5, mfc=BG, mec="none", zorder=6)
        elif e["kind"] in ("yellow", "red"):
            ax.plot(mx, y, "s", ms=10, mfc=col, mec=BG, mew=1.0, zorder=5)
        else:  # sub
            ax.plot(mx, y, "^", ms=9, mfc=GRN, mec=BG, mew=0.8, zorder=5)
        # text on the team's side
        tx = cx - 0.075 if home else cx + 0.075
        ha = "right" if home else "left"
        tcol = HOME_C if home else AWAY_C
        main = e["text"] if e["kind"] != "goal" else e["text"] + "  GOAL"
        ax.text(tx, y + 0.12, main, ha=ha, va="center", color=tcol,
                **font(9.5, True))
        if e["note"]:
            ax.text(tx, y - 0.16, e["note"], ha=ha, va="center", color=MUT,
                    **font(7.5))

    # header
    hs, asc = tl["score"].split("-")
    ax.text(cx - 0.06, n - 0.05, "{} {}".format(tl["home"], hs), ha="right",
            va="bottom", color=HOME_C, **font(13, True))
    ax.text(cx + 0.06, n - 0.05, "{} {}".format(asc, tl["away"]), ha="left",
            va="bottom", color=AWAY_C, **font(13, True))
    ax.set_title("Match replay  |  {}  ({})".format(
        tl["date"], "goals, cards, substitutions from the live event feed"),
        color=INK, loc="left", pad=18, **font(12, True))
    fig.text(0.5, 0.01,
             "exact events as they happened | data: FotMob | injuries not separately tracked (appear as subs) | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, **font(7))
    fig.tight_layout(rect=[0, 0.015, 1, 1])
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ids = sys.argv[1:] or list_matches()
    print("replaying {} match(es)".format(len(ids)))
    for mid in ids:
        tl = build_replay(mid)
        slug = "".join(ch if ch.isalnum() else "_" for ch in
                       "{}_{}".format(tl["home"], tl["away"]).lower()).strip("_")
        out = os.path.join(FIG, "wc2026_replay_{}.png".format(slug))
        render(tl, out)
        ng = sum(1 for e in tl["events"] if e["kind"] == "goal")
        nc = sum(1 for e in tl["events"] if e["kind"] in ("yellow", "red"))
        nsub = sum(1 for e in tl["events"] if e["kind"] == "sub")
        print("  {} {} {}  ({} goals, {} cards, {} subs) -> {}".format(
            tl["home"], tl["score"], tl["away"], ng, nc, nsub,
            os.path.basename(out)))


if __name__ == "__main__":
    main()
