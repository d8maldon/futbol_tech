"""Tier-1 event analytics: pass network, PPDA (pressing), field tilt, xT-added.

These are the stats that make an analysis look pro, and NONE of them need precise
tracking -- they come from the event feed. WC2026 only exposes shot-level data
publicly (FotMob), so this runs on StatsBomb open data as the capability demo; the
identical code ingests any pass-level feed (Opta/StatsBomb/in-house).

  pass network  player avg positions + pass links (one team)
  PPDA          opponent passes / our defensive actions in the press zone (low = intense)
  field tilt    share of final-third passes (territorial dominance)
  xT-added      sum of Expected-Threat gained by passes & carries (our trained 16x12 grid)

    python src/event_stats.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xt_model import NX, NY, cell

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVENTS = os.path.join(ROOT, "data", "raw", "events", "22921.json")   # France v Korea, WWC
FIG = os.path.join(ROOT, "figures")
XT = np.load(os.path.join(ROOT, "data", "processed", "xt_grid.npy")).ravel()
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; LINE = "#5e9bff"
PL, PW = 120.0, 80.0


def xt_at(x, y):
    return float(XT[cell(x, y)])


def main():
    ev = json.load(open(EVENTS, encoding="utf-8"))
    teams = []
    for e in ev:
        if e.get("team") and e["team"]["name"] not in teams:
            teams.append(e["team"]["name"])
    A, B = teams[:2]

    def is_pass(e):
        return e["type"]["name"] == "Pass"

    def completed(e):
        return is_pass(e) and not (e.get("pass", {}) or {}).get("outcome")

    # --- per-team aggregate stats ---
    stats = {}
    for tm in (A, B):
        opp = B if tm == A else A
        tp = [e for e in ev if e.get("team", {}).get("name") == tm and is_pass(e)]
        ft_team = sum(1 for e in tp if e.get("location", [0])[0] >= 80)
        # PPDA: opponent passes in their build-up 60% / our defensive actions there
        opp_passes = [e for e in ev if e.get("team", {}).get("name") == opp and is_pass(e)
                      and e.get("location", [99])[0] <= 72]
        defacts = [e for e in ev if e.get("team", {}).get("name") == tm
                   and e["type"]["name"] in ("Pressure", "Duel", "Interception", "Foul Committed")
                   and (120 - e.get("location", [0])[0]) <= 72]
        ppda = len(opp_passes) / max(len(defacts), 1)
        # xT added by passes + carries
        xt = 0.0
        for e in ev:
            if e.get("team", {}).get("name") != tm:
                continue
            loc = e.get("location"); end = None
            if e["type"]["name"] == "Pass":
                end = (e.get("pass", {}) or {}).get("end_location")
            elif e["type"]["name"] == "Carry":
                end = (e.get("carry", {}) or {}).get("end_location")
            if loc and end:
                xt += xt_at(end[0], end[1]) - xt_at(loc[0], loc[1])
        stats[tm] = {"passes": len(tp), "final_third": ft_team, "ppda": ppda, "xt": xt}

    tilt = {A: stats[A]["final_third"] / max(stats[A]["final_third"] + stats[B]["final_third"], 1),
            B: stats[B]["final_third"] / max(stats[A]["final_third"] + stats[B]["final_third"], 1)}

    # --- pass network for the higher-possession team ---
    net_team = A if stats[A]["passes"] >= stats[B]["passes"] else B
    pos, links = {}, {}
    for e in ev:
        if e.get("team", {}).get("name") != net_team or not completed(e):
            continue
        p = e.get("player", {}).get("name"); r = (e.get("pass", {}) or {}).get("recipient", {}).get("name")
        loc = e.get("location")
        if not (p and loc):
            continue
        pos.setdefault(p, []).append(loc)
        if r:
            links[(p, r)] = links.get((p, r), 0) + 1
    avg = {p: np.mean(v, 0) for p, v in pos.items() if len(v) >= 3}

    # --- render ---
    fig = plt.figure(figsize=(13.5, 6.2), dpi=150); fig.patch.set_facecolor(BG)
    axn = fig.add_axes([0.02, 0.06, 0.6, 0.84]); axs = fig.add_axes([0.66, 0.1, 0.32, 0.72])
    import homography as hg_unused  # noqa  (keep import style consistent)
    # pitch
    axn.set_facecolor("#143d2a")
    for c in ([0, 120, 0, 0], [0, 120, 80, 80], [0, 0, 0, 80], [120, 120, 0, 80], [60, 60, 0, 80]):
        axn.plot(c[:2], c[2:], color="#fff", lw=1, alpha=0.35)
    axn.add_patch(plt.Circle((60, 40), 9.15 / 105 * 120, fill=False, color="#fff", lw=1, alpha=0.35))
    mx = max(links.values()) if links else 1
    for (p, r), n in links.items():
        if p in avg and r in avg and n >= 2:
            x = [avg[p][0], avg[r][0]]; y = [avg[p][1], avg[r][1]]
            axn.plot(x, y, color=LINE, lw=0.5 + 3.5 * n / mx, alpha=0.35 + 0.5 * n / mx, zorder=2)
    involve = {p: sum(n for (a, b), n in links.items() if a == p or b == p) for p in avg}
    iv = max(involve.values()) if involve else 1
    for p, xy in avg.items():
        axn.scatter([xy[0]], [xy[1]], s=120 + 500 * involve.get(p, 0) / iv, c=LINE,
                    edgecolors="w", lw=1, zorder=4)
        axn.text(xy[0], xy[1] - 2.4, p.split()[-1], color="w", fontsize=7.5, ha="center",
                 fontfamily="Bahnschrift", zorder=5)
    axn.set_xlim(-3, 123); axn.set_ylim(-3, 83); axn.axis("off")
    axn.set_title("Pass network: {}  (node = involvement, edge = pass volume)".format(net_team),
                  color=INK, loc="left", fontfamily="Bahnschrift", fontsize=12, fontweight="bold")

    axs.axis("off"); axs.set_xlim(0, 1); axs.set_ylim(0, 1)
    rows = [("Field tilt", "{:.0%}".format(tilt[A]), "{:.0%}".format(tilt[B])),
            ("PPDA (lower = press)", "{:.1f}".format(stats[A]["ppda"]), "{:.1f}".format(stats[B]["ppda"])),
            ("xT added", "{:+.2f}".format(stats[A]["xt"]), "{:+.2f}".format(stats[B]["xt"])),
            ("Passes", str(stats[A]["passes"]), str(stats[B]["passes"]))]
    axs.text(0.45, 0.96, A.replace(" Women's", ""), color=LINE, ha="center", fontsize=9.5, fontweight="bold", fontfamily="Bahnschrift")
    axs.text(0.82, 0.96, B.replace(" Women's", ""), color="#ff7333", ha="center", fontsize=9.5, fontweight="bold", fontfamily="Bahnschrift")
    for i, (lab, va, vb) in enumerate(rows):
        y = 0.85 - i * 0.17
        axs.text(0.0, y, lab, color=MUT, fontsize=9.5, fontfamily="Bahnschrift")
        axs.text(0.45, y, va, color=INK, ha="center", fontsize=11, fontweight="bold", fontfamily="Bahnschrift")
        axs.text(0.82, y, vb, color=INK, ha="center", fontsize=11, fontweight="bold", fontfamily="Bahnschrift")
    fig.suptitle("Tier-1 event analytics  ·  {} v {}  ·  StatsBomb open data (any event feed plugs in)".format(
        A.replace(" Women's", ""), B.replace(" Women's", "")), color=INK, x=0.5, y=0.97,
        fontfamily="Bahnschrift", fontsize=13, fontweight="bold")
    out = os.path.join(FIG, "wc2026_event_analytics.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("teams:", A, "vs", B)
    for tm in (A, B):
        print("  {:<22} passes {} | field tilt {:.0%} | PPDA {:.1f} | xT added {:+.2f}".format(
            tm, stats[tm]["passes"], tilt[tm], stats[tm]["ppda"], stats[tm]["xt"]))
    print("figure:", out)


if __name__ == "__main__":
    main()
