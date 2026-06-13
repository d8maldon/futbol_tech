"""Pitch control from real positional data, the tracking-powered showpiece.

We cannot get live WC2026 tracking, but we do not need it to prove the
analytics layer. StatsBomb's open "360" data gives the position of every
visible player at every event, free, for Euro 2020 / WC 2022 / Euro 2024 /
WWC 2023 — the same repo we already use for events. This builds a pitch-control
surface (which team controls each patch of grass) from one freeze frame, the
exact visual that real tracking unlocks. When the minimap extractor comes
online it feeds this same code; the data source is the only thing that swaps.

    python src/pitch_control.py            # top-xG moment of the case match
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import urllib3
from matplotlib.path import Path as MplPath

urllib3.disable_warnings()

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
FIG = os.path.join(ROOT, "figures")
SB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
MATCH = 3794689   # Wales 0-2 Denmark, Euro 2020 R16 (our case study; has 360)

PL, PW = 120.0, 80.0   # StatsBomb pitch units
TAU = 6.0              # control falloff (m); larger = softer ownership

BG = "#0d1117"
INK = "#e6edf3"
MUT = "#7d8590"
ATT = "#5e9bff"   # team in possession
DEF = "#ff7a1a"   # defending team


def fetch(url, path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(os.path.join(RAW, "three-sixty"), exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    events = fetch("{}/events/{}.json".format(SB, MATCH),
                   os.path.join(RAW, "events", "{}.json".format(MATCH)))
    ff = fetch("{}/three-sixty/{}.json".format(SB, MATCH),
               os.path.join(RAW, "three-sixty", "{}.json".format(MATCH)))
    by_uuid = {f["event_uuid"]: f for f in ff}

    # the highest-xG shot that also has a freeze frame: a real decisive moment
    shots = [e for e in events if e["type"]["name"] == "Shot" and e.get("location")
             and e["id"] in by_uuid]
    shots.sort(key=lambda e: e["shot"].get("statsbomb_xg", 0), reverse=True)
    ev = shots[0]
    frame = by_uuid[ev["id"]]
    players = frame["freeze_frame"]
    ball = ev["location"]
    team = ev["team"]["name"]
    xg = ev["shot"].get("statsbomb_xg", 0)
    minute = ev["minute"] + ev["second"] / 60.0

    att = np.array([p["location"] for p in players if p["teammate"]], float)
    deff = np.array([p["location"] for p in players if not p["teammate"]], float)

    # pitch control: softmax of nearest-player proximity per team over a grid
    gx, gy = np.meshgrid(np.linspace(0, PL, 240), np.linspace(0, PW, 160))
    cells = np.stack([gx.ravel(), gy.ravel()], 1)

    def influence(pts):
        if len(pts) == 0:
            return np.zeros(len(cells))
        d = np.linalg.norm(cells[:, None, :] - pts[None, :, :], axis=2)
        return np.exp(-d.min(1) / TAU)

    ia, idf = influence(att), influence(deff)
    control = np.where(ia + idf > 0, ia / (ia + idf + 1e-9), 0.5).reshape(gx.shape)

    # only colour where the broadcast camera could actually see (honest)
    va = np.array(frame["visible_area"], float).reshape(-1, 2)
    inside = MplPath(va).contains_points(cells).reshape(gx.shape)
    control = np.ma.masked_where(~inside, control)

    fig, ax = plt.subplots(figsize=(10, 6.8), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#16341f")
    ax.imshow(control, origin="lower", extent=[0, PL, 0, PW], cmap="coolwarm_r",
              vmin=0, vmax=1, alpha=0.78, aspect="equal", zorder=1)
    ax.fill(va[:, 0], va[:, 1], facecolor="none", edgecolor="#ffffff",
            lw=1.0, ls=(0, (4, 3)), alpha=0.35, zorder=2)

    lc = dict(color="#ffffff", lw=1.3, alpha=0.55, zorder=3)
    ax.plot([0, 0, PL, PL, 0], [0, PW, PW, 0, 0], **lc)
    ax.plot([PL / 2, PL / 2], [0, PW], **lc)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(PL / 2 + 10 * np.cos(th), PW / 2 + 10 * np.sin(th), **lc)
    for x0 in (0, PL - 18):
        ax.plot([x0, x0 + 18, x0 + 18, x0], [18, 18, 62, 62], **lc)

    ax.scatter(att[:, 0], att[:, 1], s=170, c=ATT, edgecolors=BG, linewidths=1.5,
               zorder=5, label="{} (in possession)".format(team))
    ax.scatter(deff[:, 0], deff[:, 1], s=170, c=DEF, edgecolors=BG, linewidths=1.5,
               zorder=5, label="defending")
    ax.scatter([ball[0]], [ball[1]], s=70, c="#ffd23f", edgecolors=BG,
               linewidths=1.2, marker="o", zorder=6, label="ball")

    ax.set_xlim(-3, PL + 3); ax.set_ylim(-3, PW + 3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.annotate("", xy=(PL * 0.62, -1.5), xytext=(PL * 0.38, -1.5),
                annotation_clip=False, arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2))
    ax.text(PL / 2, -4.2, "{} attacking direction".format(team), ha="center",
            color=MUT, fontfamily="Bahnschrift", fontsize=9)
    ax.set_title("Pitch control from real positions  |  {} chance, {:.0f}'  (xG {:.2f})".format(
        team, minute, xg), color=INK, loc="left", pad=24,
        fontfamily="Bahnschrift", fontsize=14, fontweight="bold")
    ax.text(0, 1.015, "blue = space controlled by {}, orange = defenders; shaded only where the broadcast camera could see".format(team),
            transform=ax.transAxes, color=MUT, va="bottom",
            fontfamily="Bahnschrift", fontsize=9)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK,
              prop={"family": "Bahnschrift", "size": 9})
    fig.text(0.5, 0.01, "data: StatsBomb open 360 (free positional data) | {} visible players | github.com/d8maldon/hidden-timeout".format(len(players)),
             ha="center", color=MUT, fontfamily="Bahnschrift", fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(os.path.join(FIG, "pitch_control.png"), facecolor=BG)
    plt.close(fig)
    print("freeze frames in match: {}".format(len(ff)))
    print("chosen: {} shot at {:.1f}' xG {:.2f}, {} players visible".format(
        team, minute, xg, len(players)))
    print("figure: figures/pitch_control.png")


if __name__ == "__main__":
    main()
