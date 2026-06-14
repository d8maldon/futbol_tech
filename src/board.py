"""Per-match analysis board: one dossier fusing event analytics + CV snapshot.

Honest framing (this is the whole point): the ROBUST layer is the event
analytics that run on every played match -- the win-probability eval curve, the
pre-match Elo odds vs the actual result, and the goal/card/sub replay, all from
StatsBomb/FotMob/ESPN feeds. The CV layer is a best-frame tactical SNAPSHOT from
edited 480p highlights -- players detected and coloured by team, warped to a
top-down map -- and it measures only what the camera shows (never all 22, no
continuous tracking, no real formation label). Titled accordingly: tactical
snapshots + event analytics, not "full match analysis".

    python src/board.py            # a board for every played match
    python src/board.py 4667751    # one FotMob match id
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ratings as rt
from broadcast_track import detect
from live_eval import (AWAY_C, BG, GRID, HOME_C, INK, MUT, PANEL, build_timeline,
                       fetch, font, list_matches, win_prob_curve)
from replay import build_replay
import homography as hg
from tactical import assign_teams

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
CLIP = os.path.join(ROOT, "data", "clips")
PL, PW = 120.0, 80.0
ACCENT = "#ffb347"
import json as _json


def slugify(tl):
    return "".join(c if c.isalnum() else "_" for c in
                   "{}_{}".format(tl["home"], tl["away"]).lower()).strip("_")


def frames_for(slug):
    for cand in (os.path.join(CLIP, slug), os.path.join(CLIP, "frames")):
        fs = sorted(glob.glob(os.path.join(cand, "*.png")))
        if fs:
            return fs
    return []


def best_frame(frames, tries=14):
    """a wide frame where the homography succeeds and most players are on-pitch"""
    best, bestn = None, 0
    step = max(len(frames) // tries, 1)
    for fp in frames[::step]:
        H, _, _ = hg.keypoint_homography(fp)
        if H is None:
            continue
        players, _, _ = detect(fp)
        foot = np.array([[fx, fy] for fx, fy, *_ in players], float)
        if len(foot) == 0:
            continue
        top = hg.warp(H, foot)
        n = int(((top[:, 0] >= 0) & (top[:, 0] <= PL) & (top[:, 1] >= 0) & (top[:, 1] <= PW)).sum())
        if n > bestn:
            best, bestn = (fp, H, players), n
    return best


def panel_eval(ax, model, tl):
    grid, edge, ph, pa = win_prob_curve(model, tl)
    ax.set_facecolor(PANEL)
    ax.fill_between(grid, 0, edge, where=edge >= 0, color=HOME_C, alpha=0.9, interpolate=True, lw=0)
    ax.fill_between(grid, 0, edge, where=edge <= 0, color=AWAY_C, alpha=0.9, interpolate=True, lw=0)
    ax.plot(grid, edge, color=INK, lw=0.8, alpha=0.6)
    ax.axhline(0, color=INK, lw=1.0); ax.axvline(45.5, color=MUT, lw=0.7, ls=(0, (4, 3)), alpha=0.6)
    for g in tl["goals"]:
        e = np.interp(g["m"], grid, edge)
        ax.plot(g["m"], e, "o", ms=7, mfc=ACCENT, mec=BG, mew=1.2, zorder=6)
        ax.annotate("GOAL {:.0f}'".format(g["m"]), (g["m"], e), xytext=(0, 11 if e >= 0 else -11),
                    textcoords="offset points", ha="center", va="bottom" if e >= 0 else "top",
                    color=INK, **font(7, True))
    for c in tl["cards"]:
        ax.axvline(c["m"], color="#e5484d", lw=1.2, alpha=0.7)
    ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, 96)
    ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["away", "even", "home"], color=MUT, fontsize=7)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]); ax.tick_params(colors=MUT, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Win-probability eval -- who the match favoured, minute by minute",
                 color=INK, loc="left", **font(11, True))


def panel_odds(ax, tl, seed, model, norm):
    h, a = norm(tl["home"]), norm(tl["away"])
    dr = seed.get(h, rt.PROVISIONAL) - seed.get(a, rt.PROVISIONAL)
    dr += rt.HOME_ADV if h in rt.HOSTS_2026 else 0.0
    p = rt.prematch_proba(dr, model)
    ax.set_facecolor(BG)
    bars = ax.bar([0, 1, 2], [p["H"], p["D"], p["A"]], color=[HOME_C, MUT, AWAY_C], width=0.6)
    for x, k in zip([0, 1, 2], ["H", "D", "A"]):
        ax.text(x, p[k] + 0.02, "{:.0%}".format(p[k]), ha="center", color=INK, **font(8))
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([tl["home"][:9], "draw", tl["away"][:9]], color=MUT, fontsize=7.5)
    ax.set_ylim(0, max(p.values()) * 1.3); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUT)
    try:
        hs, as_ = (int(x) for x in tl["score"].split("-"))
        res = "H" if hs > as_ else ("A" if as_ > hs else "D")
        pick = max(p, key=p.get)
        verdict = "called it" if pick == res else "upset"
    except Exception:
        verdict = ""
    ax.set_title("Pre-match odds (Elo) -> {}  {}".format(tl["score"], verdict),
                 color=INK, loc="left", **font(10, True))


def panel_replay(ax, rl):
    ax.set_facecolor(BG); ax.axis("off")
    evs = [e for e in rl["events"] if e["kind"] in ("goal", "red", "yellow", "sub")]
    evs = evs[:16]
    ax.set_ylim(-len(evs) - 0.5, 0.5); ax.set_xlim(0, 1)
    sym = {"goal": ("GOAL", ACCENT), "red": ("RED", "#e5484d"),
           "yellow": ("YEL", "#f2cc0c"), "sub": ("SUB", "#3fb950")}
    for i, e in enumerate(evs):
        s, col = sym[e["kind"]]
        ax.text(0.0, -i, e["label"], color=MUT, **font(7))
        ax.text(0.13, -i, s, color=col, **font(7, True))
        ax.text(0.30, -i, "{}  {}".format(e["text"], e["note"])[:46], color=INK, **font(7.5))
    ax.set_title("Replay: goals, cards, substitutions", color=INK, loc="left", **font(10, True))


def panel_cv(axf, axt, picked):
    import cv2
    fp, H, players = picked
    img = cv2.imread(fp)
    labels, team_bgr = assign_teams(img, players)
    team_rgb = [(b[2] / 255, b[1] / 255, b[0] / 255) for b in team_bgr]
    axf.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for (fx, fy, box, _), lab in zip(players, labels):
        col = team_rgb[lab] if lab >= 0 else (0.6, 0.6, 0.6)
        x1, y1, x2, y2 = box
        axf.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.6))
    axf.axis("off")
    axf.set_title("Tactical snapshot: players by kit colour (best frame)", color=INK, loc="left", **font(10, True))
    hg.draw_pitch(axt)
    foot = np.array([[fx, fy] for fx, fy, *_ in players], np.float32)
    top = hg.warp(H, foot)
    for (x, y), lab in zip(top, labels):
        if 0 <= x <= PL and 0 <= y <= PW:
            axt.scatter([x], [y], s=110, c=[team_rgb[lab] if lab >= 0 else (0.6, 0.6, 0.6)],
                        edgecolors=BG, lw=1.2, zorder=5)
    axt.set_title("top-down (visible players only -- not all 22)", color=INK, loc="left", **font(10, True))


def build_board(mid, eval_model, pre_model, seed, norm):
    tl = build_timeline(mid)
    slug = slugify(tl)
    frames = frames_for(slug)
    picked = best_frame(frames) if frames else None

    fig = plt.figure(figsize=(13, 16 if picked else 11), dpi=150)
    fig.patch.set_facecolor(BG)
    nrows = 5 if picked else 3
    hr = [0.6, 2.2, 1.8, 2.6, 2.6] if picked else [0.6, 2.2, 2.4]
    gs = fig.add_gridspec(nrows, 2, height_ratios=hr, hspace=0.5, wspace=0.16)

    ax_b = fig.add_subplot(gs[0, :]); ax_b.axis("off"); ax_b.set_facecolor(BG)
    heat = "  |  {}C".format(tl["temp_c"]) if tl.get("temp_c") is not None else ""
    ax_b.text(0, 0.6, "{}  {}  {}".format(tl["home"], tl["score"], tl["away"]),
              color=INK, **font(20, True))
    ax_b.text(0, -0.2, "{}{}  |  WC2026 tactical snapshot + event analytics".format(tl["date"], heat),
              color=MUT, **font(10))

    panel_eval(fig.add_subplot(gs[1, :]), eval_model, tl)
    panel_odds(fig.add_subplot(gs[2, 0]), tl, seed, pre_model, norm)
    panel_replay(fig.add_subplot(gs[2, 1]), build_replay(mid))
    if picked:
        panel_cv(fig.add_subplot(gs[3, :]), fig.add_subplot(gs[4, 0]), picked)

    fig.text(0.5, 0.005, "analytics: StatsBomb/FotMob/ESPN (all matches) | CV: best-frame snapshot from edited 480p highlights, visible players only | github.com/d8maldon/hidden-timeout",
             ha="center", color=MUT, **font(7.5))
    out = os.path.join(FIG, "board_{}.png".format(slug))
    fig.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out, slug, bool(picked)


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(ROOT, "wc2026", "winprob_model.json")) as f:
        model = _json.load(f)
    import pandas as pd
    matches = pd.read_csv(os.path.join(ROOT, "data", "processed", "matches.csv"))
    seed, _ = rt.seed_ratings(matches)
    norm = rt.build_normalizer(seed)
    pre = rt.load_prematch_model()

    ids = sys.argv[1:] or list_matches()
    print("building {} board(s)".format(len(ids)))
    for mid in ids:
        try:
            out, slug, cv = build_board(mid, model, pre, seed, norm)
            print("  {}  (CV panel: {})  -> {}".format(slug, "yes" if cv else "no", os.path.basename(out)))
        except Exception as e:
            print("  match {} failed: {}".format(mid, str(e)[:80]))


if __name__ == "__main__":
    main()
