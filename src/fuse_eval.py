"""Can we infer the players the camera does NOT show? Measured in meters.

A ball-following broadcast shows only part of the pitch, so most off-ball
players are off-screen. This is the honest test of "filling in" those players:
take a StatsBomb 360 freeze frame (real positions of the visible players),
artificially HIDE the players farthest from the ball -- simulating a tighter,
ball-chasing camera -- then try to put them back from what is still visible, and
score the error in METERS against their true positions.

Two imputers are compared so the value of the prior is explicit:
  - centroid  : drop every hidden player at their team's visible centroid (dumb)
  - formation : place them on a formation template anchored to the visible team's
                extent, the ball, and the direction of play (the prior)

Honest scope: 360 only includes players an operator could see (~70% of 22), so
the hidden set here is curated-visible, not the truly-never-on-camera players a
live phone-at-TV loses -- so these meters are an OPTIMISTIC floor on the real
off-screen error. Inferred positions are guesses with uncertainty, not
measurements. (One match cached today; this is the harness, not a trained model.)

    python src/fuse_eval.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import urllib3
from scipy.optimize import linear_sum_assignment

urllib3.disable_warnings()

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
FIG = os.path.join(ROOT, "figures")
SB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
MATCH = 3794689
PL, PW = 120.0, 80.0            # canonical StatsBomb frame (matches pitch_control)
HIDE = 3                        # players hidden per team (simulate a tight shot)
MIN_VIS = 6                     # need this many visible to infer shape

BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"
MEAS = "#5e9bff"; TRUE = "#3fb950"; INF = "#ff7a1a"

# generic 10-outfield formation template in normalised (depth, width), depth 0 =
# own goal line, 1 = opponent goal line; width 0..1 across the pitch (a 4-3-3)
TEMPLATE = np.array([
    [0.10, 0.20], [0.10, 0.40], [0.10, 0.60], [0.10, 0.80],   # back 4
    [0.40, 0.30], [0.40, 0.50], [0.40, 0.70],                 # mid 3
    [0.70, 0.25], [0.72, 0.50], [0.70, 0.75],                 # front 3
])


def fetch(url, path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return json.load(open(path, encoding="utf-8"))


def template_slots(team_pts, ball, attack_right):
    """map the formation template into pitch metres for this team's footprint"""
    xs, ys = team_pts[:, 0], team_pts[:, 1]
    # depth spans from the team's own visible rear to a bit past the ball
    if attack_right:
        x0, x1 = min(xs.min(), 5.0), max(ball[0] + 10.0, xs.max())
    else:
        x1, x0 = max(xs.max(), PL - 5.0), min(ball[0] - 10.0, xs.min())
    y0, y1 = 5.0, PW - 5.0
    slots = np.empty_like(TEMPLATE)
    slots[:, 0] = x0 + TEMPLATE[:, 0] * (x1 - x0)
    slots[:, 1] = y0 + TEMPLATE[:, 1] * (y1 - y0)
    return slots


def matched_error(pred, truth):
    """min-cost assignment mean Euclidean (m); we score the SET, not identities"""
    if len(pred) == 0 or len(truth) == 0:
        return None
    d = np.linalg.norm(pred[:, None, :] - truth[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(d)
    return d[ri, ci]


def eval_frame(team_pts, ball, attack_right):
    """hide the HIDE players farthest from the ball, impute, return per-imputer
    arrays of per-player metre errors (and the bits needed to draw one frame)"""
    if len(team_pts) < MIN_VIS + 1:
        return None
    far = np.argsort(-np.linalg.norm(team_pts - ball, axis=1))[:HIDE]
    mask = np.ones(len(team_pts), bool); mask[far] = False
    visible, hidden = team_pts[mask], team_pts[~mask]

    cen = np.repeat(visible.mean(0)[None, :], len(hidden), 0)
    slots = template_slots(visible, ball, attack_right)
    # pick template slots farthest from any visible player (those are unseen)
    d_to_vis = np.linalg.norm(slots[:, None, :] - visible[None, :, :], axis=2).min(1)
    prior = slots[np.argsort(-d_to_vis)[:len(hidden)]]
    return {"visible": visible, "hidden": hidden, "cen": cen, "prior": prior,
            "e_cen": matched_error(cen, hidden), "e_prior": matched_error(prior, hidden)}


def render(frame, out):
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor("#16341f")
    ax.plot([0, 0, PL, PL, 0], [0, PW, PW, 0, 0], color="#fff", lw=1.2, alpha=0.5)
    ax.plot([PL / 2, PL / 2], [0, PW], color="#fff", lw=1, alpha=0.4)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(PL / 2 + 10 * np.cos(th), PW / 2 + 10 * np.sin(th), color="#fff", lw=1, alpha=0.4)
    for x0 in (0, PL - 18):
        ax.plot([x0, x0 + 18, x0 + 18, x0], [18, 18, 62, 62], color="#fff", lw=1, alpha=0.4)
    v, h, pr = frame["visible"], frame["hidden"], frame["prior"]
    ax.scatter(v[:, 0], v[:, 1], s=130, c=MEAS, edgecolors=BG, lw=1.3, zorder=5,
               label="measured (camera sees)")
    ax.scatter(h[:, 0], h[:, 1], s=160, facecolors="none", edgecolors=TRUE, lw=2.0,
               zorder=4, label="hidden TRUE position")
    ax.scatter(pr[:, 0], pr[:, 1], s=150, c=INF, edgecolors=BG, lw=1.0, alpha=0.55,
               marker="X", zorder=4, label="INFERRED (formation prior)")
    for p, t in zip(pr, _match_pairs(pr, h)):
        ax.plot([p[0], t[0]], [p[1], t[1]], color=INF, lw=1.0, ls=(0, (2, 2)), alpha=0.6, zorder=3)
    ax.set_xlim(-3, PL + 3); ax.set_ylim(-3, PW + 3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper center", ncol=3, frameon=False, labelcolor=INK,
              bbox_to_anchor=(0.5, 1.08), prop={"family": "Bahnschrift", "size": 9})
    ax.set_title("Inferring the players the camera hid (one team, one moment)",
                 color=INK, loc="left", pad=30, fontfamily="Bahnschrift",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "dashed = inferred-to-true error | StatsBomb 360 | inferred positions are guesses, not measurements | github.com/d8maldon/futbol_tech",
             ha="center", color=MUT, fontsize=7.5, fontfamily="Bahnschrift")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, facecolor=BG); plt.close(fig)


def _match_pairs(pred, truth):
    d = np.linalg.norm(pred[:, None, :] - truth[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(d)
    order = np.empty(len(pred), int); order[ri] = ci
    return truth[order]


def main():
    os.makedirs(FIG, exist_ok=True)
    events = fetch("{}/events/{}.json".format(SB, MATCH),
                   os.path.join(RAW, "events", "{}.json".format(MATCH)))
    ff = fetch("{}/three-sixty/{}.json".format(SB, MATCH),
               os.path.join(RAW, "three-sixty", "{}.json".format(MATCH)))
    by_uuid = {f["event_uuid"]: f for f in ff if f.get("event_uuid")}

    e_cen, e_pr = [], []
    best = None
    for ev in events:
        f = by_uuid.get(ev.get("id"))
        if not f or not ev.get("location"):
            continue
        ball = np.array(ev["location"], float)
        players = f["freeze_frame"]
        for side in (True, False):
            pts = np.array([p["location"] for p in players if p["teammate"] == side], float)
            if len(pts) < MIN_VIS + 1:
                continue
            # possessing team (teammate=True) attacks toward x=120
            res = eval_frame(pts, ball, attack_right=side)
            if res is None:
                continue
            e_cen += list(res["e_cen"]); e_pr += list(res["e_prior"])
            score = res["e_prior"].mean()
            if best is None or (len(res["hidden"]) >= 3 and 6 < score < best[0]):
                best = (score, res)

    e_cen, e_pr = np.array(e_cen), np.array(e_pr)
    print("hidden players scored: {} (over {} team-frames)".format(
        len(e_pr), len(e_pr) // HIDE))
    print("mean metres error  centroid {:.1f} m   formation-prior {:.1f} m".format(
        e_cen.mean(), e_pr.mean()))
    for d in (4, 8, 12):
        print("  within {:>2} m:  centroid {:.0%}   prior {:.0%}".format(
            d, (e_cen <= d).mean(), (e_pr <= d).mean()))
    if best:
        render(best[1], os.path.join(FIG, "wc2026_fuse_eval.png"))
        print("figure: figures/wc2026_fuse_eval.png  (example frame, prior err {:.1f} m)".format(best[0]))


if __name__ == "__main__":
    main()
