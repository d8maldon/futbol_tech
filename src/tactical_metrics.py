"""Headline tactical reads from the tracked positions: the team-level structure
that survives ~5 m zone-grade noise because it is geometry, not pixels.

Per team, per frame (from the visual_ai cache of warped positions in real metres):
  compactness   convex-hull area (m^2) of the VISIBLE block -- smaller = tighter
  width         lateral spread (m) of visible players
  depth         vertical spread (m) of visible players
  block_x       VISIBLE-BLOCK centroid up the pitch -- ball-biased, NOT true
                defensive-line height (a single broadcast shows whoever is near
                the ball, so both teams' visible centroids drift toward the ball;
                true line height needs full-22 tracking we do not have)
  formation     visible IN-MOMENT shape: visible players clustered into 3 bands
                (e.g. "5-4-2") -- the shape at that moment, NOT a base formation

Honest scope (prometheus Pass 2, Malik): these are VISIBLE-BLOCK reads. Compactness
and width survive ~5 m noise and are the robust signal; block_x and formation are
ball-biased and in-moment. Per-player marking and true line height are out of scope.

    python src/tactical_metrics.py            # over the cached Argentina match
"""
import os
import pickle

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "clips", "argentina_full", "_states.pkl")
FIG = os.path.join(ROOT, "figures")
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"
ARG_C, ALG_C = "#4dc7ff", "#ff7333"


def team_shape(pts):
    if len(pts) < 4:
        return None
    pts = np.asarray(pts, float)
    cx, cy = pts.mean(0)
    width, depth = float(np.ptp(pts[:, 1])), float(np.ptp(pts[:, 0]))
    try:
        from scipy.spatial import ConvexHull
        area = float(ConvexHull(pts).volume)      # 2D hull "volume" = area
    except Exception:
        area = width * depth
    return {"cx": float(cx), "cy": float(cy), "width": width, "depth": depth, "area": area, "n": len(pts)}


def formation(pts):
    """cluster a team's outfield players into 3 along-pitch bands -> 'back-mid-front'.
    Needs a clean count (8-11); orient so the deepest band (toward own goal) is 'back'."""
    pts = np.asarray(pts, float)
    if not (8 <= len(pts) <= 11):
        return None
    cx = pts[:, 0].mean()
    x = pts[:, 0] if cx < 52.5 else (105.0 - pts[:, 0])      # 0 = own goal line
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=3, n_init=5, random_state=0).fit(x.reshape(-1, 1))
    order = np.argsort(lab.cluster_centers_.ravel())         # back -> front
    counts = [int((lab.labels_ == k).sum()) for k in order]
    return "-".join(str(c) for c in counts)


def plausible(pts):
    return 5 <= len(pts) <= 12         # reject frames where the tracker over-spawned


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = pickle.load(open(CACHE, "rb"))["states"]
    # per-frame shape for each team on wide/tracked frames
    series = {0: [], 1: []}
    idxs = []
    for i, d in enumerate(st):
        if d["note"] == "no pitch view" or len(d["tracks"]) < 6:
            continue
        idxs.append(i)
        for c in (0, 1):
            pts = [[t[0], t[1]] for t in d["tracks"] if t[2] == c]
            series[c].append(team_shape(pts) if plausible(pts) else None)

    def smooth(v, k=31):
        v = np.array([x if x is not None else np.nan for x in v], float)
        out = np.copy(v)
        for i in range(len(v)):
            seg = v[max(0, i - k): i + k]
            seg = seg[~np.isnan(seg)]
            out[i] = seg.mean() if len(seg) else np.nan
        return out

    t = np.array(idxs) / 6.0 / 60.0      # clip minutes (6 fps)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), dpi=130, sharex=True)
    fig.patch.set_facecolor(BG)
    for c, col, name in ((0, ARG_C, "Argentina"), (1, ALG_C, "Algeria")):
        area = smooth([s["area"] if s else None for s in series[c]])
        bx = smooth([s["cx"] if s else None for s in series[c]])
        ax[0].plot(t, area, color=col, lw=1.6, label=name)
        ax[1].plot(t, bx, color=col, lw=1.6, label=name)
    for a in ax:
        a.set_facecolor("#0f1620"); a.tick_params(colors=MUT)
        for s in a.spines.values():
            s.set_color("#30363d")
        a.legend(frameon=False, labelcolor=INK, prop={"family": "Bahnschrift"})
    ax[0].set_title("Visible-block compactness (convex-hull area, m^2) -- lower = tighter",
                    color=INK, loc="left", fontfamily="Bahnschrift", fontweight="bold")
    ax[1].set_title("Visible-block centroid up the pitch (m) -- ball-biased, NOT line height",
                    color=INK, loc="left", fontfamily="Bahnschrift", fontweight="bold")
    ax[1].set_xlabel("clip minute", color=MUT, fontfamily="Bahnschrift")
    fig.suptitle("WC2026 Argentina v Algeria -- live team-shape reads from broadcast tracking",
                 color=INK, x=0.5, fontfamily="Bahnschrift", fontsize=14, fontweight="bold")
    out = os.path.join(FIG, "wc2026_tactical_metrics.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)

    # formation read on a CLEAN wide frame (a plausible ~9-11 outfielders per team)
    def score(d):
        if d["note"] == "no pitch view":
            return -1
        n0 = len([t for t in d["tracks"] if t[2] == 0])
        n1 = len([t for t in d["tracks"] if t[2] == 1])
        return min(n0, n1) if (8 <= n0 <= 11 and 7 <= n1 <= 11) else -1
    best = max(st, key=score)
    for c, name in ((0, "Argentina"), (1, "Algeria")):
        pts = [[t[0], t[1]] for t in best["tracks"] if t[2] == c]
        sh = team_shape(pts)
        print("{}: {} players | formation ~{} | compactness {:.0f} m^2 | width {:.0f} m | block_x {:.0f} m".format(
            name, len(pts), formation(pts), sh["area"] if sh else -1,
            sh["width"] if sh else -1, sh["cx"] if sh else -1))
    print("figure:", out)


if __name__ == "__main__":
    main()
