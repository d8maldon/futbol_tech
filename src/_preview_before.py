"""Render the CURRENT dashboard styling as a single static frame, from the preview
harness, so it can sit next to the redesign at the same minute and data.
Faithful port of visual_ai.render().draw() (lowercase titles + jargon kept on
purpose -- this is the 'before')."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

import _preview_data as P

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "_frames_review", "preview_before.png")
BG = "#0d1117"; INK = "#e6edf3"; MUT = "#7d8590"; BALLC = "#ffd23f"; GREEN = "#3fb950"
ARG_C, ALG_C = P.CYAN, P.ORANGE
PL, PW = P.PL, P.PW
F = {"fontfamily": "Bahnschrift"}

_GX, _GY = np.meshgrid(np.linspace(0, PL, 92), np.linspace(0, PW, 62))
_CELLS = np.stack([_GX.ravel(), _GY.ravel()], 1)


def control_surface(tracks, TAU=6.0, SEE=26.0):
    if len(tracks) < 2:
        return None
    Pp = np.array([[t[0], t[1]] for t in tracks]); tm = np.array([t[2] for t in tracks])
    d = np.linalg.norm(_CELLS[:, None, :] - Pp[None, :, :], axis=2)
    infl = lambda sel: np.exp(-d[:, sel].min(1) / TAU) if sel.any() else np.zeros(len(_CELLS))
    ia, idf = infl(tm == 0), infl(tm == 1)
    ctrl = np.where(ia + idf > 0, ia / (ia + idf + 1e-9), 0.5)
    return ctrl.reshape(_GX.shape), (d.min(1) < SEE).reshape(_GX.shape)


def main():
    m = P.build_m(); d = P.build_state(); team_rgb = P.TEAM_RGB; ti = P.PREVIEW_MIN
    mins = m["wp_mins"]
    cmap = LinearSegmentedColormap.from_list("ctrl", [team_rgb[1], (0.93, 0.93, 0.93), team_rgb[0]])
    fig = plt.figure(figsize=(16, 9), dpi=84); fig.patch.set_facecolor(BG)
    axw = fig.add_axes([0.02, 0.875, 0.96, 0.052])
    axb = fig.add_axes([0.02, 0.40, 0.455, 0.45])
    axp = fig.add_axes([0.49, 0.40, 0.265, 0.45])
    axc = fig.add_axes([0.775, 0.40, 0.205, 0.45])
    axg = fig.add_axes([0.05, 0.055, 0.32, 0.27])
    axe = fig.add_axes([0.41, 0.055, 0.23, 0.295])
    axr = fig.add_axes([0.665, 0.055, 0.315, 0.295])

    # win-prob bar
    axw.set_xlim(0, 1); axw.set_ylim(0, 1); axw.axis("off")
    ph, pdr, pa = float(m["wp_home"][ti]), float(m["wp_draw"][ti]), float(m["wp_away"][ti])
    axw.add_patch(plt.Rectangle((0, 0), ph, 1, color=ARG_C))
    axw.add_patch(plt.Rectangle((ph, 0), pdr, 1, color=(0.42, 0.42, 0.46)))
    axw.add_patch(plt.Rectangle((ph + pdr, 0), pa, 1, color=ALG_C))
    axw.text(0.008, 0.5, "ARGENTINA  {:.0%}".format(ph), va="center", ha="left", color=BG, fontsize=12, fontweight="bold", **F)
    axw.text(0.992, 0.5, "{:.0%}  ALGERIA".format(pa), va="center", ha="right", color="white", fontsize=12, fontweight="bold", **F)
    xph = float(m["wp_xg"][ti])
    axw.add_patch(plt.Rectangle((xph - 0.0015, -0.05), 0.003, 1.1, color="white", zorder=6, clip_on=False))
    axw.set_title("LIVE WIN PROBABILITY  ·  bar = score-based (OOS 0.82);  white tick = xG-deserved (ARG {:.0%})".format(xph),
                  color=MUT, loc="center", fontsize=9, **F)

    # broadcast
    axb.imshow(cv2.cvtColor(cv2.imread(d["fp"]), cv2.COLOR_BGR2RGB))
    w, h = d["wh"]; axb.set_xlim(0, w); axb.set_ylim(h, 0); axb.axis("off")
    axb.set_title("broadcast — players detected & teamed", color=INK, loc="left", fontsize=10.5, fontweight="bold", **F)

    # top-down shapes
    P.draw_pitch(axp); tr = d["tracks"]
    trd = [[t[0], PW - t[1], t[2], t[3], t[5], t[6], -t[7]] for t in tr]
    bd = [d["ball"][0], PW - d["ball"][1]]
    for c in range(2):
        grp = [t for t in trd if t[2] == c]
        for t in grp:
            axp.add_patch(Ellipse((t[0], t[1]), 2 * t[4], 2 * t[5], angle=t[6], facecolor=team_rgb[c], edgecolor="none", alpha=0.16 * t[3], zorder=4))
            axp.scatter([t[0]], [t[1]], s=110, facecolor=team_rgb[c], edgecolors=BG, lw=1.1, alpha=t[3], zorder=5)
        if len(grp) >= 3:
            P.hull(axp, np.array([[t[0], t[1]] for t in grp]), team_rgb[c])
    axp.scatter([bd[0]], [bd[1]], s=60, c=BALLC, edgecolors=BG, lw=1, zorder=7)
    sa = P.team_shape([[t[0], t[1]] for t in trd if t[2] == 0]); sb = P.team_shape([[t[0], t[1]] for t in trd if t[2] == 1])
    if sa and sb:
        axp.text(PL / 2, PW - 1.5, "compactness  ARG {:.0f}  ·  ALG {:.0f} m2".format(sa["area"], sb["area"]), ha="center", va="top", color=MUT, fontsize=7.5, **F)
    axp.set_title("top-down — shapes + 1-sigma ellipses  ·  conf {:.2f}".format(d.get("conf", 0.0)), color=INK, loc="left", fontsize=10.5, fontweight="bold", **F)

    # pitch control
    P.draw_pitch(axc)
    ctrl, vis = control_surface(trd)
    axc.imshow(np.ma.masked_where(~vis, ctrl), origin="lower", extent=[0, PL, 0, PW], cmap=cmap, vmin=0, vmax=1, alpha=0.62, aspect="equal", zorder=1.5)
    for t in trd:
        axc.scatter([t[0]], [t[1]], s=28, facecolor=team_rgb[int(t[2])], edgecolors=BG, lw=0.5, alpha=0.9 * t[3], zorder=4)
    axc.set_title("live pitch control", color=INK, loc="left", fontsize=10.5, fontweight="bold", **F)

    # xG race
    axg.set_facecolor("#0f1620")
    axg.plot(mins[:ti + 1], m["xg_h"][:ti + 1], color=ARG_C, lw=2.0)
    axg.plot(mins[:ti + 1], m["xg_a"][:ti + 1], color=ALG_C, lw=2.0)
    for s in m["shots"]:
        if s["min"] <= ti:
            cy = (m["xg_h"] if s["is_home"] else m["xg_a"])[min(s["min"], 98)]
            axg.scatter([s["min"]], [cy], s=80 if s["goal"] else 20, marker="*" if s["goal"] else "o", color=ARG_C if s["is_home"] else ALG_C, edgecolors=BG, lw=0.5, zorder=5)
    axg.axvline(ti, color=MUT, lw=0.8, ls=(0, (3, 3)))
    axg.set_xlim(0, 95); axg.set_ylim(0, max(1.5, float(m["xg_h"][-1]) + 0.25))
    axg.set_title("xG race  ·  ARG {:.2f}   ALG {:.2f}".format(float(m["xg_h"][ti]), float(m["xg_a"][ti])), color=INK, loc="left", fontsize=10, fontweight="bold", **F)
    axg.tick_params(colors=MUT, labelsize=7)
    for sp in axg.spines.values():
        sp.set_color("#30363d")

    # event ticker
    axe.set_xlim(0, 1); axe.set_ylim(0, 1); axe.axis("off")
    axe.text(0, 0.98, "MATCH EVENTS", color=INK, fontsize=10, fontweight="bold", va="top", **F)
    past = [e for e in m["events"] if e["min"] <= ti and not (e["type"] == "Substitution" and not e["player"])][-7:]
    for k, e in enumerate(reversed(past)):
        y = 0.85 - k * 0.118
        tag = {"Goal": "GOAL", "Card": "CARD", "Substitution": "SUB", "VAR": "VAR"}.get(e["type"], e["type"])
        latest = e is past[-1]
        extra = "  {}-{}".format(*e["score"]) if (e["type"] == "Goal" and e.get("score")) else (" ({})".format(e["note"]) if (e["type"] == "VAR" and e.get("note")) else "")
        col = "#ffb347" if e["type"] == "VAR" else (ARG_C if e["is_home"] else ALG_C)
        axe.text(0.0, y, "{}'".format(e["min"]), color=MUT, fontsize=9, va="center", **F)
        axe.text(0.13, y, "{} {}{}".format(tag, e["player"], extra), color=col if (latest or e["type"] == "VAR") else INK, fontsize=9.5 if latest else 8.5, fontweight="bold" if latest else "normal", va="center", **F)

    # ratings + prediction
    axr.set_xlim(0, 1); axr.set_ylim(0, 1); axr.axis("off")
    pm = m["pre_match"]
    axr.text(0, 0.98, "PRE-MATCH CALL  ·  our Elo model", color=INK, fontsize=10, fontweight="bold", va="top", **F)
    axr.text(0, 0.89, "ARG {:.0%}    draw {:.0%}    ALG {:.0%}".format(pm["p_h"], pm["p_d"], pm["p_a"]), color=MUT, fontsize=9, va="top", **F)
    axr.text(0, 0.80, "our call: ARGENTINA  ·  final {}-{}  (correct)".format(m["final_h"], m["final_a"]), color=GREEN, fontsize=9.5, fontweight="bold", va="top", **F)
    axr.text(0, 0.66, "TOP PLAYER RATINGS  ·  FotMob", color=INK, fontsize=10, fontweight="bold", va="top", **F)
    for k, r in enumerate(m["ratings"][:5]):
        y = 0.555 - k * 0.107; col = ARG_C if r["is_home"] else ALG_C
        axr.add_patch(plt.Rectangle((0.46, y - 0.03), 0.52 * r["rating"] / 10.0, 0.052, color=col, alpha=0.45))
        axr.text(0.0, y, (r["name"][:18] + ("  POTM" if r["potm"] else "")), color=(1.0, 0.84, 0.2) if r["potm"] else col, fontsize=9, va="center", **F)
        axr.text(0.985, y, "{:.2f}".format(r["rating"]), color=INK, fontsize=9.5, ha="right", va="center", fontweight="bold", **F)

    # header
    sch, sca = int(m["sc_h"][ti]), int(m["sc_a"][ti])
    fig.suptitle("visual-AI dashboard    ·    {} {}-{} {}    ·    ~{}'".format(m["home"], sch, sca, m["away"], ti), color=INK, x=0.5, y=0.978, fontsize=15, fontweight="bold", **F)

    fig.savefig(OUT, facecolor=BG); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
