"""Render the figures: detector fingerprint, xT surface, LinkedIn hero."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

BG = "#0d1117"
PANEL = "#131a23"
INK = "#e6edf3"
MUT = "#7d8590"
ACCENT = "#ffb347"   # break band
HOME = "#5e9bff"     # Italy
AWAY = "#ff7a1a"     # Netherlands
GREEN = "#27c98f"
GRAY = "#5c6773"

HOT = ["ISL 2021-22", "AFCON 2023", "Copa America 2024", "WWC 2019"]
CASE_ID = 69205      # Italy 0-2 Netherlands, WWC 2019 QF, 34C in Valenciennes
CASE_HOME, CASE_AWAY = 855, 851
CASE_BREAKS = [(29.86, 33.41), (74.48, 77.61)]


def font(size, bold=False):
    name = "Bahnschrift" if any("Bahnschrift" in f.name for f in font_manager.fontManager.ttflist) else "Segoe UI"
    return {"fontfamily": name, "fontsize": size,
            "fontweight": "bold" if bold else "normal"}


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8, length=0)
    ax.grid(axis="y", color="#2a3340", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def gaussian_smooth(t_grid, times, values, sigma):
    out = np.zeros_like(t_grid)
    for t, v in zip(times, values):
        out += v * np.exp(-0.5 * ((t_grid - t) / sigma) ** 2)
    return out / (sigma * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------- fingerprint
def fig_fingerprint(stop, matches):
    clean = stop[(stop.gap_sec >= 90) & (stop.period <= 2) & (stop.goal_before == 0)
                 & (stop.injury == 0) & (stop.card_near == 0) & stop.half_min.between(15, 45)]
    nhot = matches[matches.tournament.isin(HOT)].shape[0]
    bins = np.arange(15, 46)
    h_hot, _ = np.histogram(clean[clean.tournament.isin(HOT)].half_min, bins=bins)
    h_wc, _ = np.histogram(clean[clean.tournament == "WC 2022"].half_min, bins=bins)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    x = bins[:-1] + 0.5
    ax.bar(x, h_hot / nhot, width=0.85, color=GREEN, label="Hot tournaments ({} matches)".format(nhot))
    ax.bar(x, -h_wc / 64, width=0.85, color=GRAY, label="World Cup 2022, air-conditioned (64 matches)")
    ax.axvspan(25, 32, color=ACCENT, alpha=0.12, zorder=0)
    ax.text(25.4, 0.112, "detection window\nminutes 25-31", ha="left",
            color=ACCENT, **font(9, True))
    ax.axhline(0, color=MUT, lw=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "{:.2f}".format(abs(v))))
    ax.set_xlabel("minute of the half when a long dead-ball pause starts", color=MUT, **font(10))
    ax.set_ylabel("clean pauses >= 90 s, per match", color=MUT, **font(10))
    ax.set_title("The fingerprint of the cooling break",
                 color=INK, pad=26, loc="left", **font(16, True))
    ax.text(0, 1.02,
            "dead-ball pauses with no goal, injury or card context. Mirrored below: the same detector on the air-conditioned World Cup",
            color=MUT, va="bottom", transform=ax.transAxes, **font(8.5))
    ax.legend(loc="upper right", frameon=False, labelcolor=INK,
              prop={"family": font(9)["fontfamily"], "size": 9})
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fingerprint.png"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- xT surface
def fig_xt(xt):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    im = ax.imshow(xt.T, origin="lower", extent=[0, 120, 0, 80], cmap="magma",
                   interpolation="bilinear")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    lw, lc = 1.2, "#ffffff"
    for coords in ([0, 120, 0, 0], [0, 120, 80, 80], [0, 0, 0, 80], [120, 120, 0, 80],
                   [60, 60, 0, 80], [0, 18, 18, 18], [18, 18, 18, 62], [0, 18, 62, 62],
                   [102, 120, 18, 18], [102, 102, 18, 62], [102, 120, 62, 62],
                   [0, 6, 30, 30], [6, 6, 30, 50], [0, 6, 50, 50],
                   [114, 120, 30, 30], [114, 114, 30, 50], [114, 120, 50, 50]):
        ax.plot(coords[:2], coords[2:], color=lc, lw=lw, alpha=0.55)
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, color=lc, lw=lw, alpha=0.55))
    ax.plot([12, 108], [40, 40], "o", ms=2.5, color=lc, alpha=0.55, lw=0)
    ax.annotate("", xy=(78, -5), xytext=(42, -5), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2))
    ax.text(60, -9, "attacking direction", ha="center", color=MUT, **font(9))
    ax.set_ylim(-12, 80)
    ax.set_title("Expected Threat surface, learned from 931k passes and carries",
                 color=INK, pad=12, loc="left", **font(15, True))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("P(goal within possession from here)", color=MUT, **font(9))
    cb.ax.tick_params(colors=MUT, labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "xt_surface.png"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- hero
def fig_hero(res, threat, shots):
    fig = plt.figure(figsize=(7.2, 9), dpi=200)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.95, 2.35, 1.55, 0.30],
                          hspace=0.45, wspace=0.14,
                          left=0.075, right=0.94, top=0.97, bottom=0.025)

    # title block
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    axt.text(0, 0.93, "WORLD CUP 2026 IS PLAYING IN 30°C+ AFTERNOON HEAT",
             color=ACCENT, **font(10.5, True))
    axt.text(0, 0.52, "THE HIDDEN TIMEOUT", color=INK, **font(34, True))
    axt.text(0, 0.18, "Football has no timeouts. Except when it is hot: the cooling break.",
             color=INK, **font(11.5))
    axt.text(0, -0.04, "551 matches, a break detector built from event-stream gaps, an Expected Threat model trained from scratch.",
             color=MUT, **font(9.5))

    # momentum river: the externally documented validation case
    ax = fig.add_subplot(gs[1, :])
    style_ax(ax)
    ax.grid(False)
    case = threat[threat.match_id == CASE_ID]
    tg = np.arange(0, 95, 0.05)
    home_r = gaussian_smooth(tg, case[case.team_id == CASE_HOME]["min"].values,
                             case[case.team_id == CASE_HOME].val.values, 1.5)
    away_r = gaussian_smooth(tg, case[case.team_id == CASE_AWAY]["min"].values,
                             case[case.team_id == CASE_AWAY].val.values, 1.5)
    ax.fill_between(tg, 0, home_r, color=HOME, alpha=0.9, lw=0)
    ax.fill_between(tg, 0, -away_r, color=AWAY, alpha=0.9, lw=0)
    ax.plot(tg, home_r, color="#9cc1ff", lw=1.0)
    ax.plot(tg, -away_r, color="#ffae6b", lw=1.0)
    top = max(home_r.max(), away_r.max()) * 1.22
    ax.set_ylim(-top, top)
    ax.set_xlim(0, 95)
    ax.axhline(0, color=BG, lw=1.4)
    ax.axvline(45, color=MUT, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
    ax.text(45, top * 0.97, "HT", ha="center", va="top", color=MUT, **font(9, True))
    ann_at = [(34.8, 0.48), (49.0, 0.80)]
    for (bm, be), (tx, ty) in zip(CASE_BREAKS, ann_at):
        ax.axvspan(bm, be, color=ACCENT, alpha=0.32, zorder=1)
        ax.annotate("COOLING BREAK {:.0f}'".format(bm),
                    xy=((bm + be) / 2, top * (ty - 0.18)), xytext=(tx, top * ty),
                    color=ACCENT, ha="left", va="center",
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=2.5),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1), **font(9, True))
    goals = shots[(shots.match_id == CASE_ID) & (shots.goal == 1)]
    for _, g in goals.iterrows():
        gm = {1: 0, 2: 45}.get(g.period, 90) + g.t / 60.0
        side = 1 if g.team_id == CASE_HOME else -1
        r = np.interp(gm, tg, home_r if side > 0 else away_r)
        ax.plot([gm], [side * r], "o", ms=8, mfc=ACCENT, mec=BG, mew=1.5, zorder=5)
        ax.text(gm, side * (r + top * 0.12), "GOAL {:.0f}'".format(gm),
                ha="center", va="center", color=INK, zorder=6, **font(7.5, True))
    ax.text(1.5, top * 0.84, "ITALY 0", color=HOME, **font(12, True))
    ax.text(1.5, -top * 0.93, "NETHERLANDS 2", color=AWAY, **font(12, True))
    ax.text(1.0, -0.15, "threat per minute (xT + xG), smoothed", ha="right", color=MUT,
            transform=ax.transAxes, **font(8))
    ax.set_title("Women's World Cup 2019 quarter-final, 34°C: the detector finds both protocol breaks",
                 color=INK, pad=10, loc="left", **font(11, True))
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xticklabels(["0'", "15'", "30'", "45'", "60'", "75'", "90'"])
    ax.set_yticks([])

    # panel A: the trap
    axa = fig.add_subplot(gs[2, 0])
    style_ax(axa)
    naive_b = res["rates_break"]["tactic_after"]["pooled"]
    naive_c = res["rates_control"]["tactic_after"]["pooled"]
    fair_b = res["rates_break"]["tactic_after"]["standardized"]
    fair_c = res["rates_control"]["tactic_after"]["standardized"]
    vals = [naive_b, naive_c, fair_b, fair_c]
    xpos = [0, 0.42, 1.25, 1.67]
    cols = [GREEN, GRAY, GREEN, GRAY]
    axa.bar(xpos, vals, 0.36, color=cols)
    for x, v in zip(xpos, vals):
        axa.text(x, v + 0.02, "{:.0f}%".format(100 * v), ha="center",
                 color=INK, **font(9.5, True))
    axa.text(0.21, 0.68, "naive:\nall halves pooled", ha="center", color=MUT, **font(8.5))
    axa.text(1.46, 0.68, "fair:\nmatched by half", ha="center", color=MUT, **font(8.5))
    axa.text(0.21, 0.57, "looks huge", ha="center", color=GREEN, **font(9, True))
    axa.text(1.46, 0.57, "CI includes zero", ha="center", color=ACCENT, **font(9, True))
    axa.set_xticks([])
    axa.set_yticks([])
    axa.set_ylim(0, 0.80)
    axa.set_title("The trap: breaks live at minute 75", color=INK, loc="left", pad=10, **font(12, True))
    axa.text(0, -0.13, "share of windows with a sub or shape change in the next 5 min,\nbreak (green) vs no-break control (gray). Subs spike at 75'\nanyway; pool the halves and you invent a timeout effect.",
             color=MUT, transform=axa.transAxes, va="top", **font(8))

    # panel B: what survives
    axb = fig.add_subplot(gs[2, 1])
    axb.set_facecolor(PANEL)
    axb.set_xticks([]); axb.set_yticks([])
    for s in axb.spines.values():
        s.set_visible(False)
    axb.text(0.5, 0.88, "What actually changes", ha="center",
             color=INK, transform=axb.transAxes, **font(12, True))
    sb = res["rates_break"]["subs_in"]["standardized"]
    sc = res["rates_control"]["subs_in"]["standardized"]
    axb.text(0.5, 0.62, "{:.1f}x".format(sb / sc), ha="center",
             color=GREEN, transform=axb.transAxes, **font(24, True))
    axb.text(0.5, 0.47, "substitutions made during the pause itself\n({:.2f} vs {:.2f} expected, CI excludes zero)".format(sb, sc),
             ha="center", color=MUT, transform=axb.transAxes, **font(8.5))
    fb = res["rates_break"]["flip"]["standardized"]
    fc = res["rates_control"]["flip"]["standardized"]
    axb.text(0.5, 0.27, "{:.0f}% vs {:.0f}%".format(100 * fb, 100 * fc), ha="center",
             color="#8b949e", transform=axb.transAxes, **font(18, True))
    axb.text(0.5, 0.17, "momentum flips after break vs control:\nno detectable effect. The match itself does not change.",
             ha="center", va="top", color=MUT, transform=axb.transAxes, **font(8.5))

    # footer
    axf = fig.add_subplot(gs[3, :]); axf.axis("off")
    axf.text(0, 0.30, "{} breaks in ISL, AFCON, Copa America 2024, WWC 2019 (est. purity {:.0f}%) | WC 2022 noise floor: {:.2f}/match".format(
        res["n_breaks"], 100 * res["purity_break_set"], res["wc22_fp_per_match"]),
        color=MUT, **font(8))
    axf.text(0, -0.25, "match-clustered bootstrap CIs | StatsBomb open data | xT: 16x12 grid, value iteration | github.com/d8maldon/hidden-timeout",
             color=MUT, **font(8))

    fig.savefig(os.path.join(FIG, "hero.png"), facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    stop = pd.read_csv(os.path.join(ROOT, "stoppages.csv"))
    matches = pd.read_csv(os.path.join(ROOT, "matches.csv"))
    moves = pd.read_csv(os.path.join(ROOT, "moves.csv"))
    shots = pd.read_csv(os.path.join(ROOT, "shots.csv"))
    shots = shots[shots.period <= 4]
    xt = np.load(os.path.join(ROOT, "xt_grid.npy"))
    with open(os.path.join(ROOT, "results.json")) as f:
        res = json.load(f)

    NX, NY = xt.shape
    ok = moves[moves.ok == 1]
    sx = np.clip((ok.sx.values / 120 * NX).astype(int), 0, NX - 1)
    sy = np.clip((ok.sy.values / 80 * NY).astype(int), 0, NY - 1)
    ex = np.clip((ok.ex.values / 120 * NX).astype(int), 0, NX - 1)
    ey = np.clip((ok.ey.values / 80 * NY).astype(int), 0, NY - 1)
    off = ok.period.map({1: 0, 2: 45, 3: 90, 4: 105}).values
    threat = pd.DataFrame({"match_id": ok.match_id.values, "team_id": ok.team_id.values,
                           "min": off + ok.t.values / 60.0,
                           "val": np.maximum(0, xt[ex, ey] - xt[sx, sy])})
    soff = shots.period.map({1: 0, 2: 45, 3: 90, 4: 105}).values
    threat = pd.concat([threat, pd.DataFrame({
        "match_id": shots.match_id.values, "team_id": shots.team_id.values,
        "min": soff + shots.t.values / 60.0, "val": shots.xg.values})], ignore_index=True)

    fig_fingerprint(stop, matches)
    fig_xt(xt)
    fig_hero(res, threat, shots)
    print("figures written")


if __name__ == "__main__":
    main()
