"""Classify stoppages, then measure what changes around drinks/cooling breaks.

Design notes, learned the hard way:
- Breaks skew to second halves and hot tournaments; controls skew the other
  way. Substitutions spike around minute 70-75 regardless of breaks, so every
  comparison is reported per period and the control rate is also standardized
  to the break period mix. Pooled naive numbers are kept only to show the trap.
- Both arms are measured identically: tactical actions during the (pseudo)
  pause interval and in the 5 minutes after it, both counted from admin.csv.
- Only hot-core tournaments enter the break set. Detections elsewhere are
  statistically indistinguishable from the WC 2022 false-positive floor.
- Bootstrap resamples matches, not windows.

AUDIT-DRAFT (opt-in --strict mode; validate on the data box):
  The headline numbers below are produced by the DEFAULT (non-strict) path and
  are unchanged. Passing strict=True (CLI: --strict) turns on four audit fixes
  so the maintainer can see how each headline moves:
    #15  subs ~1.8x is partly circular: substitution stoppages are themselves
         classified AS drinks_break candidates upstream of the threat analysis.
         strict drops break rows whose pause is substitution-caused before the
         break/control comparison.
    #16  penalties are excluded from the xT model but INCLUDED in the
         momentum/threat-share analysis. strict drops penalty shots from the
         threat events so the analysis matches the xT model's filter.
    #17  the "open play only" claim is overstated: free-kick/corner shots and
         set-piece passes are retained. strict restricts the threat analysis to
         true open play (and see the corrected comment in xt_model.py).
    #18  WC 2022 is called the "noise floor" but is neither the minimum firing
         rate nor free of the same sub-stoppage contamination. strict reports a
         contamination-adjusted floor and the true minimum firing rate alongside
         the WC 2022 number.
  All four default OFF. results.json gains a "strict" block only when strict.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RNG = np.random.default_rng(42)
NX, NY = 16, 12

PRE_POST_MIN = 10.0   # momentum window length, minutes
TACTIC_WIN = 5.0      # substitution / formation-change window after the pause
EPS = 1.0 / 60.0

HOT_CORE = ["ISL 2021-22", "AFCON 2023", "Copa America 2024", "WWC 2019"]


def classify(s, strict=False):
    """Label a stoppage. strict (AUDIT-DRAFT #15, opt-in) reorders the rules so
    a substitution-caused pause is labelled "substitution" BEFORE it can fall
    into the drinks_break window. Default (strict=False) keeps the committed
    order: a sub during the 25-32' window is still counted as a drinks_break,
    which is the circularity the audit flags. Validate on the data box."""
    if s.goal_before:
        return "goal_restart"
    if s.injury:
        return "injury"
    if s.card_near:
        return "card_or_var"
    if strict and s.subs_in > 0:
        return "substitution"
    if s.gap_sec >= 90 and 25 <= s.half_min < 32 and s.period <= 2:
        return "drinks_break"
    if s.subs_in > 0:
        return "substitution"
    return "other"


def cell(x, y):
    cx = np.clip((np.asarray(x) / 120.0 * NX).astype(int), 0, NX - 1)
    cy = np.clip((np.asarray(y) / 80.0 * NY).astype(int), 0, NY - 1)
    return cx, cy


def main(strict=False):
    """Run the cooling-break analysis.

    strict (AUDIT-DRAFT, opt-in; default False keeps the committed headline
    numbers): applies findings #15-#18. See module docstring for the full list
    and the validation command. With strict=False every existing output is
    byte-for-byte unchanged; with strict=True a "strict" block is added to
    results.json and a few extra console lines are printed.
    """
    stop = pd.read_csv(os.path.join(ROOT, "stoppages.csv"))
    matches = pd.read_csv(os.path.join(ROOT, "matches.csv")).set_index("match_id")
    admin = pd.read_csv(os.path.join(ROOT, "admin.csv"))
    moves = pd.read_csv(os.path.join(ROOT, "moves.csv"))
    shots = pd.read_csv(os.path.join(ROOT, "shots.csv"))
    shots = shots[shots.period <= 4]
    # AUDIT-DRAFT #16/#17 (opt-in; validate on the data box): penalties are
    # excluded from the xT model (xt_model.py: pen == 0) but were INCLUDED in
    # the momentum/threat-share analysis, which only filtered period <= 4.
    # strict drops penalty shots so the analysis matches the model. If a
    # "set_piece"/"play_pattern" column exists, strict also restricts to true
    # open play (#17); otherwise it falls back to the penalty-only filter and
    # records that the open-play restriction was unavailable.
    strict_notes = {}
    if strict:
        if "pen" in shots.columns:
            before = len(shots)
            shots = shots[shots.pen == 0]
            strict_notes["penalty_shots_dropped"] = int(before - len(shots))
        else:
            strict_notes["penalty_shots_dropped"] = "no 'pen' column in shots.csv"
    xt = np.load(os.path.join(ROOT, "xt_grid.npy"))

    # AUDIT-DRAFT #15 (opt-in): classify with strict reordering so sub-caused
    # pauses do not enter the drinks_break / break set.
    stop["label"] = stop.apply(lambda s: classify(s, strict=strict), axis=1)
    det = stop[stop.label == "drinks_break"]
    det_by_t = det.groupby("tournament").size()
    nmatches = matches.groupby("tournament").size()
    wc22_fp_rate = det_by_t.get("WC 2022", 0) / nmatches["WC 2022"]

    print("== detector firings per tournament (WC 2022 = false positive floor) ==")
    for t in nmatches.index:
        n = det_by_t.get(t, 0)
        print("  {:20s} {:4d} in {:3d} matches ({:.3f}/match){}".format(
            t, n, nmatches[t], n / nmatches[t],
            "   <- break set" if t in HOT_CORE else ""))

    breaks = det[det.tournament.isin(HOT_CORE)].copy()
    breaks["end_min"] = breaks.match_min + breaks.gap_sec / 60.0
    n_hot_matches = int(nmatches[HOT_CORE].sum())
    expected_bg = wc22_fp_rate * n_hot_matches
    purity = (len(breaks) - expected_bg) / len(breaks)

    # AUDIT-DRAFT #18 (opt-in; validate on the data box): WC 2022 is treated as
    # the "false positive floor", but it is (a) not the minimum firing rate
    # across non-core tournaments and (b) contaminated by the same sub-stoppage
    # mislabelling as the break set. Report a true minimum firing rate and, when
    # the substitution column is present, a contamination-adjusted floor that
    # removes detections whose pause was substitution-caused. Headline purity
    # above is left on the WC 2022 number; the strict block reports alternatives.
    if strict:
        per_match_rate = {t: float(det_by_t.get(t, 0) / nmatches[t]) for t in nmatches.index}
        noncore = {t: r for t, r in per_match_rate.items() if t not in HOT_CORE}
        true_min_t = min(noncore, key=noncore.get) if noncore else None
        # contamination-adjusted WC 2022 floor: drinks_break detections in WC
        # 2022 that coincide with a substitution in the same stoppage.
        wc = det[det.tournament == "WC 2022"]
        wc_sub = int((wc.subs_in > 0).sum()) if "subs_in" in wc.columns else 0
        wc_clean = max(0, len(wc) - wc_sub)
        wc22_clean_rate = wc_clean / nmatches["WC 2022"]
        expected_bg_clean = wc22_clean_rate * n_hot_matches
        strict_notes["floor"] = {
            "wc22_fp_per_match": float(wc22_fp_rate),
            "wc22_contamination_adjusted_per_match": float(wc22_clean_rate),
            "wc22_sub_contaminated_detections": wc_sub,
            "true_min_tournament": true_min_t,
            "true_min_per_match": float(noncore[true_min_t]) if true_min_t else None,
            "purity_on_wc22_floor": float(purity),
            "purity_on_contamination_adjusted_floor":
                float((len(breaks) - expected_bg_clean) / len(breaks)) if len(breaks) else None,
            "purity_on_true_min_floor":
                float((len(breaks) - noncore[true_min_t] * n_hot_matches) / len(breaks))
                if (true_min_t and len(breaks)) else None,
        }

    # threat events: positive xT deltas of completed moves, plus shot xG
    # AUDIT-DRAFT #17 (opt-in): in strict mode restrict moves AND shots to true
    # open play when a play-pattern/set-piece column is available, so the threat
    # analysis lives up to the "open play only" claim. Recognised columns:
    # "open_play" (1 = open play), "set_piece" (1 = set piece), or
    # "play_pattern" (string "Regular Play"). Falls back to no restriction (and
    # records that) if none is present.
    if strict:
        def open_play_mask(d):
            if "open_play" in d.columns:
                return d.open_play == 1
            if "set_piece" in d.columns:
                return d.set_piece == 0
            if "play_pattern" in d.columns:
                return d.play_pattern == "Regular Play"
            return None
        mm = open_play_mask(moves)
        sm = open_play_mask(shots)
        if mm is not None:
            moves = moves[mm]
            strict_notes["moves_open_play_only"] = True
        else:
            strict_notes["moves_open_play_only"] = "no play-pattern column in moves.csv"
        if sm is not None:
            shots = shots[sm]
            strict_notes["shots_open_play_only"] = True
        else:
            strict_notes["shots_open_play_only"] = "no play-pattern column in shots.csv"
    ok = moves[moves.ok == 1]
    sx, sy = cell(ok.sx, ok.sy)
    ex, ey = cell(ok.ex, ok.ey)
    delta = np.maximum(0, xt[ex, ey] - xt[sx, sy])
    off1 = ok.period.map({1: 0, 2: 45, 3: 90, 4: 105}).values
    threat = pd.DataFrame({
        "match_id": ok.match_id.values, "team_id": ok.team_id.values,
        "min": off1 + ok.t.values / 60.0, "val": delta, "period": ok.period.values,
    })
    off2 = shots.period.map({1: 0, 2: 45, 3: 90, 4: 105}).values
    threat = pd.concat([threat, pd.DataFrame({
        "match_id": shots.match_id.values, "team_id": shots.team_id.values,
        "min": off2 + shots.t.values / 60.0, "val": shots.xg.values,
        "period": shots.period.values,
    })], ignore_index=True)
    threat = threat[threat.val > 0]

    half_end = threat.groupby(["match_id", "period"])["min"].max()
    tgroup = {k: v for k, v in threat.groupby("match_id")}
    agroup = {k: v for k, v in admin.groupby("match_id")}

    def window_stats(mid, period, bm, be):
        """threat shares and tactical actions around a real or pseudo pause,
        measured identically for both arms"""
        m = matches.loc[mid]
        df = tgroup.get(mid)
        if df is None:
            return None
        df = df[df.period == period]
        h_start = 0.0 if period == 1 else 45.0
        h_end = half_end.get((mid, period), h_start)
        pre_lo, post_hi = max(h_start, bm - PRE_POST_MIN), min(h_end, be + PRE_POST_MIN)
        pre_len, post_len = bm - pre_lo, post_hi - be
        if pre_len < 5 or post_len < 5:
            return None
        pre = df[(df["min"] >= pre_lo) & (df["min"] < bm)]
        post = df[(df["min"] > be) & (df["min"] <= post_hi)]
        out = {"match_id": mid, "period": period, "bm": bm, "be": be}
        for name, seg, ln in (("pre", pre, pre_len), ("post", post, post_len)):
            a = seg[seg.team_id == m.home_id].val.sum() / ln
            b = seg[seg.team_id == m.away_id].val.sum() / ln
            out[name + "_home"], out[name + "_away"] = a, b
            out[name + "_share"] = a / (a + b) if (a + b) > 0 else np.nan
        ad = agroup.get(mid)
        for k in ("subs_in", "shifts_in", "subs_after", "shifts_after"):
            out[k] = 0
        if ad is not None:
            during = ad[(ad.match_min >= bm - EPS) & (ad.match_min <= be + EPS)]
            after = ad[(ad.match_min > be + EPS) & (ad.match_min <= be + TACTIC_WIN)]
            out["subs_in"] = int((during.type == "Substitution").sum())
            out["shifts_in"] = int((during.type == "Tactical Shift").sum())
            out["subs_after"] = int((after.type == "Substitution").sum())
            out["shifts_after"] = int((after.type == "Tactical Shift").sum())
        return out

    rows = []
    for _, b in breaks.iterrows():
        r = window_stats(b.match_id, b.period, b.match_min, b.end_min)
        if r is None:
            continue
        r.update(tournament=b.tournament, gap_sec=b.gap_sec, kind="break")
        rows.append(r)

    med_min = float(breaks.half_min.median())
    med_gap = float(breaks.gap_sec.median())

    blocked = set(map(tuple, stop[(stop.gap_sec >= 90) & (stop.half_min.between(15, 45))]
                      [["match_id", "period"]].values))
    for mid in matches.index:
        for period in (1, 2):
            if (mid, period) in blocked or (mid, period) not in half_end.index:
                continue
            bm = med_min + (0 if period == 1 else 45.0)
            r = window_stats(mid, period, bm, bm + med_gap / 60.0)
            if r is None:
                continue
            r.update(tournament=matches.loc[mid, "tournament"], gap_sec=med_gap, kind="control")
            rows.append(r)

    df = pd.DataFrame(rows).dropna(subset=["pre_share", "post_share"])
    df["flip"] = ((df.pre_share - 0.5) * (df.post_share - 0.5) < 0).astype(float)
    df["swing"] = (df.post_share - df.pre_share).abs()
    df["tactic_during"] = ((df.subs_in + df.shifts_in) > 0).astype(float)
    df["tactic_after"] = ((df.subs_after + df.shifts_after) > 0).astype(float)
    df.to_csv(os.path.join(ROOT, "windows.csv"), index=False)

    bk, ct = df[df.kind == "break"], df[df.kind == "control"]
    w1 = float((bk.period == 1).mean())  # break period mix, fixed weights

    def rates(sub):
        out = {}
        for col in ("subs_in", "shifts_in", "subs_after", "shifts_after",
                    "tactic_during", "tactic_after", "flip", "swing"):
            out[col] = {
                "p1": float(sub[sub.period == 1][col].mean()),
                "p2": float(sub[sub.period == 2][col].mean()),
                "pooled": float(sub[col].mean()),
            }
            out[col]["standardized"] = w1 * out[col]["p1"] + (1 - w1) * out[col]["p2"]
        return out

    def cluster_boot(col, period=None, n=10000):
        """match-clustered bootstrap of break-minus-control difference,
        standardized to the break period mix unless a single period is given"""
        lo_hi = []
        samples = {}
        for name, arm in (("bk", bk), ("ct", ct)):
            sub = arm if period is None else arm[arm.period == period]
            per = {}
            for p in ((1, 2) if period is None else (period,)):
                g = sub[sub.period == p].groupby("match_id")[col].agg(["sum", "count"])
                per[p] = (g["sum"].values, g["count"].values)
            samples[name] = per
        dist = []
        for _ in range(n):
            est = {}
            for name in ("bk", "ct"):
                vals = {}
                for p, (s, c) in samples[name].items():
                    idx = RNG.integers(0, len(s), len(s))
                    tot = c[idx].sum()
                    vals[p] = s[idx].sum() / tot if tot else np.nan
                if period is None:
                    est[name] = w1 * vals[1] + (1 - w1) * vals[2]
                else:
                    est[name] = vals[period]
            d = est["bk"] - est["ct"]
            if not np.isnan(d):
                dist.append(d)
        return [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]

    res = {
        "hot_core": HOT_CORE,
        "n_breaks": int(len(bk)), "n_controls": int(len(ct)),
        "break_median_half_min": med_min, "break_median_gap_sec": med_gap,
        "detections_by_tournament": {t: int(det_by_t.get(t, 0)) for t in nmatches.index},
        "matches_by_tournament": {t: int(nmatches[t]) for t in nmatches.index},
        "wc22_fp_per_match": float(wc22_fp_rate),
        "expected_background_in_break_set": float(expected_bg),
        "purity_break_set": float(purity),
        "break_p1_share": w1,
        "control_p1_share": float((ct.period == 1).mean()),
        "rates_break": rates(bk),
        "rates_control": rates(ct),
        "ci_tactic_after_std": cluster_boot("tactic_after"),
        "ci_tactic_after_p2": cluster_boot("tactic_after", period=2),
        "ci_tactic_during_std": cluster_boot("tactic_during"),
        "ci_subs_in_std": cluster_boot("subs_in"),
        "ci_shifts_in_std": cluster_boot("shifts_in"),
        "ci_flip_std": cluster_boot("flip"),
        "ci_swing_std": cluster_boot("swing"),
    }
    # AUDIT-DRAFT (opt-in): only emit the strict block when strict is on, so the
    # default results.json is byte-for-byte unchanged.
    if strict:
        res["strict"] = strict_notes
        print("\n== AUDIT-DRAFT strict mode ON (#15-#18) ==")
        print(json.dumps(strict_notes, indent=2))
    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    cand = bk.copy()
    cand["names"] = cand.match_id.map(
        lambda i: "{} vs {} ({}, {})".format(
            matches.loc[i, "home"], matches.loc[i, "away"],
            matches.loc[i, "stage"], matches.loc[i, "date"]))
    cand["tactic"] = cand.subs_in + cand.shifts_in + cand.subs_after + cand.shifts_after
    cand = cand.sort_values(["tactic", "swing"], ascending=False)
    cols = ["names", "tournament", "period", "bm", "gap_sec", "pre_share", "post_share",
            "subs_in", "shifts_in", "subs_after", "shifts_after"]
    cand[cols].head(20).to_csv(os.path.join(ROOT, "case_candidates.csv"), index=False)
    print("\n== top case study candidates (break set only) ==")
    print(cand[cols].head(15).to_string())


if __name__ == "__main__":
    # AUDIT-DRAFT (opt-in): --strict enables findings #15-#18. Default OFF.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true",
        help="AUDIT-DRAFT opt-in: apply audit fixes #15-#18 (drop sub-caused "
             "pauses from the break set, drop penalties + restrict to open "
             "play in the threat analysis, and report a contamination-adjusted "
             "/ true-minimum detector floor). Default OFF keeps committed "
             "headline numbers unchanged.")
    args = parser.parse_args()
    main(strict=args.strict)
