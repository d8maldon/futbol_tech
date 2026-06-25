"""Agentic match-analyst loop.

Each tick it reads the live match state and DECIDES what to surface to the
dashboard -- a one-line insight for the ticker / a flagged turning point / a panel
to highlight. Two interchangeable backends:

  rule_based  : deterministic thresholds, no API key, runs anywhere   (default)
  claude      : an LLM analyst via the Anthropic SDK (set ANTHROPIC_API_KEY)

The `tools` dict exposes the match state so either backend can reason over it; the
LLM backend is given the same context() the rules use. Wire run()'s output into a
theme's ticker / an "insight" panel to turn the passive dashboard into a live
co-analyst. Designed to ride the same (m, state, ti) stream as live_dashboard.py.

    python src/analyst.py                 # rule-based, over the synthetic match
    python src/analyst.py --claude        # LLM analyst (needs ANTHROPIC_API_KEY)
"""
import argparse
import os

import dashboard_themes as T

PANELS = ("winprob", "xg", "control", "ticker", "ratings")


# ----------------------------------------------------------------- tools / context
def score(m, ti):
    return int(m["sc_h"][ti]), int(m["sc_a"][ti])


def winprob(m, ti):
    return float(m["wp_home"][ti]), float(m["wp_draw"][ti]), float(m["wp_away"][ti])


def xg(m, ti):
    return float(m["xg_h"][ti]), float(m["xg_a"][ti])


def recent_events(m, ti, k=3):
    return [e for e in m["events"] if e["min"] <= ti][-k:]


def control_share(state):
    if not state or not state.get("tracks"):
        return None
    return T.voronoi_regions(state["tracks"])[2]


tools = {"score": score, "winprob": winprob, "xg": xg,
         "recent_events": recent_events, "control_share": control_share}


def context(m, ti, state=None):
    """compact JSON-able snapshot the analyst reasons over."""
    sh, sa = score(m, ti)
    wph, wpd, wpa = winprob(m, ti)
    xgh, xga = xg(m, ti)
    ctx = {"minute": ti, "home": m["home"], "away": m["away"], "score": [sh, sa],
           "win_prob": {"home": round(wph, 3), "draw": round(wpd, 3), "away": round(wpa, 3)},
           "xg": {"home": round(xgh, 2), "away": round(xga, 2)},
           "xg_deserved_home_winprob": round(float(m["wp_xg"][ti]), 3),
           "recent_events": [{"min": e["min"], "type": e["type"], "player": e["player"]}
                             for e in recent_events(m, ti, 3)]}
    cs = control_share(state)
    if cs is not None:
        ctx["pitch_control_home"] = round(cs, 3)
    return ctx


# ----------------------------------------------------------------- rule-based analyst
def analyze(m, ti, state=None, mem=None):
    """deterministic insights for minute ti. Returns (insights, mem). Each insight =
    {kind, priority(1-3), panel, text}. `mem` carries cross-tick state."""
    mem = mem if mem is not None else {}
    seen = mem.setdefault("seen", set())
    out = []
    sh, sa = score(m, ti)
    wph = float(m["wp_home"][ti])
    H = m["home"][:3].upper()

    for e in m["events"]:
        if e["min"] != ti or id(e) in seen:
            continue
        seen.add(id(e))
        if e["type"] == "Goal":
            sc = e.get("score", (sh, sa))
            out.append({"kind": "goal", "priority": 3, "panel": "xg",
                        "text": "GOAL -{} ({}-{}). {} win prob now {:.0f}%".format(e["player"], sc[0], sc[1], H, wph * 100)})
        elif e["type"] == "VAR":
            out.append({"kind": "var", "priority": 3, "panel": "ticker",
                        "text": "VAR -{} ({})".format(e["player"], e.get("note", "under review"))})
        elif e["type"] == "Card":
            out.append({"kind": "card", "priority": 1, "panel": "ticker",
                        "text": "Booking -{}".format(e["player"])})

    last_wp = mem.get("last_wp", wph)
    if abs(wph - last_wp) >= 0.12 and ti - mem.get("last_swing", -99) >= 3:
        mem["last_swing"] = ti
        dirn = m["home"] if wph > last_wp else m["away"]
        out.append({"kind": "turning_point", "priority": 2, "panel": "winprob",
                    "text": "Turning point: momentum swings to {} ({:.0f}% -> {:.0f}%)".format(dirn, last_wp * 100, wph * 100)})

    xgh = float(m["xg_h"][ti])
    if ti >= 25 and ti - mem.get("last_xg", -99) >= 20:
        if sh - xgh >= 1.0:
            mem["last_xg"] = ti
            out.append({"kind": "xg", "priority": 1, "panel": "xg",
                        "text": "{} clinical: {} goals from just {:.1f} xG".format(H, sh, xgh)})
        elif xgh - sh >= 0.8:
            mem["last_xg"] = ti
            out.append({"kind": "xg", "priority": 1, "panel": "xg",
                        "text": "{} wasteful: {:.1f} xG but only {} goals".format(H, xgh, sh)})

    cs = control_share(state)
    if cs is not None and cs >= 0.66 and ti - mem.get("last_terr", -99) >= 15:
        mem["last_terr"] = ti
        out.append({"kind": "territory", "priority": 1, "panel": "control",
                    "text": "{} controlling {:.0f}% of the pitch".format(H, cs * 100)})

    mem["last_wp"] = wph
    return out, mem


# ----------------------------------------------------------------- claude analyst
def claude_insight(m, ti, state=None, model="claude-haiku-4-5-20251001"):
    """One LLM-authored broadcast line + panel pick. Returns dict or None (no key)."""
    import json
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=120,
        system=("You are a live football analyst feeding a broadcast dashboard. Given the "
                "match-state JSON, reply with ONE punchy line (<=18 words) and the single "
                'most relevant panel to highlight. Respond as compact JSON: '
                '{"line": "...", "panel": "winprob|xg|control|ticker|ratings"}.'),
        messages=[{"role": "user", "content": json.dumps(context(m, ti, state))}])
    text = msg.content[0].text.strip()
    try:
        d = json.loads(text)
        return {"kind": "llm", "priority": 2, "panel": d.get("panel", "ticker"), "text": d.get("line", text)}
    except Exception:
        return {"kind": "llm", "priority": 2, "panel": "ticker", "text": text}


# ----------------------------------------------------------------- loop
def run(m, states=None, use_claude=False, on_insight=None):
    """Tick over the match minute-by-minute, emitting insights. `states` (optional)
    is a per-minute state stream for spatial insights; `on_insight(ti, insight)` is
    the sink (defaults to printing -- wire it into a theme ticker)."""
    if on_insight is None:
        def on_insight(ti, ins):
            print("{:>3}'  [{:<9}] {}".format(ti, ins["panel"], ins["text"]))
    mem = {}
    state_by_min = {}
    if states is not None:
        fmin = T.compute_frame_min(states)
        for s, fm in zip(states, fmin):
            state_by_min[int(round(float(fm)))] = s
    for ti in range(0, 91):
        st = state_by_min.get(ti)
        insights, mem = analyze(m, ti, st, mem)
        if use_claude:
            li = claude_insight(m, ti, st)
            if li and (ti % 10 == 0 or insights):
                insights.append(li)
        for ins in sorted(insights, key=lambda x: -x["priority"]):
            on_insight(ti, ins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", action="store_true", help="use the LLM analyst (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--match", default="4667812")
    args = ap.parse_args()
    try:
        import match_data as MD
        m = MD.load(args.match)
    except Exception:
        import _preview_data as P
        m = P.build_m()
        print("(no FotMob cache -> using the synthetic Argentina 3-0 Algeria match)\n")
    if args.claude and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--claude set but ANTHROPIC_API_KEY is missing; falling back to rule-based.\n")
        args.claude = False
    run(m, use_claude=args.claude)


if __name__ == "__main__":
    main()
