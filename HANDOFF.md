# Session handoff — futbol_tech (formerly hidden-timeout)

**Last updated: 2026-06-16.** Continuing on another machine? Clone and read this
file first, then you're caught up:

```
git clone https://github.com/d8maldon/futbol_tech
cd futbol_tech && pip install -r requirements.txt
```

Data under `data/` is gitignored (raw event JSON, clips, models, FotMob/FIFA
caches). The scripts re-fetch what they need on first run (all HTTPS uses
`verify=False` / `curl -k` because this is a TLS-interception network).

---

## What this repo is — three pillars

1. **The cooling-break thesis** (the original "hidden timeout"). Detect FIFA
   hydration breaks in StatsBomb event data and measure what they change.
   Headline (reproduces from the data on disk): **86 hot-core breaks, ~68%
   purity**; the naive "free timeout" effect is a **minute-75 composition
   artifact**; what survives = **subs migrate into the pause ~1.8x**, **momentum
   null**. Pipeline: `download → extract → xt_model → analyze → make_figures`.

2. **The match predictor.** Self-adjusting World-Football-Elo **seeded from the
   FULL international history (~49k martj42 matches)** → P(H/D/A) via a
   multinomial-logit draw model; Monte-Carlo champion odds; leakage-free
   backtest. **Validated out-of-sample: log loss 0.86, ECE 0.019** on held-out
   2021-2026. An in-game win-probability model (the chess-engine eval bar) is
   **temperature-calibrated, OOS 0.82**. Files: `ratings, montecarlo, backtest,
   backtest_history, winprob, fixtures, squads, goalscorer, action_value,
   goal_hazard`.

3. **Computer vision + the WC2026 demo suite.** Broadcast → top-down player
   tracking (validated **~5.1 m** vs SoccerNet). Demos in `figures/`:
   `wc2026_chances.png` (xG shot map), `wc2026_highlights.png` (eval-bar swings),
   `wc2026_compilation.png` (goals-against-the-odds vs chances missed),
   `wc2026_tactical_clip.mp4/.gif` (continuous broadcast→top-down tracking),
   `wc2026_argentina_montage.mp4/.gif` (cut-aware tracking on YouTube highlights
   — Argentina 3-0 Algeria kickoff passage, blanks the pre-match graphics),
   `wc2026_argentina_clip.mp4` + the local-only `wc2026_argentina_full.mp4` (the
   flagship "visual-AI" view over the WHOLE Argentina 3-0 Algeria extended
   highlights: three synced panels — broadcast+team boxes / top-down convex-hull
   team shapes / live pitch-control probability map — with EMA-homography + Kalman
   anti-flicker; ~half the reel blanks honestly as graphics/replays/close-ups),
   `_tactical_snapshot.png`. Files: `broadcast_track, homography, track_fuse,
   validate_topdown, tactical, tactical_clip, montage_clip, visual_ai, cv_compare,
   pitch_control, minimap_track, fuse_eval, live_eval, replay, board, wc2026,
   chances, highlights, compilation`.

## Current state (2026-06-16, 18 WC2026 matches played)
- Predictor is the full-history validated engine. **Spain ~21% title favourite.**
- All demo artifacts rendered and verified; tests pass (`tests/`).
- Repo renamed `hidden-timeout` → `futbol_tech` (GitHub + in-file URLs).

## How to run
- `python run.py thesis|predict|live|all` — the orchestrator. `predict`
  force-refreshes the FIFA calendar then reruns ratings/montecarlo/backtest;
  `live` reruns wc2026/live_eval/replay. Prevents the stale-artifact drift that
  bit us before.
- Demos: `python src/chances.py`, `src/highlights.py`, `src/compilation.py`,
  `src/tactical_clip.py` (the last is slow — runs CV per frame).
- Full reproduce: README "Reproduce".

## The rigour spine (read before changing the model)
**Never fit to the games already played.** Every model change must improve the
held-out **49k OOS** (`backtest_history` / `model_search`), never the ~18 WC
games — that's overfitting (we demonstrated it, then discarded it).
- **Adopted (passed OOS):** full-history reseed; winprob temperature scaling
  (T=1.25); `gd_rem` lead×time interaction.
- **Rejected (failed OOS):** neutral-venue home bump, recency time-decay,
  tournament-draw bump, rest/congestion, and fitting the 12/18.
- **Root cause of WC2026 "misses":** small-sample variance in a draw-heavy run;
  favourites are calibrated long-run (75%=75% over 2,851 OOS games). Not a defect.
- **Council review:** `notes/council-log.md` (prometheus pass 1, signed off). Use
  the pantheon skills (prometheus=ML, athena=math, hephaestus=controls, etc.).

## Environment gotchas
- TLS-interception machine → all fetches use `verify=False` / `curl -k`.
- **WC2026 has NO public tracking data.** Positions come only from broadcast CV
  (heatmap-grade ~5 m, **visible players only**, off-screen unrecoverable —
  proven in `fuse_eval`). Do **not** promise full-22 tracking or
  pass-evaluation-from-positions on WC2026; that needs commercial/historical
  tracking (StatsBomb 360).

## Open / next steps (discussed, not built)
- Render **France–Senegal's eval bar** as a featured image (France 3-1; Mbappé
  brace incl. a **0.036-xG screamer at 91'**).
- Tune the `tactical_clip` passage (window/fps/different attack) if wanted.
- **LinkedIn post:** carousel of the demo suite (chances → highlights →
  compilation) + the MP4 hero; copy drafted in chat.
- **xG-informed live rating updates** — update ratings by *deserved* (xG)
  performance, not just results (proposed; only smaller-sample validation
  possible, since the 49k OOS set has no xG).
