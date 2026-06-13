# Methods review: what the literature teaches us, and where we can push back

A deep-read of five external bodies of work, each verified against our actual
code before anything was promoted to a recommendation. Of 37 candidate
improvements, 4 survived adversarial verification (the rest needed optical
tracking data we do not have, or were misattributed to a paper). Line
references below are real.

## Sources

- Robberechts, Van Haaren, Davis, *A Bayesian Approach to In-Game Win
  Probability in Soccer*, KDD 2021 (arXiv 1906.05029). The directly
  comparable academic model.
- Getty, Li, Yano, Gao, Hosoi (MIT Sports Lab), *Luck and the Law:
  Quantifying Chance in Fantasy Sports and Other Contests*, SIAM Review 2018
  / *Luck is Hard to Beat* (arXiv 1706.02447). The luck-vs-skill variance
  decomposition; soccer is among the highest-luck team sports.
- American Soccer Analysis, *We Have A New Win Probability Model* (2021).
  Practitioner design; found team strength carried everything, no league
  effects.
- Skripnikov/Cemek/Gillman, game-state adjustment of minute-by-minute
  production (JQAS 2026, arXiv 2508.04008).
- ML / Bayesian xG (arXiv 2301.13052; Bayes-xG) and possession value beyond
  xT (path signatures, arXiv 2508.12930).

## 1. Top improvements to adopt (ranked by impact/effort)

### 1.1 Pre-game team-strength prior in `winprob.py` (HIGHEST PRIORITY)
Add one static team-strength differential feature and refit. Source:
KDD2021 "Rating Differential", corroborated by ASA (team strength was the
load-bearing feature; no league effects).

Edit `winprob.py`: append `"strength"` (and `"strength_clock"`) to `FEATURES`
(line 23); add a `strength` arg to `feature_row` (lines 26-28) plus a
`strength * rem` interaction so the prior decays as in-game evidence
accumulates; compute `strength = home_strength - away_strength` per match in
the training loop (lines 65-92). `predict` and `live_eval.py:162` thread the
new arg through unchanged.

Build the strength scalar from event data, no new deps:
- Training: rolling pre-match per-90 (xG-for minus xG-against) over each
  team's prior N matches within the same tournament (xG already in
  `shots.csv`); cold-start to 0 (today's baseline).
- Live: tournament-to-date per-90 xG diff from the FotMob shotmaps we already
  cache (`live_eval.py:118-124`), held static from kickoff. Avoid market odds
  (scraping dependency + provenance).

Why: today every WC2026 match opens at the identical home-advantage baseline,
so a mismatch and a coin-flip both read ~50/50 at minute 0, and the
biggest-swing annotator (`live_eval.py:168`) over-fires early because the
curve has to discover the gap through xG instead of starting tilted. One
column + refit; lowest effort, highest leverage.

### 1.2 Brier + RPS on a match-grouped holdout in `winprob.py`
Replace the in-sample evaluation (lines 102-115) with leave-one-tournament-out
or `GroupKFold` keyed on `match_id`, reporting out-of-fold log loss, Brier and
RPS plus the decile calibration. Group labels are free from the existing
`matches.iterrows()` loop.

Why: the current log loss 0.79 and "within ~2 pts" calibration are in-sample
over 49,590 rows where each match contributes 90 autocorrelated rows sharing
one label (line 82). Effective-n is wildly overstated; calibration is
optimistic. Grouping the split by `match_id` is the leakage fix (not RPS, which
is a secondary ordered-outcome metric). Makes us benchmarkable against KDD2021
(ECE 0.011) and ASA.

### 1.3 Game-state reweighting on the xT/momentum river (`analyze.py`, `xt_model.py`)
Down-weight per-minute xT+xG accumulated while leading / up a man; up-weight
while trailing / down a man, before smoothing. Source: Skripnikov Fig. 6
(~0.70-0.90 leading, 1.10-1.75 trailing, multipliers by man advantage).

Edit the threat assembly in `analyze.py:80-95`: a multiplier lookup keyed on
`(score_diff, red_card_diff)` applied to each minute's `val` before the
per-team share/smoothing; reuse live in `live_eval.py`. Do NOT transplant
their shot-count coefficients onto xT; fit our own binned curves on the 551
StatsBomb matches.

Why: the river and biggest-swing annotator currently read garbage-time
pressure (a trailing team shelling a packed box) as genuine threat, inflating
output 10-75%. State-aware reweighting stops a tactical artifact being crowned
the match's defining moment.

### 1.4 Knockout handling (ET + shootout) in `winprob.py` / `live_eval.py`
`winprob.py` filters `period <= 2` (lines 51,54,57) and labels `D` on the 90'
score (line 82); `live_eval.py:162` clamps `min(minute, 90)`. Half of WC2026
is knockout, where a level match cannot end in a draw, yet the eval bar
freezes a wrong draw slice late. Route a near-90' level result through a 120'
horizon then a terminal shootout node (50/50, or strength-tilted once 1.1
lands). Honest scope: our 90'-labelled training data has ~zero ET/shootout
signal, so the terminal node is heuristic — ship as a structural-correctness
fix, not a calibration win. KDD2021 only says it *can* be extended; the
shootout node is ours.

## 2. Gaps in their papers we can improve on

- **In-game luck band (genuinely novel).** Hosoi's phi is season-level; no
  paper here has a per-match luck measure. Port the Monte-Carlo-null idea to
  one match: 10k Poisson-binomial draws over the FotMob per-shot xG profile we
  already ingest, report a 2.5/97.5 band on final P(H/D/A) ("a team with this
  shot profile fails to win X% of the time"). Fills our no-uncertainty gap and
  uses Hosoi's own soccer-is-high-luck result as the justification.
- **Cooling-break causal design.** None of the five does intervention
  analysis. Add Aoki's sharp-null permutation test (relabel break vs control,
  recompute the diff-in-diff each draw) next to `cluster_boot` in
  `analyze.py:181` as a complement to the bootstrap for our small-N effect.
- **Live multi-source pipeline.** Every paper is post-hoc archival. KDD2021
  concedes its richest features are less accurate in real time (RPS 0.138 live
  vs 0.134 offline); we are real-time-first.
- **Independent per-shot xG ground truth.** We have FotMob real xG for every
  live shot, so we can validate our 6-zone ESPN xG against a true benchmark per
  zone — better than papers that could only correlate to one xG model.
- **Public-data reproducibility.** StatsBomb open data + public feeds,
  plain-JSON model, numpy scoring, vs paid data + a 200k-iteration ADVI fit.

## 3. Where we are currently weaker (the uncomfortable truths)

- No pre-game team-strength prior (every match starts at the same baseline). → 1.1
- Autocorrelated training rows, no clustering, in-sample calibration. → 1.2
- Only log loss; no Brier/RPS for an ordered 1X2 outcome. → 1.2
- xT ignores game state; garbage-time pressure reads as real threat. → 1.3
- 6-zone ESPN xG never validated against real xG; no per-zone bias number.
  → validate each FotMob shot's `expectedGoals` against our zone value, drive
  the eval off real FotMob xG with the zone value as fallback.
- No uncertainty anywhere; eval bar and curve are hairlines. → Section 2 luck band.
- Eval bar cannot represent knockouts (`live_eval.py:162` clamp). → 1.4
- `gd / (rem + 0.1)` (winprob.py:28) blows up near 90'; a goals-then-simulate
  reframe (KDD2021/ASA) would remove it, but that is a larger re-architecture.
  Defer.

## 4. Concrete next-PR checklist (priority order)

1. Static `strength` feature + `strength*rem` in `feature_row`/`FEATURES`, refit. (1.1)
2. Replace in-sample eval with `GroupKFold`-by-`match_id`; report out-of-fold
   log loss, Brier, RPS, decile calibration. (1.2)
3. Build the live strength scalar from cached FotMob shotmaps; thread through
   `live_eval.py:162`. (1.1 live)
4. `(score_diff, red_card_diff)` multiplier (fitted on our 551 matches, binned)
   on each minute's `val` in `analyze.py:80-95`; reuse in `live_eval.py`. (1.3)
5. 10k Poisson-binomial luck band over the FotMob xG profile; render the eval
   bar as a band, not a hairline. (Section 2)
6. Knockout handling: drop `period <= 2` for knockout fixtures, 120' horizon,
   terminal shootout node; remove the `min(minute, 90)` clamp. (1.4)
7. Validate 6-zone ESPN xG vs FotMob real `expectedGoals` per zone; eval
   consumes real xG with zone value as fallback. (Section 3)
8. Sharp-null permutation test beside `cluster_boot` in `analyze.py:181`. (Section 2)
