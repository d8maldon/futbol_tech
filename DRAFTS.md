# Deferred audit fixes - DRAFTS (opt-in; validate on the data box)

_2026-06-25. The five predictor/thesis accuracy findings from AUDIT.md that need the gitignored 49k-match / StatsBomb data to re-validate. Each is implemented as an **opt-in switch that defaults to the current validated behavior** (the committed outputs and the 30-test suite are byte-identical with the switch off). Flip the switch, run the validate command on the data machine, and confirm the expected signal before adopting._

## ratings.py, backtest_history.py

### FINDING #12: the pre-match logit's intercept absorbs the average home-field edge (train window is ~72% real home games), so at dr=0 the production model returns ~H=0.404/A=0.311 -- a ~9pp / ~80-Elo phantom tilt to whichever team FIFA lists first. WC2026 group games are at neutral US/Canada/Mexico venues (fixtures.py has no neutral concept; dr_of only adds HOME_ADV for the 3 hosts), so the listed 'home' team gets that ~9pp edge purely from listing order. The OOS 0.86/ECE-0.018 numbers don't catch it because the test set is also ~73% non-neutral.

- **Approach:** Two opt-in paths in ratings.py, both defaulting to current behavior. (1) prematch_proba(dr, model, kappa=1.0, neutral=False): when neutral=True and a logit model is given, it averages the model's prediction at +dr with its H<->A-swapped prediction at -dr (mirror symmetrisation). This cancels the intercept's home/away asymmetry while preserving the rating-gap signal and draw mass, giving P(home)==P(away) exactly at dr=0, sum=1, and correct ordering at dr!=0 (verified offline). Davidson fallback is already symmetric so neutral is a no-op there. (2) fit_prematch_logit(pairs, n_matches, neutral_indicator=False): with True it fits on x=[dr/400, is_neutral] from 3-tuple pairs so the venue effect lives in an explicit coefficient instead of leaking into the intercept (AUDIT fix option a); accepts existing 2-tuple pairs unchanged in the default path via *_ unpacking. backtest_history.py: walk_forward(df, with_neutral=False) optionally emits 4-tuples (dr,o,year,neutral); a NEUTRAL_AUDIT=1-gated block in main() prints the dr=0 symmetry check and re-scores the genuinely-neutral OOS subset with both the current and neutral scorers (log-loss, accuracy, ECE-home).
- **Opt-in switch:** ratings.prematch_proba(..., neutral=True); ratings.fit_prematch_logit(..., neutral_indicator=True); backtest_history.walk_forward(..., with_neutral=True); and the env var NEUTRAL_AUDIT=1 to enable the validation block in backtest_history.main()
- **Default preserves current behavior:** yes
- **Validate:**
  ```
  cd C:/GIT/futbol_tech; $env:NEUTRAL_AUDIT=1; python src/backtest_history.py   (bash: NEUTRAL_AUDIT=1 python src/backtest_history.py)
  ```
- **Expected signal:** The [NEUTRAL_AUDIT] block prints dr=0 asymmetry |H-A| ~0.09 for the current scorer vs ~0.0000 for neutral=True (confirming symmetric P(home)=P(away) at equal ratings), and on the genuinely-neutral OOS subset the neutral scorer's log-loss and ECE(home) are <= the current scorer's. Running without the env var produces byte-identical output to today's run, and python tests/test_predictor.py still passes (offline-verified here: 4 test functions ok).

## montecarlo.py

### FINDING #13: the champion Monte-Carlo resolves knockouts with a raw Elo coin-flip (expected(dr)) that buries the draw, NOT the Davidson/multinomial draw model that ratings.py fits and validates out-of-sample -- yet the README calls montecarlo 'the very engine validated out-of-sample'. (The group-stage Poisson scorelines are intentionally retained, since GD/GF are real 2026 tiebreakers.)

- **Approach:** Added an opt-in `--draw-model {off,logit,davidson}` flag (default 'off'). New helper advance_prob(dr, draw_model): when draw_model is None it returns the unchanged expected(dr); when a model is given it pulls validated P(H/D/A) from ratings.prematch_proba and reduces the tie to a decisive winner by splitting the draw mass in proportion to the two sides' win probabilities (P(advance)=pH + pD*pH/(pH+pA)), modelling extra-time/penalties favouring the stronger side. Threaded an optional draw_model=None param through sim_knockout and sim_once (both keep their old call sites working -- the 3-arg sim_knockout in the tests is unaffected). 'logit' loads prematch_model.json via ratings.load_prematch_model() and falls back to Davidson if absent; 'davidson' forces the closed-form fallback. Opt-in runs write SEPARATE output files (sim_probs_drawmodel_<mode>.csv and wc2026_champion_drawmodel_<mode>.png) so the committed sim_probs.csv / wc2026_champion.png are never clobbered. main() gained a draw_mode='off' kwarg so the import-time default is identical.
- **Opt-in switch:** --draw-model logit (or --draw-model davidson); omit the flag, or --draw-model off, for the unchanged default
- **Default preserves current behavior:** yes
- **Validate:**
  ```
  cd C:/GIT/futbol_tech && python src/ratings.py && python src/montecarlo.py && python src/montecarlo.py --draw-model logit && python -c "import pandas as pd; a=pd.read_csv('wc2026/sim_probs.csv'); b=pd.read_csv('wc2026/sim_probs_drawmodel_logit.csv'); m=a.merge(b,on='team',suffixes=('_coin','_draw')); m['delta']=m['champion_draw']-m['champion_coin']; print('sum champion coin/draw:', round(m.champion_coin.sum(),2), round(m.champion_draw.sum(),2)); print(m.sort_values('champion_coin',ascending=False)[['team','champion_coin','champion_draw','delta']].head(15).to_string(index=False))"
  ```
- **Expected signal:** Both runs print 'knockout resolver: ...' (raw-Elo coin flip for the default, validated multinomial-logit draw model for --draw-model logit). The default run reproduces the committed wc2026/sim_probs.csv byte-for-byte (champion column unchanged). The logit run produces a champion column that still sums to ~100% but redistributes mass: favourites' title odds rise modestly and mid-tier sides fall, because the draw model no longer 50/50s every knockout draw but tilts extra-time/penalties toward the stronger side. The 29 existing tests (5 in tests/test_montecarlo.py) still pass unchanged. Reconcile the README: either label montecarlo's default knockout resolver as an Elo coin-flip approximation, or make --draw-model the default so the prose 'validated out-of-sample' claim is literally true.

## analyze.py, xt_model.py

### #15 subs ~1.8x is partly circular: substitution-caused stoppages are themselves classified as drinks_break candidates in classify() (~lines 31-42), so they enter the break set and inflate the sub-rate that is then compared against controls.

- **Approach:** Added a strict parameter to classify() that, when True, returns 'substitution' BEFORE the drinks_break window test, so sub-caused pauses never enter the break set. main(strict=...) threads it through stop.apply. Default order (drinks_break wins) is untouched.
- **Opt-in switch:** --strict CLI flag / main(strict=True) / classify(s, strict=True)
- **Default preserves current behavior:** yes
- **Validate:**
  ```
  cd /c/GIT/futbol_tech && python src/analyze.py --strict
  ```
- **Expected signal:** In strict mode, n_breaks drops (sub-caused detections removed) and rates_break['subs_in']/['subs_after'] plus ci_subs_in_std shrink toward the control arm; the ~1.8x sub migration multiplier should fall materially vs the default run. Compare results.json default vs strict.

### #16 penalties are excluded from the xT model (xt_model.py pen==0) but INCLUDED in the momentum/threat-share analysis: shots are filtered only by period<=4 (~line 57), not pen==0.

- **Approach:** In strict mode, drop shots where pen!=0 before building threat events, matching the xT model filter. Records penalty_shots_dropped count in the strict block. Guards on column presence.
- **Opt-in switch:** --strict
- **Default preserves current behavior:** yes
- **Validate:**
  ```
  cd /c/GIT/futbol_tech && python src/analyze.py --strict
  ```
- **Expected signal:** strict.penalty_shots_dropped > 0 in results.json; the momentum/threat-share quantities (pre/post share, flip, swing CIs) shift slightly since high-xG penalty spikes are removed. The momentum null should be unaffected or strengthened (still spanning 0).

### #17 the 'open play only' claim overstates filtering: free-kick & corner shots/passes are retained; xt_model.py comment (~lines 27-29) calls the filter 'open play only' when it is only non-penalty/non-shootout.

- **Approach:** Corrected the xt_model.py comment (comment-only, model output unchanged). In analyze.py strict mode, restrict moves AND shots to true open play when a play-pattern column (open_play / set_piece / play_pattern=='Regular Play') exists; otherwise record that the restriction was unavailable.
- **Opt-in switch:** --strict
- **Default preserves current behavior:** yes
- **Validate:**
  ```
  cd /c/GIT/futbol_tech && python src/analyze.py --strict
  ```
- **Expected signal:** results.json strict.moves_open_play_only / shots_open_play_only == true (if the column exists); threat-share distributions narrow as set-piece threat is removed. If the column is absent the value is a string note, telling the maintainer the data lacks a play-pattern field.

### #18 WC 2022 is called the 'noise floor' but is neither the minimum firing rate across non-core tournaments nor free of the same sub-stoppage contamination (floor/purity logic ~lines 64-77).

- **Approach:** In strict mode, compute and report (a) the true minimum per-match firing rate across non-core tournaments and which tournament it is, and (b) a contamination-adjusted WC 2022 floor that removes detections coinciding with a substitution, then recompute break-set purity under both alternative floors alongside the original WC2022 purity. Headline purity is left on the WC2022 number.
- **Opt-in switch:** --strict
- **Default preserves current behavior:** yes
- **Validate:**
  ```
  cd /c/GIT/futbol_tech && python src/analyze.py --strict
  ```
- **Expected signal:** results.json strict.floor shows wc22_contamination_adjusted_per_match <= wc22_fp_per_match and a true_min_tournament possibly != 'WC 2022' with a lower true_min_per_match; purity_on_true_min_floor / purity_on_contamination_adjusted_floor reveal whether headline purity_break_set was overstated (higher) or understated.
