# Council log

Shared review log for the pantheon councils (athena / mnemosyne / hephaestus /
prometheus). Entries are distinguished by council name. The log is institutional
memory: each pass reads it first so prior verdicts and pre-commitments hold.

## Pass 1 (2026-06-15) - prometheus

**Scope:** Audit the WC2026 match predictor and stress-test the "nothing to fix"
conclusion drawn after the model went 6/12 (argmax) on the played group games.
**Personas:** Ronald Fisher, Zoubin Ghahramani, Andrej Karpathy
**Audited:** src/ratings.py (seed_history), src/montecarlo.py, src/backtest.py,
src/backtest_history.py, src/winprob.py
**Verdict:** SOUND-WITH-FIXES

### Findings

1. **[NEW]** Ghahramani (commandment 11, calibration/eval integrity): the in-game
   winprob (winprob.py) is OVER-CONFIDENT. A single temperature (T=1.25, fit on a
   held-out calib fold) improves out-of-sample log loss 0.83 -> 0.82 and P(home)
   ECE 0.057 -> 0.045 (worst bin 11% -> 8%). Fix: added temperature scaling to
   winprob.py (3-way fit/calib/test split; T saved in the model JSON; predict()
   divides logits by T). RE-VERIFIED on a test fold disjoint from the calib fold.

2. **[NEW]** Ghahramani: a mild residual under-prediction of home wins in the mid
   deciles survives temperature scaling -- a model-form bias of the linear logit,
   not pure over-confidence. A full fix needs interaction features and its own OOS
   validation. SURFACED for an explicit user decision (the only legitimate
   non-fix); not silently deferred.

3. **[NEW]** Fisher (small-sample inference): the 6/12 "failure" is within
   sampling variance -- 12 autocorrelated tournament games; away-rated sides going
   1/12 is ~8.5% under the null. Per-class OOS calibration on 5,691 held-out
   internationals (P(H) ECE 0.017, P(D) 0.013, P(A) 0.011) confirms the PRE-MATCH
   model is sound. No fix; fitting the 12 is overfitting (demonstrated, discarded).

4. **[NEW]** Fisher: the draw model is NOT the gap. The single-feature logit
   (dr/400) is well-calibrated on P(D) out-of-sample (ECE 0.013); richer draw
   curves (dr^2, |dr|) do not beat it OOS (0.8602 vs 0.8606 = noise). No
   Dixon-Coles / Poisson-scoreline improvement is detectable on the holdout.

5. **[NEW]** Karpathy (eval footguns): pick-accuracy is the wrong headline for a
   1X2 model (argmax can never pick a draw); proper scores (RPS / log loss) +
   decisive-game accuracy now lead the scorecard. Calibration had been reported
   only for P(home win); verified here that all three classes are calibrated OOS.
   The HA/K/decay/neutral-home searches used a val/test split (no tuning on the
   test fold), so their rejections are sound negatives, correctly not adopted.

### Pre-commitment

After fix 1 (applied + verified) and an explicit decision on finding 2 (the
residual mid-range bias), prometheus signs off: the pre-match predictor is SOUND
and the in-game model is calibrated. No further additions.

### Cross-references

- Prior passes related: none (first pass)
- Pre-commitments honoured: none
- Pre-commitments newly issued: the one above
