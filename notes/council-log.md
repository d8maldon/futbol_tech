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

### Resolution of finding 2

The user chose the feature pass. A goal-difference x time-remaining interaction
(`gd_rem`), selected on the calib fold and confirmed on the held-out test fold
(0.8207 -> 0.8187), was adopted into winprob.py; no other candidate feature
beat baseline out-of-sample (the rest overfit). Finding 2 is RESOLVED, not
deferred. Combined with the temperature scaling, the in-game model is now at
OOS log loss ~0.819.

### Pre-commitment

Both conditions met -- fix 1 applied + verified, finding 2 resolved via `gd_rem`.
prometheus SIGNS OFF: the pre-match predictor is SOUND and the in-game model is
calibrated. No further additions.

### Cross-references

- Prior passes related: none (first pass)
- Pre-commitments honoured: none
- Pre-commitments newly issued: the one above

## Pass 2 (2026-06-17) - prometheus

**Scope:** NEW SCOPE (resets convergence clock) -- the broadcast-CV tactical-
analytics stack: camera_state, uncertainty, tactical_metrics, event_stats, live_mode.
**Personas:** Zoubin Ghahramani, Jitendra Malik, Fei-Fei Li
**Audited:** src/camera_state.py, src/uncertainty.py, src/tactical_metrics.py,
src/event_stats.py, src/live_mode.py
**Verdict:** SOUND-WITH-FIXES

### Findings

1. **[NEW]** Ghahramani (cmd 11, UQ integrity): uncertainty.py uses the frame's
   reprojection error as sigma_px through the Jacobian -- this is a RANDOM
   (zero-mean) error model, but the dominant broadcast error has a SYSTEMATIC bias
   (wide-shot over-stretch, the ~1-2 m offset) a Gaussian ellipse cannot represent.
   The ellipse therefore UNDERSTATES true error and is uncalibrated. Fix: coverage
   check vs the empirical leave-one-out error; report the ratio; document the
   ellipse as random-error-only with bias uncaptured.
2. **[NEW]** Ghahramani: frame_confidence conf in [0,1] is an unvalidated heuristic
   product; never shown that low conf => higher positional error. Fix: correlate
   conf with the LOO metres error; if it does not predict error it is decorative.
3. **[NEW]** Malik (cmd 5, shortcut/honesty): tactical_metrics "block height (team
   centroid)" with visible-players-only tracks whoever is near the ball, so BOTH
   teams' centroids move with the ball -- this is NOT defensive-line height and is
   misleading. Fix: relabel "visible-block centroid (ball-biased)", drop line-height
   claim.
4. **[NEW]** Malik: formation band-counts from <11 ball-biased visible players is
   the in-moment visible shape, not a base formation. Fix: label "visible in-moment
   shape", do not claim base formation.
5. **[NEW]** Malik: camera_state leans on grass-greenness -- a shortcut that fails
   on different pitch hues/lighting; the homography-ok term is the real trackability
   signal. Fix: document the shortcut + distribution-shift risk.
6. **[NEW]** Fei-Fei (cmd 2/11): camera_state thresholds tuned by eye on ONE match,
   no labeled eval, no quantified accuracy, no baseline. Fix: hand-label a sample,
   report classifier accuracy AND the trivial "homography-ok alone" baseline.
7. **[NEW]** Fei-Fei: event_stats runs on ONE StatsBomb match; PPDA/field-tilt/xT
   are a single-match capability demo, not validated; xT grid provenance uncited.
   Fix: document single-match-demo scope + xT grid provenance (931k moves, 13.6k
   shots, 16x12, in xt_meta.json).
8. **[NEW]** Fei-Fei (cmd 9): live_mode's 22 fps is per-frame DETECTION (no temporal
   Kalman tracking, which visual_ai does); the throughput claim is honest but the
   "tracking" word overreaches. Fix: scope live_mode as per-frame detection+gating
   throughput, not full tracking.

### Fixes applied (DO-NOT-DEFER; re-verified)

1. **UQ coverage (F1) + confidence validity (F2):** added `uncertainty.calibrate()`.
   On 894 held-out keypoints the ellipse holds 49% within 1-sigma (target ~39%,
   slightly conservative) and 76% within 2-sigma (target ~86%) -- a heavier-than-
   Gaussian TAIL from systematic bias, now documented as "trust the ellipse for the
   typical frame, not the wide/behind-goal outliers". corr(confidence, frame error)
   = -0.42, so confidence IS predictive, not decorative. RESOLVED + documented.
3,4. **Tactical honesty (F3,F4):** relabelled block_x -> "visible-block centroid
   (ball-biased), NOT line height"; formation -> "visible in-moment shape, not a
   base formation"; figure titles + docstring updated. RESOLVED.
5. **camera_state shortcut (F5):** documented grass-green as a shortcut + the
   distribution-shift risk; homography_ok named as the real signal. RESOLVED.
6. **camera_state accuracy/baseline (F6):** measured -- the green/player-gated
   "wide" scored 39% (3-way) vs a 57% homography-ok baseline on an eyeball N=28
   set. FIX: WIDE is now `homography_ok` (the pipeline's actual gate, whose
   accuracy is validated downstream by the ~5 m / LOO 2.5 m positional result);
   green/edges only split the non-wide frames for the fallback. We deliberately do
   NOT quote a 3-way accuracy (eyeball labels too noisy to be reliable). RESOLVED
   by anchoring on the validated signal + removing the over-claim.
7. **event_stats provenance/scope (F7):** documented xT grid provenance (931k
   moves / 13.6k shots, xt_meta.json) + the single-match capability-demo scope.
   RESOLVED.
8. **live_mode scope (F8):** documented as per-frame detection + gating throughput
   (no temporal Kalman; that is visual_ai), so "22 fps" is the gated-detection
   budget. RESOLVED.

### Pre-commitment

All eight findings resolved or honestly re-scoped and re-verified above; none
deferred. prometheus SIGNS OFF on the tactical-analytics stack as SOUND with the
documented honest scope (visible-block reads, ~5 m zone-grade, homography-anchored
gate, calibrated-in-the-bulk uncertainty). No further additions.

### Cross-references

- Prior passes related: Pass 1 (predictor; separate scope, still signed off)
- Pre-commitments honoured: none (new scope)
- Pre-commitments newly issued: the one above
