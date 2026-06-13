# hidden-timeout

Football has no timeouts. Except since 2014 it quietly does: when it is hot
enough, the referee stops play around minute 30 of each half for a cooling or
drinks break, and both benches get a mid-half conversation with the whole
team. This repo detects those breaks in free StatsBomb event data and
measures what they actually change.

Built while the 2026 World Cup, played in 30C+ North American summer
afternoons, kicks off.

## Finding the breaks

No public dataset labels cooling breaks. But they leave a fingerprint: a
dead-ball gap of 90 seconds or more, starting around minute 25-31 of either
half, with no goal, injury or card to explain it.

![fingerprint](figures/fingerprint.png)

Detector firings per match, by tournament:

| tournament              | firings/match | reading                    |
|-------------------------|---------------|----------------------------|
| ISL 2021/22 (India)     | 0.44          | breaks in most matches     |
| Copa America 2024 (USA) | 0.31          | hot-venue matches          |
| AFCON 2023 (Ivory Coast)| 0.31          | hot-venue matches          |
| WWC 2019 (France)       | 0.23          | heatwave weeks             |
| Euro 2024               | 0.14          | ~ background               |
| Euro 2020               | 0.12          | ~ background               |
| WC 2018 (Russia)        | 0.11          | = background, exactly      |
| WC 2022 (Qatar, AC)     | 0.11          | the measured noise floor   |
| WWC 2023 (winter)       | 0.09          | ~ background               |

The air-conditioned 2022 World Cup defines the false-positive floor: 0.11
firings per match (VAR checks, slow restarts). Hot tournaments run 2-4x that,
and only those four enter the break set: 89 breaks, estimated 69% genuine.

Validation against ground truth: the WWC 2019 quarter-final (Italy 0-2
Netherlands, 34C in Valenciennes) was reported at the time as requiring
cooling breaks. The detector, knowing nothing but event timestamps, finds
both: minute 29.9 of the first half and 74.5 of the match, right where the
FIFA protocol puts them.

![hero](figures/hero.png)

## World Cup 2026, live

StatsBomb-grade event streams for this tournament will not be public for
years, but two live feeds cover the analysis:

- FIFA's timeline API logs every hydration break officially (event type 83,
  "Match paused for a hydration break") and the resume (type 78), both with
  millisecond wall clocks, so break durations and subs-made-during-the-pause
  are exact, not inferred.
- ESPN's commentary describes every shot with a location phrase ("centre of
  the box", "very close range"). Those zones are calibrated against the
  13.6k open-play shots in the historical data (very close range 0.29,
  centre of box 0.13, outside 0.035, 35+ yards 0.007) to weight each chance,
  which is enough to draw the momentum rivers for live matches.

`src/wc2026.py` pulls every finished match, writes small CSVs to `wc2026/`,
and regenerates the tracker plus one river per match:

![tracker](figures/wc2026_tracker.png)

Through day 2: three matches, six official breaks, one in each half of
every match including an 8pm kickoff, all called at minutes 23-25 of the
half, a few minutes earlier than the historical norm, and zero substitutions
made during the pauses so far. The opener:

![opener](figures/wc2026_river_mexico_south_africa.png)

```
python src/wc2026.py
```

### Live win-probability eval (the chess-engine view)

`src/winprob.py` trains an in-game win-probability model on the 551
historical matches: multinomial logistic regression on goal difference,
cumulative xG difference, red-card man advantage and time remaining,
predicting P(home win / draw / away win). It is well calibrated (log loss
0.79; predicted vs observed home-win rate within ~2 points across every
decile) and saved as plain JSON so scoring needs only numpy.

`src/live_eval.py` drives it from FotMob's per-shot xG and event feed to
draw a chess.com-style evaluation bar for each match: one line, up means
the home side is favoured to win, down the away side, zero a coin flip. It
moves on goals, red cards and the slow drip of chance quality, and the
single biggest swing is annotated the way an engine flags the losing move,
with the narrative chosen from what actually happened (a collapse is called
a collapse, a comeback a comeback).

![eval](figures/wc2026_eval_south_korea_czechia.png)

Czechia led 1-0 from the 59th minute and the eval crossed into their half;
South Korea equalised at 67' and won it at 80'. The model reads the
turning point as the equalizer, not the winner. Per-match swings are
written to `wc2026/winprob_swings.csv`.

```
python src/live_eval.py            # every finished match so far
python src/live_eval.py 4667757    # a single FotMob match id
```

## The trap: breaks live at minute 75

The naive analysis is seductive. Pool everything and you get: 45% of breaks
are followed by a substitution or formation change within 5 minutes, against
26% in no-break control windows. Substitutions during the pause run 3x the
pooled control rate. It looks like coaches treat the pause as a free timeout.

It is mostly composition. Second-half breaks sit at minute 75, which is
prime substitution time with or without a break, and the break set is 63%
second halves while the control set is 61% first halves. Comparing like
halves with like:

- sub or shape change within 5 min: 45% vs 38% (control standardized to the
  break period mix), 95% CI on the gap -3 to +16 points. Not significant.
- second half only, match-clustered: 70% vs 57%, CI -3 to +28 points.
- first half only: 3% vs 7%. Nothing.

## What survives

- Substitutions migrate into the pause itself. 0.55 subs are made during the
  average break against 0.28 expected (period-standardized), roughly 2x, CI
  +0.07 to +0.49. Coaches use the dead time to make the changes they were
  about to make anyway. (Read with care: control halves cannot contain long
  sub stoppages by construction, and some detected breaks are themselves
  substitution stoppages, so part of this gap is selection.)
- Momentum does not detectably move. The team dominating threat (xT + xG)
  before the break stays dominant after it as often as in control windows:
  flips 33% vs 39%, CI -18 to +5 points; mean swing in threat share is
  indistinguishable. With 69% purity this is low-powered against subtle
  effects, so it is a no-detectable-effect result, not proof of absence.
- Formation changes during the pause: no detectable excess (CI -11 to +3
  points).

## Method

1. `download.py`, `extract.py`: 551 matches, 10 tournaments, ~1.9M events
   from [StatsBomb open data](https://github.com/statsbomb/open-data) into
   compact CSVs: dead-ball gaps with context, substitutions and tactical
   shifts, moves and shots.
2. `xt_model.py`: Expected Threat from scratch. Markov reward process on a
   16x12 grid, transitions estimated from 931k move attempts (failed moves
   absorb to zero; shootout and penalty kicks excluded from the shot model),
   solved by value iteration. `winprob.py` adds the in-game win-probability
   model used by the live eval graphs.

   ![xt](figures/xt_surface.png)

3. `analyze.py`: classifies pauses, then compares break windows against
   pseudo-breaks placed at the median break minute in halves with no long
   mid-half pause. Both arms are measured identically (actions during the
   pause interval and in the 5 minutes after, threat in 10-minute windows
   before and after). Everything is reported per period and standardized to
   the break period mix; bootstrap CIs resample matches, not windows.

## Limitations

- The detector is statistical. Purity is ~69% in the break set (75% ISL,
  ~65% AFCON and Copa, ~53% WWC 2019), measured against the WC 2022 floor.
  Contamination shrinks true differences toward zero: conservative for the
  positive findings, but it also weakens the momentum null.
- Control halves are quieter than break halves by construction (no long
  stoppage of any kind mid-half), which biases tactical comparisons in
  favor of finding a break effect; the null survives anyway.
- Threat windows are 10 minutes; slower payoffs are invisible.
- Tournaments only, and heat co-varies with competition, squad depth and
  stakes. Period standardization removes the largest confound, not all.

## Reproduce

```
pip install -r requirements.txt
python src/download.py    # ~1.5 GB of event JSON
python src/extract.py
python src/xt_model.py
python src/analyze.py
python src/make_figures.py
```

## References

- StatsBomb open data, https://github.com/statsbomb/open-data
- K. Singh, Introducing Expected Threat, 2018,
  https://karun.in/blog/expected-threat.html
- FIFA cooling break protocol (introduced at the 2014 World Cup; WBGT
  threshold, breaks around 30' and 75')
- Heat and cooling breaks at the Italy-Netherlands WWC 2019 quarter-final:
  contemporary reports, e.g. The Globe and Mail, 2019-06-28
