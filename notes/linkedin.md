# LinkedIn post draft

Attach: figures/hero.png (1440x1800, 4:5 portrait)
Optional carousel: hero.png, fingerprint.png, xt_surface.png

---

The World Cup kicked off this week in 30C+ afternoon heat, which means you
will be seeing a rule most fans ignore: the cooling break. Around minute 30
of each half, the referee stops play for about three minutes, and both
benches get the one thing football never gives them: a mid-half talk with
the whole team. A hidden timeout.

I wanted to know what coaches do with it. Nobody labels these breaks in any
public dataset, so I built a detector: a cooling break shows up in event
data as a 90+ second dead-ball gap around minute 25-31 of a half, with no
goal, injury or card to explain it.

It works. Across 551 matches of free StatsBomb data:

- Indian Super League: 0.44 firings per match. Copa America 2024 and AFCON:
  0.31. The 2019 Women's World Cup heatwave: 0.23.
- The air-conditioned 2022 World Cup: 0.11 per match, pure noise, my
  measured false-positive floor. Russia 2018 lands at exactly the same 0.11.
- The famous 34C quarter-final between Italy and the Netherlands in 2019,
  which reporters noted needed several cooling breaks: the detector finds
  both of them, blind, from timestamps alone.

Then I trained an Expected Threat model from scratch (931k passes and
carries, value iteration on a pitch grid) to measure momentum around the
89 detected breaks, and I almost published a great story.

The naive numbers: 45% of breaks are followed by a substitution or shape
change within five minutes, versus 26% without a break. Substitutions
during the pause: triple the control rate. Coaches use the pause as a free
timeout. Post written, chart rendered.

Except breaks live at minute 75. So do substitutions, break or no break.
My break sample was two-thirds second halves; my control sample was
two-thirds first halves. Compare like halves with like and the timeout
effect collapses: 45% vs 38%, confidence interval straddling zero. The
second-half sub wave was always coming; the break just happens to stand
next to it.

What actually survives:

1. Substitutions migrate INTO the pause. About 2x the expected rate of subs
   happen during the dead time itself. Coaches do not make extra changes;
   they make the same changes earlier and for free.
2. Momentum does not care. The team dominating before the break stays
   dominant after it as often as in control windows (33% vs 39% flips).

So when you watch a cooling break this summer and the commentator says it
changed the game, the data says: probably not. The bench got busier, the
match stayed the same.

The bigger lesson is about analytics, not football: my first result was a
clean, significant, completely wrong composition artifact. If your
treatment group lives at minute 75 and your control group lives at minute
30, you are not measuring the treatment.

Code, pipeline and figures: github.com/d8maldon/futbol_tech

---

Notes for posting
- post in the first days of the tournament while cooling breaks are in the
  news cycle; the first hot afternoon match is ideal
- alt text for the image: momentum chart of the Italy vs Netherlands 2019
  quarter-final with both detected cooling breaks highlighted, plus the
  naive vs corrected substitution statistics around cooling breaks
