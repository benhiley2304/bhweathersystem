# Payoff profiles ("does it pay"), replay 2012 to 2026

Built 29 Oct 2026 (execution models added 30 Oct) from the point-in-time Desk replay (`research/replay`, fixed variant). 27,076 graded rows
(prime, setup, developing with a side) across 57 markets. Script: `build.py`. Artifact: `payoff_profiles.json`,
served by `GET /api/payoff` and `GET /api/payoff/table`. Module: `bhws_core/payoff.py`.

## What was measured

For every graded row: entry at the close of the first trading day after the signal Friday, daily path for
91 calendar days, side-signed, in units of the market's trailing 52-week weekly standard deviation (one sigma).
Stops at 0.5, 0.75, 1.0 and 1.5 sigma; targets at 2 and 3 times the stop. First passage is tested on daily
closes, so intraday hits are not counted. MFE and MAE at 1, 2, 4, 8 and 13 weeks (25th, 50th, 75th
percentiles). "Dip half R" is the share of paths that traded at least half the stop distance against the
entry inside two weeks before any target. Drift rows remove each market's own in-sample mean daily return.
Halves are 2012 to 2018 and 2019 to 2026. No costs, no slippage, continuous futures with roll gaps.

Baseline: the same rules run on every Friday since 2012, both sides, for the same markets (84,854 paths).
For a driftless walk the 2R-before-1R share is one third and the 3R share one quarter, before the time cap.

Cells: archetype, archetype x side, archetype x side x confirmation state, and that x asset class, plus
each market's own history (n >= 15, labelled as colour). The panel leads with the most specific cell that
has at least 120 rows and lists the thinner ones. Binomial standard error on a share near one third is about
2 points at n = 500 and 1 point at n = 2,000, so differences of a few points inside one cell are noise.

## Findings

Negative first.

1. **The ladder as a whole does not change the first-passage odds.** All graded rows at a 1 sigma stop:
   2R first 34.8%, stop first 61.9%, unresolved 3.3%. Any week, both sides: 34.0%, 62.4%, 3.6%. Median
   13-week run for the side 2.06 sigma against a run against of 1.92 (skew 1.07); any week 1.97 and 1.97.
2. **Grade does not sort payoff.** Prime 34.9%, setup 32.6%, developing 36.0% at 1 sigma. This matches the
   replay report; the setup score orders the board, the old grade does not.
3. **Stop size does not buy odds.** At 0.5, 0.75 and 1.0 sigma the stop-first share is 62% to 63% in every
   case, because the target is always twice the stop. Wider stops (1.5 sigma) push the 2R target to 3 sigma,
   which 15% of paths never resolve inside 13 weeks, so the 2R-first share falls to 28.6%. There is no magic
   stop distance in the data; the zone decides the stop, the table only says what that distance did before.
4. **"Wait for the pullback" is path geometry, not a setup effect.** 55% of graded rows traded at least half
   the stop distance against the entry inside two weeks before any target. Any week: 56%. It supports
   entering at a zone rather than at the signal close, but it is true of almost every week, not of setups.
5. **Longs beat shorts almost everywhere, which is the decade.** All eight long classes sit at or above the
   baseline, six of eight short classes at or below. After drift removal the long advantage shrinks (with
   trend long 34.0%, pullback long 33.5%, crowd fade long 32.8%, all at or under the baseline).
6. **Classes above the baseline in both halves and after drift at 1 sigma:** counter-trend long (n 2,120:
   40.1%; 37.3% / 42.0%; drift 36.8%; skew 1.45), divergence fade long (n 585: 39.0%; 36.2% / 42.3%; drift
   36.9%) and exhaustion fade long (n 505: 38.0%; 39.2% / 36.5%; drift 36.0%). All three are longs bought
   against a falling move. Counter-trend long is the one with the sample size to mean something, and it
   agrees with the replay's "early counter-trend" finding. Before the weekly close turns (waiting) it is
   40.1%; after it has turned (confirmed, n 217) 35.1%.
7. **Classes below the baseline in both halves and after drift:** divergence fade short (n 680: 27.1%; skew
   0.62), exhaustion fade short (n 523: 27.2%; skew 0.72) and pullback in trend short (n 1,542: 31.3%).
   Divergence fade short was already the worst cell in the replay report; this says the same thing in R terms:
   the typical run against (2.9 sigma at 13 weeks) is bigger than the run for (1.8 sigma).
8. **Crowd fade timing.** Long, before the managers turn: 38.0% (35.9% / 40.5%; drift 34.0%). Long, after
   the turn: 33.1% (29.7% / 36.0%; drift 29.3%). Short after the turn: 30.6% (drift 33.0%, skew 0.69).
   Consistent with the replay's "fades after the turn are late".
9. **Context modifiers (reported, not applied).** Seasonal against 32.3% against seasonal with 37.2%
   (skew 0.89 against 1.21). Trend exhausted against the side 32.9%. Positioning against 37.2%.
10. **Asset-class cells worth knowing about, with the usual small-n warning:** counter-trend long waiting in
    FX crosses (n 456: 46.3%; 48.0% / 43.5%; drift 45.4%) is the cleanest above-baseline cell because crosses
    carry little drift. Pullback in trend long confirmed in equities (n 201: 49.3%) is the bull market.
    Crowd fade long not turned in bonds (n 287: 23.3%) and divergence fade short waiting in equities (n 134:
    16.4%) are the worst; both are fading a decade-long move.

## Execution models and profit factor (added 30 Oct 2026)

Every path is also scored in R units of the original stop under two execution models, at each stop size and
for 2R and 3R targets. Full: one unit at the entry close, stop at minus 1R, target as set, time-outs at 13 weeks
marked at the final close. Half then add: half a unit at entry; when a close reaches plus 1R before minus 1R
the other half is added at that close and the stop for the whole position moves to the lowest close of the
prior ten sessions (never below the original stop); the target stays at the original 2R or 3R.

Profit factor is gross R won divided by gross R lost across all trades in the cell, including time-outs.
1.0 is break-even before costs. For a fixed T:1 payoff with win rate w, PF = wT / (1 - w), so 34% at 2:1 is
1.03 and 24% at 3:1 is 0.95. Expectancy is the mean R per trade.

1. **The any-week row is above 1.0.** At a 1 sigma stop and 2R target, every Friday since 2012 on both sides
   gives PF 1.12 and expectancy +0.07R full size, 1.08 and +0.03R scaled. The sample drifts, time-outs are
   marked at the close and no costs are charged, so an absolute profit factor above one is not edge. Read the
   gap to the any-week row.
2. **All graded setups: PF 1.15 full, 1.11 scaled at 2R** (3R: 1.22 and 1.21). Three to four points above the
   any-week row, which is the same picture as the first-passage shares.
3. **Scaling in at 1:1 does not raise the profit factor.** In 14 of 16 archetype x side classes at 2R the
   scaled PF is level with or below the full-size PF (all graded 1.11 against 1.15); at 3R it is level
   (1.21 against 1.22). The add fires in about half of trades, and 37% of those are then stopped at the
   moved stop. What the model does change is the shape: average loss falls from -0.99R to -0.57R and
   average win from 1.92R to 1.31R, expectancy per unit of initial risk is about the same. It is a sizing
   choice, not an edge. Moving the stop under the prior ten-session low is one reading of "below the new
   low"; a breakeven stop or a swing low from intraday data would give different numbers and are not tested.
4. **3R beats 2R on profit factor almost everywhere** (all graded 1.22 against 1.15; any week 1.17 against
   1.12) at the cost of a win rate near 31% and longer holds. Consistent with the positive 13-week skew in
   the long classes.
5. **Ranking by PF matches ranking by 2R-first.** With trend long 1.44, counter-trend long 1.41, divergence
   fade long 1.36 and exhaustion fade long 1.30 at the top; trend with value short 0.99, pullback in trend
   short 0.97, exhaustion fade short 0.79 and divergence fade short 0.77 at the bottom. Shorts sit at or
   below the any-week row almost everywhere, as before.
6. **Stability of the scaled PF by half:** with trend long 1.05 / 1.83 (the second half is the equity bull
   run); counter-trend long 1.23 / 1.38 and divergence fade long 1.23 / 1.35 are steadier; crowd fade short
   1.14 / 0.93 and trend with value short 1.07 / 0.76 decay. After drift removal the long classes fall to
   1.0 to 1.2.

The Edge lab shows the table with a 2R / 3R switch; the market panel shows the two models at the plan's
stop row and target multiple, and the stress read adds a sentence with the same numbers.

## How it is shown

The "Does it pay" panel sits in the market read for any row with an archetype and a side. It shows the
payoff cone (middle half of runs for and against the side at 1, 2, 4, 8 and 13 weeks, with the any-week
medians dotted), the first-passage table at the four stop sizes with the any-week row beneath, the
stability line for the relevant row, the market's own history, and, when Ben has entered levels under
Your plan, a stress test: the nearest stop row, how often a target of that size came first, time to
target, the dip share, and flags for a stop inside the 4-week noise band or a target beyond the 13-week
75th percentile run. The engine never proposes a level. The Edge lab carries the cross-archetype table.

## Caveats

About 180 cells were tested, so a few will stand out by chance. Rules were fitted on the first half with
knowledge of the second (see the replay report). Daily closes understate stop hits. Entries at the signal
close are not how Ben trades; the zone entry changes both the stop distance and the odds, and nothing here
measures that. Replay history for a class of setups, never a probability for a live trade.
