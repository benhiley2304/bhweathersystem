# Setup score and seasonal timing: replay test

Data: the point-in-time Desk replay (research/replay), 57 markets, weekly Fridays 2012 to 2026, rows where the Desk had a side (26,901 rows with a seasonal path). Entry at the first close after Friday t. R = side-signed return over h weeks divided by the trailing 52-week weekly standard deviation times sqrt(h). "After drift" subtracts each market's average R times the side, so a long-only tailwind does not count as skill. CIs are 90% moving-block bootstrap over signal dates (13-week blocks).

Design sample: 2012-2018. Held out: 2019-2026. The points table was fitted on 2012-2018 only.

Disclosure: the seasonal timing state thresholds (1 sd veto, 1.5 sd leg, 25 TD look-ahead) were chosen after a first look at both halves split by state. Only the point values are clean out-of-sample. Treat the timing state results as partly in-sample.

## Seasonal path (point in time)

For each signal the forward seasonal path F(k), k = 0..80 trading days, is the weighted median of each completed year's move from the same trading day (years before t only, half-life 15 years, same-cycle years weighted 2.5x for index, FX and rates markets, 1.5x for FX crosses). g(k) = side x F(k) / weekly sd.

| Timing state | Rule | 2012-18 R4-8 | 2019-26 R4-8 | After drift 12-18 / 19-26 |
|---|---|---|---|---|
| Against | g falls 1 sd within 20 TD or 40 TD, no window ahead | -0.113 | -0.186 | -0.089 / -0.153 |
| Wait for window | g20 below -0.4 sd, but a dip within 25 TD is followed by a 1.5 sd leg | -0.080 | 0.000 | -0.079 / +0.004 |
| Approaching | dip 4-25 TD ahead then 1.5 sd leg, not against now | -0.013 | +0.143 | -0.034 / +0.105 |
| In window / with / flat | none of the above | +0.045 | +0.021 | +0.023 / 0.000 |

Seasonal against is the strongest and most stable finding here. It held in both halves at 4, 8 and 13 weeks.

Delayed entry (enter at the seasonal dip instead of now):

| State | Now 12-18 | At dip 12-18 | Now 19-26 | At dip 19-26 |
|---|---|---|---|---|
| Wait for window | -0.080 | +0.005 (diff +0.09, CI +0.03 to +0.16) | 0.000 | +0.051 (diff +0.06, CI -0.04 to +0.15) |
| Approaching | -0.013 | +0.030 | +0.143 | +0.113 |
| Against | -0.113 | -0.098 | -0.186 | -0.113 |

Waiting for the window rescued the "against now, window ahead" rows in both halves. It was significant in the design half only. For "approaching", delaying helped in 2012-18 and hurt in 2019-26, so it is not a rule. Waiting did not rescue outright against rows.

## Setup score

Score = 5 + structure + trend location + positioning + valuation + timing, clipped 0 to 10 in half points. Each level's points = its 2012-18 mean after-drift R4-8 minus the overall mean, shrunk by n/(n+400), one point per 0.03 R, capped at plus or minus 3.

| Score | 2012-18 n | R4 | R8 | After drift | 2019-26 n | R4 | R8 | R13 | After drift (90% CI) |
|---|---|---|---|---|---|---|---|---|---|
| 0-3.5 | 4342 | -0.083 | -0.078 | -0.106 | 5090 | +0.027 | +0.039 | +0.081 | -0.008 (-0.07, +0.05) |
| 4-5.5 | 2978 | -0.026 | -0.047 | -0.032 | 2951 | +0.015 | -0.014 | +0.039 | -0.002 (-0.05, +0.06) |
| 6-7 | 2224 | +0.064 | +0.062 | +0.049 | 2668 | +0.039 | +0.034 | +0.053 | +0.031 (-0.03, +0.09) |
| 7.5-10 | 3027 | +0.076 | +0.066 | +0.058 | 3621 | +0.049 | +0.048 | +0.047 | +0.038 (-0.02, +0.10) |

Score 6+ minus score 3.5 or lower, after drift: +0.16 (CI +0.10 to +0.23) in the design half and +0.04 (CI -0.04 to +0.12) held out. Before drift, held out: +0.01 (CI -0.08 to +0.09). Hit rates at 8 weeks are about 50% in every held-out bucket.

The old grade (2019-26, after drift): developing +0.052, prime -0.043, setup -0.044. It sorted backwards. The new score sorts in the right order at 4-8 weeks after drift, but the held-out spread is small and not significant. At 13 weeks it does not sort.

Honest read: the score is a better ranking aid than the grade, and its strongest part is the seasonal veto. It is not evidence of a tradable edge. Hit rate does not change.

## Not tested

- Macro and the engine composite (the bias) cannot be rebuilt point in time over 2012-26, so they carry no points. The bias-versus-setup relation is shown as context only. The closest proxy, setups against the trend, sorted better than setups with the trend in both halves.
- Hold horizon (2-4 vs 4-8 weeks) comes from the seasonal leg length. No archetype showed a clear best horizon.
- Event gates and the data-confidence cap are rules, not tested effects.

Scripts: seas_pit.py (path), feat.py (features), an2.py (states, delayed entry), pts2.py (points and buckets), diff.py (spread CIs). Output: points_and_results.json.
