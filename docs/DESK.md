# The Desk

The Desk is the default landing view (toggle Desk | Markets in the nav; the choice is kept in `localStorage` as `bhws_view`). It reads the whole book at once and sorts it by how the evidence lines up. It does not replace the market view or the House read; every row opens the market.

Endpoint: `GET /api/desk` (`?refresh=1` forces a rebuild). Cold builds run in the background and return 202 until ready, usually 20 to 40 seconds after a deploy. Results are cached for 30 minutes and rebuilt when the scores update.

## Pillars

Each market gets five pillars, each marked with, against or neutral relative to the side in question.

| Pillar | Source | Notes |
|---|---|---|
| Positioning | CFTC futures-only, 156-report mid-rank percentile, commercials / managers / retail | Sets the side when an extreme exists (fade the crowd). Same basis as the BHWS TradingView COT indicators. |
| Macro | Engine macro sub-score | Direction. |
| Trend | Engine trend state (Uptrend / Downtrend / Rangebound), falling back to the momentum vote | Whether price agrees. |
| Seasonal | Engine seasonal read | Used only when grade A to C, hit rate agrees with the median sign, and the median is at least 0.2%. Positive means price tends to rise; flipped for shorts. |
| Valuation | Port of `BHWS_ValuationTool.pine` (ROC 11, rescale 100) vs ZB, DXY and GC, weekly | Composite of the ZB and DXY lines. Cheap at or below -40, rich at or above +40. FX skips DXY; bonds skip ZB. |

## Classes

Aligned setup, Positioning only, Positioning vs macro, Backdrop only, Extreme lapsed, Nothing to do. Gates are shown separately: seasonality against (no adds), valuation against, non-USD top-tier events in the next 7 days, and low data confidence. USD events are shown once in the event gate rather than on every row.

The order on the board is a sort key, not a probability. Valuation and seasonality are tilts.

## Valuation buy points (index futures only)

ES, NQ, YM and RTY show the daily Pine buy-point state: BUY when the ZB and DXY lines are both below -40, exit when their average is above -25 or after 10 bars, 2-bar cooldown. The record quoted (NDX 1996 to 2026, 242 trades, 72% winners, PF 1.9, parameters tuned on 1996 to 2010) is taken from the Pine header and has not been re-derived here.

## Plan form

The House read has a plan form (side, starter date, zone, entry, stop, target, costs). It is stored on this device only (`bhws_plan_<id>`) and sent to `/api/evidence` as query parameters. BHWS never fills these in. With a zone it applies the 25% depth stop rule; with levels it reports net reward to risk and the break-even hit rate. Reward to risk alone is not expectancy.

## Limitations

Crosses have no house COT and are classed on macro and trend only. The weekly brief is assembled from the numbers on the page, with no model text. Hidden concentration uses return correlations of the live sides and reports effective bets.

## Market read (per market)

`bhws_core/marketread.py` turns one desk row plus the engine payload into the read shown at the top of each market's Overview tab. It is pure and deterministic: same inputs, same words. No model call.

Output per market (`DESK.rows[].read`):

- `headline`, one line. For "nothing to do" markets it names the split, e.g. "No setup: trend and seasonality lean long and macro leans short."
- `stance`: desk class, side, lean (side if there is one, else macro sign, else pillar count), trigger.
- `summary`: two to four plain sentences.
- `pillars`: positioning, macro, trend, seasonal, valuation. Each has a sign (green up, red down), whether it is with or against the side or lean, short facts and the tab that holds the detail.
- `watch`: what would change it (distance to a fresh extreme, valuation thresholds, seasonal timing, marginal macro votes).
- `events`: upcoming top-tier prints for the market's currencies.
- `engine`: the 1 to 10 composite and the engine's lean, quoted as is and never inverted. If it disagrees with the read, both are shown and neither is adjusted.

A macro vote that clears the 0.5 deadband by less than 1.0 is flagged as marginal. The read is a description of the evidence, not a probability.

## Where the desk shows up

There is no separate Desk view any more. One `/api/desk` build (polled on load, 202 while warming) feeds:

- the sidebar, ordered by setup class by default (toggle to Score for the composite ranking);
- the "This week" band on the home page (brief, live setups, event gate, what changed, concentration);
- the market read on each Overview tab.
