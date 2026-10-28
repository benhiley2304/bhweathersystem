# The Desk

The Desk is the default landing view (toggle Desk | Markets in the nav; the choice is kept in `localStorage` as `bhws_view`). It reads the whole book at once and sorts it by how the evidence lines up. It does not replace the market view or the House read; every row opens the market.

Endpoint: `GET /api/desk` (`?refresh=1` forces a rebuild). Cold builds run in the background and return 202 until ready, usually 20 to 40 seconds after a deploy. Results are cached for 30 minutes and rebuilt when the scores update.

## Pillars and roles

The Desk follows the report hierarchy rather than the composite score (`bhws_core/strategy.py`). Each pillar has a role.

| Pillar | Role | Source and rule |
|---|---|---|
| Positioning | Payoff | CFTC futures-only, 156-report mid-rank. Three parts: a playbook extreme (stronger when commercials sit on the other side), a fund manager divergence (managers peaked at the 80th+ 3 or more reports ago and are 10+ points off it while price makes a new 26-report extreme beyond where it stood at that peak), and structure (commercials at 20 or below against a long crowd, or the mirror). For equity indices it is a size warning only, never the side. |
| Valuation | Payoff | Port of `BHWS_ValuationTool.pine` (ROC 11, rescale 100) on daily and weekly bars. References: ZB and DXY for indices and commodities, ZB, gold and euro for the dollar, ZB and gold for FX, DXY and gold for bonds. A reference counts when both timeframes agree (one at 40+, the other 20+ the same way) or one is at 40+ and the other quiet; opposite reads are shown as split and count for neither side. |
| Trend | Context and timing | Engine momentum detail. Running, pullback (last four weeks against an intact trend), extended (outer 15% of the 52-week range with the 13-week move stalled) or range. A healthy trend against the side vetoes Prime and Setup; an extended one does not count against a fade. |
| Seasonality | Minor tilt | Forward path from today: weighted median of each past year's move over the next 30, 60 and 90 trading days, crossing the year end, using the chart's blend weights. Read on 60 TD with 30 TD as the check. Positive means price tends to rise. Inside a small band (median under 0.15 of the half-IQR or under 0.2%) it reads flat to up or flat to down and does not count. |
| Macro | Hit rate | Engine macro direction. Against the side it lowers the hit rate and shows a size-down note; it never blocks a grade on its own. |

## Grades and setup types

Prime: two payoff sources agree, or one plus a healthy trend and the seasonal, with seasonality not against. Setup: one payoff source with support (trend, seasonality or macro), an exhausted trend, or a live playbook extreme. Developing: a payoff source fighting a healthy trend or with nothing behind it. Backdrop only: trend and macro agree with no payoff edge. A payoff pillar firmly on the other side, low data confidence, or macro and seasonality both against each cost one grade.

Setup types: Crowd fade, Divergence fade, Exhaustion fade, Pullback in trend, Trend with value, With trend, Counter-trend, Range. Confirmation is stated per type: the playbook state for crowd fades, a completed weekly close through the 10-week average for fades and counter-trend ideas, and the daily buy point (or both daily lines under -40, "In zone") for index pullbacks.

The ordering number weights positioning 3 (indices 1), valuation 2.5, trend 2, macro 1.5 and seasonality 1. It is a sort key only: the weights encode the written hierarchy and were not fitted or backtested, and nothing here is a probability. Zones and stops are always Ben's.

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
