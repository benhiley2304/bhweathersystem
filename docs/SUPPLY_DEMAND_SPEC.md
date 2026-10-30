# Supply and demand zones: rules, calibration and evidence

Added 30 Oct 2026 (v12). Code: `bhws_core/zones.py` (detector), `bhws_zones_api.py` (`GET /api/zones`), the
Levels panel in `index.html` (`v12 js: zones`), research in `research/zones/`.

The zones are candidates for Ben's chart. They mirror how he draws levels by hand in the BH Macro reports
("1D BH ZONES", "1W · COT BH ZONES"). They are not signals, the quality score is not a probability, and the
replay below did not find a payoff lift from zone confluence on weekly bars. Entries, stops and targets stay
his: the panel pre-fills a plan from a zone when asked and never saves it.

## 1. What a zone is (the rules the detector applies)

Read from 32 zones hand-marked across Issues 01 to 11 (Jul to Oct 2026):

- A zone is the whole consolidation (the base) immediately before an impulsive leg out. Demand when the leg
  is up (drop-base-rally DBR, rally-base-rally RBR), supply when it is down (rally-base-drop RBD,
  drop-base-drop DBD). The leg in decides D or R by its sign; a leg in under a quarter ATR is noted as weak.
- The proximal edge is the body extreme of the base nearest to price; the distal edge is the wick extreme
  behind it. Ben sometimes draws wick to wick on a long base (sugar 13.2 to 15.4); the wide scale covers that.
- Two scales per timeframe. Tight: bodies, capped near one ATR weekly or two daily, at most 8 (weekly) or 15
  (daily) bars. Wide: the full consolidation including wicks and the bars around it, up to three ATR weekly or
  five daily, at most 16 or 45 bars. "Supply within supply" is a tight zone inside a wide one.
- The leg out must be impulsive: close-to-close move of at least 1.0 ATR (weekly) or 1.2 ATR (daily) within
  three bars and at least 0.9 (weekly) or 0.8 (daily) times the base height. Close-to-close is used because
  Yahoo opens are unreliable on several futures and FX series.
- The base is built backwards from the leg-out bar and stops at a bar whose own range spans the band once the
  band is real (two bars and half an ATR); every extension length is a candidate and the longest one that
  still passes the departure test is kept.
- Fresh: no bar has reached the proximal edge since the leg out. Tested n: reached n times without a close
  through the distal edge. Broken: a close beyond the distal edge. Flips are not modelled.
- Stop convention from the reports: a quarter of the zone height beyond the distal edge (NQ: zone 28,800 to
  29,563, stop 28,631.75). The "Use this zone" button applies exactly that and labels it as a suggestion.
- Nesting: a daily zone inside a weekly one is tagged "inside weekly".
- Score 0 to 100, ordinal: leg-out strength against the base (35) and in ATR (20), tightness (15), reversal
  shape over continuation (10), freshness (20, decaying with tests), quartered when broken. It ranks the list.

Not modelled: 4-hour and hourly zones (Ben uses them for entries), volume, zone flips, the discretionary
"rejection candle inside the zone" entry.

## 2. Calibration against Ben's zones (in-sample)

`research/zones/validate.py`: 28 of 32 hand-marked zones overlap at least 50 percent with a detected zone of
the same kind that existed before the report date. Misses: NQ daily 29,220 to 29,800 (Jul), NQ daily 30,370
to 30,600, cotton daily 83.3 to 89 and 71 to 77.5. Median edge overlap is about 51 percent, so the match is
on the band, not the exact edges.

This is a fit, not a validation: the parameters were tuned on the same 32 zones. It says the detector speaks
Ben's language; it does not say it predicts anything.

## 3. Data

Yahoo Finance: six years daily, fifteen years weekly, in memory for six hours per market. Continuous
futures where the backend scores a cash proxy (NQ=F for ^NDX, ES=F, YM=F, RTY=F, 6E=F, 6B=F, 6J=F, 6A=F,
6C=F, 6S=F, 6N=F, 6M=F; DX-Y.NYB for the dollar index). Known defects: SI=F daily has about 14 percent
single-print bars (open = high = low = close), JPY=X and other spot series have bad opens and closes. The
panel shows a note when more than 5 percent of daily bars are single prints and recommends the weekly zones.
Weekly bars are the reliable layer.

## 4. Evidence (replay 2012 to 2026, 57 or 58 markets)

All replay history, no costs or slippage, nothing here is a probability for a live trade.

### 4.1 First visit to a zone, no other filter (`research/zones/replay.py`)

20,825 zones. Held = the close travelled one zone height away before any close through the distal edge,
within 8 weekly or 20 daily bars of the first visit. Held about 45 percent weekly and 48 percent daily;
failed about 45 percent. Tight zones held more than wide (49 vs 40 percent), strong leg-outs more than weak
(48 vs 39 percent at 1.5x the base), short bases more than long (47 vs 34 percent for 2 or fewer vs 5 or more
bars). A shifted control band (the same band moved 1.5 heights away from the base) held less, 34 percent,
but the control is unfair for weak leg-outs, and on the clean subset (leg out 2.5x or more) the zone and the
control held the same, 49 vs 48 percent. Traded blind at the proximal edge with the quarter-height stop and
a 2R target, every weekly zone paid profit factor 1.01 at 8 bars and 1.02 at 13 bars (`blind_first_touch_13w.json`).
Zones on their own are not an edge in this data.

### 4.2 Zones with a Desk read (`research/zones/confluence.py`, 27,076 graded setup weeks)

Three executions were compared on the same setups:

1. Resting order at the nearest fresh weekly zone ahead of price (within 2 ATR), filled within 4 weeks,
   entry at the proximal edge, stop a quarter height beyond the distal, 13 weekly bars. Point in time.
   Orders 11,412, filled 39 percent. 2R: win 39 percent, profit factor 1.13, +0.08R (n 4,493). 3R: 1.06.
   Prime 1.18 at 2R. Halves 1.04 (2012 to 2018) and 1.22 (2019 to 2026), so not robust. Counter-trend 1.35,
   With trend 1.43, Crowd fade 1.24; Pullback in trend 0.96, Exhaustion fade 0.68. This is level with the
   engine's Monday-close entry at a 1-sigma stop (all graded 1.15 at 2R).
2. Entry at the signal Friday close while inside the zone or within a quarter height of it: fresh 1.05 at
   2R, tested 1.04, blind baseline 1.02. No lift.
3. The trap: entering at the proximal edge intrabar during the signal week and keeping only weeks that did
   not close through the zone gives fresh 1.39 at 2R and 1.55 for prime. Both choices use information from
   after the entry. It is reported here so the number is never quoted as evidence.

### 4.3 The waiting trade-off

Split the resting-order setups by whether price came back to the zone. When it did (39 percent), the
engine's Monday-close entry lost badly (profit factor 0.27) and the zone entry rescued the trade (1.13).
When it did not (61 percent), the Monday-close entry paid 2.16 and the zone trader was never in. Per order:
+0.08R from entering at the close, +0.03R from waiting for the zone. The split is by a future event, so it
describes the trade-off, not a rule: the zone fixes where you are wrong cheaply; it does not pick the
trades that work, and waiting forgoes the runners. Median risk at the zone was 0.94 ATR, similar to the
engine's 1-sigma stop.

### 4.4 What this means for the panel

The Levels panel is a location and risk tool: where the bases are, which are fresh, how strong the leg out
was, and what the plan looks like against them. The evidence line in the panel says so. A finer test (4-hour
zones, daily fills, costs) could change the picture; it has not been run.

## 5. Roadmap

- Daily-bar fills for the resting order (weekly bars resolve same-bar ambiguity against the trade).
- Zone flips and "rejection candle inside" entries.
- A Pine indicator that draws the same rules on TradingView, so chart and tool agree.
- Cost and slippage assumptions per market in every profit factor.
