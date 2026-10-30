# Supply and demand zones: rules, calibration and evidence

Added 30 Oct 2026 (v12). Code: `bhws_core/zones_v2.py` (the detector the API uses since the second pass,
section 0), `bhws_core/zones.py` (the first detector, kept for the replay in section 4), `bhws_zones_api.py`
(`GET /api/zones`), the Levels panel in `index.html` (`v12 js: zones`), research in `research/zones/`.

The zones are candidates for Ben's chart. They mirror how he draws levels by hand in the BH Macro reports
("1D BH ZONES", "1W · COT BH ZONES"). They are not signals, the quality score is not a probability, and the
replay below did not find a payoff lift from zone confluence on weekly bars. Entries, stops and targets stay
his: the panel pre-fills a plan from a zone when asked and never saves it.

## 0. Second detector (30 Oct 2026, evening): the three-step method

Ben's note on the first Levels chart: scaling and level clarity were poor and the zones did not read like the
ones he draws. He pointed at Bernd Skorupinski's Supply and Demand Pro Trader Series on YouTube as the
reference. The second detector applies that method as taught; the panel was redrawn to match his TradingView
charts (dark, tall, full range visible, the base boxed over the base candles only, the zone extended to the
right as a translucent band, bold major axis ticks, a right margin, one or two zones per side).

Method rules, each cited to the lesson it comes from. These are the method's rules, not evidence of payoff.

- A zone is leg in, base, leg out. Demand when the leg out is up (DBR, RBR), supply when it is down (RBD,
  DBD). "Small bodies build the base, a big body is the exit." Find the leg out first, then the base, then the
  leg in (lesson 6, https://www.youtube.com/watch?v=Hb0QCNABlzc).
- The explosive candle: body above 70 percent of its range and abnormally bigger than the previous few
  candles; or two decisive candles in a row (same lesson). Proximal line across the base bodies, distal line
  across the base wicks; preferred (tight) version from the last one or two base candles, wider version from
  the whole base (in-depth guide, https://www.youtube.com/watch?v=U5BJiBRFkjI).
- Five qualifiers: departure, base duration (six candles or fewer), freshness measured to the right only,
  profit margin of at least 1:2 to the next opposing zone, arrival (freshness lesson,
  https://www.youtube.com/watch?v=ZEhWr_3ehLc). Originality: a base that forms at an older zone of the same
  kind is a reaction, not an original move (method map, https://www.youtube.com/watch?v=I0GO9e5RAuY; supply
  and demand against support and resistance, https://www.youtube.com/watch?v=VV4z8C2lMf4).
- Location on the curve between the nearest weekly demand and supply, split at 0.33 and 0.66, and direction
  from the last swing highs and lows; the action matrix (demand only low, or at equilibrium with an uptrend;
  supply only high, or at equilibrium with a downtrend) is a rule set (mechanical decisions,
  https://www.youtube.com/watch?v=CI9emApkg-g).
- Level on top of level: two zones of one kind whose gap is no taller than either zone are one structure
  (https://www.youtube.com/watch?v=2XvYSbMpznA). "Always trade covered": a lower-timeframe zone inside a
  higher-timeframe zone of the same kind.
- Stop: Ben's convention, a quarter of the zone height beyond the distal edge. Entries and targets are his.

How the code applies them (`bhws_core/zones_v2.py`):

- Sizes are in units of the typical range: a rolling median true range over 20 bars, so one crash bar does
  not hide the next base. Leg out accepted by any of: explosive (range 1.35 times the median of the previous
  five or six ranges and a close-to-close move of 0.80 weekly / 0.85 daily typical ranges), two decisive
  candles (0.55 / 0.60 each), a strong run (2.0 / 2.2 within four to six bars), a slow departure (2.5 within
  six to ten bars without coming back), or the origin of a swing (3.5 within 26 / 40 bars before any close
  comes back through the launch). Ben's own zones include bases left by sustained moves, so the explosive
  candle is sufficient, not necessary; zones not left by one explosive candle carry a note.
- Base, backwards from the bar before the leg out: a bar joins while its body sits at least half inside the
  body band so far (padded by 0.35 typical ranges), the body band stays under 1.7 and the wick band under 3.0
  typical ranges, and no decisive bar widens the band (that bar is the leg in). Up to twelve bars; more than
  six is kept as a wide zone and marked, matching the sugar and bitcoin bases Ben drew.
- Edges: proximal at the body extreme of the base nearest price, distal at the wick extreme behind it (Ben
  draws the wider version body to wick). The preferred proximal edge (last one or two base candles) is kept
  for the freshness rule: a test must reach it; a wick into the wider band only is noted.
- Freshness to the right: visits are counted with hysteresis (a visit lasts until price has left the proximal
  edge by half a zone height), the visit in progress on the last two bars is the arrival and not a test, and a
  weekly test that stayed within a quarter of the preferred version is "shallow". Broken is a close beyond the
  distal edge.
- Originality: a base formed inside an older unbroken zone of the same kind (half its height overlapping), or
  whose wick extreme sits at that zone, is marked a reaction and scored down (never removed). Lookback three
  years weekly, eighteen months daily and two-day. Parts of one stacked structure are not reactions to each
  other.
- Stacking: unbroken zones of one kind join when the gap is no taller than either zone, the structure stays
  within 3.5 typical ranges and 2.5 times its tallest part, the bases formed within 400 days (weekly) or 200
  days (daily, two-day) of each other, and at most three bases. The structure is drawn as one band with each
  base boxed.
- Selection for the chart: a zone price sits inside is always shown; otherwise the anchor per side is the
  nearest fresh or shallow zone; one more joins when it sits close behind the anchor or is tested and closer.
  Two per side at most. Everything else is a candidate behind the "all candidates" toggle.
- Two-day bars are paired from a fixed anchor (the first bar of the earliest full calendar year) so the pairing
  does not shift from one session to the next; Yahoo spot FX daily bars (open printed equal to the close, close
  lagged a day) are repaired before detection and the panel says so.
- Score 0 to 100, ordinal, following the five qualifiers (departure 35, base 20, freshness 25, leg-out candle
  10, originality 10); it ranks candidates and is not a probability.

Calibration against the six charts Ben drew on 30 Oct 2026 (nine zones: SI weekly 45.5 to 50.5; SB weekly
21.0 to 23.6; BTC two-day 75.5 to 78.8k and 58 to 65k; HG daily 5.81 to 6.00; USD/JPY daily 159.4 to 160.4
and 153.0 to 154.3; EUR/USD two-day 1.1085 to 1.1195 and 1.0755 to 1.0845): `research/zones/validate_v2.py`
finds an unbroken detected zone of the same kind with a base inside his base window for all nine. The
parameters were tuned on those nine, so this is in-sample agreement with hand-drawn boxes, not validation and
not payoff evidence. The live panel reproduces them as SI 45.85 to 50.75 (two bases, Oct 27 and Nov 17 2025),
SB 21.08 to 24.62 (two bases; the Sep 23 to Dec 2 2024 base shallow-tested), BTC 74.9 to 79.8k and 62.2 to
64.9k, HG 5.79 to 6.04 (two bases), USD/JPY 159.7 to 160.2 and 152.9 to 154.5, EUR/USD 1.107 to 1.124 on the
research frame; the live Yahoo two-day frame for EUR/USD gives a narrower 1.1141 to 1.1188 because one base
bar's body falls outside the body band (the detector's version of his zone is the preferred one, not the wider).

Not replayed: the second detector has no payoff replay yet. The evidence line in the panel still quotes the
first detector's weekly replay (section 4) and says so. Nothing in sections 0 to 3 is a claim of edge.

## 1. What a zone is (the rules the first detector applied)

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

- Replay the second detector (weekly and daily, first visit and Desk-read confluence) the way section 4 did
  for the first, including the two-day frame.
- EUR/USD two-day: decide whether a doji base bar may carry the previous bar into the base (Ben's wider box)
  without loosening every other base.

- Daily-bar fills for the resting order (weekly bars resolve same-bar ambiguity against the trade).
- Zone flips and "rejection candle inside" entries.
- A Pine indicator that draws the same rules on TradingView, so chart and tool agree.
- Cost and slippage assumptions per market in every profit factor.
