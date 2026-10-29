# Desk rules v2

Live from 29 October 2026. Selected with `BHWS_RULESET` (default `v2`; set `v1` to run the old ladder). Both ladders are computed on every Desk build: `class` is the live grade, `class_v1` the old one, and the pair is written to the grade record (`ledger_grades`) once per market per day so the change can be judged forward rather than assumed.

## What changed

Four rules, each taken from a finding in `research/replay/REPORT.md` (2012 to 2026 held-out replay, 57 markets, 13-week hold). The replay numbers below are what was found there. They are the reason for the rule, not a forecast for it.

1. Counter-trend is no longer capped. Under v1 a healthy trend against the side held any setup at Developing. Under v2 a setup with a payoff anchor (positioning or valuation firmly with the side) grades on its anchors and supports like any other row. Without an anchor it is still nothing to do. Replay: payoff sources against a running trend, before the weekly close turned, +0.097 R, +0.09 after drift, positive in both halves. Waiting for the 10-week close to turn took it to -0.01.
2. Seasonal against caps Prime and Setup at Developing. "Against" here is the seasonal pillar read: the 60 trading day forward seasonal path (12 weeks) runs against the side by a meaningful share of its own cross-year spread, with the 30 TD path not pointing the other way. This is the measure the replay tested. It is not the 4 to 8 week timing state used by the setup score (side-signed path at 20 and 40 TD in weekly standard deviations), so a row can show "seasonal flat" on timing and still carry this cap; the cap text quotes the 60 TD median so the two can be read side by side. Under v1 it only blocked Prime. Replay: -0.11 R in 2012 to 2018 and -0.19 R in 2019 to 2026, negative at 4, 8 and 13 weeks.
3. Divergence-fade shorts are capped at Developing. Replay: -0.27 R, -0.10 after drift, negative in both halves, median adverse run 0.75 against a favourable run of 0.47.
4. Crowd fades after the managers have turned are capped at Developing. The turn closes the window rather than confirming the setup. Replay: -0.07 R after the turn, -0.08 after drift, negative in both halves; the same fades before the turn made +0.04.

Nothing else moved. No weights were refitted, the setup score is unchanged, and the composite bias is untouched.

## What a cap is

A cap is a veto with a stated reason. It lowers the grade to Developing and nothing else: the side, archetype, setup score, timing state and pillars are reported exactly as before. The reasons are returned on the Desk row as `caps` (a list of strings), shown on the market read under the Desk line and as a small "capped" tag on the board. A capped row never disappears, so the setup is still visible while the reason for not promoting it is on screen.

Downgrades (payoff pillar firmly on the other side, macro and seasonality both against, low data confidence) still apply before the caps, as under v1.

## Confirmation text

Under v2 the counter-trend playbook keeps the same state keys (Waiting, Confirmed) so nothing downstream breaks, but the label and text now describe the replay evidence: "Early window" while the weekly close is still against the side (the window the replay found payoff in) and "Turned" once it is through (the window closing). Crowd fade reports the window as open or closed by the managers' turn.

## Forward record

Every Desk build appends one row per market per day to the grade record with `grade`, `grade_v1`, side, lean, archetype, confirmation state, caps and the close on the day. Rows are pruned after three years. The record is the basis for the forward comparison of the two ladders; until it has enough history the Edge lab shows this week's counts under each rule set and nothing more.

The v1 ladder stays in the code as the shadow rule set. It is not deleted because the replay that motivated v2 was fitted on the same sample it was found in; the forward record is the test.

## What this is not

- Not a backtest of v2. The rules are drawn from the replay, so applying them to the same history would only restate it.
- Not a change to expectancy. The replay R figures are averages over past signal weeks with no costs, roll handling or carry.
- Not a level. Caps say what not to promote; zones, entries and stops remain the trader's.
