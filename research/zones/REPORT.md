# Supply and demand zones: research notes

Full write-up, rules and caveats: `docs/SUPPLY_DEMAND_SPEC.md`. Scripts here:

- `validate.py`: calibration against 32 hand-marked zones from Issues 01 to 11 (28 of 32 overlap, in-sample).
- `replay.py`: first-visit reactions for 20,825 zones, 58 markets, 2012 to 2026 (`zone_reactions.json`,
  `blind_first_touch_13w.json`). Blind first-touch trading at 2R: profit factor 1.01 weekly, 0.90 daily.
- `confluence.py`: zones combined with the Desk's graded setups (`zone_confluence.json`). Point-in-time resting
  order at the nearest fresh weekly zone: 39 percent filled, 2R profit factor 1.13 (n 4,493), level with the
  engine's Monday-close entry (1.15). The look-ahead version (entry at the intrabar touch plus a filter on the
  week's close) showed 1.39 and is reported as a trap, not a result.

Negative findings first: no blind edge, no measured lift from fresh over tested zones once point in time, no
robust split across halves (1.04 then 1.22). Zones define risk; they do not pick trades.

Second detector, point in time (`replay_v2.py`, `zone_replay_v2.json`, 30 Oct 2026): first visit after the
zone became knowable, resting order at the proximal edge, stop a quarter height beyond the distal, full against
half-then-add, control band 1.5 heights before the zone. Blind: PF 1.05 weekly, 1.10 two-day, 1.12 daily.
Method grade (matrix, base of six or fewer, departure of three heights or more, covered on lower frames): 1.29,
1.27, 1.34, both halves above 1.2; the rest 0.98 to 1.08; the control band pays in the graded contexts too
(1.31 to 1.60), so the lift is location and direction, the zone sets the stop. Loud leg outs (Yahoo volume,
coherent series only) showed no lift. Half then add trailed full size in every cell. Details in
`docs/SUPPLY_DEMAND_SPEC.md` section 0.1.
