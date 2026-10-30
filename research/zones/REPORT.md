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
