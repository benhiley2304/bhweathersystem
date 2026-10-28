# BHWS evidence layer

The evidence layer runs alongside the existing engine. It does not change `/api/scores` or any score. It adds one read-only endpoint, one dashboard panel and an optional lock on the write routes.

## `GET /api/evidence?market=<ID>`

For a market with a single CFTC contract it returns:

- `positioning`: commercials, fund managers (legacy non-commercial) and retail (non-reportable) on the house basis. That means CFTC legacy futures-only, a trailing 156-report window and the mid-rank percentile, the same method as the BHWS TradingView COT indicators. Each block carries the observation date (Tuesday), the CFTC release date (normally Friday 15:30 ET, with holiday delays) and whether the report is current.
- `cross_check`: the engine's own futures-and-options combined figures, on its 520-week index. They are shown, never reconciled.
- `disagreements`: turn counts that differ by basis, and percentile gaps of 10 points or more.
- `playbook`: the Issue 11 state machine. It covers the extreme at the 90th/10th percentile of 156 reports, the add (two reports turning plus a weekly close through the 10-week average) and the six-report test window. BHWS does not draw zones. Without a starter date the clock is labelled "Past test window" rather than "Expired". `macro_fit` uses the macro-context family (regime and macro, counted once), not the overall engine direction.
- `read`: bias, setup and trade kept apart.
  - The bias is judged without positioning.
  - It reads "Unsupported" when positioning is the only thing behind the engine direction, and "Mixed" when the families net out.
  - The trade is never produced from BHWS data. It needs the trader's own zone, entry and stop.
- `data_confidence`: High, Medium, Low or Insufficient, from per-input provenance stamps. A failed or stale critical input cannot grade High, and nothing missing is filled in.

Crosses, ICE and crypto markets have no single CFTC contract and return 422.

Futures-only data is fetched from `https://publicreporting.cftc.gov/resource/6dca-aqww.json` and prices from yfinance. Both are cached for six hours per process.

## Dashboard

The Positioning tab shows a "House read" panel under the COT cards. It lazy-loads `/api/evidence` and shows a plain message, not substitute numbers, if the call fails. The "Setup" conviction tier is displayed as "Firm" so it cannot be read as a positioning setup. The backend value is unchanged.

## Admin lock

When the `BHWS_ADMIN_TOKEN` environment variable is set, every `/api/inject-*` and `/api/clear-*` request must send the same value in the `X-BHWS-Admin-Token` header, or it gets a 401. With the variable unset, behaviour is unchanged. The inject scripts read the token from the environment or from a local `.bhws_admin_token` file kept outside the repo.

## Research

`research/playbook_pit_study.py` runs a point-in-time test of the Issue 11 rule. Entries are taken only after each report is public, returns are measured in 12-week volatility units and costs are deducted. Set `BHWS_COT_BASIS=futures_only|combined` to choose the basis. Results are in `research/results/`. Across 26 markets since 2006, none of the variants produced a positive mean after costs that holds up on both bases. The scaled version cuts average loss mainly through position size.

## Tests

`pip install -r requirements-dev.txt && python -m pytest -q tests`. The network is stubbed with fixtures in `tests/fixtures/`.
