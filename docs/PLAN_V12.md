# BH Weather System v12: top-down plan

Prepared 29 Oct 2026 against HEAD d0a96f1 (v11, live). Everything below is grounded in the codebase, the 2012 to 2026 desk replay (`research/replay/REPORT.md`) and the setup-score study (`research/score/REPORT.md`). Where a claim is replay evidence it says so; where it is an engineering or design judgement it says that instead.

## 1. Objective and organising principle

The tool exists to do two things, in this order:

1. Find favourably asymmetric swing setups: positions where the historical payoff distribution for that kind of setup is right-skewed and where your zone, stop and target sit sensibly inside it.
2. Read probable direction: an honest view of which way the pillars lean, how strongly, and what would change the view.

The organising principle for v12 is that the **setup**, not the market or the score, becomes the core object. A setup has a thesis (archetype and side), a payoff profile (what happened historically to setups like it), a timing window (seasonal path and catalysts inside the hold), an invalidation (your zone, and the cause-level trigger behind it), a book role (what it adds or duplicates), and a ledger (your plan and the realised outcome). Every screen, number and chart in v12 serves one of those six fields or is removed.

Scores stay as ranking aids. They are never turned into probabilities and the composite is never inverted.

## 2. Where we are (honest baseline)

**Evidence.** The replay over 57 markets, weekly, point in time, shows the grade does not sort outcomes: Prime mean R +0.047 (n 2,447), Setup -0.006, Developing +0.051, against always-long +0.069; after drift Prime is -0.020. The setup score beats the grade as a ranking aid but its held-out edge is small: top bucket minus bottom bucket +0.04 R after drift (CI -0.04 to +0.12), with hit rates near 50% in every bucket. The real information is in pockets. Positive: counter-trend before the weekly close turns +0.097 (skew 1.29, both halves positive), counter-trend longs +0.155 (skew 1.45), crowd fade before managers turn +0.044. Negative, and currently not vetoed in the live rules: divergence-fade shorts -0.269 (skew 0.63), crowd fade after managers have turned -0.068, seasonal-against on Prime or Setup -0.120, with-trend shorts -0.114, exhaustion fade graded Prime -0.185 (n 72). The macro pillar and the bias have never been tested because they cannot be rebuilt point in time. The playbook study found no variant positive after costs.

**Live rules (confirmed in `bhws_core/strategy.py` today).** Trend against caps at Developing (the replay says it should not). Seasonal against only blocks Prime (the replay says it is the strongest veto on Prime and Setup). Crowd fades are not gated on the managers' turn. Divergence-fade shorts are not capped. Desk sort weights (positioning 3, valuation 2.5, trend 2, macro 1.5, seasonality 1) are unfitted.

**Engineering.** `index.html` is 51,022 lines and 2.5 MB: 1,018 KB of CSS with 9,457 `!important` rules, 1,451 KB of inline JS in 10 blocks, 1,878 inline style attributes, and four generations of runtime DOM rewrites (v8 to v11) layered over the original render functions. There is no frontend build or test suite. `backend.py` is 19,663 lines in one file. Four stores (Forex Factory labour and macro, Fed pricing, supply) live in memory, so every deploy wipes them and needs a 5 minute re-injection routine plus a desk rebuild. There is no persistence anywhere: plans live in the browser's localStorage, so there is no outcome ledger and the calibration loop never closes. Known data faults: FRED 404 on LRHUTTTTCHM156S, COT code errors on 044601, 098662 and 090741. No Postgres or key-value instance exists on the Render workspace.

The design and clarity passes in v8 to v11 were real improvements, but they were patches on a structure that fights them. A leap needs the structure to change.

## 3. Strategy, logic and scoring (the edge layer)

### 3.1 Rules v2 (needs your sign-off)

Apply the replay's four findings as live rules, with the old grade retired from the interface and kept only in the Edge lab for comparison:

| Change | Evidence | Effect on the board |
|---|---|---|
| Counter-trend setups with a live positioning or valuation anchor are graded on their anchors, not capped at Developing | +0.097 R before the weekly close turns, +0.155 for longs, both halves positive | Early fades surface as Setup or Prime instead of hiding in Developing |
| Seasonal against is a hard veto on Prime and Setup (max Developing) | -0.120 R, the strongest negative pocket | Fewer Prime setups, and the ones left are cleaner |
| Crowd fade window closes when managers have turned | -0.068 R after the turn vs +0.044 before | "Turned" becomes a timing state, not a confirmation |
| Divergence-fade shorts capped at Developing | -0.269 R, skew 0.63 | Removes the worst archetype from the top of the board |

Weights stay as they are: the study found no stable improvement and I will not fit to a sample this size. From the day rules v2 go live, every grade is written to the ledger (section 3.6) so the forward record starts immediately.

### 3.2 Payoff profile engine (the asymmetry object)

Replace the single MFE/MAE median with a full first-passage study per setup class. For each archetype × side × confirmation state × asset class, from the point-in-time replay:

- P(reach +2R before -1R), P(+3R before -1R), with the stop set at k × weekly σ for k in {0.5, 0.75, 1.0, 1.5} (because your stops are yours, the engine shows the whole row rather than picking one).
- MFE and MAE at 4, 8 and 13 weeks: median, 25th and 75th percentiles, skew.
- Time to target distribution, and how often the path went through -0.5R first (the "do I get a better entry by waiting" question).
- Stability: both halves of the sample, and after the drift adjustment, shown side by side with n.

Displayed as a **payoff cone** (section 6) with your plan overlaid. The cone is labelled as replay history for setups like this one, never as the probability for this trade.

### 3.3 Plan stress test

The plan form already takes zone, entry, stop and target. v12 evaluates them, never generates them: "In 212 past setups of this class, a stop this far from entry (0.8σ) was hit before the target in 46% of cases; median time to target 5 weeks; 31% saw a better entry within 2 weeks." It also flags when the stop sits inside the historical noise band (MAE 25th percentile) and when the target sits beyond the MFE 75th percentile. Levels and zones remain yours; the engine only reports how similar structures resolved.

### 3.4 Timing and catalysts

The seasonal path and sweet-spot logic stay, with the sign convention stated on the chart. Added: a **catalyst map** per market built from release-day reactions since 2012 (realised |move| in σ units on CPI, NFP, FOMC, PMI and the market's own supply releases; direction conditional on surprise sign where we have the surprise). Each setup then shows which binary events fall inside the hold window and their historical reaction size, so you can size down or wait. US series come from the FRED release calendar; non-US coverage will be partial and labelled as such.

### 3.5 Invalidation at cause level

Beyond your zone, each setup carries a cause-level trigger derived from the book's factor exposure (section 3.7): "this is a USD-weakness trade; 4 other legs share it; a DXY weekly close above your reference invalidates the cluster." This is descriptive (which factor the leg loads on), not a level you have not set.

### 3.6 Ledger and calibration (the compounding loop)

Plans, grades, payoff-profile snapshots and outcomes are persisted (section 5.1). A calibration view compares realised R by archetype against the replay cone, so after 20 to 30 closed trades you can see whether live results sit inside the historical distribution. This is the only mechanism that turns the tool from a reader of history into evidence about your edge.

### 3.7 Book layer

Rolling 52-week regressions give each market a factor loading on USD, duration, risk and broad commodities. The book view shows net exposure per factor across open setups and plans, effective independent bets (already in `exposure.py`), and which candidate setups add diversification versus duplicate an existing bet.

### 3.8 Things that will not change

No probabilities from scores. No expectancy implied from R:R. No manufactured levels. No fitted weights. Stale or missing data is shown as such, never silently filled. Negative findings stay on screen.

## 4. Intelligence and data

- **Point-in-time climate history to 2012.** Climate pillars are price-based (VIX, credit proxies, breadth, curve) so a PIT series is buildable. That lets the payoff engine condition on regime and lets us test whether climate adds anything to setup outcomes (reported either way).
- **Point-in-time macro.** US macro via FRED ALFRED vintages and release dates, so the macro pillar and bias can finally be replayed. Until that passes, macro stays labelled as context.
- **Catalyst statistics** as above, stored as a precomputed artifact.
- **Your managers' turn hypothesis** tested explicitly: entry on the first one or two weekly reports of managers reversing from an extreme, adds on continued reversal with price and macro alignment. The crowd-fade result above is adjacent but not the same test. Result published either way.
- **Data quality page.** Freshness, provenance and failures for every feed, including the three COT codes and the FRED series that error today. Nothing fails silently.
- **Desk brief with falsifiable claims.** Each market read ends with "what would change this" in the same terms the pillars use, so the brief can be marked right or wrong a week later.

## 5. Engineering and efficiency

### 5.1 Persistence (removes the post-deploy routine)

A separate private `bhws-state` repository (no deploy hook) holds `ff_event_store.json`, `supply_event_store.json`, Fed pricing, plans and the ledger. The backend loads it at boot and writes through on change; the injection scripts and the Forex Factory watcher write to it too. Deploys stop wiping anything. If you later move Render to a paid plan, the same interface points at Postgres instead.

### 5.2 Precomputed evidence artifacts

The replay, payoff profiles, catalyst statistics and PIT climate history are produced by a nightly scheduled task and committed as JSON. The live backend serves them instantly; the 45 second history cold load goes away for everything that is not genuinely live.

### 5.3 Backend split

`backend.py` divides into data ingestion, scoring engine, desk and evidence API, and serving. Behaviour is pinned by the existing 96 tests plus golden-file tests on `scores_full_snapshot.json` before anything moves.

### 5.4 Frontend rebuild (strangler pattern)

A Vite build with ES modules replaces the single file. Pages are rebuilt one at a time inside a new shell while the old page keeps serving until its replacement passes QA, so there is never a regression week. Targets: under 600 KB initial load, zero runtime DOM rewrites, zero `!important` in new code, a component library (card, stat, caption, info popover, tab, table) and one chart kit (SVG, shared axes, hover crosshair, annotations, tokens). Playwright screenshot tests at 390, 1366 and 1920 replace manual tab walks. Order: Home and board, Setup dossier, Positioning, Seasonal, Macro and Climate, Valuation and Trend, Edge lab.

### 5.5 Hygiene

Fix the FRED and COT code faults, rotate the GitHub token referenced in `bh_system_reference.md`, health-check and rate-limit the backend calls the frontend makes on load.

## 6. Design system and signature graphics

**Tokens.** Two type families, five sizes, an 8 pt spacing scale, one radius scale, one elevation. Colour roles are fixed: bull green `#22C97A`, bear red `#E05252`, commercials blue `#3D7BD6`, managers gold `#EAB850` (gold is also the "against the side" colour), ivory `#F1E3BE` for prime surfaces, neutral greys for everything that carries no direction. Green means bullish and red means bearish everywhere, including charts and the ledger.

**Chart kit.** Every chart uses the same axis style, gridline weight, hover crosshair, legend placement and annotation style. Signature graphics:

1. **Payoff cone.** MFE and MAE fan at 4, 8, 13 weeks for the setup class with your stop and target drawn as horizontal lines through it, and the first-passage table beneath.
2. **Positioning phase plot.** Commercials net versus managers net as a path through time, extremes shaded, the current point marked, so a crowd fade or a managers' turn is visible as a shape rather than two gauges.
3. **Seasonal fan with sweet spot.** Existing ensemble path, unified to the kit, sign convention printed.
4. **Catalyst strip.** The hold window as a timeline with release markers sized by historical reaction.
5. **Book wheel.** Factor exposures of the open book and the candidate setup, so duplication is visible at a glance.
6. **Setup map and archetype heatmap.** Kept, rebuilt in the kit.

Motion is restrained and purposeful (state change, reveal, hover), and respects reduced-motion settings.

## 7. Workflow and screens

The Sunday routine is five screens, in the order you work:

1. **Desk.** Climate, this week's catalysts, and the board ranked by evidence quality (anchors, payoff skew for the class, timing state), not by score theatre.
2. **Setup dossier.** One page per candidate with the six fields: thesis, payoff cone, timing and catalysts, invalidation and plan stress test, book fit, evidence with n.
3. **Factor tabs.** Positioning, Seasonal, Macro and Climate, Valuation and Trend, each answering "what does this pillar say about this setup" first and showing the full chart second.
4. **Ledger.** Open plans, outcomes, calibration against the cones.
5. **Edge lab.** The replay, the rules comparison (v1 vs v2), the hypothesis tests, and the data quality page.

The Sunday Setup chart pack generates from the dossiers, so the report and the tool agree by construction.

## 8. Build order and gates

| Phase | Scope | Gate to pass |
|---|---|---|
| 0 | Decisions below; `bhws-state` persistence; ledger schema; rules v2 behind a flag; nightly evidence artifact task | Deploy with no injection routine; grades written to ledger; tests green |
| 1 | Payoff profile engine; plan stress test; PIT climate history; catalyst statistics; data quality page; managers' turn test | Research reports published in `research/`, negative results included; artifacts served |
| 2 | New frontend shell, chart kit and tokens; Desk and Setup dossier rebuilt; screenshot tests | Old and new pages side by side pass QA at three widths; initial load under 600 KB |
| 3 | Factor tabs, Ledger, Edge lab rebuilt; book layer; report pack from dossiers; old file retired | No runtime rewrites remain; backend split complete |
| 4 | Forward review after roughly 12 weeks of ledger | Calibration view populated; rules v2 kept, adjusted or reverted on evidence |

Phases 0 and 1 are where the edge work lives. Phase 2 and 3 are where the visual leap lives. I would not start the visual rebuild before the persistence and evidence engines exist, because the new screens are designed around objects (payoff cone, ledger, catalyst map) that do not exist yet.

## 9. Decisions I need from you

1. **Rules v2.** Adopt the four replay-backed changes as the live grading rules, with the old grade visible only in the Edge lab? My recommendation is yes.
2. **Stop convention for the payoff tables.** Show the full k × σ row (0.5 to 1.5σ) and let your plan pick the row, or lead with one convention (your 25% zone rule needs zone width, which the data does not have)? My recommendation is the full row.
3. **Persistence.** The free `bhws-state` repository now, or a paid Render Postgres plus disk? My recommendation is the repository now; the interface is the same either way.

Everything else in this plan I will take under the existing authorisation, in the order above, verifying each step live before reporting it done.
