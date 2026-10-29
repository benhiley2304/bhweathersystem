# State layer

`bhws_core/state.py` holds the seven stores that used to live only in memory: `ff_event_store`, `supply_event_store`, `fed_pricing`, `fed_pricing_history`, `ff_macro_injected`, `ledger`, `ledger_grades`.

## Tiers

- **Local files** next to the code. Two of them (`ff_event_store.json`, `supply_event_store.json`) are committed seeds; the rest are git-ignored and written at runtime. A seed has no stamp, so it counts as "age 0" and any import or restore is allowed to extend it.
- **Remote** (optional): a Render Postgres via `DATABASE_URL`, or a private GitHub repository via `BHWS_STATE_REPO` and `BHWS_STATE_TOKEN`. Neither is configured yet, so `/api/state/status` reports `remote: none, durable: false`. Every deploy starts from the seeds.
- **Stamps**: `.bhws_state_stamps.json` records when each store was last saved by this process, independent of file mtimes (which a build resets).

## Export and import

- `GET /api/admin/state/export` returns every store with its stamp (0 for a seed).
- `POST /api/admin/state/import?only_newer=true` takes that bundle back.

Keyed stores are **merged**, not replaced: `ff_event_store` and `supply_event_store` union by key, `ff_macro_injected` keeps the later `time` per currency, `fed_pricing_history` unions by date, `ledger` unions entries by id (later `updated_at` wins), `ledger_grades` unions by (day, market). Where both sides hold a key, the newer side wins. `fed_pricing` is a snapshot and stays newest-wins. The same rules apply when a remote tier is restored at boot. Results per store: `imported`, `merged`, `local newer`, `unchanged`.

This matters because the first request after a deploy refreshes the Forex Factory feed and re-saves the store, which used to make the import see the short live copy as "newer" and drop the injected labour history. With merge semantics the history comes back on import regardless of order.

## Post-deploy routine (until a remote tier exists)

1. Before pushing, export the bundle.
2. After both services are live, import it with `only_newer=true`.
3. `GET /api/desk?refresh=1`, then poll `/api/desk` until 200.

Admin endpoints require the `x-bhws-admin-token` header and return 401 without it.
