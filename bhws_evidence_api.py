"""HTTP surface for the evidence layer, kept out of backend.py on purpose.

backend.py calls register(app, markets=..., scores_cache=..., cot_cache=...)
once at the end of the module. Everything here is additive:

  GET /api/evidence?market=6E
      House-method positioning (CFTC legacy futures-only, 156-report
      mid-rank, the TradingView basis), the backend's own combined-basis
      numbers as a labelled cross-check, observation vs release dates,
      freshness, the Issue 11 playbook state, a data-confidence grade and the
      bias / setup / trade read. Read-only.

  Admin guard (opt-in)
      When the env var BHWS_ADMIN_TOKEN is set, the inject and clear-cache
      routes require header X-BHWS-Admin-Token with that value. When it is
      unset, behaviour is unchanged, so this can ship before the inject
      scripts are updated.
"""
from __future__ import annotations

import asyncio
import hmac
import math
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from bhws_core import cot_calendar, percentile, playbook, readout, riskreward
from bhws_core.provenance import InputStamp, data_confidence

FUTURES_ONLY_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
PROTECTED_PREFIXES = ("/api/inject-", "/api/clear-")
ADMIN_HEADER = "x-bhws-admin-token"
_CACHE_TTL = 6 * 3600
_FO_CACHE: Dict[str, Any] = {}
_PX_CACHE: Dict[str, Any] = {}
_PREV_READ: Dict[str, Dict[str, Any]] = {}

METHOD_NOTE = ("Positioning percentiles: CFTC legacy futures-only, trailing 156 reports, mid-rank "
               "(same as the BHWS TradingView indicators). Observation = Tuesday; release = CFTC "
               "publication, normally Friday 15:30 ET. The cross-check block uses the backend engine's "
               "futures-and-options combined data and is shown, not reconciled.")


def admin_ok(headers, token: Optional[str]) -> bool:
    if not token:
        return True
    got = headers.get(ADMIN_HEADER) or headers.get(ADMIN_HEADER.title()) or ""
    return hmac.compare_digest(str(got), str(token))


def _fetch_futures_only(code: str) -> pd.DataFrame:
    import httpx
    sel = ("report_date_as_yyyy_mm_dd,comm_positions_long_all,comm_positions_short_all,"
           "noncomm_positions_long_all,noncomm_positions_short_all,"
           "nonrept_positions_long_all,nonrept_positions_short_all,open_interest_all")
    params = {"$select": sel,
              "$where": f"cftc_contract_market_code='{code}' AND report_date_as_yyyy_mm_dd >= '2019-01-01'",
              "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": "2000"}
    r = httpx.get(FUTURES_ONLY_API, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.date
    for c in df.columns:
        if c.endswith("_all"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["comm_net"] = df.comm_positions_long_all - df.comm_positions_short_all
    df["mgr_net"] = df.noncomm_positions_long_all - df.noncomm_positions_short_all
    df["retail_net"] = df.nonrept_positions_long_all - df.nonrept_positions_short_all
    return df.drop_duplicates("date").reset_index(drop=True)


def _fetch_weekly_close(yf_ticker: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(yf_ticker).history(period="5y", interval="1d", auto_adjust=False)
    s = h["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


async def _cached(store, key, fn, *args):
    hit = store.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1], hit[2]
    val = await asyncio.to_thread(fn, *args)
    stamp = datetime.now(timezone.utc).isoformat()
    store[key] = (time.time(), val, stamp)
    return val, stamp


def _align_closes(obs_dates, daily: pd.Series):
    """Weekly close known after each report's release: last close on/before release day."""
    out = []
    for d in obs_dates:
        rel = pd.Timestamp(cot_calendar.release_for(d).released)
        s = daily[:rel]
        out.append(float(s.iloc[-1]) if len(s) else float("nan"))
    return out


def _macro_sign(engine_payload: Optional[dict]) -> Optional[int]:
    """Sign of the macro-context family (regime + macro, counted once).

    The playbook's macro fit must come from macro context alone, not the
    overall engine direction, which also counts positioning."""
    votes = (engine_payload or {}).get("factor_votes")
    if not votes or not any(k in votes for k in readout.FAMILIES["macro context"]):
        return None
    return readout._family_votes(votes).get("macro context", 0)


def build_evidence(market: dict, fo: Optional[pd.DataFrame], fo_retrieved: Optional[str],
                   daily: Optional[pd.Series], px_retrieved: Optional[str],
                   engine_payload: Optional[dict], engine_updated: Optional[str],
                   combined_df: Optional[pd.DataFrame], now: Optional[datetime] = None,
                   plan: Optional[dict] = None) -> dict:
    """plan (optional, trader-supplied, never generated): side (+1/-1), starter
    (date), zone_proximal/zone_distal, entry/stop/target, spread."""
    now = now or datetime.now(timezone.utc)
    plan = plan or {}
    mid = market["id"]
    stamps = []
    ev: Dict[str, Any] = {"market": mid, "name": market.get("name"),
                          "generated_at": now.isoformat(), "method": METHOD_NOTE}
    if market.get("cot_note"):
        ev["quote_direction"] = market["cot_note"]

    # ── positioning: house basis ─────────────────────────────────────────
    if fo is not None and len(fo):
        last_obs = fo.date.iloc[-1]
        fr = cot_calendar.freshness(last_obs, now)
        stamps.append(InputStamp(name="cot_futures_only", source=FUTURES_ONLY_API,
                                 expected_frequency="weekly", observed_at=last_obs.isoformat(),
                                 released_at=fr["released_at_utc"], retrieved_at=fo_retrieved,
                                 transformations=["net = long - short", "midrank 156 reports"]))
        ev["positioning"] = {
            "basis": "CFTC legacy futures-only",
            "observed": last_obs.isoformat(), "released": fr["released"],
            "freshness": fr["status"], "reports_behind": fr["reports_behind"],
            "commercials": percentile.describe(fo.comm_net.tolist(), "futures_only"),
            "managers": percentile.describe(fo.mgr_net.tolist(), "futures_only"),
            "retail": percentile.describe(fo.retail_net.tolist(), "futures_only"),
            "managers_chg_13w": (int(fo.mgr_net.iloc[-1] - fo.mgr_net.iloc[-14]) if len(fo) > 14 else None),
        }
    else:
        stamps.append(InputStamp(name="cot_futures_only", source=FUTURES_ONLY_API,
                                 expected_frequency="weekly", note="fetch failed or no rows"))
        ev["positioning"] = None

    # ── cross-check: backend combined basis, shown not reconciled ────────
    if combined_df is not None and len(combined_df) and "lspec_net" in combined_df:
        cd = combined_df.dropna(subset=["lspec_net"])
        if "date" in cd.columns:
            cd = cd.sort_values("date")
            cobs = pd.Timestamp(cd["date"].iloc[-1]).date()
        elif not isinstance(cd.index, pd.RangeIndex):
            cobs = pd.Timestamp(cd.index[-1]).date()
        else:
            cobs = None
        cnet = cd.lspec_net.tolist()
        cturn = None   # filled once the playbook fade side is known
        ev["cross_check"] = {
            "basis": "CFTC legacy futures+options combined (backend engine)",
            "managers_net": int(cnet[-1]) if cnet else None,
            "managers_turning_reports": cturn,
            "observed": cobs.isoformat() if cobs else None,
            "managers_midrank_156": round(percentile.midrank_percentile(cd.lspec_net.tolist()) or float("nan"), 1),
            "managers_minmax_520": round(percentile.minmax_index(cd.lspec_net.tolist(), 520) or float("nan"), 1),
            "commercials_midrank_156": round(percentile.midrank_percentile(cd.comm_net.tolist()) or float("nan"), 1),
            "commercials_minmax_520": round(percentile.minmax_index(cd.comm_net.tolist(), 520) or float("nan"), 1),
        }
        if ev.get("positioning"):
            a = ev["positioning"]["managers"]["percentile"]
            b = ev["cross_check"]["managers_minmax_520"]
            if a is not None and b is not None and not math.isnan(b):
                ev["cross_check"]["managers_gap_vs_house"] = round(b - a, 1)
    else:
        ev["cross_check"] = None

    # ── price ────────────────────────────────────────────────────────────
    if daily is not None and len(daily):
        stamps.append(InputStamp(name="price", source=f"yfinance {market.get('yf')}",
                                 expected_frequency="daily",
                                 observed_at=pd.Timestamp(daily.index[-1]).date().isoformat(),
                                 retrieved_at=px_retrieved))
    else:
        stamps.append(InputStamp(name="price", source=f"yfinance {market.get('yf')}",
                                 expected_frequency="daily", note="fetch failed"))

    # ── engine payload ───────────────────────────────────────────────────
    stamps.append(InputStamp(name="engine_scores", source="/api/scores",
                             expected_frequency="daily", observed_at=engine_updated,
                             critical=True, note=None if engine_payload else "no cached scores"))

    # ── playbook ─────────────────────────────────────────────────────────
    if fo is not None and len(fo) >= 20 and daily is not None and len(daily):
        closes = _align_closes(fo.date.tolist(), daily)
        keep = [i for i, c in enumerate(closes) if not math.isnan(c)]
        zone = None
        if plan.get("zone_proximal") is not None and plan.get("zone_distal") is not None:
            zone = playbook.Zone(float(plan["zone_proximal"]), float(plan["zone_distal"]), source="trader")
        pb = playbook.evaluate([fo.date.iloc[i] for i in keep],
                               [float(fo.mgr_net.iloc[i]) for i in keep],
                               [closes[i] for i in keep],
                               macro_bias=_macro_sign(engine_payload), zone=zone,
                               book_side=plan.get("side"), starter_date=plan.get("starter"))
        ev["playbook"] = pb
    else:
        ev["playbook"] = {"status": "Insufficient data"}

    # ── basis disagreements: report both, never reconcile ────────────────
    ev["disagreements"] = []
    pb = ev["playbook"]; cc = ev.get("cross_check") or {}
    fade = pb.get("direction")
    if fade and combined_df is not None and "lspec_net" in getattr(combined_df, "columns", []):
        cnet = combined_df.dropna(subset=["lspec_net"])
        if "date" in cnet.columns:
            cnet = cnet.sort_values("date")
        cnet = cnet.lspec_net.tolist()
        t = 0
        for k in range(len(cnet) - 1, 0, -1):
            if (cnet[k] - cnet[k - 1]) * fade > 0:
                t += 1
            else:
                break
        cc["managers_turning_reports"] = t
        h = pb.get("managers_turning_reports")
        if h is not None and h != t:
            ev["disagreements"].append({
                "field": "managers turning (reports)",
                "house_futures_only": h, "combined_futures_options": t,
                "note": "Turn count differs by basis. The TradingView indicator (futures-only) is the house number.",
            })
    if cc.get("managers_gap_vs_house") is not None and abs(cc["managers_gap_vs_house"]) >= 10:
        ev["disagreements"].append({
            "field": "managers percentile",
            "house_futures_only_midrank156": (ev.get("positioning") or {}).get("managers", {}).get("percentile"),
            "combined_minmax520": cc.get("managers_minmax_520"),
            "note": "Different basis and method; the gap is shown, not reconciled.",
        })

    ev["data_confidence"] = data_confidence(stamps, now)
    prev = _PREV_READ.get(mid)
    prev_read = None
    obs_key = (ev.get("positioning") or {}).get("observed")
    if prev and prev.get("report") != obs_key:
        prev_read = prev.get("read")
    levels = None
    if plan.get("side") in (1, -1) and all(plan.get(k) is not None for k in ("entry", "stop", "target")):
        rr = riskreward.reward_risk(float(plan["entry"]), float(plan["stop"]), float(plan["target"]),
                                    int(plan["side"]), spread=float(plan.get("spread") or 0.0))
        levels = {"side": plan["side"], "entry": plan["entry"], "stop": plan["stop"],
                  "target": plan["target"], "reward_risk": rr}
        ev["plan"] = levels
    elif plan:
        ev["plan"] = {"side": plan.get("side"), "starter": plan.get("starter"),
                      "note": "side, entry, stop and target are all needed for reward:risk"}
    ev["read"] = readout.build(engine_payload or {"id": mid}, evidence=ev, previous=prev_read,
                               trader_levels=levels)
    if obs_key and not plan and (not prev or prev.get("report") != obs_key):
        _PREV_READ[mid] = {"report": obs_key, "read": ev["read"], "prev": prev_read}
    elif prev and prev.get("prev") is not None:
        ev["read"]["changes"] = readout.diff(prev["prev"], ev["read"])
    return ev


def register(app, *, markets, scores_cache, cot_cache=None):
    from fastapi import Request
    from fastapi.responses import JSONResponse

    by_id = {m["id"]: m for m in markets}

    @app.middleware("http")
    async def _bhws_admin_guard(request: Request, call_next):
        path = request.url.path
        if path.startswith(PROTECTED_PREFIXES):
            token = os.environ.get("BHWS_ADMIN_TOKEN")
            if not admin_ok(request.headers, token):
                return JSONResponse({"error": "admin token required"}, status_code=401)
            resp = await call_next(request)
            resp.headers["X-BHWS-Auth"] = "enforced" if token else "disabled"
            return resp
        return await call_next(request)

    @app.get("/api/evidence")
    async def get_evidence(market: str, side: Optional[str] = None, starter: Optional[str] = None,
                           zone_proximal: Optional[float] = None, zone_distal: Optional[float] = None,
                           entry: Optional[float] = None, stop: Optional[float] = None,
                           target: Optional[float] = None, spread: Optional[float] = None):
        mid = (market or "").strip().upper()
        m = by_id.get(mid)
        if m is None:
            return JSONResponse({"error": f"Unknown market '{market}'"}, status_code=404)
        if not m.get("cftc_code"):
            return JSONResponse({"error": f"{mid} has no CFTC code (cross, ICE or crypto market)",
                                 "market": mid}, status_code=422)
        fo = fo_ret = daily = px_ret = None
        try:
            fo, fo_ret = await _cached(_FO_CACHE, m["cftc_code"], _fetch_futures_only, m["cftc_code"])
        except Exception as e:  # missing data is reported, never filled in
            print(f"[evidence] futures-only fetch failed for {mid}: {e!r}", flush=True)
        try:
            daily, px_ret = await _cached(_PX_CACHE, m.get("yf"), _fetch_weekly_close, m.get("yf"))
        except Exception as e:
            print(f"[evidence] price fetch failed for {mid}: {e!r}", flush=True)
        payload = (scores_cache or {}).get("data") or {}
        eng = next((x for x in payload.get("markets") or [] if x.get("id") == mid), None)
        comb = None
        if cot_cache is not None:
            hit = cot_cache.get(m["cftc_code"])
            comb = hit if isinstance(hit, pd.DataFrame) else None
        plan = {}
        sd = {"long": 1, "short": -1, "1": 1, "-1": -1}.get((side or "").strip().lower())
        if sd:
            plan["side"] = sd
        if starter:
            try:
                plan["starter"] = date.fromisoformat(starter[:10])
            except ValueError:
                return JSONResponse({"error": "starter must be YYYY-MM-DD"}, status_code=422)
        for k, v in (("zone_proximal", zone_proximal), ("zone_distal", zone_distal), ("entry", entry),
                     ("stop", stop), ("target", target), ("spread", spread)):
            if v is not None and math.isfinite(v):
                plan[k] = v
        ev = build_evidence(m, fo, fo_ret, daily, px_ret, eng, payload.get("updated_at"), comb,
                            plan=plan or None)
        return JSONResponse(_json_safe(ev))

    return app


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if hasattr(o, "item"):
        try:
            return _json_safe(o.item())
        except Exception:
            return str(o)
    return o
