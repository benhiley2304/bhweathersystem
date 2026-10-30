"""HTTP surface for the Desk. Registered from backend.py, additive only.

  GET /api/desk            the weekly read across the book (202 while warming)
  GET /api/desk?refresh=1  rebuild now (still returns the last good build first)

Fetching is batched so the Desk does not hammer anyone: one Socrata query for
every CFTC code (futures-only, since 2019) and one yfinance download for every
price series plus the valuation references. Results are written into the same
caches /api/evidence uses, so the Desk and the House read panel agree.

Missing data is reported per market and never filled in.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

import bhws_evidence_api as evapi
from bhws_core import desk, marketread, percentile, setup_score, strategy, valuation

REFS = {"zb": "ZB=F", "dxy": "DX-Y.NYB", "gc": "GC=F", "eur": "6E=F"}
DESK_TTL = 30 * 60
_STATE: Dict[str, Any] = {"data": None, "built": 0.0, "key": None, "building": False,
                          "error": None, "started": None}
_STATE_STORE = None   # set by backend.py via set_state_store(); optional


_CAP_SHORT = (("seasonal against", "60 TD seasonal path against"), ("divergence-fade short", "divergence short"),
              ("managers have turned", "managers turned"))


def _cap_short(c: str) -> str:
    for k, v in _CAP_SHORT:
        if str(c).startswith(k):
            return v
    return str(c).split(":")[0]


def set_state_store(store) -> None:
    global _STATE_STORE
    _STATE_STORE = store
_LOCK = threading.Lock()
_KICK: Optional[Callable[[], bool]] = None   # set by register(); lets backend warm the desk at boot


def kick() -> bool:
    """Start a desk rebuild if one is not running. Must be called on the event loop thread."""
    return bool(_KICK and _KICK())


def is_ready() -> bool:
    with _LOCK:
        return _STATE["data"] is not None


def _prep_fo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.date
    for c in df.columns:
        if c.endswith("_all"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["comm_net"] = df.comm_positions_long_all - df.comm_positions_short_all
    df["mgr_net"] = df.noncomm_positions_long_all - df.noncomm_positions_short_all
    df["retail_net"] = df.nonrept_positions_long_all - df.nonrept_positions_short_all
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def batch_futures_only(codes) -> Dict[str, pd.DataFrame]:
    import httpx
    codes = sorted(set(c for c in codes if c))
    sel = ("cftc_contract_market_code,report_date_as_yyyy_mm_dd,comm_positions_long_all,"
           "comm_positions_short_all,noncomm_positions_long_all,noncomm_positions_short_all,"
           "nonrept_positions_long_all,nonrept_positions_short_all,open_interest_all")
    inlist = ",".join(f"'{c}'" for c in codes)
    rows = []
    offset = 0
    while True:
        params = {"$select": sel,
                  "$where": f"cftc_contract_market_code in({inlist}) AND report_date_as_yyyy_mm_dd >= '2019-01-01'",
                  "$order": "cftc_contract_market_code,report_date_as_yyyy_mm_dd", "$limit": "50000",
                  "$offset": str(offset)}
        r = httpx.get(evapi.FUTURES_ONLY_API, params=params, timeout=60)
        r.raise_for_status()
        part = r.json()
        rows.extend(part)
        if len(part) < 50000:
            break
        offset += 50000
    df = pd.DataFrame(rows)
    out = {}
    if df.empty:
        return out
    for code, g in df.groupby("cftc_contract_market_code"):
        out[code] = _prep_fo(g)
    return out


def batch_prices(tickers) -> Dict[str, pd.Series]:
    import yfinance as yf
    tickers = sorted(set(t for t in tickers if t))
    data = yf.download(tickers, period="5y", interval="1d", auto_adjust=False,
                       group_by="ticker", threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            s = data[t]["Close"].dropna() if isinstance(data.columns, pd.MultiIndex) else data["Close"].dropna()
        except Exception:
            continue
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None)
            out[t] = s.astype(float)
    return out


def _weekly(s: pd.Series) -> pd.Series:
    return s.resample("W-FRI").last().dropna()


def _valuation_for(m: dict, px: Dict[str, pd.Series]) -> Optional[dict]:
    sym = px.get(m.get("yf"))
    if sym is None or len(sym) < 300:
        return None
    skip = set()
    if m.get("category") in ("fx", "fx_cross") or m["id"] == "DX":
        skip.add("dxy")      # an FX pair against the dollar index is mostly itself
    if m.get("category") == "bond":
        skip.add("zb")
    wk = {"sym": _weekly(sym)}
    frame = pd.DataFrame({"sym": wk["sym"]})
    for k, t in REFS.items():
        if k == "eur" or k in skip or px.get(t) is None:
            continue
        frame[k] = _weekly(px[t])
    frame = frame.dropna()
    if len(frame) < valuation.ROC_LEN + valuation.RESCALE_LEN:
        return None
    weekly = {c: frame[c].tolist() for c in frame.columns}
    daily = None
    if m["id"] in valuation.EQUITY_INDEX_IDS and px.get(REFS["zb"]) is not None and px.get(REFS["dxy"]) is not None:
        d = pd.DataFrame({"sym": sym, "zb": px[REFS["zb"]], "dxy": px[REFS["dxy"]]}).dropna()
        daily = {c: d[c].tolist() for c in d.columns}
    res = valuation.assess(m["id"], weekly, daily)
    try:
        refs = valuation.refs_for(m["id"], m.get("category"))
        cols = {"sym": sym}
        for k in refs:
            if px.get(REFS[k]) is not None:
                cols[k] = px[REFS[k]]
        dd = pd.DataFrame(cols).dropna()
        dl = valuation.daily_lines(m["id"], m.get("category"), {c: dd[c].tolist() for c in dd.columns})
        if dl is not None:
            dl["last_bar"] = dd.index[-1].date().isoformat()
            if res is None:
                res = {"timeframe": "weekly", "lines": {}}
            res["daily"] = dl
    except Exception:
        pass
    if res is not None:
        res["refs_used"] = [k for k in ("zb", "dxy", "gc") if k in weekly]
        res["weekly_lines_note"] = "weekly lines use ZB, DXY and gold for every market (context)"
        res["skipped"] = sorted(skip)
        res["last_bar"] = frame.index[-1].date().isoformat()
    return res


def _body(x) -> Optional[dict]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    b = getattr(x, "body", None)
    if b is not None:
        try:
            return json.loads(b)
        except Exception:
            return None
    return None


def _weekly_state(daily: Optional[pd.Series]) -> Optional[dict]:
    """Last COMPLETED weekly close and its 10-week average (W-FRI bars).
    The week in progress is excluded so a midweek print is never called a
    weekly close."""
    if daily is None or len(daily) < 80:
        return None
    w = daily.resample("W-FRI").last().dropna()
    last_day = daily.index[-1]
    if len(w) and w.index[-1] > last_day.normalize() + pd.Timedelta(hours=23):
        w = w.iloc[:-1]
    if len(w) < 11:
        return None
    sma = float(w.iloc[-10:].mean())
    c = float(w.iloc[-1])
    return {"close": round(c, 6), "sma10": round(sma, 6), "week_ending": w.index[-1].date().isoformat(),
            "above": c > sma, "basis": "last completed weekly close (W-FRI) vs 10-week simple average"}


def _divergence_for(fo: Optional[pd.DataFrame], daily: Optional[pd.Series]) -> Optional[dict]:
    if fo is None or daily is None or len(fo) < 60:
        return None
    pct = percentile.midrank_series(fo.mgr_net.tolist())
    dates = pd.to_datetime(fo["date"])
    px = daily.reindex(daily.index.union(dates)).ffill().reindex(dates)
    prices = [None if pd.isna(x) else float(x) for x in px.tolist()]
    out = strategy.manager_divergence(prices, pct)
    out["basis"] = ("26 reports; managers peaked at or above the 80th (or trough at or below the 20th) "
                    "3+ reports ago, now 10+ pts off it, while price sits within 2% of its 26-report extreme "
                    "and beyond where it stood at that peak. Heuristic, not backtested.")
    return out


def build_desk(markets, scores_payload, cot_cache, events_payload, fed_payload,
               now: Optional[datetime] = None, seasonal_provider: Optional[Callable] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    t0 = time.time()
    notes = []
    codes = [m.get("cftc_code") for m in markets if m.get("cftc_code")]
    fo_ret = datetime.now(timezone.utc).isoformat()
    try:
        fo_all = batch_futures_only(codes)
        for c, df in fo_all.items():
            evapi._FO_CACHE[c] = (time.time(), df, fo_ret)
    except Exception as e:
        notes.append(f"CFTC batch failed ({e.__class__.__name__}); using cached rows where present")
        fo_all = {c: v[1] for c, v in evapi._FO_CACHE.items() if v and v[1] is not None}
    px_ret = datetime.now(timezone.utc).isoformat()
    tickers = [m.get("yf") for m in markets] + list(REFS.values())
    try:
        px = batch_prices(tickers)
        for t, s in px.items():
            evapi._PX_CACHE[t] = (time.time(), s, px_ret)
    except Exception as e:
        notes.append(f"price batch failed ({e.__class__.__name__}); using cached series where present")
        px = {}
    for t in tickers:
        if t not in px and evapi._PX_CACHE.get(t):
            px[t] = evapi._PX_CACHE[t][1]
    missing_px = [t for t in tickers if t not in px]
    if missing_px:
        notes.append("no price for " + ", ".join(missing_px))

    eng_by = {x.get("id"): x for x in (scores_payload or {}).get("markets") or []}
    events = (events_payload or {}).get("events") or []
    rows = []
    for m in markets:
        mid = m["id"]
        eng = eng_by.get(mid)
        fo = fo_all.get(m.get("cftc_code")) if m.get("cftc_code") else None
        daily = px.get(m.get("yf"))
        ev = None
        changes = None
        if m.get("cftc_code"):
            comb = None
            if cot_cache is not None:
                hit = cot_cache.get(m["cftc_code"])
                comb = hit if isinstance(hit, pd.DataFrame) else None
            try:
                ev = evapi.build_evidence(m, fo, fo_ret if fo is not None else None, daily,
                                          px_ret if daily is not None else None, eng,
                                          (scores_payload or {}).get("updated_at"), comb, now)
            except Exception as e:
                notes.append(f"{mid}: evidence failed ({e.__class__.__name__})")
            if fo is not None and len(fo) >= 30:
                changes = desk.positioning_changes(fo.mgr_net.tolist(), fo.comm_net.tolist(),
                                                   fo.retail_net.tolist())
        try:
            val = _valuation_for(m, px)
        except Exception as e:
            val = None
            notes.append(f"{mid}: valuation failed ({e.__class__.__name__})")
        try:
            div = _divergence_for(fo, daily)
        except Exception as e:
            div = None
            notes.append(f"{mid}: divergence failed ({e.__class__.__name__})")
        sf = None
        spath = None
        if seasonal_provider is not None:
            try:
                sy = seasonal_provider(mid)
                if sy and sy.get("years"):
                    sf = strategy.seasonal_forward(sy["years"], sy["weights"], sy["current_td"], sy["year"])
                    spath = setup_score.seasonal_path(sy["years"], sy["weights"], sy["current_td"], sy["year"])
            except Exception as e:
                notes.append(f"{mid}: seasonal path failed ({e.__class__.__name__})")
        row = desk.build_row(market=m, engine=eng, evidence=ev, macro=evapi._macro_sign(eng),
                             changes=changes, valuation=val, events=events, now=now,
                             divergence=div, seas_fwd=sf, weekly=_weekly_state(daily))
        try:
            sdw = None
            if daily is not None and len(daily) > 300:
                wr = _weekly(daily).pct_change().dropna().iloc[-52:]
                sdw = float(wr.std()) if len(wr) >= 40 else None
            row["setup"] = setup_score.compute(row, spath, sdw, now)
            if spath:
                row["setup"]["path"] = {"F": spath["F"][::2], "step": 2, "n": spath["n"]}
            # A replay veto on the grade also takes the row off the trade-ready
            # list: the readiness label cannot say "ready" while the grade says
            # stand aside. Wait rows are left alone (the replay found waiting for
            # the seasonal window rescued seasonal-against rows in both halves).
            if row.get("caps") and (row["setup"].get("ready") or {}).get("key") == "ready":
                row["setup"]["ready"] = {"key": "veto", "label": "Stand aside: " + "; ".join(_cap_short(c) for c in row["caps"]),
                                         "from": "ready", "reason": "replay veto on the grade"}
        except Exception as e:
            row["setup"] = None
            notes.append(f"{mid}: setup score failed ({e.__class__.__name__})")
        rows.append(row)
    def _rk(r):
        su = r.get("setup") or {}
        sc = su.get("score")
        rk = {"ready": 0, "wait": 1, "building": 2, "weak": 3, "veto": 4}.get((su.get("ready") or {}).get("key"), 5)
        return (0 if (r.get("side") and sc is not None) else 1, rk, -(sc or 0), r["order"], r["name"] or "")
    rows.sort(key=_rk)

    # Clusters among markets that currently have a side worth acting on
    acting = {r["id"]: r["side"] for r in rows
              if r["side"] and r["class"]["key"] in ("prime", "setup", "developing", "backdrop")}
    wr = {}
    for m in markets:
        if m["id"] in acting and px.get(m.get("yf")) is not None:
            w = _weekly(px[m["yf"]]).pct_change().dropna()
            wr[m["id"]] = w
    if len(wr) >= 2:
        frame = pd.DataFrame(wr).dropna().iloc[-52:]
        cl = desk.correlation_clusters({k: frame[k].tolist() for k in frame.columns},
                                       {k: acting[k] for k in frame.columns})
    else:
        cl = {"clusters": [], "effective_bets": None, "n": len(wr)}
    cl["basis"] = "52 weekly returns, signed by the side the Desk would take; single-link at 0.5"

    regime = (scores_payload or {}).get("regime")
    br = desk.brief(regime=regime, fed=fed_payload, events=events, rows=rows, clusters=cl, now=now)
    changed = desk.what_changed(rows, now)
    fed_facts = (br.get("facts") or {}).get("fed")
    for r in rows:
        try:
            r["read"] = marketread.build_read(r, eng_by.get(r["id"]), fed_facts, now)
        except Exception as e:
            r["read"] = None
            notes.append(f"{r['id']}: market read failed ({e.__class__.__name__})")

    cot_obs = sorted({(r.get("positioning") or {}).get("observed") for r in rows if r.get("positioning")} - {None})
    cot_rel = sorted({(r.get("positioning") or {}).get("released") for r in rows if r.get("positioning")} - {None})
    last_bars = sorted(str(s.index[-1].date()) for s in px.values() if len(s))
    stamps = [
        {"name": "Positioning (CFTC futures-only)", "observed": cot_obs[-1] if cot_obs else None,
         "released": cot_rel[-1] if cot_rel else None, "retrieved": fo_ret,
         "note": None if len(cot_obs) <= 1 else f"{len(cot_obs)} different report dates across markets"},
        {"name": "Engine scores", "observed": (scores_payload or {}).get("updated_at")},
        {"name": "Prices (yfinance daily)", "observed": last_bars[-1] if last_bars else None,
         "retrieved": px_ret, "note": f"{len(missing_px)} missing" if missing_px else None},
        {"name": "Economic calendar", "observed": (events_payload or {}).get("generated_utc")},
        {"name": "Fed funds pricing", "observed": (((fed_payload or {}).get("snapshots") or {}).get("now") or {}).get("date")},
    ]
    classes = {}
    classes_v1 = {}
    for r in rows:
        classes[r["class"]["key"]] = classes.get(r["class"]["key"], 0) + 1
        k1 = r.get("class_v1") or r["class"]["key"]
        classes_v1[k1] = classes_v1.get(k1, 0) + 1
    # Forward record: one row per market per build day, both rule sets, so
    # v1 and v2 can be compared on live outcomes later without a re-run.
    try:
        from bhws_core import ledger as _ledger
        if _STATE_STORE is not None:
            day = now.date().isoformat()
            grows = []
            for r in rows:
                s_ = px.get(next((m.get("yf") for m in markets if m["id"] == r["id"]), None))
                close = float(s_.iloc[-1]) if s_ is not None and len(s_) else None
                grows.append(_ledger.grade_row(r["id"], {"grade": r["class"], "grade_v1": r.get("class_v1"),
                                                         "side": r["side"], "lean": r.get("lean"),
                                                         "archetype": r.get("archetype"), "confirm": r.get("confirm"),
                                                         "caps": r.get("caps")}, r.get("setup"), close))
            _STATE_STORE.save("ledger_grades", _ledger.append_grades(_STATE_STORE.load("ledger_grades"), day, grows))
    except Exception as e:
        notes.append(f"grade record not written ({e.__class__.__name__})")
    return {
        "generated_at": now.isoformat(),
        "ruleset": strategy.RULESET,
        "counts_v1": classes_v1,
        "build_seconds": round(time.time() - t0, 1),
        "method": {
            "positioning": evapi.METHOD_NOTE,
            "ordering": ("Sorted by readiness (trade-ready, wait for window, building, weak, seasonal veto), "
                         "then by setup score. Setup score points were set on the 2012-18 replay and tested on "
                         "2019-26; it sorts weakly in the right direction out of sample. Not a probability."),
            "setup_score": ("5 + structure (archetype x confirmation) + trend location + positioning payoff + "
                            "valuation + seasonal timing, clipped 0-10, half-point steps. Timing comes from the "
                            "weighted-median seasonal path over the next 80 TD in units of one weekly standard "
                            "deviation: against (path falls >1 sd within 20-40 TD), wait (dip now, leg >=1.5 sd "
                            "after it within 25 TD), approaching, in window, with or flat."),
            "seasonal": ("Forward path from today: blend-weighted median of each completed year's move over "
                         "the next 20/30/60/90 trading days (crosses the year end). Positive means price tends to "
                         "rise. 60 TD is primary, 30 TD must not disagree."),
            "valuation": ("BHWS Valuation Tool port: 11-bar ROC spread vs reference, rescaled to -100..+100 over "
                          "100 bars. Daily lines drive the read (ZB+DXY for indices and commodities, ZB+gold+euro "
                          "for the dollar, ZB+gold for FX, DXY+gold for bonds); weekly lines are context."),
            "trend": ("Engine momentum detail: pullback = last 4 weeks against an intact trend; extended = outer "
                      "15% of the 52-week range with the 13-week move stalled. An extended trend does not count "
                      "against a fade."),
            "divergence": "Fund managers lower high in net length against a higher high in price (and mirror). Heuristic.",
            "classes": [dict(key=k, label=l, tone=t, note=n) for k, l, r, t, n in desk.CLASSES],
        },
        "counts": classes,
        "rows": rows,
        "clusters": cl,
        "changed": changed,
        "brief": br,
        "regime": None if not regime else {k: regime.get(k) for k in ("regime", "regime_label", "score_10", "vix_level")},
        "stamps": stamps,
        "notes": notes,
    }


def register(app, *, markets, scores_cache, cot_cache=None,
             events_provider: Optional[Callable] = None, fed_provider: Optional[Callable] = None,
             seasonal_provider: Optional[Callable] = None):
    from fastapi.responses import JSONResponse

    async def _call(fn):
        # Providers may be sync (run off the event loop) or async.
        if asyncio.iscoroutinefunction(fn):
            return await fn()
        r = await asyncio.to_thread(fn)
        return (await r) if asyncio.iscoroutine(r) else r

    async def _rebuild():
        try:
            payload = (scores_cache or {}).get("data") or {}
            evp = fedp = None
            if events_provider:
                try:
                    evp = _body(await _call(events_provider))
                except Exception as e:
                    print(f"[desk] events provider failed: {e!r}", flush=True)
            if fed_provider:
                try:
                    fedp = _body(await _call(fed_provider))
                except Exception as e:
                    print(f"[desk] fed provider failed: {e!r}", flush=True)
            data = await asyncio.to_thread(build_desk, markets, payload, cot_cache, evp, fedp, None,
                                           seasonal_provider)
            with _LOCK:
                _STATE.update(data=evapi._json_safe(data), built=time.time(),
                              key=payload.get("updated_at"), error=None)
        except Exception as e:
            traceback.print_exc()
            with _LOCK:
                _STATE["error"] = repr(e)
        finally:
            with _LOCK:
                _STATE["building"] = False

    def _kick():
        with _LOCK:
            if _STATE["building"]:
                return False
            _STATE["building"] = True
            _STATE["started"] = time.time()
        asyncio.get_event_loop().create_task(_rebuild())
        return True

    global _KICK
    _KICK = _kick

    @app.get("/api/desk")
    async def get_desk(refresh: int = 0):
        payload = (scores_cache or {}).get("data") or {}
        if not payload:
            return JSONResponse({"status": "warming", "reason": "engine scores not built yet"}, status_code=202)
        with _LOCK:
            data, built, key, building, err = (_STATE["data"], _STATE["built"], _STATE["key"],
                                               _STATE["building"], _STATE["error"])
        stale = (data is None or time.time() - built > DESK_TTL or key != payload.get("updated_at"))
        if (stale or refresh) and not building:
            _kick()
            building = True
        if data is None:
            return JSONResponse({"status": "warming", "error": err,
                                 "reason": "building the desk (one CFTC query, one price batch)"},
                                status_code=202)
        out = dict(data)
        out["cache"] = {"age_seconds": round(time.time() - built), "rebuilding": bool(building),
                        "scores_key_matches": key == payload.get("updated_at")}
        return JSONResponse(out)

    return app
