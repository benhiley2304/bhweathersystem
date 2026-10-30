"""BHWS supply and demand zones API (additive): GET /api/zones.

/api/zones?market=SI[&tf=W|D|both][&refresh=1]
    Weekly and daily supply/demand zones near the current price for one market, detected with the rules in
    bhws_core.zones (base before an impulsive leg out, proximal edge at the body extreme, distal edge at the
    wick extreme). Candles are included so the front end can draw the chart. Zones are a calibrated proxy
    for the zones Ben draws by hand (28 of 32 hand-marked zones matched in the calibration set, which is the
    same set the parameters were tuned on). They are candidates for his judgement, not signals.

Data: Yahoo Finance continuous futures where available (NQ=F rather than ^NDX and so on), six years of
daily bars and fifteen years of weekly bars, cached in memory for six hours per market. Spot FX crosses and
the dollar index use the cash series. Known Yahoo defects (single-print daily bars on some futures, bad
opens on FX spot) are handled by using close-to-close moves for the impulse tests and by preferring weekly
zones for the higher-timeframe read.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from fastapi.responses import JSONResponse

from bhws_core import zones as Z

# Futures series where the backend scores a cash proxy.
FUTURES_OVERRIDE = {
    "ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "RTY": "RTY=F",
    "6E": "6E=F", "6B": "6B=F", "6J": "6J=F", "6A": "6A=F", "6C": "6C=F", "6S": "6S=F", "6N": "6N=F", "6M": "6M=F",
    "DX": "DX-Y.NYB",
}
CACHE_TTL = 6 * 3600
DAILY_YEARS = 6
WEEKLY_YEARS = 15
MAX_DIST_ATR = {"D": 6.0, "W": 5.0}
CANDLES = {"D": 260, "W": 200}
PER_SIDE = 3

DEFINITIONS = {
    "zone": "A base (one or more bars of consolidation) followed by an impulsive leg out. Demand when the leg is up "
            "(DBR, RBR), supply when it is down (RBD, DBD). The proximal edge is the body extreme of the base nearest "
            "price, the distal edge the wick extreme furthest from price.",
    "scales": "Tight: the base measured on bodies, capped at about one ATR (weekly) or two (daily). Wide: the whole "
              "consolidation including wicks and the bars around it, up to three ATR (weekly) or five (daily). Ben's "
              "'supply within supply' is a tight zone inside a wide one.",
    "fresh": "No bar has reached the proximal edge since the leg out. Tested n: reached n times without a close through "
             "the distal edge. Broken: a close beyond the distal edge.",
    "score": "0 to 100 ordinal quality: strength of the leg out against the base and in ATR, tightness, reversal over "
             "continuation, freshness. It ranks candidates. It is not a probability and carries no payoff evidence.",
    "suggestion": "'Use this zone' pre-fills entry at the proximal edge and a stop a quarter of the zone height beyond "
                  "the distal edge, which is the convention in Ben's reports. Both remain his to set.",
    "evidence": "Replay 2012-2026, 57 markets, weekly zones: a graded Desk read with a resting order at the nearest fresh "
                "weekly zone paid 1.13 profit factor at 2R when filled (39% of orders filled), level with the engine's "
                "Monday-close entry. Waiting rescues the retracing trades and forgoes the runners. Zone confluence is a "
                "location and risk tool here, not a measured lift in win rate. See docs/SUPPLY_DEMAND_SPEC.md.",
}

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def _candles(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    d = df.tail(n)
    out = []
    for ts, r in d.iterrows():
        out.append({"t": ts.strftime("%Y-%m-%d"), "o": round(float(r["Open"]), 6), "h": round(float(r["High"]), 6),
                    "l": round(float(r["Low"]), 6), "c": round(float(r["Close"]), 6)})
    return out


def _clean(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return None
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)]
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    df.index = df.index.normalize()
    return df if len(df) >= 60 else None


def _fetch(symbol: str, yf_with_timeout: Callable) -> Dict[str, Optional[pd.DataFrame]]:
    import yfinance as yf
    tk = yf.Ticker(symbol)
    daily = yf_with_timeout(tk.history, period=f"{DAILY_YEARS}y", interval="1d", auto_adjust=False, label=f"zones {symbol} D")
    time.sleep(0.5)
    weekly = yf_with_timeout(tk.history, period=f"{WEEKLY_YEARS}y", interval="1wk", auto_adjust=False, label=f"zones {symbol} W")
    return {"D": _clean(daily), "W": _clean(weekly)}


def _build(mid: str, symbol: str, yf_with_timeout: Callable) -> Dict[str, Any]:
    data = _fetch(symbol, yf_with_timeout)
    notes: List[str] = []
    if data["D"] is None and data["W"] is None:
        return {"error": f"no price history for {symbol}", "market": mid, "symbol": symbol}
    out: Dict[str, Any] = {"market": mid, "symbol": symbol, "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
                           "zones": {}, "nearest": {}, "candles": {}, "atr": {}, "all_counts": {}}
    ref = data["D"] if data["D"] is not None else data["W"]
    price = float(ref["Close"].iloc[-1])
    out["price"] = round(price, 6)
    out["asof"] = ref.index[-1].strftime("%Y-%m-%d")
    detected: Dict[str, List[Z.Zone]] = {}
    for tf in ("W", "D"):
        df = data[tf]
        if df is None:
            notes.append(f"{tf}: no data")
            continue
        if tf == "D":
            single = float(((df["High"] == df["Low"])).mean())
            if single > 0.05:
                notes.append(f"daily: {single:.0%} of bars are single prints on Yahoo; weekly zones are more reliable here")
        zs = Z.detect(df, tf)
        detected[tf] = zs
        atr_now = float(Z.atr(df).iloc[-1])
        out["atr"][tf] = round(atr_now, 6)
        out["all_counts"][tf] = len(zs)
        out["candles"][tf] = _candles(df, CANDLES[tf])
    if "D" in detected and "W" in detected:
        Z.mark_nesting(detected["D"], detected["W"])
    for tf, zs in detected.items():
        atr_now = out["atr"][tf]
        near = Z.live(zs, price, atr_now, MAX_DIST_ATR[tf])
        # unbroken zones only, closest first within each kind, capped per side
        dem = sorted([z for z in near if z.kind == "demand" and z.proximal <= price], key=lambda z: price - z.proximal)[:PER_SIDE]
        sup = sorted([z for z in near if z.kind == "supply" and z.proximal >= price], key=lambda z: z.proximal - price)[:PER_SIDE]
        inside = [z for z in near if z.lo <= price <= z.hi and z not in dem and z not in sup][:2]
        keep = dem + sup + inside
        out["zones"][tf] = [z.to_dict() for z in sorted(keep, key=lambda z: -z.proximal)]
        out["nearest"][tf] = {"demand": dem[0].to_dict() if dem else None, "supply": sup[0].to_dict() if sup else None}
    out["data_note"] = "; ".join(notes) if notes else "Yahoo continuous futures / cash series; closes used for impulse tests."
    out["definitions"] = DEFINITIONS
    return out


def register(app, *, markets, yf_with_timeout: Callable) -> None:
    by_id = {m["id"]: m for m in markets}

    @app.get("/api/zones")
    def zones(market: str, tf: Optional[str] = "both", refresh: Optional[str] = None):
        mid = (market or "").upper().strip()
        m = by_id.get(mid)
        if not m:
            return JSONResponse({"error": f"unknown market {mid}"}, status_code=404)
        symbol = FUTURES_OVERRIDE.get(mid) or m.get("yf")
        if not symbol:
            return JSONResponse({"error": f"no price symbol for {mid}"}, status_code=422)
        now = time.time()
        with _LOCK:
            hit = _CACHE.get(mid)
        if hit and not refresh and now - hit["t"] < CACHE_TTL:
            res = hit["v"]
        else:
            try:
                res = _build(mid, symbol, yf_with_timeout)
            except Exception as e:  # pragma: no cover
                return JSONResponse({"error": f"zone build failed: {e!r}", "market": mid}, status_code=502)
            if "error" not in res:
                with _LOCK:
                    _CACHE[mid] = {"t": now, "v": res}
        if "error" in res:
            return JSONResponse(res, status_code=502)
        want = (tf or "both").upper()
        if want in ("D", "W"):
            res = dict(res)
            res["zones"] = {want: res["zones"].get(want, [])}
            res["candles"] = {want: res["candles"].get(want, [])}
            res["nearest"] = {want: res["nearest"].get(want)}
        return JSONResponse(res, headers={"Cache-Control": "public, max-age=1800"})
