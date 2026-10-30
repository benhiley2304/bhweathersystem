"""BHWS supply and demand zones API (additive): GET /api/zones.

/api/zones?market=SI[&tf=W|2D|D|both][&refresh=1]
    Weekly, two-day and daily supply/demand zones near the current price for one market, detected with the
    three-step method in bhws_core.zones_v2 (leg out first, then leg in, then base; proximal edge at the body
    extreme, distal edge at the wick extreme; freshness measured to the right; stacked levels shown as one
    structure). Candles are included so the front end can draw the chart from the earliest key zone to now.

    The detector was calibrated on 32 zones from Ben's reports (Issues 01-11) and checked against six charts he
    drew on 30 Oct 2026 (nine zones: SI and SB weekly, BTC and EUR/USD two-day, HG and USD/JPY daily). Those
    checks are agreement with hand-drawn boxes, not payoff evidence. Zones are candidates for his judgement.

Data: Yahoo Finance continuous futures where available (NQ=F rather than ^NDX and so on); spot FX for the
euro, pound, aussie and kiwi so the levels match the EUR/USD style charts Ben draws. Six years of daily bars,
fifteen years of weekly bars, cached in memory for six hours per market. Yahoo spot FX daily bars print the
open equal to the close and lag the close by a day; they are repaired before detection (see zones_v2.repair_spot_fx).
Some futures print single-price daily bars; those markets are flagged and the weekly read is preferred.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from fastapi.responses import JSONResponse

from bhws_core import zones_v2 as Z

# Futures series where the backend scores a cash proxy, and spot FX where Ben charts the cash pair.
FUTURES_OVERRIDE = {
    "ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "RTY": "RTY=F",
    "6E": "EURUSD=X", "6B": "GBPUSD=X", "6A": "AUDUSD=X", "6N": "NZDUSD=X",
    "6J": "6J=F", "6C": "6C=F", "6S": "6S=F", "6M": "6M=F",
    "DX": "DX-Y.NYB",
}
CACHE_TTL = 6 * 3600
DAILY_YEARS = 6
WEEKLY_YEARS = 15
MAX_DIST = 8.0                      # typical ranges from price to the proximal edge
CANDLES = {"W": 260, "2D": 260, "D": 320}
CANDLES_MAX = 420
PER_SIDE = 2

DEFINITIONS = {
    "zone": "Leg in, base, leg out. The base is a consolidation whose candle bodies overlap; the leg out is an explosive "
            "candle or a sustained move away from it. Demand when the leg out is up (DBR, RBR), supply when it is down "
            "(RBD, DBD). Proximal edge: the body extreme of the base nearest price. Distal edge: the wick extreme behind it. "
            "The preferred (tight) version uses the last one or two base candles; it is drawn as a thin inner line.",
    "fresh": "Measured to the right of the leg out only. Fresh: no prior visit reached the preferred version. Shallow "
             "(weekly): prior visits stayed within a quarter of the preferred version. Tested n: n completed prior visits "
             "reached it without a close beyond the distal edge. Broken: a close beyond the distal edge. The visit in "
             "progress is the arrival, not a test.",
    "stack": "Level on top of level: two or three zones of one kind whose gaps are no taller than either zone are one "
             "structure, one band with each base boxed. A newer zone formed inside an older one of the same kind is the "
             "level shown; the older is named on it.",
    "score": "0 to 100 ordinal, following the five zone qualifiers: departure (swing in zone heights and run in typical "
             "ranges), base duration, freshness, leg-out candle quality, originality. It ranks candidates; it is not a "
             "probability and carries no payoff evidence.",
    "curve": "Where price sits between the nearest weekly demand below and weekly supply above, split at 0.33 and 0.66 "
             "into low, equilibrium and high. The action matrix (demand when low, supply when high, equilibrium only "
             "with trend) is the method's rule, reported as such, not as measured evidence.",
    "suggestion": "'Use this zone' pre-fills entry at the proximal edge and a stop a quarter of the zone height beyond "
                  "the distal edge, which is the convention in Ben's reports. Both remain his to set.",
    "evidence": "Replay 2012-2026, 57 markets, weekly zones (first detector): a graded Desk read with a resting order at "
                "the nearest fresh weekly zone paid 1.13 profit factor at 2R when filled (39% of orders filled), level "
                "with the engine's Monday-close entry. Zone confluence is a location and risk tool here, not a measured "
                "lift in win rate. The second detector has not been replayed yet. See docs/SUPPLY_DEMAND_SPEC.md.",
}

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


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
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) >= 60 else None


def _fetch(symbol: str, yf_with_timeout: Callable) -> Dict[str, Optional[pd.DataFrame]]:
    import yfinance as yf
    tk = yf.Ticker(symbol)
    daily = yf_with_timeout(tk.history, period=f"{DAILY_YEARS}y", interval="1d", auto_adjust=False, label=f"zones {symbol} D")
    time.sleep(0.5)
    weekly = yf_with_timeout(tk.history, period=f"{WEEKLY_YEARS}y", interval="1wk", auto_adjust=False, label=f"zones {symbol} W")
    return {"D": _clean(daily), "W": _clean(weekly)}


def _frames(data: Dict[str, Optional[pd.DataFrame]], notes: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    d = data.get("D")
    if d is not None:
        if Z.looks_like_spot_fx_defect(d):
            d = Z.repair_spot_fx(d)
            notes.append("daily spot FX bars repaired (Yahoo prints open equal to close and lags the close by a day)")
        single = float((d["High"] == d["Low"]).mean())
        if single > 0.05:
            notes.append(f"daily: {single:.0%} of bars are single prints on Yahoo; weekly zones are more reliable here")
        out["D"] = d
        out["2D"] = Z.resample_2d(d)
    if data.get("W") is not None:
        out["W"] = data["W"]
    return out


def _zone_dict(z: Z.Zone, price: float, weekly: List[Z.Zone], opposing: List[Z.Zone], curve_label: str, trend: str) -> Dict[str, Any]:
    d = z.to_dict()
    d["inside"] = bool(z.lo <= price <= z.hi)
    d["dist_typ"] = round(float((price - z.proximal) / z.atr if z.kind == "demand" else (z.proximal - price) / z.atr), 2) if z.atr else None
    d["room"] = Z.room(z, opposing)
    cov = Z.covered(z, weekly) if z.tf != "W" else None
    d["covered_by"] = None if cov is None else {"lo": float(cov.lo), "hi": float(cov.hi), "base_start": pd.Timestamp(cov.base_start).strftime("%Y-%m-%d"), "fresh_state": cov.fresh_state}
    d["action"] = Z.action(z.kind, curve_label, trend)
    d["title"] = f"{'Weekly' if z.tf == 'W' else 'Two-day' if z.tf == '2D' else 'Daily'} {z.kind}"
    return d


def _build(mid: str, symbol: str, yf_with_timeout: Callable) -> Dict[str, Any]:
    data = _fetch(symbol, yf_with_timeout)
    notes: List[str] = []
    if data["D"] is None and data["W"] is None:
        return {"error": f"no price history for {symbol}", "market": mid, "symbol": symbol}
    frames = _frames(data, notes)
    ref = frames.get("D", frames.get("W"))
    price = float(ref["Close"].iloc[-1])
    out: Dict[str, Any] = {"market": mid, "symbol": symbol, "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
                           "price": round(price, 6), "asof": ref.index[-1].strftime("%Y-%m-%d"), "detector": "v2",
                           "zones": {}, "candidates": {}, "candles": {}, "typ": {}, "all_counts": {}, "nearest": {}}
    detected: Dict[str, List[Z.Zone]] = {}
    live: Dict[str, List[Z.Zone]] = {}
    for tf, df in frames.items():
        zs = Z.detect(df, tf)
        detected[tf] = zs
        typ = float(Z.typical_range(df).iloc[-1])
        out["typ"][tf] = round(typ, 6)
        out["all_counts"][tf] = len(zs)
        live[tf] = Z.live(zs, price, typ, MAX_DIST)
    weekly_live = Z.stack(live.get("W", []), price) if "W" in live else []
    cur = Z.curve(price, weekly_live)
    trend = Z.direction(frames["W"]) if "W" in frames else {"label": "unknown", "highs": [], "lows": [], "rule": ""}
    for tf, df in frames.items():
        stacked = Z.stack(live[tf], price)
        shown, rest = Z.select(stacked, price, PER_SIDE, typ=out["typ"][tf])
        opp = stacked
        out["zones"][tf] = [_zone_dict(z, price, weekly_live, opp, cur["label"], trend["label"]) for z in sorted(shown, key=lambda z: -(z.lo + z.hi))]
        out["candidates"][tf] = [_zone_dict(z, price, weekly_live, opp, cur["label"], trend["label"]) for z in sorted(rest, key=lambda z: -(z.lo + z.hi))]
        dem = [z for z in shown if z.kind == "demand"]
        sup = [z for z in shown if z.kind == "supply"]
        out["nearest"][tf] = {"demand": max(dem, key=lambda z: z.proximal).to_dict() if dem else None,
                              "supply": min(sup, key=lambda z: z.proximal).to_dict() if sup else None}
        # candles back to the earliest key zone base (plus a margin), bounded
        n = CANDLES[tf]
        if shown:
            first = min(pd.Timestamp(z.base_start) for z in shown)
            pos = int(df.index.searchsorted(first))
            n = max(n, min(CANDLES_MAX, len(df) - pos + 12))
        out["candles"][tf] = _candles(df, n)
    out["curve"] = cur
    out["direction"] = trend
    out["read"] = _read(mid, price, cur, trend, out["zones"])
    out["data_note"] = "; ".join(notes) if notes else "Yahoo continuous futures / cash series."
    out["definitions"] = DEFINITIONS
    return out


def _fmt(v: float, price: float) -> str:
    if price >= 1000:
        return f"{v:,.0f}"
    if price >= 100:
        return f"{v:.1f}"
    if price >= 10:
        return f"{v:.2f}"
    if price >= 1:
        return f"{v:.4f}"
    return f"{v:.5f}"


def _read(mid: str, price: float, cur: Dict, trend: Dict, zones: Dict[str, List[Dict]]) -> List[str]:
    """Short method read, labelled as rules. No levels are invented: everything quoted is a detected zone edge."""
    lines: List[str] = []
    lab = cur.get("label", "unknown")
    if cur.get("demand") and cur.get("supply") and cur.get("frac") is not None:
        lines.append(f"Weekly curve: price sits {int(round(100 * cur['frac']))}% of the way from weekly demand "
                     f"{_fmt(cur['demand']['lo'], price)}-{_fmt(cur['demand']['hi'], price)} to weekly supply "
                     f"{_fmt(cur['supply']['lo'], price)}-{_fmt(cur['supply']['hi'], price)}: {lab}.")
    elif lab in ("very low", "very high"):
        side = cur.get("demand") if lab == "very low" else cur.get("supply")
        if side:
            lines.append(f"Weekly curve: price is inside the weekly {side['kind']} zone {_fmt(side['lo'], price)}-{_fmt(side['hi'], price)}: {lab}.")
    else:
        lines.append(f"Weekly curve: {lab}; the location rule needs a weekly zone on both sides.")
    if trend.get("label") != "unknown":
        lines.append(f"Weekly direction: {trend['label']} ({trend.get('rule', '')}).")
    # the nearest key zone on the daily or two-day chart, and what the matrix says about it
    for tf in ("D", "2D"):
        zs = zones.get(tf) or []
        near = sorted(zs, key=lambda z: (0 if z.get("inside") else 1, abs(z["dist_typ"]) if z.get("dist_typ") is not None else 99))
        if near:
            z = near[0]
            act = z.get("action") or {}
            cov = "covered by a weekly zone of the same kind" if z.get("covered_by") else "not inside a weekly zone of the same kind"
            lines.append(f"Nearest {z['title'].lower()} {_fmt(z['lo'], price)}-{_fmt(z['hi'], price)} ({z['fresh_state']}"
                         f"{', price inside' if z.get('inside') else ''}), {cov}. Method matrix: {act.get('text', 'no rule')}.")
            break
    lines.append("These are the method's rules applied to detected zones, not measured edge. Entries, stops and targets stay yours.")
    return lines


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
        if want in ("D", "W", "2D"):
            res = dict(res)
            for k in ("zones", "candidates", "candles", "nearest"):
                res[k] = {want: res[k].get(want)}
        return JSONResponse(res, headers={"Cache-Control": "public, max-age=1800"})
