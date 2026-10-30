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

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

from bhws_core import zones_v2 as Z

REPLAY_PATH = Path(__file__).resolve().parent / "research" / "zones" / "zone_replay_v2.json"


def _load_replay() -> Dict[str, Any]:
    try:
        return json.loads(REPLAY_PATH.read_text())
    except Exception:
        return {"meta": {}, "cells": {}}


REPLAY = _load_replay()

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
    "evidence": "Point-in-time replay of this detector, 58 markets (weekly 2011-2026, two-day and daily 2020-2026): a resting "
                "order at the proximal edge on the first visit after the zone became knowable, stop a quarter height beyond "
                "the distal edge, 2R target, time-outs marked at the final close. All zones: profit factor 1.05 weekly, 1.10 "
                "two-day, 1.12 daily before costs. Zones that pass the method's own grade (matrix allows it, base of six bars "
                "or fewer, departure of three heights or more, covered by a weekly zone on lower timeframes): 1.29, 1.27 and "
                "1.34, with 38 to 40% of resolved visits reaching 2R first, and both halves of the sample above 1.2. The rest: "
                "0.98 to 1.08. High-volume leg outs did not hold better where Yahoo volume is coherent. Half then add at 1R "
                "trailed full size. Replay history, not a track record; parameters were tuned on nine of Ben's zones.",
    "grade": "The method's grade is its own four rules applied to the zone now: the action matrix allows it at today's curve "
             "location and weekly direction, the base is six bars or fewer, the leg out ran at least three zone heights before "
             "the first return, and on the two-day and daily the zone sits inside a weekly zone of the same kind. Pass or fail "
             "per rule; it is a filter the replay was split on, not a probability.",
    "volume": "Volume candles: candle width follows the bar's volume against the median of the previous twenty bars in the same "
              "volume regime (thick is loud, thin is quiet), the TradingView volume-candle cue. Yahoo continuous futures switch "
              "contracts, so volume is coherent on about four series in ten; where it is not, candles are drawn at a fixed width "
              "and no volume tier is shown.",
}
GRADE_SWING = 3.0
GRADE_BASE = 6

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_BOARD: Dict[str, Any] = {"building": False, "started": 0.0, "finished": 0.0, "done": 0, "total": 0, "failed": []}
BOARD_PACE = 2.0          # seconds between markets in the background warm (Yahoo is shared with the engine)
NEAR_TYP = 1.5            # "near" a zone: within this many typical ranges of the proximal edge


def _candles(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    d = df.tail(n)
    vol, ref = Z.volume_reference(df)
    rv = None
    if vol is not None:
        r = np.clip(vol / ref, 0.15, 6.0)
        rv = pd.Series(r, index=df.index).tail(n)
    out = []
    for i, (ts, r) in enumerate(d.iterrows()):
        c = {"t": ts.strftime("%Y-%m-%d"), "o": round(float(r["Open"]), 6), "h": round(float(r["High"]), 6),
             "l": round(float(r["Low"]), 6), "c": round(float(r["Close"]), 6)}
        if rv is not None:
            x = float(rv.iloc[i])
            c["v"] = float(r.get("Volume", 0) or 0)
            c["rv"] = round(x, 2) if np.isfinite(x) else None
        out.append(c)
    return out


def _grade(z: Z.Zone, act: Dict[str, Any], covered: bool) -> Dict[str, Any]:
    rules = [
        {"k": "matrix", "ok": bool(act.get("allowed")), "t": "action matrix allows it" if act.get("allowed") else ("matrix: stand aside" if act.get("allowed") is False else "matrix: location unknown")},
        {"k": "base", "ok": z.base_bars <= GRADE_BASE, "t": f"base {z.base_bars} bars" + ("" if z.base_bars <= GRADE_BASE else " (over six)")},
        {"k": "departure", "ok": z.swing_h >= GRADE_SWING, "t": f"departure {'30+' if z.swing_h >= 30 else f'{z.swing_h:.1f}'} heights" + ("" if z.swing_h >= GRADE_SWING else " (under three)")},
    ]
    if z.tf != "W":
        rules.append({"k": "covered", "ok": bool(covered), "t": "covered by a weekly zone" if covered else "not covered by a weekly zone"})
    return {"ok": all(r["ok"] for r in rules), "rules": rules}


def _evidence_for(tf: str) -> Dict[str, Any]:
    cells = REPLAY.get("cells", {})
    pick = {"all": tf, "grade": f"{tf}|grade", "rest": f"{tf}|rest", "grade_first": f"{tf}|grade|half|first",
            "grade_second": f"{tf}|grade|half|second", "grade_demand": f"{tf}|grade|demand", "grade_supply": f"{tf}|grade|supply",
            "vol_high": f"{tf}|grade|vol|high", "vol_not_high": f"{tf}|grade|vol|not_high"}
    keep = ("n", "markets", "win1", "win2", "win3", "pf_full_2", "pf_full_3", "pf_scale_2", "pf_scale_3", "exp_full_2", "held", "control_pf_2", "control_win2")
    out = {k: {kk: v.get(kk) for kk in keep} for k, key in pick.items() if (v := cells.get(key))}
    out["horizon"] = (REPLAY.get("meta", {}).get("horizon") or {}).get(tf)
    out["built"] = REPLAY.get("meta", {}).get("built")
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
    d["grade"] = _grade(z, d["action"], cov is not None)
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
                           "zones": {}, "candidates": {}, "candles": {}, "typ": {}, "all_counts": {}, "nearest": {},
                           "volume": {}, "evidence": {}}
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
        out["volume"][tf] = bool(Z.volume_reference(df)[0] is not None)
        out["evidence"][tf] = _evidence_for(tf)
    if not any(out["volume"].values()):
        notes.append("Yahoo volume is not coherent for this series (contract switches or a cash series); candles are drawn at a fixed width")
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


def _nearest_by_kind(zs: List[Dict[str, Any]], price: float, kind: str) -> Optional[Dict[str, Any]]:
    """Nearest key zone of a kind: the one containing price, else the closest on its own side of price."""
    cands = [z for z in zs if z.get("kind") == kind]
    inside = [z for z in cands if z.get("inside")]
    if inside:
        return inside[0]
    side = [z for z in cands if (z["hi"] <= price if kind == "demand" else z["lo"] >= price)]
    if not side:
        return None
    return max(side, key=lambda z: z["proximal"]) if kind == "demand" else min(side, key=lambda z: z["proximal"])


def _board_entry(res: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-market summary for the board: nearest demand and supply per timeframe, and the best 'at a level'
    state per kind (inside beats near; weekly beats daily beats two-day)."""
    price = res.get("price")
    out: Dict[str, Any] = {"price": price, "asof": res.get("asof"), "tf": {}, "at": {}}
    order = {"W": 0, "D": 1, "2D": 2}
    for tf, zs in (res.get("zones") or {}).items():
        row = {}
        for kind in ("demand", "supply"):
            z = _nearest_by_kind(zs or [], price, kind)
            if not z:
                continue
            d = {"lo": z["lo"], "hi": z["hi"], "proximal": z["proximal"], "fresh": z.get("fresh_state"), "inside": bool(z.get("inside")),
                 "dist_typ": z.get("dist_typ"), "grade": bool((z.get("grade") or {}).get("ok")), "pattern": z.get("pattern"),
                 "covered": bool(z.get("covered_by")) if tf != "W" else None}
            d["state"] = "inside" if d["inside"] else ("near" if d["dist_typ"] is not None and abs(d["dist_typ"]) <= NEAR_TYP else "far")
            row[kind] = d
            if d["state"] != "far":
                cur = out["at"].get(kind)
                rank = (0 if d["state"] == "inside" else 1, order.get(tf, 9))
                if cur is None or rank < cur["_rank"]:
                    out["at"][kind] = {**d, "tf": tf, "_rank": rank}
        out["tf"][tf] = row
    for kind in list(out["at"]):
        out["at"][kind].pop("_rank", None)
    return out


def _warm(items: List[Any], yf_with_timeout: Callable) -> None:
    try:
        for mid, symbol in items:
            now = time.time()
            with _LOCK:
                hit = _CACHE.get(mid)
            if hit and now - hit["t"] < CACHE_TTL:
                _BOARD["done"] += 1
                continue
            try:
                res = _build(mid, symbol, yf_with_timeout)
                if "error" not in res:
                    with _LOCK:
                        _CACHE[mid] = {"t": time.time(), "v": res}
                else:
                    _BOARD["failed"].append(mid)
            except Exception:
                _BOARD["failed"].append(mid)
            _BOARD["done"] += 1
            time.sleep(BOARD_PACE)
    finally:
        _BOARD["building"] = False
        _BOARD["finished"] = time.time()


def register(app, *, markets, yf_with_timeout: Callable) -> None:
    by_id = {m["id"]: m for m in markets}

    def _symbol(mid: str) -> Optional[str]:
        m = by_id.get(mid)
        return (FUTURES_OVERRIDE.get(mid) or m.get("yf")) if m else None

    @app.get("/api/zones/board")
    def zones_board(warm: Optional[str] = "1"):
        """Board summary across markets from the zone cache; kicks a paced background warm for markets not cached."""
        items = [(m["id"], _symbol(m["id"])) for m in markets if _symbol(m["id"])]
        now = time.time()
        with _LOCK:
            cached = {mid: _CACHE[mid] for mid, _ in items if mid in _CACHE and now - _CACHE[mid]["t"] < CACHE_TTL}
        missing = [(mid, sym) for mid, sym in items if mid not in cached]
        if missing and warm != "0" and not _BOARD["building"]:
            _BOARD.update(building=True, started=now, done=0, total=len(missing), failed=[])
            threading.Thread(target=_warm, args=(missing, yf_with_timeout), daemon=True, name="zones-board-warm").start()
        out = {mid: _board_entry(c["v"]) for mid, c in cached.items()}
        complete = not missing
        body = {"markets": out, "complete": complete, "building": bool(_BOARD["building"]), "cached": len(cached),
                "total": len(items), "progress": {"done": _BOARD["done"], "total": _BOARD["total"], "failed": list(_BOARD["failed"])},
                "near_typ": NEAR_TYP, "note": "Nearest key demand and supply zone per timeframe from the same detector as /api/zones; "
                "state is inside, near (within 1.5 typical ranges of the proximal edge) or far. Location, not a signal."}
        return JSONResponse(body, status_code=200 if complete else 202, headers={"Cache-Control": "no-store"})

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
            for k in ("zones", "candidates", "candles", "nearest", "volume", "evidence"):
                res[k] = {want: (res.get(k) or {}).get(want)}
        return JSONResponse(res, headers={"Cache-Control": "public, max-age=1800"})
