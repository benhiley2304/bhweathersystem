"""Port of the BHWS TradingView Valuation Tool (v8.1), same maths as the Pine.

    roc    = 100 * (close - close[N]) / close[N]          N = 11
    spread = roc(chart) - roc(reference)
    value  = spread rescaled to -100..+100 over the last M bars   M = 100

References follow the Campus defaults: ZB (30-year bond futures) and DXY on,
gold off but computed. Thresholds: +-75 over/undervalued.

Buy points (Pine "Auto" mode: US equity indices only, daily bars, where the
rule was tested): BUY when BOTH ZB and DXY lines are below -40; EXIT when
their average climbs back above -25 or after 10 bars; one trade at a time;
re-arm 2 bars after an exit. The Pine header documents the BHWS backtest
(NDX 1996-2026, 242 trades, 72% winners, avg +1.2%, PF 1.9; tuned on
1996-2010 with some sight of later years). That record is quoted, not
re-derived here.

Everything here is computed on bar closes the caller supplies. Nothing is
filled in: a series that is too short returns None.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math

ROC_LEN = 11
RESCALE_LEN = 100
TH_UP = 75.0
TH_DN = -75.0
BUY_BELOW = -40.0
EXIT_ABOVE = -25.0
MAX_HOLD = 10
COOL = 2
EQUITY_INDEX_IDS = {"ES", "NQ", "YM", "RTY"}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def roc(closes: Sequence[float], n: int = ROC_LEN) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i, c in enumerate(closes):
        if i < n:
            out.append(None)
            continue
        base = closes[i - n]
        out.append(None if _nan(c) or _nan(base) or base == 0 else 100.0 * (c - base) / base)
    return out


def rescale(values: Sequence[Optional[float]], m: int = RESCALE_LEN) -> List[Optional[float]]:
    """Pine f_rescale: ta.highest/ta.lowest over the last m bars (inclusive).
    Pine's highest/lowest return na until m bars of non-na data exist."""
    out: List[Optional[float]] = []
    for i, v in enumerate(values):
        if v is None or i < m - 1:
            out.append(None)
            continue
        win = values[i - m + 1:i + 1]
        if any(w is None for w in win):
            out.append(None)
            continue
        hh, ll = max(win), min(win)
        out.append(0.0 if hh == ll else (v - ll) / (hh - ll) * 200.0 - 100.0)
    return out


def valuation_line(sym: Sequence[float], ref: Sequence[float],
                   n: int = ROC_LEN, m: int = RESCALE_LEN) -> List[Optional[float]]:
    if len(sym) != len(ref):
        raise ValueError("sym and ref must be aligned to the same bars")
    rs, rr = roc(sym, n), roc(ref, n)
    spread = [None if a is None or b is None else a - b for a, b in zip(rs, rr)]
    return rescale(spread, m)


def state(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v >= TH_UP:
        return "Overvalued"
    if v >= 40:
        return "Rich"
    if v <= TH_DN:
        return "Undervalued"
    if v <= BUY_BELOW:
        return "Cheap"
    return "Fair"


def buy_points(v_zb: Sequence[Optional[float]], v_dxy: Sequence[Optional[float]],
               closes: Sequence[float]) -> Dict:
    """Run the Pine buy-point engine over aligned daily bars. Returns the
    current state and the trades it produced (for display, not a backtest)."""
    in_trade = False
    e_px = e_bar = None
    x_bar = -100000
    trades = []
    for i in range(len(closes)):
        a, b = v_zb[i], v_dxy[i]
        if a is None or b is None:
            continue
        v_sig = max(a, b)
        v_ex = (a + b) / 2.0
        if in_trade:
            held = i - e_bar
            if v_ex > EXIT_ABOVE or held >= MAX_HOLD:
                trades.append({"entry_bar": e_bar, "exit_bar": i, "ret": closes[i] / e_px - 1})
                in_trade = False
                x_bar = i
        elif v_sig < BUY_BELOW and i - x_bar >= COOL:
            in_trade = True
            e_px, e_bar = closes[i], i
    last = len(closes) - 1
    cur = {"in_trade": in_trade}
    if in_trade:
        cur.update(bars_held=last - e_bar, open_ret=closes[last] / e_px - 1)
    return {"current": cur, "trades": trades}


def refs_for(market_id: str, category: Optional[str]) -> List[str]:
    """References that carry information for this asset, as Ben reads them.

    Equity indices, commodities and crypto: ZB and DXY (the Pine defaults).
    Gold: ZB and DXY. The dollar index: ZB, gold and the euro. FX majors and
    crosses: ZB and gold (DXY is mostly the pair itself). Bonds: DXY and gold.
    """
    if market_id == "DX":
        return ["zb", "gc", "eur"]
    if category in ("fx", "fx_cross"):
        return ["zb", "gc"]
    if category == "bond" or market_id in ("ZB", "ZN", "ZF", "ZT", "R"):
        return ["dxy", "gc"]
    return ["zb", "dxy"]


def daily_lines(market_id: str, category: Optional[str], daily: Dict[str, Sequence[float]]) -> Optional[Dict]:
    """Valuation Tool lines on daily bars (the timeframe Ben's chart uses)
    against the asset's own references. `daily` is aligned {'sym': [...], ref: [...]}."""
    sym = daily.get("sym")
    if not sym or len(sym) < ROC_LEN + RESCALE_LEN + 5:
        return None
    lines = {}
    for k in refs_for(market_id, category):
        ref = daily.get(k)
        if ref is None or len(ref) != len(sym):
            continue
        line = valuation_line(sym, ref)
        cur = line[-1]
        p5 = line[-6] if len(line) >= 6 else None
        lines[k] = {"value": None if cur is None else round(cur, 1),
                    "chg_5d": None if cur is None or p5 is None else round(cur - p5, 1),
                    "state": state(cur)}
    if not lines:
        return None
    return {"timeframe": "daily", "roc": ROC_LEN, "rescale": RESCALE_LEN, "lines": lines,
            "refs": list(lines.keys())}


def assess(market_id: str, weekly: Dict[str, Sequence[float]],
           daily: Optional[Dict[str, Sequence[float]]] = None) -> Optional[Dict]:
    """weekly/daily: {'sym': [...], 'zb': [...], 'dxy': [...], 'gc': [...]} aligned.

    Weekly lines describe the swing backdrop. Daily buy points are run only
    for US equity indices, matching the Pine 'Auto' mode."""
    sym = weekly.get("sym")
    if not sym or len(sym) < ROC_LEN + RESCALE_LEN:
        return None
    out: Dict = {"timeframe": "weekly", "roc": ROC_LEN, "rescale": RESCALE_LEN, "lines": {}}
    for k in ("zb", "dxy", "gc"):
        ref = weekly.get(k)
        if ref is None or len(ref) != len(sym):
            continue
        if market_id in {"ZB"} and k == "zb" or market_id == "DX" and k == "dxy" or market_id == "GC" and k == "gc":
            continue   # a reference that is the chart itself carries no information
        line = valuation_line(sym, ref)
        cur = line[-1]
        prev4 = line[-5] if len(line) >= 5 else None
        out["lines"][k] = {"value": None if cur is None else round(cur, 1),
                           "chg_4w": None if cur is None or prev4 is None else round(cur - prev4, 1),
                           "state": state(cur)}
    vals = [x["value"] for k, x in out["lines"].items() if k in ("zb", "dxy") and x["value"] is not None]
    if vals:
        avg = sum(vals) / len(vals)
        out["composite"] = {"value": round(avg, 1), "state": state(avg),
                            "both_cheap": len(vals) == 2 and max(vals) < BUY_BELOW,
                            "both_rich": len(vals) == 2 and min(vals) > -BUY_BELOW}
    if market_id in EQUITY_INDEX_IDS and daily and daily.get("sym") and daily.get("zb") and daily.get("dxy"):
        d = daily
        if len(d["sym"]) == len(d["zb"]) == len(d["dxy"]) and len(d["sym"]) > ROC_LEN + RESCALE_LEN:
            vz = valuation_line(d["sym"], d["zb"])
            vd = valuation_line(d["sym"], d["dxy"])
            bp = buy_points(vz, vd, d["sym"])
            out["daily_buy_point"] = {
                "rule": "both ZB and DXY below -40; exit when their average is above -25 or after 10 bars",
                "zb": None if vz[-1] is None else round(vz[-1], 1),
                "dxy": None if vd[-1] is None else round(vd[-1], 1),
                "active": bp["current"]["in_trade"],
                "bars_held": bp["current"].get("bars_held"),
                "record_note": ("BHWS test in the Pine header: NDX 1996-2026, 242 trades, 72% winners, "
                                "avg +1.2%, PF 1.9. Tuned on 1996-2010, so later years are slightly flattering."),
            }
    return out
