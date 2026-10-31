"""Price series for the Levels panel: Yahoo daily and weekly feeds with the last two years rebuilt from hourly bars.

Why. Yahoo's daily feed prints settlement-only bars (high = low = close) for many COMEX, NYMEX and ICE contracts:
on 30 Oct 2026 platinum 67% of days, palladium 79%, CAD 33%, cocoa 23%, cotton and coffee 15%, silver 14%, and it
carries no volume on most of those series. Those are the "dots instead of candles" charts. The hourly feed for the
same symbols has real ranges and volume (0% single prints across platinum, silver, cocoa, gold and the S&P on the
same day). So daily bars are rebuilt from hourly bars by exchange session for the 730 days Yahoo serves, and spliced
onto the daily feed for older history. Weekly bars over the rebuilt span are resampled from the rebuilt daily bars.

Session rule. A CME Globex trading day runs 18:00 to 17:00 New York, so hourly bars at or after 18:00 belong to the
next calendar day; forex uses 17:00. Checked against the feed on the series whose daily feed is clean (ES, corn):
rebuilt highs and lows match the feed to the tick, opens within 0.05%, closes within 0.1% (the feed's close is the
settlement, the rebuilt close is the last trade).

Intraday bars. Four-hour and eight-hour bars are anchored at 18:00 New York (the Globex open), so the 8-hour bars
are roughly the Asian, European and US sessions. They exist to refine a daily or weekly level and to show context
inside one, never to find a level on their own: a level that is not visible on the daily is not traded.

Everything here is a pure function of the frames it is given except fetch_series, which calls Yahoo through the
caller's timeout wrapper.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

NY = "America/New_York"
DAILY_YEARS = 6
WEEKLY_YEARS = 15
HOURLY_PERIOD = "730d"       # the most Yahoo serves at 1h
MIN_BARS_DAY = 3             # a rebuilt day needs this many hourly bars (holiday half-days keep more than this)
SPLICE_MAX_MED_DIFF = 0.02   # median |rebuilt close - feed close| over the overlap, as a fraction, before we refuse to splice


def cutoff_hour(symbol: str) -> Optional[int]:
    """Hour (New York) at which a new session day starts; None when the daily feed should be used as is."""
    s = (symbol or "").upper()
    if s.endswith("-USD"):          # crypto: 24/7, Yahoo's UTC daily bars are fine
        return None
    if s.endswith("=X"):            # spot forex: 17:00 New York
        return 17
    return 18                       # CME Globex and the ICE US contracts Yahoo carries


def _ny_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx
    return idx.tz_convert(NY).tz_localize(None)


def session_days(idx: pd.DatetimeIndex, cutoff: int) -> pd.DatetimeIndex:
    """Session date for each intraday bar start (naive New York time)."""
    idx = _ny_naive(idx)
    nxt = (idx + pd.Timedelta(days=1)).normalize()
    return pd.DatetimeIndex(np.where(idx.hour >= cutoff, nxt, idx.normalize()))


def rebuild_daily(h1: pd.DataFrame, cutoff: int, drop_weekends: bool = True) -> pd.DataFrame:
    """Daily OHLCV by exchange session from hourly bars."""
    if h1 is None or len(h1) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    x = h1.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    days = session_days(pd.DatetimeIndex(x.index), cutoff)
    g = x.groupby(days)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum() if "Volume" in x.columns else 0.0,
                        "bars": g["Close"].size()})
    out = out[out["bars"] >= MIN_BARS_DAY]
    if drop_weekends:
        out = out[out.index.dayofweek < 5]
    out = out.drop(columns=["bars"]).sort_index()
    if len(out) > 1:
        out = out.iloc[1:]            # the first session is usually partial
    return out


def resample_intraday(h1: pd.DataFrame, hours: int, anchor_hour: int = 18) -> pd.DataFrame:
    """Session-anchored intraday bars (naive New York time, bar start as the index)."""
    if h1 is None or len(h1) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    x = h1.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    x.index = _ny_naive(pd.DatetimeIndex(x.index))
    origin = pd.Timestamp(x.index[0].date()) + pd.Timedelta(hours=anchor_hour) - pd.Timedelta(days=1)
    r = x.resample(f"{hours}h", origin=origin, label="left", closed="left")
    out = pd.DataFrame({"Open": r["Open"].first(), "High": r["High"].max(), "Low": r["Low"].min(), "Close": r["Close"].last(),
                        "Volume": r["Volume"].sum() if "Volume" in x.columns else 0.0})
    return out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def quality(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"n": 0, "single_pct": None, "zero_volume_pct": None}
    single = float((df["High"] == df["Low"]).mean())
    zv = float((df["Volume"].fillna(0) <= 0).mean()) if "Volume" in df.columns else 1.0
    return {"n": int(len(df)), "single_pct": round(single * 100, 1), "zero_volume_pct": round(zv * 100, 1)}


def splice_daily(feed: Optional[pd.DataFrame], rebuilt: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Rebuilt bars where they exist, the feed before that. Refuses when the two disagree on the overlap."""
    if rebuilt is None or len(rebuilt) < 40:
        return {"df": feed, "spliced": False, "reason": "no hourly history"}
    if feed is None or len(feed) == 0:
        return {"df": rebuilt, "spliced": True, "from": rebuilt.index[0], "reason": "feed empty"}
    both = feed.join(rebuilt, how="inner", rsuffix="_r")
    if len(both) >= 20:
        med = float((np.abs(both["Close_r"] - both["Close"]) / both["Close"].abs()).median())
        if not np.isfinite(med) or med > SPLICE_MAX_MED_DIFF:
            return {"df": feed, "spliced": False, "reason": f"hourly and daily closes disagree (median {med:.1%})"}
    start = rebuilt.index[0]
    older = feed[feed.index < start]
    cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in feed.columns else [])
    rb = rebuilt.copy()
    for c in cols:
        if c not in rb.columns:
            rb[c] = 0.0
    out = pd.concat([older[cols], rb[cols]]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return {"df": out, "spliced": True, "from": start, "reason": None}


def rebuild_weekly(feed_w: Optional[pd.DataFrame], daily: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Weekly bars indexed by Monday (Yahoo's convention): the feed before `start`, resampled daily bars after."""
    d = daily[daily.index >= start]
    if len(d) < 10:
        return feed_w
    r = d.resample("W-FRI", label="right", closed="right")
    wk = pd.DataFrame({"Open": r["Open"].first(), "High": r["High"].max(), "Low": r["Low"].min(), "Close": r["Close"].last(),
                       "Volume": r["Volume"].sum() if "Volume" in d.columns else 0.0}).dropna(subset=["Open"])
    wk.index = wk.index - pd.Timedelta(days=4)        # Friday label to the Monday of that week
    first_monday = wk.index[0]
    if feed_w is None or len(feed_w) == 0:
        return wk
    older = feed_w[feed_w.index < first_monday]
    cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in feed_w.columns else [])
    for c in cols:
        if c not in wk.columns:
            wk[c] = 0.0
    out = pd.concat([older[cols], wk[cols]]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _clean(df: Optional[pd.DataFrame], normalize: bool = True) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return None
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)]
    if normalize:
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) >= 60 else None


def fetch_series(symbol: str, yf_with_timeout: Callable, *, intraday: bool = True, pace: float = 0.5,
                 daily_repair: Optional[Callable[[pd.DataFrame], Optional[pd.DataFrame]]] = None) -> Dict[str, Any]:
    """Daily (spliced), weekly, 4-hour and 8-hour frames plus a provenance record for one Yahoo symbol.

    `daily_repair` is applied to the daily feed before the splice (the spot FX open/close repair lives in zones_v2);
    it returns the repaired frame or None to leave the feed alone."""
    import yfinance as yf
    tk = yf.Ticker(symbol)
    daily = _clean(yf_with_timeout(tk.history, period=f"{DAILY_YEARS}y", interval="1d", auto_adjust=False, label=f"zones {symbol} D"))
    repaired = None
    if daily is not None and daily_repair is not None:
        try:
            repaired = daily_repair(daily)
        except Exception:
            repaired = None
        if repaired is not None:
            daily = repaired
    time.sleep(pace)
    weekly = _clean(yf_with_timeout(tk.history, period=f"{WEEKLY_YEARS}y", interval="1wk", auto_adjust=False, label=f"zones {symbol} W"))
    meta: Dict[str, Any] = {"symbol": symbol, "feed": quality(daily), "rebuilt": None, "spliced": False, "source": "Yahoo daily feed",
                            "intraday": False, "note": None, "feed_repaired": repaired is not None}
    out: Dict[str, Any] = {"D": daily, "W": weekly, "4H": None, "8H": None, "meta": meta}
    cut = cutoff_hour(symbol)
    if not intraday or cut is None:
        meta["note"] = "daily feed used as is" + (" (24/7 series)" if cut is None else "")
        return out
    time.sleep(pace)
    h1 = yf_with_timeout(tk.history, period=HOURLY_PERIOD, interval="1h", auto_adjust=False, label=f"zones {symbol} 1h", timeout=40)
    if h1 is None or len(h1) < 500:
        meta["note"] = "no hourly history from Yahoo; daily feed used as is"
        return out
    h1 = h1.dropna(subset=["Open", "High", "Low", "Close"])
    h1 = h1[(h1["High"] >= h1["Low"]) & (h1["Close"] > 0)]
    rebuilt = rebuild_daily(h1, cut, drop_weekends=True)
    meta["rebuilt"] = quality(rebuilt)
    sp = splice_daily(daily, rebuilt)
    if sp["spliced"]:
        out["D"] = sp["df"]
        start = pd.Timestamp(sp["from"])
        out["W"] = rebuild_weekly(weekly, sp["df"], start)
        meta.update(spliced=True, rebuilt_from=start.strftime("%Y-%m-%d"), source="Yahoo hourly bars rebuilt by session (last two years) over the daily feed")
        meta["used"] = quality(out["D"][out["D"].index >= start])
    else:
        meta["note"] = sp["reason"]
    try:
        for key, hours in (("4H", 4), ("8H", 8)):
            bars = resample_intraday(h1, hours, anchor_hour=cut)
            out[key] = bars if len(bars) >= 60 else None
        meta["intraday"] = out["4H"] is not None
        if out["4H"] is not None:
            meta["intraday_from"] = out["4H"].index[0].strftime("%Y-%m-%d")
            meta["intraday_asof"] = out["4H"].index[-1].strftime("%Y-%m-%d %H:%M")
    except Exception as e:  # pragma: no cover
        meta["note"] = f"intraday resample failed: {e!r}"
    return out
