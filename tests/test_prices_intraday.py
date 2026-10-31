"""Session-aligned daily rebuild, splice, intraday resample and lower-timeframe refinement."""
import numpy as np
import pandas as pd
import pytest

from bhws_core import prices as P
from bhws_core import zones_intraday as ZI


def _hourly(start="2026-03-02 18:00", days=12, base=100.0, seed=1):
    """Globex-style hourly bars: 18:00 to 16:00 New York, Sunday 18:00 open, no Saturday."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="America/New_York")
    idx = idx[(idx.hour != 17)]
    idx = idx[~((idx.dayofweek == 5) | ((idx.dayofweek == 4) & (idx.hour >= 17)) | ((idx.dayofweek == 6) & (idx.hour < 18)))]
    n = len(idx)
    close = base + np.cumsum(rng.normal(0, 0.3, n))
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.4, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.4, n)
    vol = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_session_days_roll_at_cutoff():
    idx = pd.DatetimeIndex(["2026-03-03 17:00", "2026-03-03 18:00", "2026-03-04 02:00", "2026-03-04 16:00"]).tz_localize("America/New_York")
    days = P.session_days(idx, 18)
    assert [d.strftime("%m-%d") for d in days] == ["03-03", "03-04", "03-04", "03-04"]
    days_fx = P.session_days(idx, 17)
    assert days_fx[0].strftime("%m-%d") == "03-04"


def test_rebuild_daily_matches_session_extremes_and_drops_weekends():
    h1 = _hourly()
    d = P.rebuild_daily(h1, 18)
    assert (d.index.dayofweek < 5).all()
    assert len(d) >= 7
    # one session: 18:00 previous evening through 16:00 on the labelled day
    day = d.index[2]
    sess = h1[(h1.index >= (day - pd.Timedelta(hours=6)).tz_localize("America/New_York")) & (h1.index < (day + pd.Timedelta(hours=17)).tz_localize("America/New_York"))]
    assert d.loc[day, "High"] == pytest.approx(sess["High"].max())
    assert d.loc[day, "Low"] == pytest.approx(sess["Low"].min())
    assert d.loc[day, "Open"] == pytest.approx(sess["Open"].iloc[0])
    assert d.loc[day, "Close"] == pytest.approx(sess["Close"].iloc[-1])
    assert d.loc[day, "Volume"] == pytest.approx(sess["Volume"].sum())
    assert (d["High"] > d["Low"]).all()


def test_resample_intraday_anchored_at_globex_open():
    h1 = _hourly()
    h4 = P.resample_intraday(h1, 4, 18)
    hours = set(h4.index.hour)
    assert hours <= {18, 22, 2, 6, 10, 14}
    h8 = P.resample_intraday(h1, 8, 18)
    assert set(h8.index.hour) <= {18, 2, 10}
    assert (h4["High"] >= h4["Low"]).all() and len(h8) < len(h4) < len(h1)


def test_splice_prefers_rebuilt_and_refuses_on_disagreement():
    days = pd.bdate_range("2025-01-01", periods=400)
    feed = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0 + np.arange(400) * 0.01, "Volume": 0.0}, index=days)
    rebuilt = feed.iloc[300:].copy()
    rebuilt["High"] += 1.0
    rebuilt["Low"] -= 1.0
    rebuilt["Volume"] = 500.0
    sp = P.splice_daily(feed, rebuilt)
    assert sp["spliced"] and sp["from"] == rebuilt.index[0]
    out = sp["df"]
    assert len(out) == 400
    assert (out.loc[rebuilt.index, "High"] - out.loc[rebuilt.index, "Low"] > 1.9).all()
    assert (out.iloc[:300]["High"] == out.iloc[:300]["Low"]).all()
    # a rebuilt series on a different price level (wrong contract) is refused
    bad = rebuilt.copy()
    bad[["Open", "High", "Low", "Close"]] *= 1.3
    sp2 = P.splice_daily(feed, bad)
    assert not sp2["spliced"] and "disagree" in sp2["reason"]


def test_rebuild_weekly_is_monday_indexed_and_uses_rebuilt_span():
    days = pd.bdate_range("2025-01-06", periods=120)   # a Monday
    daily = pd.DataFrame({"Open": 1.0, "High": np.linspace(1, 2, 120), "Low": 0.5, "Close": 1.5, "Volume": 1.0}, index=days)
    feed_w = pd.DataFrame({"Open": 1.0, "High": 9.0, "Low": 0.1, "Close": 1.0, "Volume": 1.0},
                          index=pd.date_range("2024-01-01", periods=80, freq="W-MON"))
    out = P.rebuild_weekly(feed_w, daily, pd.Timestamp("2025-02-03"))
    assert (out.index.dayofweek == 0).all()
    rebuilt_part = out[out.index >= pd.Timestamp("2025-02-03")]
    assert len(rebuilt_part) >= 15 and (rebuilt_part["High"] < 2.01).all()
    assert (out[out.index < pd.Timestamp("2025-02-03")]["High"] == 9.0).all()


def test_cutoff_hour_by_symbol_type():
    assert P.cutoff_hour("PL=F") == 18
    assert P.cutoff_hour("EURUSD=X") == 17
    assert P.cutoff_hour("BTC-USD") is None
    assert P.cutoff_hour("DX-Y.NYB") == 18


def _zone(kind="demand", lo=95.0, hi=100.0, tf="D"):
    return {"tf": tf, "kind": kind, "lo": lo, "hi": hi, "proximal": hi if kind == "demand" else lo, "distal": lo if kind == "demand" else hi,
            "base_start": "2026-03-04", "base_end": "2026-03-06", "legout_start": "2026-03-09", "legout_end": "2026-03-10"}


def _h4_for_demand():
    """Four-hour bars: a base drifting in the upper part of a 95-100 demand zone, then a leg out on 9 March."""
    rows = []
    t = pd.Timestamp("2026-03-03 18:00")
    price = 97.0
    while t < pd.Timestamp("2026-03-08 18:00"):
        if t.dayofweek < 5 or (t.dayofweek == 6 and t.hour >= 18):
            o = price
            c = price + (0.3 if (t.hour in (2, 10)) else -0.2)
            rows.append((t, o, max(o, c) + 0.3, min(o, c) - 0.6, c, 100.0))
            price = min(max(c, 97.5), 99.5)
        t += pd.Timedelta(hours=4)
    # leg out: 9 March session
    for k, (o, c) in enumerate([(99.0, 101.5), (101.5, 104.0), (104.0, 106.0), (106.0, 107.0), (107.0, 108.0), (108.0, 108.5)]):
        rows.append((pd.Timestamp("2026-03-08 18:00") + pd.Timedelta(hours=4 * k), o, c + 0.2, o - 0.2, c, 300.0))
    for k in range(6):
        o = 108.5 + k * 0.3
        rows.append((pd.Timestamp("2026-03-09 18:00") + pd.Timedelta(hours=4 * k), o, o + 0.5, o - 0.2, o + 0.3, 200.0))
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume"]).set_index("t")
    # pad history so the median range reference exists
    hist = pd.DataFrame({"Open": 96.0, "High": 96.6, "Low": 95.4, "Close": 96.0, "Volume": 100.0},
                        index=pd.date_range("2026-02-02 18:00", periods=120, freq="4h"))
    return pd.concat([hist, df]).sort_index()


def test_refine_demand_zone_tightens_to_upper_part():
    h4 = _h4_for_demand()
    r = ZI.refine(_zone(), h4, "4H", cutoff=18)
    assert r["available"] and r["tightens"]
    assert 95.0 <= r["lo"] < r["hi"] <= 100.0
    assert r["frac"] < ZI.TIGHTEN_MAX
    assert "upper" in r["note"]
    assert r["legout_at"].startswith("2026-03-08 18:00") or r["legout_at"].startswith("2026-03-0")
    assert 1 <= r["bars"] <= ZI.BASE_MAX


def test_refine_reports_missing_coverage_and_never_widens():
    h4 = _h4_for_demand()
    late = h4[h4.index >= "2026-03-05"]
    r = ZI.refine(_zone(), late, "4H", cutoff=18)
    assert r["available"] is False and "start" in r["note"]
    # a weekly zone refined by daily bars that span the whole level does not tighten
    days = pd.bdate_range("2026-02-02", periods=40)
    daily = pd.DataFrame({"Open": 96.5, "High": 100.5, "Low": 94.5, "Close": 99.5, "Volume": 1.0}, index=days)
    daily.iloc[-5:, daily.columns.get_loc("Close")] = 110.0
    daily.iloc[-5:, daily.columns.get_loc("Open")] = 105.0
    daily.iloc[-5:, daily.columns.get_loc("High")] = 111.0
    daily.iloc[-5:, daily.columns.get_loc("Low")] = 104.0
    z = _zone(tf="W")
    z.update(base_start="2026-02-02", base_end="2026-03-09", legout_start="2026-03-16", legout_end="2026-03-23")
    r2 = ZI.refine(z, daily, "D", cutoff=18)
    assert r2["available"] and r2["tightens"] is False
    assert "edges stand" in r2["note"]
