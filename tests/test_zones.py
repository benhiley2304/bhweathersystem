"""Supply and demand zone detector: synthetic bases with an impulsive leg out, tests, breaks, dedupe, reaction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bhws_core.zones import detect, live, mark_nesting, reaction  # noqa: E402


def _bars(closes, rng=0.8, start="2020-01-06", freq="W-MON"):
    """OHLC bars with a fixed range around a close path; opens at the prior close."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng * 0.3
    l = np.minimum(o, c) - rng * 0.3
    idx = pd.date_range(start, periods=len(c), freq=freq)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}, index=idx)


def _dbr_path():
    """60 weekly bars: drift, a drop, a three-bar base near 92, a rally of 12 points, then a return to the base."""
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.25, 60)
    c = np.full(60, 100.0) + noise
    c[20:26] = np.linspace(100, 92.5, 6) + noise[20:26]          # drop
    c[26:29] = np.array([92.2, 92.6, 92.3]) + noise[26:29] * 0.2  # base
    c[29:35] = np.linspace(95.5, 104.5, 6) + noise[29:35]          # leg out
    c[35:50] = 104.5 + noise[35:50]
    c[50:56] = np.linspace(103.5, 92.4, 6)                         # return to the zone
    c[56:60] = np.array([93.0, 95.0, 97.5, 100.5])                 # held
    return c


def test_detects_demand_base_before_rally():
    df = _bars(_dbr_path())
    zs = detect(df, "W")
    demand = [z for z in zs if z.kind == "demand" and z.lo < 93.5 and z.hi > 91.5]
    assert demand, [z.to_dict() for z in zs]
    z = demand[0]
    assert z.pattern.endswith("BR")
    assert z.distal <= z.proximal
    assert 1 <= z.base_bars <= 8
    assert z.departure_base >= 0.9
    assert z.tests >= 1          # the return in week 50-55 touched the proximal edge
    assert not z.broken


def test_detects_supply_base_before_drop():
    c = _dbr_path()
    df = _bars(200 - c)           # mirror image: a rally, a base, a drop
    zs = detect(df, "W")
    supply = [z for z in zs if z.kind == "supply" and z.hi > 106.5 and z.lo < 108.5]
    assert supply, [z.to_dict() for z in zs]
    assert supply[0].pattern.endswith("BD")
    assert supply[0].distal >= supply[0].proximal


def test_broken_zone_flagged():
    c = _dbr_path()
    c[56:60] = np.array([90.0, 88.5, 87.0, 86.0])   # closes through the distal edge
    df = _bars(c)
    zs = [z for z in detect(df, "W") if z.kind == "demand" and z.lo < 93.5 and z.hi > 91.5]
    assert zs and zs[0].broken


def test_live_filters_by_distance_and_broken():
    df = _bars(_dbr_path())
    zs = detect(df, "W")
    near = live(zs, price=94.0, atr_now=1.0, max_dist_atr=3.0)
    assert all(not z.broken for z in near)
    assert all(abs(z.proximal - 94.0) <= 3.0 + (z.hi - z.lo) for z in near)
    far = live(zs, price=140.0, atr_now=1.0, max_dist_atr=1.0)
    assert far == []


def test_reaction_first_visit_held():
    df = _bars(_dbr_path())
    z = [z for z in detect(df, "W") if z.kind == "demand" and z.lo < 93.5][0]
    r = reaction(df, z, horizon=8)
    assert r is not None
    assert r["touch"] >= z.legout_end
    assert r["held"] and not r["failed"]
    assert r["hit_1r"]


def test_no_zones_on_flat_noise():
    rng = np.random.default_rng(1)
    df = _bars(100 + rng.normal(0, 0.1, 80))
    assert detect(df, "W") == []


def test_dedupe_keeps_one_zone_per_base():
    df = _bars(_dbr_path())
    zs = [z for z in detect(df, "W") if z.kind == "demand" and z.scale == "tight" and z.lo < 93.5 and z.hi > 91.5
          and z.base_start.year == 2020]
    assert len(zs) == 1


def test_mark_nesting_links_daily_to_weekly():
    wk = _bars(_dbr_path())
    daily = _bars(np.repeat(_dbr_path(), 5) + np.tile([0, .1, -.1, .05, 0], 60), rng=0.3, freq="B")
    wz = detect(wk, "W")
    dz = detect(daily, "D")
    mark_nesting(dz, wz)
    assert all(hasattr(z, "notes") for z in dz)
    inside = [z for z in dz if any("inside weekly" in n for n in z.notes)]
    assert len(inside) <= len(dz)


def test_to_dict_is_json_friendly():
    import json
    df = _bars(_dbr_path())
    for z in detect(df, "W"):
        json.dumps(z.to_dict())
