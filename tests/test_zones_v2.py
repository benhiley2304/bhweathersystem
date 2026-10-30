"""Second supply and demand detector (bhws_core.zones_v2): the three-step method on synthetic bars, the Yahoo
spot FX repair, deterministic two-day bars, visit hysteresis, stacking caps, selection and coverage, and the
/api/zones build with a fake fetcher."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bhws_core import zones_v2 as Z  # noqa: E402
from bhws_core.zones_v2 import Zone  # noqa: E402


def _bars(closes, rng=0.8, start="2024-01-02"):
    """Daily OHLC bars: open at the prior close, wicks 0.3*rng beyond the body."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng * 0.3
    l = np.minimum(o, c) - rng * 0.3
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}, index=idx)


def _dbr(n=140):
    """Drift at 100, a seven-bar drop to 92.6, a three-bar base, an explosive leg out and a run to 100, a flat
    stretch, one return whose wick reaches the preferred version, then a move away."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.2, n)
    c = np.full(n, 100.0) + noise
    c[40:47] = np.linspace(99.5, 92.6, 7) + noise[40:47] * 0.5
    c[47:50] = np.array([92.4, 92.7, 92.5]) + noise[47:50] * 0.2
    c[50], c[51], c[52], c[53] = 95.6, 97.4, 98.9, 100.2
    c[54:90] = 101.5 + noise[54:90]
    c[90:97] = np.linspace(101, 93.4, 7)
    c[97] = 92.9
    c[98:104] = np.array([93.8, 95.0, 96.3, 97.5, 98.5, 99.6])
    c[104:] = 100.5 + noise[104:]
    return c


def _demand_zones(zs):
    return [z for z in zs if z.kind == "demand" and z.lo < 93.0 and z.hi > 92.0]


def test_detects_drop_base_rally_demand_with_body_and_wick_edges():
    df = _bars(_dbr())
    zs = Z.detect(df, "D")
    dem = _demand_zones(zs)
    assert dem, [(z.kind, z.lo, z.hi) for z in zs]
    z = sorted(dem, key=lambda z: z.base_start)[0]
    assert z.pattern == "DBR"
    assert z.base_bars == 3
    assert z.explosive is True
    # proximal at the body extreme of the base, distal at the wick extreme behind it
    base = df.loc[z.base_start:z.base_end]
    assert z.proximal == pytest.approx(float(base[["Open", "Close"]].max().max()), abs=1e-9)
    assert z.distal == pytest.approx(float(base["Low"].min()), abs=1e-9)
    assert z.body_ratio is not None and z.body_ratio > 0.7
    # the single return reached the preferred version once and left: tested x1, not visiting now
    assert z.tests == 1 and z.fresh_state == "tested" and z.visiting_now is False
    assert z.broken is None


def test_reaction_from_an_older_zone_is_marked_not_original():
    df = _bars(_dbr())
    zs = Z.detect(df, "D")
    dem = sorted(_demand_zones(zs), key=lambda z: z.base_start)
    assert len(dem) >= 2
    origin, reaction = dem[0], dem[-1]
    assert origin.original is True
    assert reaction.original is False and "demand zone from" in reaction.origin_note
    assert Z.score(reaction) < Z.score(Zone(**{**reaction.__dict__, "original": True}))


def test_detects_rally_base_drop_supply_mirror():
    c = 200.0 - _dbr()          # mirror: the drop-base-rally becomes rally-base-drop around 100
    df = _bars(c)
    zs = Z.detect(df, "D")
    sup = [z for z in zs if z.kind == "supply" and z.hi > 107.0 and z.lo < 108.0]
    assert sup, [(z.kind, z.lo, z.hi) for z in zs]
    z = sorted(sup, key=lambda z: z.base_start)[0]
    assert z.pattern == "RBD" and z.base_bars == 3
    base = df.loc[z.base_start:z.base_end]
    assert z.proximal == pytest.approx(float(base[["Open", "Close"]].min().min()), abs=1e-9)
    assert z.distal == pytest.approx(float(base["High"].max()), abs=1e-9)


def test_broken_zone_is_marked_and_scored_down():
    c = _dbr()
    c[120:] = 88.0              # a close below the distal edge breaks the demand zone
    df = _bars(c)
    zs = Z.detect(df, "D")
    dem = sorted(_demand_zones(zs), key=lambda z: z.base_start)
    assert dem and dem[0].fresh_state == "broken" and dem[0].broken is not None
    assert Z.score(dem[0]) < 30
    assert dem[0] not in Z.live(zs, 88.0, 0.8)


def test_spot_fx_defect_detected_and_repaired():
    idx = pd.bdate_range("2024-01-01", periods=80)
    close = 1.10 + np.cumsum(np.random.default_rng(1).normal(0, 0.002, 80))
    bad = pd.DataFrame({"Open": close, "Close": close, "High": close + 0.003, "Low": close - 0.003}, index=idx)
    assert Z.looks_like_spot_fx_defect(bad) is True
    good = _bars(100 + np.cumsum(np.random.default_rng(2).normal(0, 0.5, 80)))
    assert Z.looks_like_spot_fx_defect(good) is False
    fixed = Z.repair_spot_fx(bad)
    assert np.allclose(fixed["Open"].to_numpy(), close)
    assert np.allclose(fixed["Close"].to_numpy()[:-1], close[1:])
    assert fixed["Close"].iloc[-1] == pytest.approx(close[-1])
    assert (fixed["High"] >= fixed[["Open", "Close"]].max(axis=1)).all()
    assert (fixed["Low"] <= fixed[["Open", "Close"]].min(axis=1)).all()


def test_two_day_bars_are_anchored_and_do_not_repair_when_a_day_is_added():
    df = _bars(100 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, 300)), start="2023-10-02")
    r = Z.resample_2d(df)
    assert abs(len(r) - len(df) / 2) <= 1.5
    # anchored on the first bar of the first full calendar year: Jan 1 2024 is a Monday, so (Jan 1, Jan 2) is one bar
    assert pd.Timestamp("2024-01-02") in r.index and pd.Timestamp("2024-01-01") not in r.index
    # high and low cover both days, open is the first day's open, close the second day's close
    first, second = df.loc["2024-01-01"], df.loc["2024-01-02"]
    bar = r.loc["2024-01-02"]
    assert bar["Open"] == first["Open"] and bar["Close"] == second["Close"]
    assert bar["High"] == max(first["High"], second["High"]) and bar["Low"] == min(first["Low"], second["Low"])
    # adding one more day leaves every earlier bar unchanged
    longer = _bars(100 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, 300)), start="2023-10-02")
    extra = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000]},
                         index=[longer.index[-1] + pd.offsets.BDay(1)])
    r2 = Z.resample_2d(pd.concat([longer, extra]))
    common = r.index[:-1]
    pd.testing.assert_frame_equal(r.loc[common], r2.loc[common])


def _zone(kind, lo, hi, tf="D", base="2024-03-01", state="fresh", tests=0, atr=1.0, pref=None):
    proximal, distal = (hi, lo) if kind == "demand" else (lo, hi)
    b = pd.Timestamp(base)
    return Zone(tf=tf, kind=kind, pattern="DBR" if kind == "demand" else "RBD", proximal=proximal, distal=distal,
                pref_proximal=pref if pref is not None else proximal, base_start=b, base_end=b + pd.Timedelta(days=2),
                legout_start=b + pd.Timedelta(days=3), legout_end=b + pd.Timedelta(days=5), base_bars=3,
                height=hi - lo, height_atr=(hi - lo) / atr, atr=atr, run_atr=2.5, run_bars=3, swing_h=4.0,
                body_ratio=0.8, explosive=True, base_small=0.2, legin_atr=-2.0, tests=tests, fresh_state=state)


def test_visit_hysteresis_counts_one_test_for_a_hovering_visit():
    z = _zone("demand", 98.0, 100.0, pref=99.5)
    # bars after the leg out (index 0 is the leg-out end): a touch, a hover just above the zone, a touch again,
    # a departure, a second visit, then a move far away
    L = np.array([104, 104, 104, 99.4, 100.3, 99.6, 102.0, 103.0, 103.0, 99.2, 103.0, 104.0, 104.0, 104.0])
    H = L + 1.5
    C = L + 0.8
    idx = pd.bdate_range("2024-03-11", periods=len(L))
    Z._walk_forward(z, H, L, C, idx, 0, 1, Z.DEFAULTS["D"])
    assert z.tests == 2, z.__dict__
    assert z.visiting_now is False and z.fresh_state == "tested"
    assert z.last_test == idx[9]


def test_visit_in_progress_is_the_arrival_not_a_test():
    z = _zone("demand", 98.0, 100.0, pref=99.5)
    L = np.array([104, 104, 104, 103, 103, 102, 101, 99.7, 99.6])
    H = L + 1.5
    C = L + 0.8
    idx = pd.bdate_range("2024-03-11", periods=len(L))
    Z._walk_forward(z, H, L, C, idx, 0, 1, Z.DEFAULTS["D"])
    assert z.visiting_now is True and z.tests == 0 and z.fresh_state == "fresh"


def test_stack_joins_neighbours_within_caps_only():
    a = _zone("demand", 90.0, 91.0, base="2024-03-01")
    b = _zone("demand", 91.5, 92.3, base="2024-04-01")        # gap 0.5 <= min height: joins
    c = _zone("demand", 94.0, 95.0, base="2024-05-01")        # gap 1.7 > min height: separate
    far = _zone("demand", 88.0, 89.0, base="2022-01-10")      # base too old (200 days on the daily): separate
    out = Z.stack([a, b, c, far], price=100.0)
    merged = [z for z in out if z.parts]
    assert len(merged) == 1
    m = merged[0]
    assert m.lo == pytest.approx(90.0) and m.hi == pytest.approx(92.3)
    assert len(m.parts) == 2 and m.proximal == pytest.approx(92.3) and m.distal == pytest.approx(90.0)
    assert any("level on level: 2 bases" in n for n in m.notes)
    assert len(out) == 3


def test_stack_respects_height_cap_and_three_base_limit():
    zs = [_zone("supply", 100.0 + i * 1.2, 101.0 + i * 1.2, base=f"2024-0{3 + i // 3}-{10 + (i % 3) * 5:02d}") for i in range(5)]
    out = Z.stack(zs, price=90.0)
    assert all(len(z.parts) <= 3 for z in out)
    for z in out:
        assert z.height <= 3.5 * z.atr + 1e-9


def test_select_shows_inside_zone_and_prefers_fresh_anchor():
    inside = _zone("demand", 99.0, 100.5, base="2024-06-01")
    tested_near = _zone("demand", 96.0, 97.0, base="2024-05-01", state="tested", tests=1)
    fresh_far = _zone("demand", 92.0, 93.0, base="2024-04-01")
    fresh_further = _zone("demand", 85.0, 86.0, base="2024-02-01")
    sup_fresh = _zone("supply", 104.0, 105.0, base="2024-05-15")
    sup_tested = _zone("supply", 106.0, 107.0, base="2024-03-15", state="tested", tests=2)
    shown, rest = Z.select([inside, tested_near, fresh_far, fresh_further, sup_fresh, sup_tested], price=100.0, per_side=2, typ=1.0)
    assert inside in shown
    dem = [z for z in shown if z.kind == "demand"]
    assert len(dem) <= 2 and fresh_further not in shown
    sup = [z for z in shown if z.kind == "supply"]
    assert sup_fresh in sup and len(sup) <= 2
    ids_shown, ids_rest = {id(z) for z in shown}, {id(z) for z in rest}
    assert ids_shown.isdisjoint(ids_rest) and len(ids_shown) + len(ids_rest) == 6
    # without an inside zone the nearest fresh zone anchors, and a tested zone closer to price joins it
    shown2, _ = Z.select([tested_near, fresh_far, fresh_further], price=100.0, per_side=2, typ=1.0)
    assert fresh_far in shown2 and tested_near in shown2 and fresh_further not in shown2


def test_covered_requires_half_of_the_zone_inside_a_weekly_zone_of_the_same_kind():
    w = _zone("demand", 90.0, 95.0, tf="W")
    assert Z.covered(_zone("demand", 93.0, 94.0), [w]) is w
    assert Z.covered(_zone("demand", 94.5, 96.5), [w]) is None        # a quarter inside
    assert Z.covered(_zone("supply", 93.0, 94.0), [w]) is None        # wrong kind


def test_curve_location_and_action_matrix():
    dem = _zone("demand", 90.0, 92.0, tf="W")
    sup = _zone("supply", 108.0, 110.0, tf="W")
    assert Z.curve(95.0, [dem, sup])["label"] == "low"
    assert Z.curve(100.0, [dem, sup])["label"] == "equilibrium"
    assert Z.curve(106.0, [dem, sup])["label"] == "high"
    assert Z.curve(91.0, [dem, sup])["label"] == "very low"
    assert Z.curve(109.0, [dem, sup])["label"] == "very high"
    assert Z.curve(95.0, [sup])["label"] == "no weekly demand below"
    assert Z.action("demand", "low", "down")["allowed"] is True and Z.action("demand", "low", "down")["tier"] == "weak"
    assert Z.action("demand", "high", "up")["allowed"] is False
    assert Z.action("supply", "equilibrium", "down")["allowed"] is True
    assert Z.action("supply", "equilibrium", "up")["allowed"] is False
    assert Z.action("demand", "very low", "up")["tier"] == "best"


def test_api_build_with_fake_fetcher(monkeypatch):
    import bhws_zones_api as A

    daily = _bars(_dbr(400), start="2023-03-01")
    weekly = daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    monkeypatch.setattr(A, "_fetch", lambda symbol, yfw: {"D": daily, "W": weekly})
    res = A._build("TEST", "TEST=F", lambda fn, *a, **k: fn(*a, **k))
    assert res["detector"] == "v2" and res["market"] == "TEST"
    assert set(res["zones"]) == {"D", "2D", "W"} and set(res["candles"]) == {"D", "2D", "W"}
    for tf in ("D", "2D", "W"):
        assert len(res["candles"][tf]) >= 60
        for z in res["zones"][tf]:
            assert z["lo"] < z["hi"] and z["kind"] in ("demand", "supply")
            assert "room" in z and "covered_by" in z and "action" in z and "inside" in z
            assert z["fresh_state"] in ("fresh", "shallow", "tested")
    assert isinstance(res["read"], list) and res["read"]
    assert res["read"][-1].startswith("These are the method's rules")
    assert "evidence" in res["definitions"] and "not been replayed" in res["definitions"]["evidence"]
