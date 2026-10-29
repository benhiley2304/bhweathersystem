import math
from datetime import datetime

from bhws_core import setup_score as S


def _path(fn, n=20):
    return {"F": [fn(k) for k in range(S.K + 1)], "H": [0.5] * (S.K + 1), "n": n}


def _row(side=1, arch="Counter-trend", cstate="Waiting", tr="against", p="with", v="with",
         eng="Bearish", conf="High", gates=None):
    return {"side": side, "lean": side, "archetype": arch, "confirm": {"state": cstate},
            "pillars": {"trend": {"rel": tr}, "positioning": {"rel": p}, "valuation": {"rel": v}},
            "engine": {"direction": eng}, "data_confidence": conf, "gates": gates or []}


SD = 0.02  # 2% weekly sd -> 1 sd = 2%


def test_against_is_veto():
    o = S.compute(_row(), _path(lambda k: -0.12 * k), SD)  # falls ~2.4% by 20 TD
    assert o["timing"]["state"] == "against"
    assert o["ready"]["key"] == "veto"


def test_wait_for_window_projects_higher_score():
    # dip to -1.5% at 15 TD, then +6% rally
    f = lambda k: (-0.2 * k) if k <= 15 else (-3.0 + 0.3 * (k - 15))
    o = S.compute(_row(), _path(f), SD, datetime(2026, 10, 29))
    t = o["timing"]
    assert t["state"] == "wait" and t["entry_td"] == 15
    assert o["ready"]["key"] == "wait"
    assert o["projected"]["score"] == min(10.0, o["score"] + 3.0)
    assert t["entry_date"] > "2026-10-29"
    assert t["hold_weeks"][1] <= 8


def test_in_window_now():
    o = S.compute(_row(), _path(lambda k: 0.15 * k), SD)
    assert o["timing"]["state"] == "in_window"
    assert o["timing"]["entry_td"] == 0


def test_short_side_flips_path():
    o = S.compute(_row(side=-1, eng="Bullish"), _path(lambda k: -0.15 * k), SD)
    assert o["timing"]["state"] == "in_window"
    assert o["relation"]["key"] == "counter"
    assert "Bias bullish, setup short" in o["relation"]["text"]


def test_score_bounds_and_parts():
    o = S.compute(_row(arch="Crowd fade", cstate="Turned", tr="with", p="neutral", v="neutral"), _path(lambda k: 0.0), SD)
    assert 0 <= o["score"] <= 10
    assert sum(p["pts"] for p in o["parts"]) + 5 == o["score"]
    assert o["score"] * 2 == math.floor(o["score"] * 2)


def test_low_confidence_cap():
    o = S.compute(_row(conf="Low"), _path(lambda k: 0.0), SD)
    assert o["score"] <= 6


def test_no_side():
    o = S.compute(_row(side=0), None, SD)
    assert o["score"] is None and o["ready"]["key"] == "none"


def test_missing_path_is_na_not_invented():
    o = S.compute(_row(), None, SD)
    assert o["timing"]["state"] == "na"
    assert [p for p in o["parts"] if p["key"] == "timing"][0]["pts"] == 0.0


def test_seasonal_path_basic():
    years = {str(y): [float(i) * 0.01 for i in range(252)] for y in range(2000, 2026)}
    w = {str(y): 1.0 for y in range(2000, 2026)}
    p = S.seasonal_path(years, w, 100, 2026)
    assert p and p["n"] == 26 and p["F"][0] == 0.0 and p["F"][10] > 0
    # crosses year end
    p2 = S.seasonal_path(years, w, 240, 2026)
    assert p2["F"][40] is not None
