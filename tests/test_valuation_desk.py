import math
from datetime import datetime, timezone

import numpy as np

from bhws_core import desk, valuation

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


# ── valuation (Pine port) ───────────────────────────────────────────────
def test_roc_matches_pine_formula():
    c = [100.0] * 11 + [110.0]
    r = valuation.roc(c)
    assert r[10] is None and math.isclose(r[11], 10.0)


def test_rescale_bounds_and_flat():
    v = valuation.rescale(list(range(100)))
    assert v[98] is None and math.isclose(v[99], 100.0)
    flat = valuation.rescale([5.0] * 120)
    assert flat[-1] == 0.0


def test_valuation_line_sign():
    # chart rising faster than reference -> rich (line near +100)
    sym = [100 * (1.01 ** i) for i in range(160)]
    ref = [100.0 + 0.01 * i for i in range(160)]
    sym[-1] *= 1.10
    line = valuation.valuation_line(sym, ref)
    assert line[-1] > 75
    assert valuation.state(line[-1]) == "Overvalued"
    assert valuation.state(-50) == "Cheap" and valuation.state(-80) == "Undervalued"


def test_buy_points_entry_exit_and_cooldown():
    n = 30
    zb = [0.0] * n
    dx = [0.0] * n
    closes = [100.0 + i for i in range(n)]
    for i in range(5, 8):
        zb[i] = dx[i] = -60.0
    bp = valuation.buy_points(zb, dx, closes)
    assert len(bp["trades"]) == 1
    t = bp["trades"][0]
    assert t["entry_bar"] == 5 and t["exit_bar"] == 8   # average back above -25 on bar 8
    # max hold: stays cheap for 15 bars -> exit after 10
    zb2 = [-60.0] * n
    bp2 = valuation.buy_points(zb2, zb2, closes)
    assert bp2["trades"][0]["exit_bar"] - bp2["trades"][0]["entry_bar"] == 10
    # re-arm needs 2 bars after the exit
    assert bp2["trades"][1]["entry_bar"] - bp2["trades"][0]["exit_bar"] >= 2


def test_assess_skips_self_reference_and_short_series():
    assert valuation.assess("GC", {"sym": [1.0] * 50}) is None
    n = 200
    rng = np.random.default_rng(1)
    s = list(np.cumprod(1 + rng.normal(0, 0.01, n)) * 100)
    out = valuation.assess("ZB", {"sym": s, "zb": s, "dxy": list(reversed(s))})
    assert "zb" not in out["lines"] and "dxy" in out["lines"]


# ── desk ────────────────────────────────────────────────────────────────
def _pb(status, direction):
    return {"status": status, "direction": direction, "managers_turning_reports": 1}


def test_seasonal_tilt_requires_grade_and_agreeing_hit_rate():
    assert desk.seasonal_tilt({"median_pct": 1.2, "hit_rate": 65, "reliability_grade": "B"})["sign"] == 1
    assert desk.seasonal_tilt({"median_pct": -1.2, "hit_rate": 35, "reliability_grade": "A"})["sign"] == -1
    weak = desk.seasonal_tilt({"median_pct": 1.2, "hit_rate": 65, "reliability_grade": "D"})
    assert weak["sign"] == 0 and "grade" in weak["why"]
    contra = desk.seasonal_tilt({"median_pct": 1.2, "hit_rate": 45, "reliability_grade": "B"})
    assert contra["sign"] == 0
    assert desk.seasonal_tilt(None)["sign"] is None


def test_classify_aligned_vs_macro_against():
    seas0 = {"sign": 0}
    a = desk.classify(pb=_pb("Not turned", 1), macro=1, trend=0, seas=seas0, val_sign=None, conf="High")
    assert a["class"]["key"] == "aligned" and a["side"] == 1 and a["primary_edge"] == "Positioning"
    b = desk.classify(pb=_pb("Not turned", 1), macro=-1, trend=0, seas=seas0, val_sign=None, conf="High")
    assert b["class"]["key"] == "payoff_vs_macro"
    c = desk.classify(pb=_pb("Not turned", 1), macro=1, trend=0, seas={"sign": -1}, val_sign=None, conf="High")
    assert c["class"]["key"] == "payoff" and "seasonality against: no adds" in c["gates"]


def test_classify_backdrop_lapsed_none():
    d = desk.classify(pb={"status": "No extreme", "direction": 0}, macro=-1, trend=-1,
                      seas={"sign": 0}, val_sign=None, conf="High")
    assert d["class"]["key"] == "backdrop" and d["side"] == -1
    e = desk.classify(pb={"status": "Extreme persisting", "direction": -1}, macro=0, trend=1,
                      seas={"sign": 0}, val_sign=None, conf="High")
    assert e["class"]["key"] == "lapsed"
    f = desk.classify(pb={"status": "No extreme", "direction": 0}, macro=1, trend=-1,
                      seas={"sign": 0}, val_sign=None, conf="High")
    assert f["class"]["key"] == "none" and f["side"] == 0


def test_order_prefers_aligned_then_triggered_and_penalises_bad_data():
    s = {"sign": 0}
    turned = desk.classify(pb=_pb("Turned", 1), macro=1, trend=1, seas=s, val_sign=None, conf="High")
    waiting = desk.classify(pb=_pb("Not turned", 1), macro=1, trend=1, seas=s, val_sign=None, conf="High")
    vs = desk.classify(pb=_pb("Turned", 1), macro=-1, trend=1, seas=s, val_sign=None, conf="High")
    low = desk.classify(pb=_pb("Turned", 1), macro=1, trend=1, seas=s, val_sign=None, conf="Low")
    assert turned["order"] < waiting["order"] < vs["order"]
    assert low["order"] > turned["order"] and "data confidence low" in low["gates"]


def test_positioning_changes_flags_new_extreme():
    mgr = [100 + (i * 37) % 200 for i in range(200)] + [150, 10000]
    out = desk.positioning_changes(mgr, mgr, mgr)
    assert out["managers"]["pct"] > 90 and out["managers"]["event"] == "entered extreme"
    assert out["managers"]["chg_1"] is not None


def test_event_gate_filters_currency_and_window():
    ev = [{"name": "CPI", "currency": "USD", "impact": "High", "datetime_utc": "2026-10-29T12:30:00+00:00"},
          {"name": "BoJ", "currency": "JPY", "impact": "High", "datetime_utc": "2026-10-30T03:00:00+00:00"},
          {"name": "PMI", "currency": "EUR", "impact": "Medium", "datetime_utc": "2026-10-29T09:00:00+00:00"},
          {"name": "Old", "currency": "USD", "impact": "High", "datetime_utc": "2026-10-01T12:30:00+00:00"}]
    g = desk.event_gate(["EUR", "USD"], ev, NOW)
    assert [x["name"] for x in g] == ["CPI"]
    assert desk.currencies({"id": "GBPCHF", "category": "fx_cross", "base_leg": "6B", "quote_leg": "6S"}) == ["GBP", "CHF"]
    assert desk.currencies({"id": "6J", "category": "fx"}) == ["JPY", "USD"]
    assert desk.currencies({"id": "GC", "category": "commodity"}) == ["USD"]


def test_trend_sign_from_state():
    assert desk.trend_sign({"trend_state": "Downtrend · short-term flat"}) == -1
    assert desk.trend_sign({"trend_state": "Rangebound · no clear trend"}) == 0
    assert desk.trend_sign({"factor_votes": {"momentum": 1}}) == 1


def test_clusters_find_one_bet():
    rng = np.random.default_rng(0)
    f = rng.normal(0, 1, 52)
    rets = {"A": list(f + rng.normal(0, 0.2, 52)), "B": list(f + rng.normal(0, 0.2, 52)),
            "C": list(-f + rng.normal(0, 0.2, 52)), "D": list(rng.normal(0, 1, 52))}
    # long A, long B, short C: all the same bet
    out = desk.correlation_clusters(rets, {"A": 1, "B": 1, "C": -1, "D": 1})
    assert out["clusters"] and set(out["clusters"][0]["members"]) == {"A", "B", "C"}
    assert out["clusters"][0]["effective_bets"] < 1.5
    assert out["effective_bets"] < 4


def test_brief_numbers_come_from_inputs():
    rows = [{"id": "6E", "name": "Euro", "side": 1, "class": {"key": "aligned"},
             "playbook": {"status": "Turned"}},
            {"id": "SB", "name": "Sugar", "side": -1, "class": {"key": "payoff_vs_macro"},
             "playbook": {"status": "Not turned"}}]
    fed = {"snapshots": {"now": {"meetings": [{"date": "2026-12-09", "cum_bp": 24.7, "p_hike_cum": 98.6}]}}}
    b = desk.brief(regime={"regime": "Neutral", "score_10": 5.2, "vix_level": 17.19}, fed=fed,
                   events=[], rows=rows, clusters=None, now=NOW)
    text = " ".join(b["paragraphs"])
    assert "+25bp by the 9 Dec FOMC" in text and "99% implied odds" in text
    assert "Euro long" in text and "Sugar has macro against" in text
    assert "not a probability" in text
