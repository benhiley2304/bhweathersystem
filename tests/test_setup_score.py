import math
from datetime import datetime

from bhws_core import setup_score as S


def _path(fn, n=20):
    return {"F": [fn(k) for k in range(S.K + 1)], "H": [0.5] * (S.K + 1), "n": n}


def _row(side=1, arch="Counter-trend", cstate="Waiting", tr="against", p="with", v="with",
         eng="Bearish", conf="High", gates=None, trend_dir=None, macro=None, state="running"):
    if trend_dir is None:
        trend_dir = -side if tr == "against" else side if tr == "with" else 0
    return {"side": side, "lean": side, "archetype": arch, "confirm": {"state": cstate},
            "pillars": {"trend": {"rel": tr}, "positioning": {"rel": p}, "valuation": {"rel": v}},
            "trend": {"dir": trend_dir, "state": state}, "macro_sign": macro,
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
    o = S.compute(_row(side=-1, eng="Bullish", macro=1), _path(lambda k: -0.15 * k), SD)
    assert o["timing"]["state"] == "in_window"
    assert o["relation"]["key"] == "counter"
    assert "Setup short against the backdrop" in o["relation"]["text"]


def test_backdrop_reads_trend_and_macro_together():
    # trend down and macro short: bearish; a long against it is counter-trend
    bd = S.backdrop({"trend": {"dir": -1, "state": "running"}, "macro_sign": -1})
    assert bd["label"] == "bearish" and bd["dir"] == -1 and bd["total"] == -2
    assert bd["text"].startswith("Backdrop bearish: trend down, running, macro leans short")
    # one pillar only is a lean; opposite signs are mixed; nothing is neutral
    assert S.backdrop({"trend": {"dir": 1, "state": "pullback"}, "macro_sign": None})["label"] == "leans bullish"
    mixed = S.backdrop({"trend": {"dir": 1, "state": "running"}, "macro_sign": -1})
    assert mixed["label"] == "mixed" and mixed["dir"] == 0
    assert S.backdrop({"trend": {"dir": 0, "state": "range"}, "macro_sign": 0})["label"] == "neutral"
    marginal = S.backdrop({"trend": {"dir": 0, "state": "range"}, "macro_sign": 1, "macro_marginal": True})
    assert "marginally leans long" in marginal["text"]
    # the relation is built from the backdrop, never from the engine composite
    o = S.compute(_row(side=1, eng="Bearish", trend_dir=1, macro=1), _path(lambda k: 0.15 * k), SD)
    assert o["relation"]["key"] == "aligned" and o["backdrop"]["label"] == "bullish"
    o2 = S.compute(_row(side=1, eng="Bullish", trend_dir=1, macro=-1), _path(lambda k: 0.15 * k), SD)
    assert o2["relation"]["key"] == "mixed"


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


def test_approaching_is_a_note_not_a_gate():
    from bhws_core import setup_score as ss
    # dip at 8 TD, then a strong leg; near term not against
    F = [0.0] * 81
    for k in range(81):
        F[k] = -0.2 * min(k, 8) / 8 if k <= 8 else -0.2 + 0.25 * (k - 8)
    t = ss.timing_state({"F": F, "n": 20}, 1, 0.02)
    assert t["state"] == "approaching"
    assert t["pool"] == "approaching"
