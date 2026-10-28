"""Strategy layer: grades and setup types follow the written hierarchy.

These tests pin behaviour, not edge. The weights and thresholds encode the
report rules (positioning and valuation are the payoff, trend is context and
timing, seasonality a minor tilt, macro the hit rate); none were fitted.
"""
from bhws_core import strategy as S

DX_ENGINE = {"trend_state": "Uptrend · short-term flat",
             "scores": {"momentum": {"detail": {"pct_range": 94.6, "roc4w_pct": 0.62, "roc13w_pct": 0.03,
                                                 "roc26w_pct": 1.63, "sma200_pct_diff": 2.64,
                                                 "efficiency_ratio": 0.27}}}}
NQ_ENGINE = {"trend_state": "Uptrend · short-term pullback (entry-watch for longs)",
             "scores": {"momentum": {"detail": {"pct_range": 81.6, "roc4w_pct": -3.6, "roc13w_pct": -3.8,
                                                 "roc26w_pct": 3.4, "sma200_pct_diff": 6.5,
                                                 "efficiency_ratio": 0.12}}}}
RUN_DOWN = {"trend_state": "Downtrend · momentum aligned (continuation)",
            "scores": {"momentum": {"detail": {"pct_range": 7.0, "roc4w_pct": -0.1, "roc13w_pct": -3.8,
                                                "roc26w_pct": -7.3, "sma200_pct_diff": -10.4}}}}


def _fwd(m30, m60, spread=1.0):
    return {30: {"median": m30, "p25": m30 - spread, "p75": m30 + spread, "hit_up": 50, "n": 30},
            60: {"median": m60, "p25": m60 - spread, "p75": m60 + spread, "hit_up": 50, "n": 30}}


def _val(daily, weekly):
    return {"daily": {"lines": {k: {"value": v} for k, v in daily.items()}},
            "lines": {k: {"value": v} for k, v in weekly.items()}}


def test_trend_profile_states():
    assert S.trend_profile(DX_ENGINE)["state"] == "extended"
    assert S.trend_profile(NQ_ENGINE)["state"] == "pullback"
    assert S.trend_profile(RUN_DOWN)["state"] == "running"
    assert S.trend_profile({"trend_state": "Rangebound · no clear trend"})["state"] == "range"


def test_divergence_detects_lower_high_in_managers():
    prices = [100 + i * 0.1 for i in range(30)]          # steady higher highs
    mgr = [50.0] * 30
    mgr[15] = 98.0                                          # peak 14 reports ago
    for i in range(16, 30):
        mgr[i] = 75.0                                       # well off the peak
    out = S.manager_divergence(prices, mgr)
    assert out["dir"] == -1 and "lower high" in out["text"]
    # no divergence when managers keep making highs
    mgr2 = [60 + i for i in range(30)]
    assert not S.manager_divergence(prices, mgr2).get("dir")


def test_valuation_needs_timeframes_to_agree():
    rich = S.valuation_read(_val({"zb": 45, "gc": 50}, {"zb": 30, "gc": 70}))
    assert rich["sign"] == -1 and rich["strength"] >= 0.5
    split = S.valuation_read(_val({"zb": -57, "gc": 39, "eur": -5}, {"zb": 27, "gc": 69}))
    assert split["sign"] == -1 and "zb" in split["split"] and split["cheap"] == []
    fair = S.valuation_read(_val({"zb": -60, "dxy": 60}, {"zb": -50, "dxy": 50}))
    assert fair["sign"] == 0
    daily_only = S.valuation_read(_val({"zb": -47, "dxy": -49}, {"zb": -7, "dxy": -56}))
    assert daily_only["sign"] == 1 and set(daily_only["cheap"]) == {"zb", "dxy"}


def test_seasonal_forward_wraps_the_year_end():
    years = {"2020": [i * 0.01 for i in range(252)], "2021": [i * 0.02 for i in range(252)],
             "2026": [0.0] * 100}
    out = S.seasonal_forward(years, {"2020": 1.0, "2021": 1.0}, current_td=240, current_year=2026,
                             horizons=(30,))
    assert out[30]["n"] == 1            # only 2020 has a following year in the sample


def test_seasonal_flat_band():
    assert S.seasonal_read(_fwd(-0.34, -0.38, 2.7))["sign"] == 0
    up = S.seasonal_read(_fwd(3.4, 4.3, 2.0))
    assert up["sign"] == 1 and up["word"] == "tends up"
    assert S.seasonal_read(_fwd(1.0, -1.0, 0.5))["sign"] == 0      # 30 and 60 disagree


def _assess(engine, pos, pb, div, val, fwd, macro, equity=False):
    m = {"id": "NQ" if equity else "DX", "category": "equity" if equity else "fx"}
    return S.assess(market=m, engine=engine, pos=pos, pb=pb, div=div, val=val, seas_fwd=fwd,
                    macro=macro, macro_marginal=False, conf="High")


def test_dx_like_exhaustion_fade_grades_prime_short():
    pos = {"commercials": 18, "managers": 79, "retail": 89}
    div = {"dir": -1, "text": "lower high"}
    val = _val({"zb": -57, "gc": 39, "eur": -5}, {"zb": 27, "gc": 69})
    a = _assess(DX_ENGINE, pos, {"status": "No extreme", "direction": -1}, div, val, _fwd(-0.34, -0.38, 2.7), 1)
    assert a["side"] == -1 and a["grade"]["key"] == "prime"
    assert a["archetype"] == "Divergence fade"
    assert a["rel"]["trend"] == "exhausted" and a["rel"]["macro"] == "against"
    assert any("size down" in x for x in a["sizing"])


def test_nq_like_pullback_is_prime_long_and_positioning_only_warns():
    pos = {"commercials": 2, "managers": 97, "retail": 76}
    val = _val({"zb": -47, "dxy": -49}, {"zb": -7, "dxy": -56})
    a = _assess(NQ_ENGINE, pos, {"status": "Not turned", "direction": -1}, None, val, _fwd(3.4, 4.3, 2.0), -1,
                equity=True)
    assert a["side"] == 1 and a["grade"]["key"] == "prime"
    assert a["archetype"] == "Pullback in trend"
    assert a["rel"]["positioning"] == "against"
    assert any("size warning" in x for x in a["sizing"])


def test_cheap_into_a_running_downtrend_is_not_a_fade():
    val = _val({"zb": -67, "dxy": -45}, {"zb": -63, "dxy": -69})
    a = _assess(RUN_DOWN, {"commercials": 56, "managers": 34, "retail": 79},
                {"status": "No extreme", "direction": 0}, None, val, _fwd(-0.3, 3.3, 3.0), 0)
    assert a["rel"]["trend"] == "against"
    assert a["grade"]["key"] in ("developing", "none")
