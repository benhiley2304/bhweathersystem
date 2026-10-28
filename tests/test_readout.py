import json, pathlib
from bhws_core.readout import build, _family_votes

BASE = pathlib.Path(__file__).parent / "fixtures" / "scores_6E_2026-10-28.json"


def load():
    return json.loads(BASE.read_text())


def test_regime_and_macro_count_once():
    fv = _family_votes({"regime": -1, "macro": -1, "cot": 1})
    assert fv["macro context"] == -1 and len(fv) == 2


def test_bias_is_not_a_trade():
    m = load()
    r = build(m, evidence={"data_confidence": {"grade": "High"},
                           "playbook": {"status": "Not turned", "direction": 1, "fade_side": "long"}})
    assert r["trade"]["state"] == "No executable trade from BHWS data"
    assert r["setup"]["state"] == "Positioning setup"
    assert r["setup"]["vs_bias"] == "against the bias"     # 6E: bearish backdrop, long fade


def test_low_conviction_reads_mixed_not_bearish():
    m = load()
    assert abs(m["conviction"]) < 0.5
    r = build(m, evidence={"data_confidence": {"grade": "High"}})
    assert r["bias"]["state"] == "Mixed"


def test_insufficient_data_blocks_setup():
    m = load()
    r = build(m, evidence={"data_confidence": {"grade": "Insufficient", "reasons": ["cot: missing"]},
                           "playbook": {"status": "Add eligible", "direction": 1}})
    assert r["bias"]["state"] == "Insufficient data"
    assert r["setup"]["state"] == "Not assessed"


def test_change_since_previous():
    m = load()
    prev = build(m, evidence={"data_confidence": {"grade": "High"}, "playbook": {"status": "No extreme"}})
    cur = build(m, evidence={"data_confidence": {"grade": "Low"},
                             "playbook": {"status": "Not turned", "direction": 1}}, previous=prev)
    assert any(c.startswith("setup:") for c in cur["changes"])
    assert any(c.startswith("data confidence:") for c in cur["changes"])


def test_bias_resting_on_positioning_alone_is_unsupported():
    # Sugar on 28 Oct 2026: engine Bearish, only bearish vote is COT, macro bullish.
    m = {"id": "SB", "bias_sign": -1, "conviction": -0.5, "direction": "Bearish", "tier": "Watch",
         "factor_votes": {"seasonal": 0, "regime": 0, "macro": 1, "momentum": 0, "cot": -1}}
    ev = {"data_confidence": {"grade": "High"},
          "playbook": {"status": "Past test window", "direction": -1, "fade_side": "short",
                       "managers_turning_reports": 0, "sma10": 18.752, "close_through_10w": False}}
    r = build(m, evidence=ev)
    assert r["bias"]["state"] == "Unsupported"
    assert "macro context" in r["bias"]["note"]
    assert r["positioning_vs_bias"] == "with the bias"
    assert r["setup"]["state"] == "Extreme; clock runs from your starter"
    assert any("selling two reports" in c for c in r["confirm_if"])
    assert any("below the 10-week average, 18.752" in c for c in r["confirm_if"])
    assert r["trade"]["state"] == "No executable trade from BHWS data"


def test_cot_vote_never_counts_as_bias_support():
    m = {"id": "X", "bias_sign": 1, "conviction": 2.0,
         "factor_votes": {"cot": 1, "macro": 1}}
    r = build(m, evidence={"data_confidence": {"grade": "High"}})
    assert [s["family"] for s in r["support"]] == ["macro context"]
