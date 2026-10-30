"""Crowd map: research loader, measured gate text and the /api/complex builder."""
import json
from pathlib import Path

import pytest

from bhws_core import crowd
from bhws_core.desk import _crowd_history
import bhws_complex_api as capi

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "research" / "positioning" / "complex_evidence.json"


def test_evidence_file_present_and_consistent():
    ev = json.loads(EV.read_text())
    for k in ("built", "method", "complexes", "market_complex", "haircut", "haircut_pooled", "relative"):
        assert k in ev
    # every market in a complex has a haircut record and vice versa
    for mid, ck in ev["market_complex"].items():
        assert mid in ev["complexes"][ck]["ids"]
        assert mid in ev["haircut"]
    nq = ev["haircut"]["NQ"]
    assert nq["all_13"]["n"] > 500 and nq["long_crowded_13"]["n"] > 50
    # the short block is the short trade's return (sign flipped), so all_short == -all
    assert abs(nq["all_short_13"]["mean_pct"] + nq["all_13"]["mean_pct"]) < 1e-6


@pytest.mark.parametrize("mgr,comm,exp", [
    (96.8, 1.9, "crowded_long"), (5.0, 95.0, "crowded_short"), (92.0, 40.0, "managers_high"),
    (8.0, 50.0, "managers_low"), (50.0, 50.0, "none"), (None, 50.0, "none"), (90.0, 10.0, "crowded_long"),
])
def test_crowd_state_thresholds(mgr, comm, exp):
    assert crowd.crowd_state(mgr, comm) == exp


def test_same_side_text_reports_haircut_for_nq_longs():
    sh = crowd.side_history("NQ", 1)
    assert sh and sh["lift_pct"] < 0
    txt = crowd.same_side_text("NQ", 1)
    assert txt.startswith("crowd on the same side") and "Size down." in txt
    assert "13 weeks" in txt and "independent" in txt


def test_same_side_text_reports_persistence_when_no_penalty():
    txt = crowd.same_side_text("GC", 1)
    assert txt and "Size down." not in txt and "persist" in txt


def test_fade_text_verdicts_follow_the_numbers():
    si = crowd.fade_text("SI", 1)      # silver longs against a crowded short: measured lift
    assert si and "a measured lift" in si
    kc = crowd.fade_text("KC", 1)      # coffee: no lift on the crude test
    assert kc and ("no measured difference" in kc or "a measured drag" in kc)
    assert crowd.fade_text("EURJPY", 1) is None   # no COT series
    assert crowd.same_side_text("NQ", 0) is None


def test_desk_crowd_history_gate_only_when_same_side_and_penalised():
    nq = _crowd_history("NQ", 1, {"managers": 96.8, "commercials": 1.9})
    assert nq["state"] == "crowded_long" and nq["same_side"] and nq["gate"] and nq["fact"]
    gc = _crowd_history("GC", 1, {"managers": 95, "commercials": 5})
    assert gc["gate"] is None and gc["fact"]            # persistence: fact, no size gate
    si = _crowd_history("SI", 1, {"managers": 8, "commercials": 93})
    assert si["fade"] and si["gate"] is None and si["fact"].startswith("Fading the crowd")
    assert _crowd_history("NQ", 0, {"managers": 96.8, "commercials": 1.9}) is None
    assert _crowd_history("EURJPY", 1, None) is None
    mid = _crowd_history("KC", 1, {"managers": 12.9, "commercials": 85.2})
    assert mid["state"] == "none" and mid["fact"] is None


def _rows():
    def r(i, name, m, c, rt, cls, side, arch):
        return {"id": i, "name": name, "side": side, "archetype": arch, "class": {"key": cls, "label": cls.title()},
                "positioning": {"managers": m, "commercials": c, "retail": rt, "observed": "2026-10-20",
                                "released": "2026-10-23", "changes": {"managers": {"chg_4": -3.0}}}}
    return [r("NQ", "NASDAQ", 96.8, 1.9, 76.1, "prime", 1, "Pullback in trend"),
            r("ES", "S&P 500", 29.7, 59.4, 60.6, "prime", 1, "Pullback in trend"),
            r("YM", "Dow", 54.2, 43.9, 27.7, "prime", 1, "Trend with value"),
            r("RTY", "Russell 2000", 27.7, 71.0, 38.7, "none", 0, None),
            {"id": "EURJPY", "name": "EUR/JPY", "side": 1, "class": {"key": "developing"}, "positioning": None}]


def test_complex_build_equity():
    rows = _rows()
    by = {r["id"]: r for r in rows}
    out = capi.build("NQ", rows, by)
    assert out["complex"]["key"] == "equity"
    assert [p["id"] for p in out["peers"]] == ["NQ", "YM", "ES", "RTY"]      # sorted by managers
    assert out["me"]["state"] == "crowded_long" and out["me"]["class"] == "prime"
    assert out["history"]["crowded_long"]["n"] > 50
    assert out["history"]["fade_short_as_long"]["mean_pct"] > 0            # flipped from the short block
    assert set(out["relative"]["pairs"]) == {"ES>NQ", "NQ>ES", "NQ>YM", "NQ>RTY", "YM>NQ", "RTY>NQ"}
    read = " ".join(out["read"])
    assert "97th" in read and "2nd" in read and "crowded long" in read
    assert "Most crowded long in the complex: NQ" in read and "Least: RTY" in read
    assert "location, not a signal" in read


def test_complex_build_without_cot():
    rows = _rows()
    by = {r["id"]: r for r in rows}
    out = capi.build("EURJPY", rows, by)
    assert out["complex"] is None and out["peers"] == []


def test_complex_endpoint_statuses():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    state = {"data": {"rows": []}, "built": None}
    capi.register(app, markets=[{"id": "NQ", "name": "NASDAQ"}, {"id": "ES", "name": "S&P 500"}], desk_state_getter=lambda: state)
    c = TestClient(app)
    assert c.get("/api/complex?market=NQ").status_code == 202
    assert c.get("/api/complex?market=XX").status_code == 404
    state["data"]["rows"] = _rows()
    r = c.get("/api/complex?market=nq")
    assert r.status_code == 200
    j = r.json()
    assert j["market"] == "NQ" and j["complex"]["label"] == "US equity indices" and j["evidence_built"]
