"""Integration tests: backend app + evidence layer, network stubbed."""
from datetime import datetime, timezone
import pathlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend
import bhws_evidence_api as ev

FX = pathlib.Path(__file__).parent / "fixtures"


def _fo(code):
    c = pd.read_csv(FX / "cot_fo_6E_tail300.csv")
    c["date"] = pd.to_datetime(c["date"]).dt.date
    c["retail_net"] = 0.0          # fixture has no non-reportable columns
    return c


def _px(ticker):
    p = pd.read_csv(FX / "px_6E_tail1600.csv", index_col=0, parse_dates=True)["Close"]
    return p.astype(float)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ev, "_fetch_futures_only", _fo)
    monkeypatch.setattr(ev, "_fetch_weekly_close", _px)
    ev._FO_CACHE.clear(); ev._PX_CACHE.clear(); ev._PREV_READ.clear()
    return TestClient(backend.app)   # no context manager: startup warmers do not run


def test_evidence_route_registered():
    paths = {r.path for r in backend.app.routes}
    assert "/api/evidence" in paths


def test_unknown_market_404(client):
    assert client.get("/api/evidence?market=ZZZ").status_code == 404


def test_evidence_payload_shape(client):
    r = client.get("/api/evidence?market=6E")
    assert r.status_code == 200
    j = r.json()
    pos = j["positioning"]
    assert pos["basis"] == "CFTC legacy futures-only"
    assert pos["observed"] == "2026-10-20" and pos["released"] == "2026-10-23"
    assert pos["managers"]["method"] == "midrank_156w"
    assert "playbook" in j and "status" in j["playbook"]
    assert j["read"]["trade"]["state"] == "No executable trade from BHWS data"
    # engine scores are not warmed in tests -> must not read as high confidence
    assert j["data_confidence"]["grade"] in ("Insufficient", "Low")


def test_failed_fetch_is_reported_not_filled(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(ev, "_fetch_futures_only", boom)
    monkeypatch.setattr(ev, "_fetch_weekly_close", boom)
    ev._FO_CACHE.clear(); ev._PX_CACHE.clear()
    j = TestClient(backend.app).get("/api/evidence?market=6E").json()
    assert j["positioning"] is None
    assert j["playbook"]["status"] == "Insufficient data"
    assert j["data_confidence"]["grade"] == "Insufficient"
    assert j["read"]["setup"]["state"] == "Not assessed"


def test_stale_cot_downgrades_confidence():
    m = next(x for x in backend.MARKETS if x["id"] == "6E")
    fo = _fo("099741")
    stale = fo[fo.date <= pd.Timestamp("2026-09-29").date()]
    eng = {"id": "6E", "bias_sign": -1, "conviction": -1.2, "factor_votes": {"macro": -1}}
    now = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    e = ev.build_evidence(m, stale, now.isoformat(), _px("x"), now.isoformat(), eng,
                          "2026-10-28T11:00:00+00:00", None, now=now)
    assert e["positioning"]["freshness"] == "stale"
    assert e["data_confidence"]["grade"] == "Low"


def test_admin_guard_off_by_default(monkeypatch):
    monkeypatch.delenv("BHWS_ADMIN_TOKEN", raising=False)
    r = TestClient(backend.app).get("/api/clear-narrative-cache")
    assert r.status_code != 401


def test_admin_guard_enforced_when_token_set(monkeypatch):
    monkeypatch.setenv("BHWS_ADMIN_TOKEN", "s3cret")
    c = TestClient(backend.app)
    assert c.get("/api/clear-narrative-cache").status_code == 401
    assert c.post("/api/inject-fed-pricing", json={}).status_code == 401
    ok = c.get("/api/clear-narrative-cache", headers={"X-BHWS-Admin-Token": "s3cret"})
    assert ok.status_code != 401
    # read routes stay public
    assert c.get("/api/evidence?market=ZZZ").status_code == 404


def test_basis_disagreement_is_reported_not_reconciled():
    """Issue 11 case: sugar managers kept buying on futures-only but sold on
    the combined basis in the week to 20 Oct 2026."""
    m = next(x for x in backend.MARKETS if x["id"] == "6E")
    fo = _fo("x").copy()
    n = len(fo)
    # force a crowded long that is still being added to on futures-only
    fo["mgr_net"] = [float((i * 37) % 100) for i in range(n - 4)] + [200.0, 210.0, 220.0, 230.0]
    comb = pd.DataFrame({"date": fo.date, "lspec_net": fo.mgr_net.copy(), "comm_net": -fo.mgr_net})
    comb.loc[comb.index[-1], "lspec_net"] = comb.lspec_net.iloc[-2] - 5   # combined: selling 1 week
    now = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    e = ev.build_evidence(m, fo, now.isoformat(), _px("x"), now.isoformat(), None, None, comb, now=now)
    assert e["playbook"]["managers_turning_reports"] == 0
    d = [x for x in e["disagreements"] if x["field"] == "managers turning (reports)"]
    assert d and d[0]["house_futures_only"] == 0 and d[0]["combined_futures_options"] == 1
    assert e["positioning"]["managers"]["percentile"] > 90        # house number untouched


def test_macro_fit_uses_macro_family_not_engine_direction():
    # Engine says Bearish (bias_sign -1) only because COT votes bearish;
    # macro context is bullish. The playbook macro fit must follow macro.
    eng = {"id": "SB", "bias_sign": -1,
           "factor_votes": {"cot": -1, "macro": 1, "regime": 0, "seasonal": 0}}
    assert ev._macro_sign(eng) == 1
    assert ev._macro_sign({"factor_votes": {"cot": -1}}) is None
    assert ev._macro_sign(None) is None
    assert ev._macro_sign({"factor_votes": {"macro": 1, "regime": -1}}) == 0


def test_cot_fo_series_shape(client):
    r = client.get("/api/cot-fo-series?market=6E")
    assert r.status_code == 200
    j = r.json()
    n = len(j["dates"])
    assert j["basis"] == "futures_only_legacy" and j["window"] == 156
    for k in ("comm", "mgr", "retail"):
        assert set(j[k]) == {"net", "pct", "p10", "p90"}
        assert len(j[k]["net"]) == n and len(j[k]["pct"]) == n
    assert len(j["close"]) == n and len(j["release"]) == n
    # early reports have too little history for a percentile: left empty, not filled
    assert j["mgr"]["pct"][0] is None
    last = j["mgr"]["pct"][-1]
    assert last is not None and 0 <= last <= 100
    assert j["mgr"]["p10"][-1] <= j["mgr"]["p90"][-1]


def test_cot_fo_series_midrank_matches_house_percentile():
    import numpy as np
    fo = _fo("099741")
    got = ev._fo_group_series(fo)["mgr"]["pct"][-1]
    w = fo["mgr_net"].to_numpy(float)[-156:]
    cur = w[-1]
    want = 100 * ((w < cur).sum() + 0.5 * ((w == cur).sum() - 1)) / (len(w) - 1)
    assert abs(got - want) < 1e-9


def test_cot_fo_series_no_code_is_422(client):
    m = next((x for x in backend.MARKETS if not x.get("cftc_code")), None)
    if m is None:
        pytest.skip("every market has a CFTC code")
    assert client.get(f"/api/cot-fo-series?market={m['id']}").status_code == 422


def test_security_headers_on_every_route(client):
    r = client.get("/api/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert r.headers.get("x-frame-options") == "DENY"
    assert "max-age=" in r.headers.get("strict-transport-security", "")
    # admin routes are never cacheable; public routes keep their own cache policy
    ra = client.get("/api/admin/state/export")
    assert ra.headers.get("cache-control") == "no-store"
    assert r.headers.get("cache-control") != "no-store"


def test_state_status_reports_seed_bundle(client):
    j = client.get("/api/state/status").json()
    assert "seed_bundle" in j and isinstance(j["seed_bundle"].get("present"), bool)
