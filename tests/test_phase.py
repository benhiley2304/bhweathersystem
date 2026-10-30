"""Crowd phase artifact: integrity of the per-market files, agreement with the crowd research, and the API."""
import json
import math
from pathlib import Path

import pytest

import bhws_phase_api as papi

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "research" / "positioning" / "phase"
EV = ROOT / "research" / "positioning" / "complex_evidence.json"


def test_index_lists_every_file():
    idx = json.loads((PHASE / "index.json").read_text())
    files = {p.stem for p in PHASE.glob("*.json")} - {"index"}
    assert set(idx["markets"]) == files
    assert len(files) >= 30
    assert idx["method"] == "midrank_156w" and idx["basis"] == "futures_only_legacy"


@pytest.mark.parametrize("mid", ["NQ", "GC", "KC", "6B", "ZB", "BTC"])
def test_market_file_shape(mid):
    d = json.loads((PHASE / f"{mid}.json").read_text())
    n = d["n"]
    for k in ("dates", "mgr", "comm", "retail", "close", "r4", "r8", "r13"):
        assert len(d[k]) == n, k
    assert d["dates"] == sorted(d["dates"]) and d["from"] == d["dates"][0] and d["to"] == d["dates"][-1]
    for k in ("mgr", "comm", "retail"):
        vals = [v for v in d[k] if v is not None]
        assert vals and min(vals) >= 0.0 and max(vals) <= 100.0
    # the most recent 13 weeks cannot have a 13-week outcome; earlier weeks mostly do
    assert all(v is None for v in d["r13"][-3:])
    known = sum(1 for v in d["r13"] if v is not None)
    assert known >= n - 20


def test_nq_corner_matches_crowd_research():
    """The phase cloud and the Desk's measured crowd sentence come from the same frame."""
    d = json.loads((PHASE / "NQ.json").read_text())
    ev = json.loads(EV.read_text())["haircut"]["NQ"]
    rows = [(d["mgr"][i], d["comm"][i], d["r13"][i]) for i in range(d["n"]) if d["mgr"][i] is not None and d["comm"][i] is not None]
    corner = [r for m, c, r in rows if m >= 90 and c <= 10]
    known = [r for r in corner if r is not None]
    assert len(corner) == ev["n_long_crowded"]
    assert len(known) == ev["long_crowded_13"]["n"]
    assert abs(100 * sum(known) / len(known) - ev["long_crowded_13"]["mean_pct"]) < 0.05
    assert abs(sum(1 for r in known if r > 0) / len(known) - ev["long_crowded_13"]["pos"]) < 0.005
    assert math.ceil(len(corner) / 8) == ev["n_long_crowded_thin"]


def test_loader_guards_paths_and_unknowns():
    assert papi.load("NQ")["market"] == "NQ"
    assert papi.load("nq")["market"] == "NQ"
    assert papi.load("ZZZ") is None
    assert papi.load("../index") is None
    assert papi.load("") is None


def test_api_routes():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    papi.register(app)
    c = TestClient(app)
    r = c.get("/api/phase")
    assert r.status_code == 200 and "NQ" in r.json()["markets"]
    r = c.get("/api/phase?market=gc")
    assert r.status_code == 200 and r.json()["market"] == "GC" and r.headers.get("cache-control")
    assert c.get("/api/phase?market=EURJPY").status_code == 404
