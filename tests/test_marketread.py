"""Market read: deterministic prose from live fields (snapshot 28 Oct 2026, strategy grades)."""
import json
import re
from datetime import datetime
from pathlib import Path

from bhws_core import marketread

FX = json.loads((Path(__file__).parent / "fixtures" / "marketread_snapshot.json").read_text())
NOW = datetime.fromisoformat(FX["now"])


def _read(mid):
    row = next(r for r in FX["rows"] if r["id"] == mid)
    return row, marketread.build_read(row, FX["eng"].get(mid), FX["fed"], NOW)


def _all_text(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [t for v in x for t in _all_text(v)]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _all_text(v)]
    return []


def test_every_market_builds_and_has_five_pillars():
    for r in FX["rows"]:
        out = marketread.build_read(r, FX["eng"].get(r["id"]), FX["fed"], NOW)
        assert [p["key"] for p in out["pillars"]] == ["positioning", "valuation", "trend", "seasonal", "macro"]
        assert out["headline"]


def test_no_em_dashes_or_probability_language():
    for r in FX["rows"]:
        out = marketread.build_read(r, FX["eng"].get(r["id"]), FX["fed"], NOW)
        for t in _all_text(out):
            assert "\u2014" not in t
            assert not re.search(r"\bprobability\b|\bchance of\b", t, re.I)


def test_pillar_signs_match_desk_row():
    for r in FX["rows"]:
        out = marketread.build_read(r, FX["eng"].get(r["id"]), FX["fed"], NOW)
        for p in out["pillars"]:
            if p["key"] == "positioning" and not r.get("positioning"):
                continue
            assert p["sign"] == (r["pillars"][p["key"]]["sign"] or 0), (r["id"], p["key"])


def test_es_marginal_macro_and_dx_engine_disagreement_are_stated():
    row, out = _read("ES")
    mac = next(p for p in out["pillars"] if p["key"] == "macro")
    assert mac["marginal"] is True
    assert any("marginal" in s for s in out["summary"])
    _, dx = _read("DX")
    assert dx["engine"]["disagrees"] is True
    assert "Neither is adjusted" in dx["engine"]["text"]
    # composite is quoted, never inverted
    assert f"{FX['eng']['DX']['weighted_score']:.1f}" in dx["engine"]["text"]


def test_dx_reads_as_a_divergence_fade_and_nq_as_a_pullback():
    _, dx = _read("DX")
    assert dx["stance"]["key"] == "prime" and dx["stance"]["side"] == -1
    assert dx["archetype"] == "Divergence fade"
    assert dx["headline"].startswith("Short: divergence fade.")
    tr = next(p for p in dx["pillars"] if p["key"] == "trend")
    assert tr["rel"] == "exhausted"
    pos = next(p for p in dx["pillars"] if p["key"] == "positioning")
    assert any("lower high" in f for f in pos["facts"])
    _, nq = _read("NQ")
    assert nq["stance"]["key"] == "prime" and nq["stance"]["side"] == 1
    assert nq["archetype"] == "Pullback in trend"
    pos = next(p for p in nq["pillars"] if p["key"] == "positioning")
    assert pos["rel"] == "against" and any("size warning" in f for f in pos["facts"])
    assert any("size warning only for an index" in s for s in nq["summary"])


def test_es_buy_point_surfaces_with_its_quoted_record():
    _, out = _read("ES")
    val = next(p for p in out["pillars"] if p["key"] == "valuation")
    assert val["buy_point"] is True
    assert any("Pine header" in f for f in val["facts"])


def test_numbers_in_positioning_come_from_the_row():
    row, out = _read("ZN")
    pos = next(p for p in out["pillars"] if p["key"] == "positioning")
    m = round(row["positioning"]["managers"])
    assert f"fund managers {m}" in pos["facts"][0]


def test_cross_has_no_house_cot_and_no_fed_line():
    _, out = _read("EURJPY")
    pos = next(p for p in out["pillars"] if p["key"] == "positioning")
    assert pos["rel"] == "na"
    mac = next(p for p in out["pillars"] if p["key"] == "macro")
    assert not any("Fed funds" in f for f in mac["facts"])
    assert any("EUR" in f for f in mac["facts"])


def test_seasonal_sign_convention_is_stated():
    _, out = _read("NQ")
    sea = next(p for p in out["pillars"] if p["key"] == "seasonal")
    assert any("Positive means price tends to rise" in f for f in sea["facts"])
    assert any("60 TD +4.28%" in f for f in sea["facts"])
    assert sea["sign"] == 1


def test_missing_engine_payload_does_not_invent_numbers():
    row = dict(next(r for r in FX["rows"] if r["id"] == "CL"))
    out = marketread.build_read(row, None, None, NOW)
    mac = next(p for p in out["pillars"] if p["key"] == "macro")
    assert "n/a" in mac["facts"][0]
    assert not any("Fed funds" in f for f in mac["facts"])
