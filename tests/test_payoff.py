"""Payoff profile engine: lookup chain, stress test wording, and the no-level-generation rule."""
from bhws_core import payoff as PO


def test_artifact_loads_and_has_baseline():
    D = PO.load()
    assert D["cells"]["all"]["n"] > 20000
    assert "baseline" in D and D["baseline"]["all"]["n"] > 50000
    ks = [r["k"] for r in D["cells"]["all"]["first_passage"]]
    assert ks == [0.5, 0.75, 1.0, 1.5]


def test_lookup_falls_back_to_wider_cell():
    p = PO.lookup("Counter-trend", 1, "Waiting", "crypto", market="BTC")
    assert p["available"]
    # crypto cell exists but is below MIN_USE, so the chain must show it unused and lead with a wider cell
    used = [c for c in p["chain"] if c["used"]]
    assert len(used) == 1 and used[0]["n"] >= PO.MIN_USE
    assert p["baseline"] is not None and p["baseline"]["n"] > 0


def test_lookup_without_archetype_is_unavailable():
    p = PO.lookup(None, 0, None, "fx")
    assert not p["available"] and "archetype" in p["reason"]


def test_stress_rows_and_flags():
    p = PO.lookup("Crowd fade", 1, "Not turned", "commodity", market="SI")
    s = PO.stress(p, stop_sd=0.3, target_sd=6.0, r_multiple=20.0)
    assert s["k"] == 0.5 and s["uses"] == "3R"
    assert any("outside the tested range" in n for n in s["notes"])
    kinds = [f["v"] for f in s["flags"]]
    assert "neg" in kinds                     # stop inside noise band and target beyond p75 both flag
    assert all(0 <= v <= 1 for v in (s["row"]["target2"], s["row"]["stop"], s["row"]["neither"]))
    assert abs(s["row"]["target2"] + s["row"]["stop"] + s["row"]["neither"] - 1) < 0.01
    # no level is ever proposed
    txt = " ".join(s["sentences"] + [f["t"] for f in s["flags"]])
    assert "should move" not in txt and "recommend" not in txt


def test_stress_requires_levels():
    p = PO.lookup("Counter-trend", -1, "Waiting", "fx")
    assert PO.stress(p, None, None) is None
    assert PO.stress(p, 0, None) is None


def test_table_is_sorted_and_has_baseline():
    t = PO.table(level_max=2, k=1.0)
    assert t["baseline"]["k"] == 1.0
    assert all(r["level"] <= 2 for r in t["rows"])
    assert abs(t["baseline"]["target2"] - 1 / 3) < 0.05   # unconditional walk sits near the gambler's-ruin share
