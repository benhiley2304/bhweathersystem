import json
import os
import time

import pytest

from bhws_core import ledger as L
from bhws_core.state import StateStore


def test_state_store_local_roundtrip_and_status(tmp_path):
    st = StateStore(str(tmp_path), env={})
    assert st.remote_kind == "none" and st.status()["durable"] is False
    st.save("ledger", {"entries": [1, 2]})
    assert st.load("ledger") == {"entries": [1, 2]}
    assert os.path.exists(tmp_path / "ledger.json") and not os.path.exists(tmp_path / "ledger.json.tmp")
    assert st.status()["files"]["ledger"]["present"] is True
    assert st.restore_all(["ledger"]) == {"ledger": "no remote"}


def test_state_store_export_import_newer_only(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    sa, sb = StateStore(str(a), env={}), StateStore(str(b), env={})
    sa.save("fed_pricing", {"months": [1]})
    sb.save("fed_pricing", {"months": [0]})
    sb._stamp("fed_pricing", time.time() - 3600)   # b's copy is an hour older
    out = sb.import_bundle(sa.export())
    assert out["fed_pricing"] == "imported" and sb.load("fed_pricing") == {"months": [1]}
    # the other way round, b's copy now carries a's stamp, so a keeps its own
    out2 = sa.import_bundle(sb.export())
    assert out2["fed_pricing"] == "local newer"
    # a seed file (came with the checkout, never written by the store) has no
    # stamp, so an imported copy always wins over it however fresh its mtime
    c = tmp_path / "c"; c.mkdir()
    (c / "fed_pricing.json").write_text('{"months": ["seed"]}')
    sc = StateStore(str(c), env={})
    assert sc.status()["files"]["fed_pricing"]["seed"] is True
    assert sc.import_bundle(sa.export())["fed_pricing"] == "imported" and sc.load("fed_pricing") == {"months": [1]}
    # and a seed exported (mtime 0) never overwrites a stamped local copy
    d = tmp_path / "d"; d.mkdir()
    (d / "fed_pricing.json").write_text('{"months": ["seed"]}')
    assert sa.import_bundle(StateStore(str(d), env={}).export())["fed_pricing"] == "local newer"
    # unknown names are skipped, never written
    out3 = sb.import_bundle({"items": {"evil": {"body": {}, "mtime": time.time()}}})
    assert out3 == {"evil": "skipped"} and not os.path.exists(b / "evil.json")


def test_state_restore_hook_runs_on_import(tmp_path):
    st = StateStore(str(tmp_path), env={})
    seen = {}
    st.on_restore("ff_macro_injected", lambda doc: seen.update(doc))
    st.import_bundle({"items": {"ff_macro_injected": {"body": {"GBP": {"data": {"score": 6}, "time": 1}},
                                                       "mtime": time.time()}}})
    assert seen == {"GBP": {"data": {"score": 6}, "time": 1}}


def test_ledger_derive_uses_bens_levels_only():
    d = L.derive({"zone_hi": 102, "zone_lo": 100, "entry": 101, "stop": 99, "target": 107}, 1, sd_weekly=0.02)
    assert d["r_multiple"] == 3.0 and d["stop_distance"] == 2 and d["zone_width"] == 2
    assert d["stop_outside_zone"] is True and d["warnings"] == []
    assert d["stop_sd"] == pytest.approx(0.99, abs=0.01)
    # stop inside the zone is flagged, not corrected
    d2 = L.derive({"zone_hi": 102, "zone_lo": 100, "entry": 101, "stop": 100.5, "target": 107}, 1)
    assert d2["stop_outside_zone"] is False and any("inside" in w for w in d2["warnings"])
    # missing levels give None, never a default
    d3 = L.derive({"entry": 101}, 1)
    assert d3["r_multiple"] is None and d3["stop_distance"] is None
    # short side geometry
    d4 = L.derive({"entry": 100, "stop": 102, "target": 94}, -1)
    assert d4["r_multiple"] == 3.0


def test_ledger_entry_lifecycle_and_calibration():
    e = L.new_entry({"market": "gc", "side": 1, "plan": {"zone_type": "dbr", "entry": 100, "stop": 98, "target": 106}},
                    context={"archetype": "Counter-trend", "grade": "setup"}, now=1000.0)
    assert e["market"] == "GC" and e["plan"]["zone_type"] == "DBR" and e["status"] == "planned"
    assert e["derived"]["r_multiple"] == 3.0 and e["context"]["grade"] == "setup"
    e = L.update_entry(e, {"status": "open", "plan": {"target": 108}}, now=1001.0)
    assert e["status"] == "open" and e["derived"]["r_multiple"] == 4.0
    e = L.close_entry(e, {"exit_px": 104, "note": "half target"}, now=1002.0)
    assert e["status"] == "closed" and e["outcome"]["r_realised"] == 2.0
    e2 = L.close_entry(L.new_entry({"market": "CL", "side": -1, "plan": {"entry": 80, "stop": 82, "target": 74}},
                                   context={"archetype": "Crowd fade", "grade": "prime"}), {"exit_px": 83})
    assert e2["outcome"]["r_realised"] == -1.5
    cal = L.calibration([e, e2])
    assert cal["all"]["n"] == 2 and cal["all"]["hit_rate"] == 0.5 and cal["all"]["enough"] is False
    assert cal["by_archetype"]["Counter-trend"]["mean_r"] == 2.0
    assert cal["by_grade"]["prime"]["n"] == 1
    with pytest.raises(ValueError):
        L.new_entry({"market": "GC", "side": 0})
    with pytest.raises(ValueError):
        L.new_entry({"market": "GC", "side": 1, "plan": {"zone_type": "XYZ"}})


def test_grade_record_replaces_same_day_and_prunes():
    st = L.append_grades(None, "2026-10-30", [{"market": "GC", "grade": "setup", "grade_v1": "developing"}])
    st = L.append_grades(st, "2026-10-30", [{"market": "GC", "grade": "prime", "grade_v1": "developing"},
                                            {"market": "CL", "grade": "none", "grade_v1": "none"}])
    assert len(st["rows"]) == 2 and {r["grade"] for r in st["rows"]} == {"prime", "none"}
    st = L.append_grades(st, "2019-01-04", [{"market": "GC", "grade": "setup"}], keep_days=365)
    assert any(r["day"] == "2019-01-04" for r in st["rows"])
    st = L.append_grades(st, "2026-10-31", [{"market": "GC", "grade": "setup"}], keep_days=365)
    assert all(r["day"] != "2019-01-04" for r in st["rows"]) and len(st["rows"]) == 3


def test_state_api_routes(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import bhws_state_api as api

    store = StateStore(str(tmp_path), env={})
    desk = {"data": {"generated_at": "2026-10-30T00:00:00", "rows": [
        {"id": "GC", "class": {"key": "setup"}, "class_v1": "developing", "ruleset": "v2", "caps": [],
         "side": 1, "archetype": "Counter-trend", "confirm": {"state": "Waiting", "window": "open"},
         "setup": {"score": 6.5, "ready": {"key": "ready"}, "timing": {"state": "in_window"}, "bias": {"dir": 1}},
         "pillars": {"positioning": {"rel": "with"}}, "primary_edge": "positioning", "gates": []}]}}
    app = FastAPI()
    api.register(app, store=store, desk_state_getter=lambda: desk, sd_weekly_getter=lambda m: 0.02)
    c = TestClient(app)

    assert c.get("/api/state/status").json()["durable"] is False
    r = c.post("/api/admin/ledger", json={"market": "GC", "side": 1,
                                          "plan": {"zone_type": "DBR", "zone_hi": 4010, "zone_lo": 3990,
                                                   "entry": 4000, "stop": 3985, "target": 4060}})
    assert r.status_code == 200
    ent = r.json()["entry"]
    assert ent["context"]["grade"] == "setup" and ent["context"]["grade_v1"] == "developing"
    assert ent["context"]["confirm_window"] == "open" and ent["derived"]["r_multiple"] == 4.0
    assert c.post("/api/admin/ledger", json={"market": "GC", "side": 0}).status_code == 422
    r = c.patch(f"/api/admin/ledger/{ent['id']}", json={"status": "open"})
    assert r.json()["entry"]["status"] == "open"
    assert c.patch("/api/admin/ledger/nope", json={}).status_code == 404
    r = c.post(f"/api/admin/ledger/{ent['id']}/close", json={"exit_px": 4030})
    assert r.json()["entry"]["outcome"]["r_realised"] == 2.0
    lst = c.get("/api/admin/ledger").json()
    assert lst["n"] == 1 and lst["calibration"]["all"]["n"] == 1
    # persisted to the local tier
    assert json.load(open(tmp_path / "ledger.json"))["entries"][0]["status"] == "closed"
    # export / import round trip through the API
    bundle = c.get("/api/admin/state/export").json()
    assert "ledger" in bundle["items"]
    store2 = StateStore(str(tmp_path / "other"), env={}) if (tmp_path / "other").mkdir() is None else None
    assert store2.import_bundle(bundle)["ledger"] == "imported"
    # grade record endpoint
    store.save("ledger_grades", L.append_grades(None, "2026-10-30",
                                                [{"market": "GC", "grade": "setup", "grade_v1": "developing", "caps": []}]))
    g = c.get("/api/admin/ledger/grades", params={"market": "gc"}).json()
    assert g["days"][0]["v2"] == {"setup": 1} and g["days"][0]["v1"] == {"developing": 1}


def test_cap_short_labels():
    from bhws_desk_api import _cap_short
    assert _cap_short("seasonal against: replay -0.120 R on prime and setup") == "60 TD seasonal path against"
    assert _cap_short("divergence-fade short: replay -0.269 R") == "divergence short"
    assert _cap_short("managers have turned: crowd fade window closed") == "managers turned"
    assert _cap_short("something new: detail") == "something new"


def test_state_import_merges_keyed_stores(tmp_path):
    from bhws_core.state import merge
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    sa, sb = StateStore(str(a), env={}), StateStore(str(b), env={})
    # a holds the long injected history, b was refreshed from a short live feed after boot
    sa.save("ff_event_store", {"USD|NFP|2026-03-06": {"actual": "1"}, "USD|CPI|2026-10-14": {"actual": "old"}})
    sb.save("ff_event_store", {"USD|CPI|2026-10-14": {"actual": "new"}, "USD|CPI|2026-11-12": {"actual": "2"}})
    out = sb.import_bundle(sa.export())          # a's export is older than b's local copy
    assert out["ff_event_store"] == "merged"
    got = sb.load("ff_event_store")
    assert set(got) == {"USD|NFP|2026-03-06", "USD|CPI|2026-10-14", "USD|CPI|2026-11-12"}
    assert got["USD|CPI|2026-10-14"] == {"actual": "new"}   # local (newer) wins the shared key
    # snapshot stores stay newest-wins
    sa.save("fed_pricing", {"months": [1], "time": 1})
    sb.save("fed_pricing", {"months": [2], "time": 2})
    assert sb.import_bundle(sa.export())["fed_pricing"] == "local newer" and sb.load("fed_pricing")["months"] == [2]
    # per-currency macro injections keep the later time per key
    m, ch = merge("ff_macro_injected", {"GBP": {"time": 5}, "EUR": {"time": 1}}, {"EUR": {"time": 9}}, incoming_is_newer=False)
    assert m == {"GBP": {"time": 5}, "EUR": {"time": 9}} and ch
    # grade record unions by day and market
    g, _ = merge("ledger_grades", {"rows": [{"day": "2026-10-28", "market": "GC", "grade": "setup"}]},
                 {"rows": [{"day": "2026-10-29", "market": "GC", "grade": "prime"}]}, incoming_is_newer=True)
    assert [(r["day"], r["grade"]) for r in g["rows"]] == [("2026-10-28", "setup"), ("2026-10-29", "prime")]


def test_state_seed_bundle_applies_on_fresh_checkout_and_never_imports_ledger(tmp_path):
    # a live store with injected histories and a private ledger
    live = tmp_path / "live"; live.mkdir()
    sl = StateStore(str(live), env={})
    sl.save("fed_pricing", {"months": [1, 2, 3]})
    sl.save("ledger", {"entries": [{"id": "x", "market": "SI"}]})
    seed = sl.seed_bundle()
    assert seed["seed"] is True and "ledger" not in seed["items"] and "fed_pricing" in seed["items"]
    # a fresh checkout with a stale per-file seed and the committed bundle
    fresh = tmp_path / "fresh"; (fresh / "state").mkdir(parents=True)
    (fresh / "fed_pricing.json").write_text('{"months": ["stale"]}')
    bundle_with_ledger = dict(seed); bundle_with_ledger["items"] = dict(seed["items"])
    bundle_with_ledger["items"]["ledger"] = {"body": {"entries": [{"id": "leak"}]}, "mtime": time.time()}
    (fresh / "state" / "seed_bundle.json").write_text(json.dumps(bundle_with_ledger))
    sf = StateStore(str(fresh), env={})
    out = sf.restore_all(["fed_pricing", "ledger"])
    assert out["fed_pricing"] == "seed imported" and sf.load("fed_pricing") == {"months": [1, 2, 3]}
    assert out["ledger"] == "no remote" and sf.load("ledger") is None   # ledger never read from the seed
    st = sf.status()
    assert st["seed_bundle"]["present"] is True and st["seed_bundle"]["last_apply"]["fed_pricing"] == "imported"
    # a restart with newer local data keeps the local copy
    sf.save("fed_pricing", {"months": [9]})
    sf._stamp("fed_pricing", time.time() + 10)
    out2 = sf.restore_all(["fed_pricing"])
    assert out2["fed_pricing"] == "no remote" and sf.load("fed_pricing") == {"months": [9]}
    # no bundle on disk is a quiet no-op
    bare = tmp_path / "bare"; bare.mkdir()
    assert StateStore(str(bare), env={}).apply_seed() == {}
