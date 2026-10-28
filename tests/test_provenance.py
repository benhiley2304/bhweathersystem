from datetime import datetime, timezone
from bhws_core.provenance import InputStamp, data_confidence

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def cot(**kw):
    base = dict(name="cot", source="CFTC legacy futures-only", expected_frequency="weekly",
                observed_at="2026-10-20", released_at="2026-10-23T19:30:00+00:00",
                retrieved_at="2026-10-28T06:00:00+00:00")
    base.update(kw)
    return InputStamp(**base)


def price(**kw):
    base = dict(name="price", source="yfinance", expected_frequency="daily",
                observed_at="2026-10-27", retrieved_at="2026-10-28T06:00:00+00:00")
    base.update(kw)
    return InputStamp(**base)


def test_all_fresh_is_high():
    assert data_confidence([cot(), price()], NOW)["grade"] == "High"


def test_stale_critical_input_caps_confidence():
    r = data_confidence([cot(observed_at="2026-09-29", released_at="2026-10-02"), price()], NOW)
    assert r["grade"] == "Low" and any("cot: stale" in x for x in r["reasons"])


def test_fallback_never_reads_as_normal():
    r = data_confidence([cot(), price(fallback="timeout_neutral")], NOW)
    assert r["grade"] == "Insufficient"


def test_missing_critical_is_insufficient():
    r = data_confidence([cot(observed_at=None, released_at=None), price()], NOW)
    assert r["grade"] == "Insufficient"


def test_noncritical_problem_is_medium():
    pcr = InputStamp(name="pcr", source="cboe", expected_frequency="daily",
                     observed_at="2026-10-01", critical=False)
    assert data_confidence([cot(), price(), pcr], NOW)["grade"] == "Medium"
