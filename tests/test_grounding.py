from bhws_core.grounding import check, sanitize

FACTS = {"t10y": 4.87, "hy_oas_bps": 291, "cpi_yoy": 3.4, "nfp_k": 162, "label": "EFFR 4.33%"}


def test_supported_numbers_pass():
    r = check("The 10-year sits at 4.87% with HY spreads at 291bp and CPI 3.4%.", FACTS)
    assert r["grounded"], r


def test_invented_number_is_flagged():
    r = check("Gold looks set for 2,650 with spreads at 291bp.", FACTS)
    assert not r["grounded"]
    assert any(i["token"].startswith("2,650") for i in r["issues"])


def test_probability_language_flagged():
    r = check("There is a 70% chance the Fed hikes.", FACTS)
    assert not r["grounded"]


def test_counts_and_years_allowed():
    assert check("Two of three legs agree, as in 2022, over 6 weeks.", FACTS)["grounded"]


def test_sanitize_drops_only_bad_sentences():
    s = sanitize("Spreads are 291bp. Copper targets 5.40 next. CPI is 3.4%.", FACTS)
    assert s["text"] == "Spreads are 291bp. CPI is 3.4%."
    assert s["dropped"] == ["Copper targets 5.40 next."]
