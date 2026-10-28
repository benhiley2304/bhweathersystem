import numpy as np
import pytest
from bhws_core.exposure import effective_bets, clusters, first_factor_share
from bhws_core.riskreward import reward_risk


def test_effective_bets_limits():
    assert effective_bets(np.eye(5)) == pytest.approx(5.0)
    assert effective_bets(np.ones((4, 4))) == pytest.approx(1.0)


def test_signs_matter():
    c = np.array([[1.0, -0.9], [-0.9, 1.0]])
    # long A, short B with negative correlation = same bet twice
    assert effective_bets(c, [1, -1]) == pytest.approx(effective_bets(np.array([[1, .9], [.9, 1]])))


def test_dollar_legs_cluster():
    c = np.array([[1, .7, .6, .0], [.7, 1, .65, .1], [.6, .65, 1, .0], [0, .1, 0, 1]])
    g = clusters(c, ["EUR", "GBP", "CHF", "NG"], [1, 1, 1, 1])
    assert g[0] == ["EUR", "GBP", "CHF"] or sorted(g[0]) == ["CHF", "EUR", "GBP"]
    assert first_factor_share(c) > 0.5


def test_rr_with_frictions_and_no_implied_expectancy():
    r = reward_risk(100, 98, 106, 1, spread=0.1, slippage=0.05)
    assert r["rr_gross"] == 3.0 and r["rr_net"] < 3.0
    assert r["expectancy"] is None and "does not imply" in r["expectancy_note"]


def test_rr_short_and_invalid():
    assert reward_risk(100, 102, 94, -1)["valid"]
    assert not reward_risk(100, 98, 106, -1)["valid"]
    assert not reward_risk(100, 99.9, 100.1, 1, spread=0.3)["valid"]
