"""Reward:risk with frictions, and what it does and does not mean.

An attractive reward:risk is not positive expectancy. The break-even hit rate
is shown so the reader can compare it with an evidenced hit rate, if there is
one. If there is none, the output says so instead of implying an edge.
"""
from __future__ import annotations

from typing import Optional


def reward_risk(entry: float, stop: float, target: float, side: int, *,
                spread: float = 0.0, slippage: float = 0.0,
                financing: float = 0.0, evidenced_hit_rate: Optional[float] = None,
                hit_rate_n: Optional[int] = None) -> dict:
    """All frictions in price units per unit position, charged on a round trip.

    side: +1 long, -1 short. Levels must be supplied by the trader; this
    function never proposes them.
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    risk_gross = (entry - stop) * side
    reward_gross = (target - entry) * side
    if risk_gross <= 0 or reward_gross <= 0:
        return {"valid": False, "reason": "stop and target must sit either side of entry for this side"}
    cost = spread + 2 * slippage + financing
    risk = risk_gross + cost
    reward = reward_gross - cost
    if reward <= 0:
        return {"valid": False, "reason": "frictions exceed the reward"}
    rr = reward / risk
    be = risk / (risk + reward)
    out = {
        "valid": True,
        "rr_gross": round(reward_gross / risk_gross, 2),
        "rr_net": round(rr, 2),
        "breakeven_hit_rate": round(be, 3),
        "frictions": {"spread": spread, "slippage_each_way": slippage, "financing": financing},
    }
    if evidenced_hit_rate is None:
        out["expectancy"] = None
        out["expectancy_note"] = ("No evidenced hit rate for this setup. A good reward:risk on its own "
                                  "does not imply positive expectancy.")
    else:
        e = evidenced_hit_rate * rr - (1 - evidenced_hit_rate)
        out["expectancy_r"] = round(e, 3)
        out["expectancy_note"] = (f"Uses an evidenced hit rate of {evidenced_hit_rate:.0%}"
                                  + (f" (n={hit_rate_n})" if hit_rate_n else "")
                                  + ". Illustrative, not a forecast.")
    return out
