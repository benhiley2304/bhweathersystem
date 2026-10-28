"""Per-market read that keeps bias, setup and trade apart.

Built on top of the existing /api/scores market payload. It does not replace
the engine; it restates what the engine and the evidence layer know in the
house order, and it refuses to promote a bias into a trade.

  bias      direction of the slow backdrop and its horizon (context)
  setup     is there an attractive asymmetric situation now (positioning
            extreme + playbook state), and what would confirm or kill it
  trade     executable only with the trader's own zone and levels; BHWS never
            produces entry, stop or target on its own

Correlated inputs are counted once. regime and macro are both built from the
same rates and macro data, so they form one "macro context" family.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

FAMILIES = {
    "macro context": ("regime", "macro"),
    "positioning": ("cot",),
    "calendar": ("seasonal",),
    "price trend": ("momentum",),
    "valuation": ("relval",),
    "options sentiment": ("pcr",),
}
# Issue 10 roles. Used for wording only, never as weights.
ROLES = {
    "positioning": "sets the payoff",
    "macro context": "lifts the hit rate",
    "calendar": "a minor tilt",
    "price trend": "confirmation",
    "valuation": "context",
    "options sentiment": "context",
}


def _family_votes(votes: Dict[str, int]) -> Dict[str, int]:
    out = {}
    for fam, keys in FAMILIES.items():
        vals = [int(votes.get(k) or 0) for k in keys if k in votes]
        if not vals:
            continue
        s = sum(vals)
        out[fam] = (s > 0) - (s < 0)
    return out


def build(market: Dict[str, Any], *, evidence: Optional[Dict[str, Any]] = None,
          previous: Optional[Dict[str, Any]] = None,
          trader_levels: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    evidence = evidence or {}
    conf = (evidence.get("data_confidence") or {}).get("grade", "Unknown")
    votes = market.get("factor_votes") or {}
    fam = _family_votes(votes)
    bias_sign = int(market.get("bias_sign") or 0)
    conviction = float(market.get("conviction") or 0.0)

    # ── bias ────────────────────────────────────────────────────────────
    # Positioning is not a directional input in the house framework (Issue 10:
    # it "sets the payoff"; Issue 04: an extreme "is not itself a buy signal").
    # So the bias is judged on the other families only, and positioning is
    # reported against the bias separately.
    backdrop = {f: v for f, v in fam.items() if f != "positioning"}
    support = [f for f, v in backdrop.items() if bias_sign and v == bias_sign]
    conflict = [f for f, v in backdrop.items() if bias_sign and v == -bias_sign]
    bias_note = None
    if conf == "Insufficient":
        bias_state = "Insufficient data"
    elif bias_sign == 0:
        bias_state = "Neutral"
    elif not support:
        bias_state = "Unsupported"
        bias_note = ("The engine direction has no support outside positioning"
                     + (f" and is opposed by {', '.join(conflict)}" if conflict else "")
                     + ". Positioning sets the payoff, not the direction.")
    elif abs(conviction) < 0.5 or len(conflict) >= len(support):
        bias_state = "Mixed"      # direction exists but the evidence nets out
    else:
        bias_state = "Bullish" if bias_sign > 0 else "Bearish"
    pos_v = fam.get("positioning")
    positioning_vs_bias = (None if not (pos_v and bias_sign) else
                           ("with the bias" if pos_v == bias_sign else "against the bias"))

    # ── setup (positioning-led, per Issue 10/11) ─────────────────────────
    pb = evidence.get("playbook") or {}
    pb_status = pb.get("status")
    pb_dir = pb.get("direction") or 0
    if conf == "Insufficient" or not pb:
        setup_state = "Not assessed"
    elif pb_status in ("No extreme", None):
        setup_state = "No positioning setup"
    elif pb_status in ("Expired", "Extreme persisting"):
        setup_state = "Lapsed"
    elif pb_status == "Past test window":
        setup_state = "Extreme; clock runs from your starter"
    elif pb_status == "With the crowd":
        setup_state = "Crowd with the book"
    else:
        setup_state = "Positioning setup"
    setup_vs_bias = None
    if pb_dir and bias_sign:
        setup_vs_bias = "aligned" if pb_dir == bias_sign else "against the bias"

    # ── trade: never produced by BHWS on its own ─────────────────────────
    if trader_levels and all(k in trader_levels for k in ("entry", "stop")):
        rr = trader_levels.get("reward_risk") or {}
        trade = {"state": "Executable plan (your levels)" if rr.get("valid", True) else "Levels do not form a trade",
                 "levels": trader_levels, "source": "trader (manual)",
                 "note": rr.get("reason") or ("Reward:risk alone is not expectancy. Break-even hit rate "
                                              f"{rr['breakeven_hit_rate']:.0%} after the frictions entered."
                                              if rr.get("breakeven_hit_rate") is not None else None)}
    else:
        trade = {"state": "No executable trade from BHWS data",
                 "needs": "your weekly zone, entry and stop"}

    confirm, invalidate = [], []
    if pb_status in ("Not turned", "One more week", "Hold; no add", "Past test window"):
        word = "buying" if pb_dir > 0 else "selling"
        confirm.append(f"managers {word} two reports running (now {pb.get('managers_turning_reports', 0)})")
        if pb.get("sma10") is not None and pb.get("close_through_10w") is not True:
            side = "above" if pb_dir > 0 else "below"
            confirm.append(f"a weekly close {side} the 10-week average, {pb['sma10']:.6g} at the last report")
        elif pb.get("close_through_10w") is True:
            confirm.append("price already through the 10-week average; needs to hold")
    if pb_status in ("Not turned", "One more week"):
        invalidate.append(f"no turn by report {6} after the extreme (playbook expiry)")
    if (pb.get("starter") or {}).get("stop") is not None:
        invalidate.append(f"price beyond {pb['starter']['stop']} (zone stop rule)")

    read = {
        "market": market.get("id"),
        "bias": {
            "state": bias_state,
            "horizon": "weeks (macro, calendar and trend backdrop)",
            "engine_direction": market.get("direction"),
            "engine_conviction": conviction,
            "engine_tier": market.get("tier"),
            "note": bias_note,
            "judged_on": "macro context, calendar, price trend, valuation, options sentiment (positioning excluded)",
        },
        "positioning_vs_bias": positioning_vs_bias,
        "support": [{"family": f, "role": ROLES[f]} for f in support],
        "conflict": [{"family": f, "role": ROLES[f]} for f in conflict],
        "setup": {"state": setup_state, "playbook_status": pb_status,
                  "fade_side": pb.get("fade_side"), "vs_bias": setup_vs_bias,
                  "explain": pb.get("explain")},
        "trade": trade,
        "confirm_if": confirm,
        "wrong_if": invalidate,
        "data_confidence": conf,
        "data_notes": (evidence.get("data_confidence") or {}).get("reasons", []),
        "positioning": evidence.get("positioning"),
    }
    read["changes"] = diff(previous, read) if previous else []
    return read


def diff(prev: Dict[str, Any], cur: Dict[str, Any]) -> List[str]:
    out = []
    def g(d, *ks):
        for k in ks:
            d = (d or {}).get(k)
        return d
    for label, path in (("bias", ("bias", "state")), ("setup", ("setup", "state")),
                        ("playbook", ("setup", "playbook_status")),
                        ("data confidence", ("data_confidence",))):
        a, b = g(prev, *path), g(cur, *path)
        if a != b:
            out.append(f"{label}: {a} -> {b}")
    pa = g(prev, "positioning", "managers", "percentile")
    pb = g(cur, "positioning", "managers", "percentile")
    if pa is not None and pb is not None and abs(pa - pb) >= 5:
        out.append(f"managers percentile: {pa} -> {pb}")
    return out
