"""The Desk: one weekly read across the book, built the way the reports work.

The composite 1-10 score is not the verdict (Issue 10/11). Each market gets a
grade from bhws_core.strategy (Prime, Setup, Developing, Backdrop, None) and
a setup type (crowd fade, exhaustion fade, pullback in trend ...). Roles:

    positioning   payoff (house basis: CFTC futures-only, 156-report mid-rank,
                  same as the TradingView indicators). Size warning only for
                  equity indices.
    valuation     payoff (Valuation Tool, daily lines, asset-specific refs).
    trend         context and timing: healthy, pullback, or extended.
    macro         hit rate.
    seasonality   minor tilt, forward path at 30/60 TD (positive = tends up).

Nothing here proposes levels. Every function is pure: callers pass data in,
nothing is fetched here.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import crowd as _crowd
from . import exposure, percentile, strategy

FX_CCY = {"6E": "EUR", "6J": "JPY", "6B": "GBP", "6A": "AUD", "6C": "CAD",
          "6S": "CHF", "6N": "NZD", "6M": "MXN", "DX": "USD"}

LIVE = ("Not turned", "One more week", "Turned", "Past test window", "Add eligible", "Hold; no add")
LAPSED = ("Extreme persisting", "Expired")

CLASSES = strategy.GRADES
CLASS = strategy.GRADE

TRIGGER_RANK = {"Turned": 0, "Add eligible": 0, "One more week": 1, "Not turned": 2,
                "Hold; no add": 2, "Past test window": 3}

TRIGGER_TEXT = {
    "Turned": "Triggered: managers turned two reports and price closed through the 10-week average",
    "Add eligible": "Add eligible on the playbook",
    "One more week": "One more week: turn or close through the 10-week average still missing",
    "Not turned": "Waiting: extreme in place, managers not yet moving against it",
    "Past test window": "Extreme older than six reports; the clock runs from your starter",
    "Hold; no add": "Holding; no add yet",
}


def ordinal(x: Optional[float]) -> str:
    """'2nd', '10.4th' near the 10/90 lines so a left-extreme at 10.4 reads right."""
    if x is None:
        return "n/a"
    if 9.5 <= x <= 10.5 or 89.5 <= x <= 90.5:
        return f"{x:.1f}th"
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _sgn(x) -> int:
    if x is None:
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def currencies(market: Dict[str, Any]) -> List[str]:
    mid = market.get("id")
    if market.get("category") == "fx_cross":
        legs = [FX_CCY.get(market.get("base_leg")), FX_CCY.get(market.get("quote_leg"))]
        return [c for c in legs if c]
    if mid in FX_CCY:
        return [FX_CCY[mid]] if mid == "DX" else [FX_CCY[mid], "USD"]
    return ["USD"]


def seasonal_tilt(sd: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Seasonal tilt for the next window. Positive means price tends to rise.

    Usable only when the reliability grade is A-C and the hit rate agrees
    with the median's sign; otherwise the tilt is 0 and says why."""
    if not sd or sd.get("median_pct") is None:
        return {"sign": None, "usable": False, "why": "no seasonal data"}
    med = float(sd.get("median_pct"))
    hit = sd.get("hit_rate")
    grade = sd.get("reliability_grade")
    out = {"median_pct": round(med, 2), "hit_rate": hit, "grade": grade,
           "n_years": sd.get("n_years"),
           "window_td": [sd.get("horizon_td_start"), sd.get("horizon_td_end")],
           "far_median_pct": sd.get("far_median_pct"),
           "entry_timing": sd.get("entry_timing"),
           "days_to_goldilocks": sd.get("days_to_goldilocks"),
           "goldilocks_dir": sd.get("goldilocks_dir"),
           "plan_note": sd.get("plan_note")}
    agrees = hit is not None and ((med > 0 and hit > 50) or (med < 0 and hit < 50))
    if grade in ("A", "B", "C") and agrees and abs(med) >= 0.2:
        out.update(sign=_sgn(med), usable=True, why=None)
    else:
        why = ("reliability grade " + str(grade)) if grade not in ("A", "B", "C") else \
              ("hit rate does not back the median" if not agrees else "median too small to matter")
        out.update(sign=0, usable=False, why=why)
    return out


def trend_sign(engine: Dict[str, Any]) -> Optional[int]:
    """Long-term trend from the engine's trend_state ("Uptrend ...",
    "Downtrend ...", "Rangebound ..."), falling back to the momentum vote."""
    ts = str(engine.get("trend_state") or "")
    if ts.startswith("Uptrend"):
        return 1
    if ts.startswith("Downtrend"):
        return -1
    if ts.startswith("Rangebound"):
        return 0
    v = (engine.get("factor_votes") or {}).get("momentum")
    return None if v is None else _sgn(v)


def valuation_sign(val: Optional[Dict[str, Any]]) -> Optional[int]:
    if not val or not val.get("composite"):
        return None
    v = val["composite"]["value"]
    return 1 if v <= -40 else (-1 if v >= 40 else 0)


def _rel(v: Optional[int], side: int) -> str:
    if v is None:
        return "n/a"
    if side == 0 or v == 0:
        return "neutral"
    return "with" if v == side else "against"


def positioning_changes(mgr: Sequence[float], comm: Sequence[float], retail: Sequence[float]) -> Dict:
    """Change matters more than level (B4). 1- and 4-report moves in house
    percentile for all three groups, plus extremes entered or left."""
    out = {}
    for key, s in (("managers", mgr), ("commercials", comm), ("retail", retail)):
        p = percentile.midrank_series(list(s))
        cur = p[-1] if p else None
        p1 = p[-2] if len(p) >= 2 else None
        p4 = p[-5] if len(p) >= 5 else None
        d = {"pct": None if cur is None else round(cur, 1),
             "chg_1": None if cur is None or p1 is None else round(cur - p1, 1),
             "chg_4": None if cur is None or p4 is None else round(cur - p4, 1)}
        ext = lambda x: x is not None and (x >= percentile.EXTREME_HI or x <= percentile.EXTREME_LO)
        if cur is not None and p1 is not None:
            if ext(cur) and not ext(p1):
                d["event"] = "entered extreme"
            elif ext(p1) and not ext(cur):
                d["event"] = "left extreme"
        out[key] = d
    return out


def event_gate(ccys: Iterable[str], events: Sequence[Dict[str, Any]], now: datetime,
               days: int = 7) -> List[Dict[str, Any]]:
    """High-impact releases for the market's currencies in the next `days`.
    House rule: no adds into a top-tier print."""
    ccys = set(ccys)
    end = now + timedelta(days=days)
    out = []
    for e in events or []:
        if (e.get("impact") or "").lower() != "high" or e.get("currency") not in ccys:
            continue
        try:
            t = datetime.fromisoformat(str(e.get("datetime_utc")).replace("Z", "+00:00"))
        except Exception:
            continue
        if now - timedelta(hours=2) <= t <= end:
            out.append({"name": e.get("name"), "currency": e.get("currency"),
                        "datetime_utc": t.isoformat(), "forecast": e.get("forecast") or None,
                        "prior": e.get("prior") or None})
    out.sort(key=lambda x: x["datetime_utc"])
    return out


def _crowd_history(mid: str, side: int, posd: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Measured crowd history for the row: state, the sentence for the gate list (same side, haircut) and the
    sentence for the positioning facts (same side persistence, or a fade). None when there is no research file
    or no COT, so callers keep the generic wording."""
    if not posd or not side:
        return None
    try:
        mg, cm = posd.get("managers"), posd.get("commercials")
        state = _crowd.crowd_state(mg, cm)
        same = (state == "crowded_long" and side == 1) or (state == "crowded_short" and side == -1)
        fade = (state == "crowded_long" and side == -1) or (state == "crowded_short" and side == 1)
        out: Dict[str, Any] = {"state": state, "complex": _crowd.complex_of(mid), "gate": None, "fact": None,
                               "same_side": same, "fade": fade}
        sh = _crowd.side_history(mid, side)
        if sh:
            out["same_side_13"] = {"crowd": sh["crowd"], "base": sh["base"], "n": sh["n"], "n_thin": sh["n_thin"],
                                   "lift_pct": sh["lift_pct"]}
        if same:
            txt = _crowd.same_side_text(mid, side)
            if txt:
                out["fact"] = txt[0].upper() + txt[1:]
                if sh and (sh.get("lift_pct") or 0) < 0:
                    out["gate"] = txt
        elif fade:
            txt = _crowd.fade_text(mid, side)
            if txt:
                out["fact"] = txt[0].upper() + txt[1:]
        return out
    except Exception:
        return None


def _macro_marginal(engine: Dict[str, Any]) -> bool:
    sc = engine.get("scores") or {}
    out = False
    for k in ("regime", "macro"):
        v = (sc.get(k) or {}).get("score")
        if v is None:
            continue
        d = abs(float(v) - 5.0)
        if d >= 1.0:
            return False
        if d >= 0.5:
            out = True
    return out


def build_row(*, market: Dict[str, Any], engine: Optional[Dict[str, Any]],
              evidence: Optional[Dict[str, Any]], macro: Optional[int],
              changes: Optional[Dict[str, Any]], valuation: Optional[Dict[str, Any]],
              events: Sequence[Dict[str, Any]], now: datetime,
              divergence: Optional[Dict[str, Any]] = None,
              seas_fwd: Optional[Dict[int, Dict[str, Any]]] = None,
              weekly: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    engine = engine or {}
    ev = evidence or {}
    pb = ev.get("playbook") or {}
    sd = ((engine.get("scores") or {}).get("seasonal") or {}).get("detail")
    seas_window = seasonal_tilt(sd)
    conf = (ev.get("data_confidence") or {}).get("grade") or ("Unknown" if ev else "No house COT")
    pos = ev.get("positioning") or {}
    posd = None if not pos else {
        "observed": pos.get("observed"), "released": pos.get("released"),
        "freshness": pos.get("freshness"),
        "managers": (pos.get("managers") or {}).get("percentile"),
        "commercials": (pos.get("commercials") or {}).get("percentile"),
        "retail": (pos.get("retail") or {}).get("percentile"),
        "changes": changes,
    }
    A = strategy.assess(market=market, engine=engine, pos=posd, pb=pb, div=divergence, val=valuation,
                        seas_fwd=seas_fwd, macro=macro, macro_marginal=_macro_marginal(engine),
                        conf=conf if ev else "Medium", weekly=weekly)
    side = A["side"]
    rd = A["reads"]
    rel = A["rel"]
    pillars = {
        "positioning": {"rel": rel["positioning"], "sign": rd["positioning"].get("sign") or 0,
                        "strength": rd["positioning"].get("strength"), "role": "payoff" if not
                        rd["positioning"].get("equity") else "size warning (index)"},
        "valuation": {"rel": rel["valuation"], "sign": rd["valuation"].get("sign") or 0,
                      "strength": rd["valuation"].get("strength"), "role": "payoff"},
        "trend": {"rel": rel["trend"], "sign": rd["trend"].get("dir") or 0, "state": rd["trend"].get("state"),
                  "role": "context and timing"},
        "seasonal": {"rel": rel["seasonal"], "sign": rd["seasonal"].get("sign") or 0,
                     "strength": rd["seasonal"].get("strength"), "role": "minor tilt"},
        "macro": {"rel": rel["macro"], "sign": macro, "role": "hit rate"},
    }
    gates = list(A["sizing"])
    crowd_hist = _crowd_history(market["id"], side, posd)
    if crowd_hist and crowd_hist.get("gate"):
        # measured history replaces the generic index wording; the generic line stays when there is no evidence
        gates = [x for x in gates if not x.startswith("crowd on the same side")]
        gates.append(crowd_hist["gate"])
    if conf in ("Low", "Insufficient"):
        gates.append(f"data confidence {conf.lower()}")
    ccys = currencies(market)
    gate_ev = event_gate(ccys, events, now)
    local = [e for e in gate_ev if e["currency"] != "USD"]
    if local and side:
        gates.append(f"{local[0]['currency']} {local[0]['name']} this week")
    g = A["grade"]
    trig = None
    cf = A.get("confirm") or {}
    if cf.get("kind") == "playbook":
        trig = cf.get("text") or TRIGGER_TEXT.get(cf.get("state"))
    elif cf.get("text"):
        trig = cf["text"]
    order = g["rank"] * 100 - int(round(A["total"] * 10))
    if conf in ("Low", "Insufficient"):
        order += 50
    row = {
        "id": market["id"], "name": market.get("name"), "category": market.get("category"),
        "currencies": ccys,
        "class": g, "class_v1": A.get("grade_v1"), "ruleset": A.get("ruleset"), "caps": A.get("caps") or [],
        "side": side, "lean": A["lean"], "archetype": A["archetype"],
        "side_basis": ("+".join(A["anchors"]) if A["anchors"] else ("trend+macro" if g["key"] == "backdrop" else None)),
        "primary_edge": ", ".join(A["anchors"]) or None,
        "trigger": trig, "confirm": A.get("confirm"), "order": order,
        "alignment": A["total"], "alignment_other_side": A["other_side_total"],
        "pillars": pillars, "with": len(A["for"]), "against": len(A["against"]),
        "for_list": A["for"], "against_list": A["against"], "downgrades": A["downgrades"],
        "gates": gates,
        "positioning": posd,
        "positioning_read": rd["positioning"],
        "crowd_history": crowd_hist,
        "divergence": A["divergence"],
        "playbook": {k: pb.get(k) for k in ("status", "direction", "fade_side", "managers_turning_reports",
                                             "close_through_10w", "sma10", "price", "reports_since_extreme",
                                             "macro_fit", "explain") if k in pb} if pb else None,
        "disagreements": ev.get("disagreements") or [],
        "seasonal": seas_window,
        "seasonal_fwd": rd["seasonal"],
        "valuation": valuation,
        "valuation_read": rd["valuation"],
        "macro_sign": macro,
        "macro_marginal": rd["macro"].get("marginal"),
        "trend": dict(rd["trend"], sign=rd["trend"].get("dir"), state_label=engine.get("trend_state")),
        "weekly": weekly,
        "engine": {"direction": engine.get("direction"), "conviction": engine.get("conviction"),
                   "tier": ("Firm" if engine.get("tier") == "Setup" else engine.get("tier")),
                   "note": "engine composite shown for reference; it is not the verdict"},
        "events": gate_ev,
        "data_confidence": conf,
        "data_notes": (ev.get("data_confidence") or {}).get("reasons", []),
        "quote_direction": market.get("cot_note") if market.get("category") in ("fx", "fx_cross") else None,
    }
    if not ev:
        row["positioning_note"] = ("No single CFTC contract. Positioning is not on the house basis here; "
                                   "the read uses valuation, trend, seasonality and macro.")
    return row


def correlation_clusters(weekly_returns: Dict[str, Sequence[float]], sides: Dict[str, int],
                         threshold: float = 0.5, min_obs: int = 30) -> Dict[str, Any]:
    """Signed 52-week return correlation of the markets with a side.
    'These three are one bet' falls out as a cluster."""
    names = [n for n in weekly_returns if sides.get(n)]
    if len(names) < 2:
        return {"clusters": [], "effective_bets": None, "n": len(names)}
    L = min(len(weekly_returns[n]) for n in names)
    if L < min_obs:
        return {"clusters": [], "effective_bets": None, "n": len(names), "note": "not enough history"}
    M = np.array([list(weekly_returns[n])[-L:] for n in names], dtype=float)
    ok = ~np.isnan(M).any(axis=0)
    M = M[:, ok]
    if M.shape[1] < min_obs:
        return {"clusters": [], "effective_bets": None, "n": len(names), "note": "not enough overlapping weeks"}
    corr = np.corrcoef(M)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    signs = [sides[n] for n in names]
    groups = [g for g in exposure.clusters(corr, names, signs, threshold) if len(g) >= 2]
    idx = {n: i for i, n in enumerate(names)}
    detail = []
    for g in groups:
        ii = [idx[n] for n in g]
        sub = corr[np.ix_(ii, ii)]
        sg = [signs[i] for i in ii]
        s = np.outer(sg, sg) * sub
        off = s[~np.eye(len(ii), dtype=bool)]
        detail.append({"members": g, "sides": {n: sides[n] for n in g},
                       "avg_signed_corr": round(float(off.mean()), 2),
                       "effective_bets": round(exposure.effective_bets(sub, sg), 1)})
    detail.sort(key=lambda d: -len(d["members"]))
    return {"clusters": detail, "effective_bets": round(exposure.effective_bets(corr, signs), 1),
            "n": len(names), "weeks": int(M.shape[1]), "threshold": threshold}


def what_changed(rows: Sequence[Dict[str, Any]], now: datetime) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        ch = ((r.get("positioning") or {}).get("changes")) or {}
        for g in ("managers", "commercials", "retail"):
            d = ch.get(g) or {}
            if d.get("event"):
                out.append({"id": r["id"], "kind": "extreme",
                            "group": g,
                            "text": f"{r['name']}: {g} {d['event']} ({ordinal(d['pct'])} percentile, "
                                    f"{d['chg_1']:+.0f} pts on the report)"})
        m = ch.get("managers") or {}
        if m.get("chg_1") is not None and abs(m["chg_1"]) >= 20 and not m.get("event"):
            out.append({"id": r["id"], "kind": "move",
                        "group": "managers",
                        "text": f"{r['name']}: managers moved {m['chg_1']:+.0f} pts in one report, now {ordinal(m['pct'])}"})
        pb = r.get("playbook") or {}
        if pb.get("status") == "Turned":
            out.append({"id": r["id"], "kind": "trigger",
                        "text": f"{r['name']}: playbook turned ({'long' if pb.get('direction', 0) > 0 else 'short'} fade)"})
        elif pb.get("status") == "One more week" and pb.get("managers_turning_reports") == 1:
            out.append({"id": r["id"], "kind": "turn",
                        "text": f"{r['name']}: managers turned for the first report against the extreme"})
        s = r.get("seasonal") or {}
        dg = s.get("days_to_goldilocks")
        if r.get("side") and isinstance(dg, (int, float)) and 0 < dg <= 10 and s.get("goldilocks_dir") is not None:
            gd = s["goldilocks_dir"]
            gd = gd if isinstance(gd, int) else (1 if str(gd).lower() in ("long", "up", "1") else -1)
            if gd == r["side"]:
                out.append({"id": r["id"], "kind": "seasonal",
                            "text": f"{r['name']}: seasonal window for the {'long' if gd > 0 else 'short'} opens in about {int(dg)} trading days"})
    order = {"trigger": 0, "turn": 1, "extreme": 2, "move": 3, "seasonal": 4}
    grp = {"managers": 0, "commercials": 1, "retail": 2}
    out.sort(key=lambda x: (order.get(x["kind"], 9), grp.get(x.get("group"), 0)))
    return out


def _names(rows, n=6):
    xs = [r["name"] for r in rows][:n]
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _side_word(s):
    return "long" if s > 0 else "short"


def brief(*, regime: Optional[Dict[str, Any]], fed: Optional[Dict[str, Any]],
          events: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]],
          clusters: Optional[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
    """Deterministic prose. Every number comes from the inputs, so it can be
    checked against the facts block returned with it."""
    paras, facts = [], {}
    if regime:
        r = regime.get("regime")
        facts["regime"] = {"label": r, "score_10": regime.get("score_10"), "vix": regime.get("vix_level")}
        s = f"Risk backdrop reads {str(r).lower()}"
        if regime.get("score_10") is not None:
            s += f" ({regime['score_10']}/10 on the engine's risk score"
            s += f", VIX {regime['vix_level']:.1f})" if regime.get("vix_level") is not None else ")"
        paras.append(s + ".")
    if fed:
        meets = ((fed.get("snapshots") or {}).get("now") or {}).get("meetings") or []
        up = [m for m in meets if str(m.get("date")) >= now.date().isoformat()]
        if up:
            m0 = up[0]
            far = up[min(len(up) - 1, 3)]
            facts["fed"] = {"next": m0, "later": far}
            def _d(x):
                try:
                    d = datetime.fromisoformat(str(x)[:10])
                    return f"{d.day} {d.strftime('%b')} {d.year}" if d.year != now.year else f"{d.day} {d.strftime('%b')}"
                except Exception:
                    return str(x)
            s = (f"Fed funds futures price {m0.get('cum_bp', 0):+.0f}bp by the {_d(m0['date'])} FOMC "
                 f"({m0.get('p_hike_cum', 0):.0f}% implied odds of at least one hike) and "
                 f"{far.get('cum_bp', 0):+.0f}bp by {_d(far['date'])}.")
            paras.append(s)
    prime = [r for r in rows if r["class"]["key"] == "prime"]
    setups = [r for r in rows if r["class"]["key"] == "setup"]
    dev = [r for r in rows if r["class"]["key"] == "developing"]
    live = prime + setups
    turned = [r for r in rows if r.get("archetype") == "Crowd fade"
              and (r.get("playbook") or {}).get("status") in ("Turned", "Add eligible")]
    omw = [r for r in rows if r.get("archetype") == "Crowd fade"
           and (r.get("playbook") or {}).get("status") == "One more week"]
    facts["counts"] = {"prime": len(prime), "setup": len(setups), "developing": len(dev),
                       "turned": len(turned), "one_more_week": len(omw)}

    def _lbl(r):
        return f"{r['name']} {_side_word(r['side'])}" + (f" ({r['archetype'].lower()})" if r.get("archetype") else "")
    def _sc(r):
        return ((r.get("setup") or {}).get("score"))
    scored = [r for r in rows if r.get("side") and _sc(r) is not None]
    if scored:
        ready = sorted([r for r in scored if (r["setup"].get("ready") or {}).get("key") == "ready"],
                       key=lambda r: -_sc(r))
        waits = sorted([r for r in scored if (r["setup"].get("ready") or {}).get("key") == "wait"],
                       key=lambda r: (r["setup"]["timing"].get("entry_td") or 99))
        vetoed = [r for r in scored if (r["setup"].get("ready") or {}).get("key") == "veto"]
        facts["counts"].update({"ready": len(ready), "wait": len(waits), "veto": len(vetoed)})
        live = ready
        if ready:
            paras.append("Trade-ready on the setup score: " + ", ".join(
                f"{_lbl(r)} {_sc(r):.1f}" for r in ready[:6]) + ".")
        if waits:
            paras.append("Waiting for a seasonal window: " + ", ".join(
                f"{r['name']} {_side_word(r['side'])} in {r['setup']['timing']['entry_td']} TD" for r in waits[:6]) + ".")
        if vetoed:
            paras.append(f"Standing aside (seasonal timing against or a replay veto): {_names(vetoed)}.")
    else:
        if prime:
            paras.append(f"Prime this week: " + ", ".join(_lbl(r) for r in prime[:6]) + ".")
        if setups:
            paras.append(f"Setups: " + ", ".join(_lbl(r) for r in setups[:8]) + ".")
    if not live:
        paras.append("Nothing is trade-ready on the setup score this week. That is a result, not a gap.")
    against_macro = [r for r in live if (r.get("pillars") or {}).get("macro", {}).get("rel") == "against"]
    if against_macro:
        paras.append(f"{_names(against_macro)} {'runs' if len(against_macro) == 1 else 'run'} against the macro read, "
                     "which lowers the hit rate rather than the payoff: size down.")
    if turned or omw:
        bits = []
        if turned:
            bits.append(f"turned and through the 10-week average: {_names(turned)}")
        if omw:
            bits.append(f"one more week: {_names(omw)}")
        paras.append("Playbook triggers, " + "; ".join(bits) + ".")
    for c in (clusters or {}).get("clusters", [])[:2]:
        mem = c["members"]
        nm = {r["id"]: r["name"] for r in rows}
        lbl = _names([{"name": f"{nm.get(x, x)} {_side_word(c['sides'][x])}"} for x in mem], n=8)
        paras.append(f"{lbl} are close to one bet (average signed correlation {c['avg_signed_corr']:.2f}, "
                     f"{c['effective_bets']:.1f} effective of {len(mem)}).")
    hi = [e for e in events or [] if (e.get("impact") or "").lower() == "high"]
    hi_up = []
    for e in hi:
        try:
            t = datetime.fromisoformat(str(e["datetime_utc"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if t >= now - timedelta(hours=2):
            hi_up.append((t, e))
    hi_up.sort(key=lambda x: x[0])
    if hi_up:
        facts["events"] = [e for _, e in hi_up[:8]]
        txt = ", ".join(f"{e['currency']} {e['name']} {t.strftime('%a %H:%M')} UTC" for t, e in hi_up[:5])
        paras.append(f"Top-tier prints ahead: {txt}. No adds into them.")
    paras.append("Ordering is by setup score. A ranking aid tested on a held-out replay. Not a probability. Zones are yours.")
    return {"paragraphs": paras, "facts": facts}
