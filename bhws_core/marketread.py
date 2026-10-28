"""Market read: a deterministic, per-market narrative built from live fields.

Replaces the hand-written Overview narratives. Every sentence is assembled
from numbers already on the Desk row and the engine payload, so it can be
checked against the panels beneath it. No model text, no invented levels,
no probabilities.

Inputs
  row     a Desk row (bhws_core.desk.build_row)
  eng     the engine market payload from /api/scores (scores, factor_votes ...)
  fed     {"next": meeting, "later": meeting} from the Desk brief facts, or None

Output keys
  headline   one line, the verdict in plain words
  stance     class key/label, side, lean and what the lean is based on
  summary    2-3 sentences
  pillars    positioning, macro, trend, seasonal, valuation: sign, rel, facts[]
  watch      what would change the read
  risks      gates and events
  engine     how the 1-10 composite reads, and whether it disagrees
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

DEADBAND = 0.5          # engine abstains inside +/-0.5 of 5.0
MARGINAL = 1.0          # a vote that clears the deadband by less than this is marginal
EXT_LO, EXT_HI = 10, 90

_SIDE = {1: "long", -1: "short", 0: "flat"}
_LEAN = {1: "long", -1: "short"}


def _f(x, nd=1, sign=False):
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v) < 0.5 * 10 ** (-nd):
        v = 0.0
        sign = False
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "\u2212")


def _ord(x) -> str:
    if x is None:
        return "n/a"
    n = int(round(float(x)))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _score(eng, k):
    s = ((eng or {}).get("scores") or {}).get(k) or {}
    return s.get("score"), (s.get("detail") or s)


def _vote_word(score) -> str:
    if score is None:
        return "no data"
    d = float(score) - 5.0
    if abs(d) < DEADBAND:
        return "abstains"
    w = "bullish" if d > 0 else "bearish"
    return w + (", marginal" if abs(d) < MARGINAL else "")


def _is_marginal(score) -> bool:
    if score is None:
        return False
    d = abs(float(score) - 5.0)
    return DEADBAND <= d < MARGINAL


def _join(xs: List[str]) -> str:
    xs = [x for x in xs if x]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _date(x) -> str:
    try:
        d = datetime.fromisoformat(str(x)[:10])
        return f"{d.day} {d.strftime('%b')}"
    except Exception:
        return str(x)


# ── pillars ──────────────────────────────────────────────────────────────
def _positioning(row) -> Dict[str, Any]:
    p = row.get("positioning")
    pb = row.get("playbook") or {}
    pil = (row.get("pillars") or {}).get("positioning") or {}
    facts: List[str] = []
    if not p:
        facts.append("No house COT for this market (crosses and some contracts have none).")
        return {"key": "positioning", "title": "Positioning", "sign": 0, "rel": "na",
                "word": "n/a", "facts": facts, "tab": "cot"}
    facts.append(f"Commercials {_ord(p.get('commercials'))}, fund managers {_ord(p.get('managers'))}, "
                 f"retail {_ord(p.get('retail'))} percentile (futures-only, 156-report mid-rank, "
                 f"observed {_date(p.get('observed'))}).")
    ch = (p.get("changes") or {}).get("managers") or {}
    if ch.get("chg_1") is not None:
        facts.append(f"Fund managers {_f(ch.get('chg_1'), 1, True)} pts on the latest report, "
                     f"{_f(ch.get('chg_4'), 1, True)} over four.")
    if pb.get("explain"):
        facts.append(pb["explain"])
    m = p.get("managers")
    word = "no extreme"
    if pil.get("rel") == "lapsed":
        word = "extreme, clock run out"
    elif m is not None and (m <= EXT_LO or m >= EXT_HI):
        word = f"extreme, fade {pb.get('fade_side') or ''}".strip()
    elif pil.get("sign"):
        word = f"recent extreme, fade {pb.get('fade_side') or ''}".strip()
    return {"key": "positioning", "title": "Positioning", "sign": pil.get("sign") or 0,
            "rel": pil.get("rel") or "none", "word": word, "facts": facts, "tab": "cot"}


_US_WORDS = {
    "Growth": ("growth data beating forecasts", "growth data missing forecasts", "growth data in line"),
    "Jobs": ("labour data beating", "labour data missing", "labour data in line"),
    "Inflation": ("inflation printing above forecast", "inflation printing below forecast", "inflation in line"),
    "Rates (2Y)": ("2Y yield above its 21-day average", "2Y yield below its 21-day average", "2Y flat on its 21-day average"),
}


def _driver_phrase(d) -> str:
    lbl = d.get("label") or ""
    inp = d.get("input")
    w = _US_WORDS.get(lbl)
    if w is None or inp is None:
        return f"{lbl} {_f(inp, 2, True)}"
    return w[0] if inp > 0.15 else w[1] if inp < -0.15 else w[2]


def _top_releases(fx_side: Dict[str, Any], n=2) -> List[str]:
    out = []
    for cat, rows in (fx_side.get("cat_details") or {}).items():
        for r in rows or []:
            if r.get("age_d") is None or r["age_d"] > 14 or r.get("z") is None:
                continue
            out.append((abs(r["z"]) * (r.get("weight") or 1), r))
    out.sort(key=lambda x: -x[0])
    res = []
    for _, r in out[:n]:
        res.append(f"{r['name']} {r.get('actual')} vs {r.get('forecast')} forecast ({r['age_d']}d ago)")
    return res


def _macro(row, eng, fed) -> Dict[str, Any]:
    pil = (row.get("pillars") or {}).get("macro") or {}
    rs, rdet = _score(eng, "regime")
    ms, mdet = _score(eng, "macro")
    facts: List[str] = []
    facts.append(f"Risk regime {_f(rs)} ({_vote_word(rs)}), macro data {_f(ms)} ({_vote_word(ms)}). "
                 "Together they set the macro direction.")
    rl = (rdet or {}).get("label")
    if rl and rl not in ("Neutral", ""):
        facts.append(f"Regime read for this market: {rl}.")
    fxd = (mdet or {}).get("fx_detail")
    if fxd:
        if mdet.get("reason"):
            facts.append("Data surprises: " + mdet["reason"] + ".")
        f = fxd.get("foreign") or {}
        rel = _top_releases(f)
        if rel:
            facts.append(f"Biggest recent {f.get('currency', '')} prints: " + "; ".join(rel) + ".")
    else:
        drv = (mdet or {}).get("drivers") or []
        helps = [_driver_phrase(d) for d in drv if (d.get("directional") or 0) > 0.05]
        hurts = [_driver_phrase(d) for d in drv if (d.get("directional") or 0) < -0.05]
        if helps or hurts:
            s = []
            if helps:
                s.append("helping: " + _join(helps))
            if hurts:
                s.append("hurting: " + _join(hurts))
            facts.append("US data for this market, " + "; ".join(s) + ".")
        sup = (mdet or {}).get("supply")
        if sup and sup.get("rows"):
            seen, bits = set(), []
            for r in sup["rows"]:
                if r.get("minor") or r.get("key") in seen:
                    continue
                seen.add(r.get("key"))
                fc = r.get("forecast")
                bits.append(f"{r.get('title')} {r.get('actual')}" + (f" vs {fc}" if fc and fc != "\u2014" else "")
                            + (f" ({r.get('tag')})" if r.get("tag") else ""))
                if len(bits) >= 2:
                    break
            facts.append(f"{sup.get('title')}: " + "; ".join(bits) + "."
                         + ("" if sup.get("scored") else " Shown, not scored."))
    usd = "USD" in (row.get("currencies") or []) or row.get("category") not in ("fx_cross",)
    if fed and usd and fed.get("next"):
        n0, far = fed["next"], fed.get("later") or {}
        facts.append(f"Fed funds futures price {_f(n0.get('cum_bp'), 0, True)}bp by the {_date(n0.get('date'))} FOMC"
                     + (f" and {_f(far.get('cum_bp'), 0, True)}bp by {_date(far.get('date'))}" if far else "") + ".")
    marginal = [n for n, s in (("regime", rs), ("macro", ms)) if _is_marginal(s)]
    sign = pil.get("sign") or 0
    word = ("leans " + _LEAN[sign]) if sign else "no lean"
    if sign and marginal:
        word += ", marginal"
    return {"key": "macro", "title": "Macro", "sign": sign, "rel": pil.get("rel") or "neutral",
            "word": word, "facts": facts, "tab": "macro", "marginal": bool(sign and marginal)}


def _trend(row, eng) -> Dict[str, Any]:
    pil = (row.get("pillars") or {}).get("trend") or {}
    _, md = _score(eng, "momentum")
    md = md or {}
    facts = []
    if md.get("sma200_pct_diff") is not None:
        facts.append(f"Price {_f(abs(md['sma200_pct_diff']))}% {'above' if md['sma200_pct_diff'] >= 0 else 'below'} "
                     f"the 200-day; {_ord(md.get('pct_range'))} percentile of its 52-week range.")
    if md.get("roc_lt_pct") is not None:
        facts.append(f"26-week change {_f(md.get('roc_lt_pct'), 1, True)}%, 4-week {_f(md.get('roc_st_pct'), 1, True)}%.")
    st = (row.get("trend") or {}).get("state")
    if st:
        facts.append(f"Engine trend state: {st}.")
    sign = pil.get("sign") or 0
    return {"key": "trend", "title": "Trend", "sign": sign, "rel": pil.get("rel") or "neutral",
            "word": ("up" if sign > 0 else "down" if sign < 0 else "flat"), "facts": facts, "tab": "momentum"}


def _seasonal(row) -> Dict[str, Any]:
    pil = (row.get("pillars") or {}).get("seasonal") or {}
    s = row.get("seasonal") or {}
    facts = []
    if s.get("median_pct") is None:
        facts.append("No seasonal statistics for this window.")
        return {"key": "seasonal", "title": "Seasonality", "sign": 0, "rel": "neutral", "word": "n/a",
                "facts": facts, "tab": "seasonal"}
    w = s.get("window_td") or []
    facts.append(f"Median {_f(s['median_pct'], 2, True)}% over trading days {w[0] if w else '?'} to {w[-1] if w else '?'}, "
                 f"up in {s.get('hit_rate')}% of {s.get('n_years')} years, grade {s.get('grade')}. "
                 "Positive means price tends to rise.")
    if s.get("plan_note"):
        facts.append(s["plan_note"].rstrip(".") + ".")
    if not s.get("usable"):
        facts.append("Not used as a tilt" + (f": {s['why']}" if s.get("why") else "") + ".")
    sign = pil.get("sign") or 0
    return {"key": "seasonal", "title": "Seasonality", "sign": sign, "rel": pil.get("rel") or "neutral",
            "word": ("tends up" if sign > 0 else "tends down" if sign < 0 else "no usable tilt"),
            "facts": facts, "tab": "seasonal"}


def _valuation(row) -> Dict[str, Any]:
    pil = (row.get("pillars") or {}).get("valuation") or {}
    v = row.get("valuation") or {}
    facts = []
    L = v.get("lines") or {}
    if not L:
        facts.append("Valuation Tool lines unavailable for this market.")
    else:
        facts.append("Weekly Valuation Tool: " + ", ".join(
            f"vs {k.upper()} {_f(x.get('value'), 0, True)} ({x.get('state')})" for k, x in L.items()) + ".")
        c = v.get("composite") or {}
        if c.get("value") is not None:
            facts.append(f"Composite {_f(c['value'], 0, True)}, {c.get('state', '').lower()}. "
                         "Below \u221240 supports longs, above +40 supports shorts.")
        if v.get("skipped"):
            facts.append("Skipped " + ", ".join(x.upper() for x in v["skipped"]) + " (too close to the market itself).")
    bp = v.get("daily_buy_point") or {}
    if bp.get("active"):
        facts.append(f"Daily buy point is live (bar {bp.get('bars_held')} of 10): ZB {_f(bp.get('zb'), 0, True)}, "
                     f"DXY {_f(bp.get('dxy'), 0, True)}. {bp.get('record_note') or ''}".strip())
    sign = pil.get("sign") or 0
    word = "cheap" if sign > 0 else "rich" if sign < 0 else "fair"
    if bp.get("active"):
        word += ", buy point live"
    return {"key": "valuation", "title": "Valuation", "sign": sign, "rel": pil.get("rel") or "neutral",
            "word": word, "facts": facts, "tab": "relval", "buy_point": bool(bp.get("active"))}


# ── assembly ─────────────────────────────────────────────────────────────
_NAMES = {"positioning": "positioning", "macro": "macro", "trend": "trend",
          "seasonal": "seasonality", "valuation": "valuation"}


def build_read(row: Dict[str, Any], eng: Optional[Dict[str, Any]], fed: Optional[Dict[str, Any]] = None,
               now: Optional[datetime] = None) -> Dict[str, Any]:
    eng = eng or {}
    cls = row.get("class") or {}
    ck = cls.get("key")
    side = int(row.get("side") or 0)
    pillars = [_positioning(row), _macro(row, eng, fed), _trend(row, eng), _seasonal(row), _valuation(row)]
    P = {p["key"]: p for p in pillars}

    longs = [_NAMES[p["key"]] for p in pillars if p["sign"] > 0]
    shorts = [_NAMES[p["key"]] for p in pillars if p["sign"] < 0]
    lean = 0
    if side:
        lean = side
    elif P["macro"]["sign"]:
        lean = P["macro"]["sign"]
    elif len(longs) != len(shorts):
        lean = 1 if len(longs) > len(shorts) else -1

    name = row.get("name") or row.get("id")
    sw = _SIDE[side].capitalize()
    trig = row.get("trigger")
    if ck == "aligned":
        headline = f"{sw} setup: positioning and macro agree."
    elif ck == "payoff":
        headline = f"{sw} on positioning. Macro does not confirm yet."
    elif ck == "payoff_vs_macro":
        headline = f"{sw} positioning setup, against the macro backdrop."
    elif ck == "backdrop":
        headline = f"{sw} backdrop, but no crowd to fade."
    elif ck == "lapsed":
        headline = "Crowd extreme has outlasted the playbook clock."
    else:
        headline = None  # filled from the pillar split below

    pb = row.get("playbook") or {}
    summary: List[str] = []
    split = []
    if longs:
        split.append(_join(longs) + (" lean" if len(longs) > 1 else " leans") + " long")
    if shorts:
        split.append(_join(shorts) + (" lean" if len(shorts) > 1 else " leans") + " short")
    if ck in ("aligned", "payoff", "payoff_vs_macro"):
        mg = (row.get("positioning") or {}).get("managers")
        if mg is not None and (mg <= EXT_LO or mg >= EXT_HI):
            s = f"Fund managers sit at the {_ord(mg)} percentile, so the fade is {_SIDE[side]}."
        else:
            n = pb.get("reports_since_extreme")
            s = (f"Fund managers reached an extreme {n} reports ago and are now at the {_ord(mg)}, "
                 f"so the fade is still {_SIDE[side]}.") if n is not None else f"The fade is {_SIDE[side]}."
        if trig:
            s += f" Trigger: {trig}."
        summary.append(s)
        ag = [_NAMES[p["key"]] for p in pillars[1:] if p["rel"] == "against"]
        if ag:
            summary.append(f"Against it: {_join(ag)}. Size for that.")
    elif ck == "backdrop":
        summary.append(f"Macro and trend point {_SIDE[side]} with no positioning extreme behind them. "
                       "Trade it only off your own level.")
    elif ck == "lapsed":
        summary.append("The extreme is older than the six-report test window without a turn. Respect the trend until it breaks.")
    else:
        if split:
            headline = "No setup: " + _join(split) + "."
        else:
            headline = "No setup and no lean this week."
        if longs and shorts:
            summary.append("The pieces disagree, which is a reason to wait rather than average them into a neutral score.")
        if not row.get("positioning"):
            summary.append("There is no house positioning read here, so a setup would have to come from the backdrop alone.")
        elif not (P["positioning"]["word"].startswith("extreme")):
            summary.append("Positioning is not at an extreme, so there is no crowd to lean on.")
    if P["macro"].get("marginal"):
        summary.append("The macro lean is marginal: the vote only just clears the engine's deadband.")

    watch: List[str] = []
    pos = row.get("positioning")
    if pos and pos.get("managers") is not None and EXT_LO < pos["managers"] < EXT_HI:
        m = pos["managers"]
        near = EXT_LO if m < 50 else EXT_HI
        watch.append(f"Fund managers at the {_ord(m)}: a fresh extreme needs the {_ord(near)} or beyond "
                     f"({_f(abs(m - near), 0)} pts away).")
    pb = row.get("playbook") or {}
    if ck in ("aligned", "payoff", "payoff_vs_macro") and pb.get("status"):
        watch.append(f"Playbook state: {pb['status']}"
                     + (f", managers turning {pb['managers_turning_reports']} report(s)" if pb.get("managers_turning_reports") is not None else "")
                     + (", price through the 10-week average" if pb.get("close_through_10w") else ", price not through the 10-week average") + ".")
    vc = ((row.get("valuation") or {}).get("composite") or {}).get("value")
    if vc is not None and 25 <= abs(vc) < 40:
        thr = "\u221240" if vc < 0 else "+40"
        watch.append(f"Valuation composite {_f(vc, 0, True)}: it becomes a {'long' if vc < 0 else 'short'} tilt past {thr}.")
    s = row.get("seasonal") or {}
    if s.get("usable") and s.get("plan_note"):
        watch.append("Seasonal timing: " + s["plan_note"].rstrip(".") + ".")
    if P["macro"].get("marginal"):
        watch.append("A small move in the risk regime or macro score would remove the macro lean.")

    risks = list(row.get("gates") or [])
    evs = []
    for e in (row.get("events") or [])[:4]:
        try:
            t = datetime.fromisoformat(str(e["datetime_utc"]).replace("Z", "+00:00"))
            if now and t < now:
                continue
            evs.append({"name": e.get("name"), "currency": e.get("currency"), "datetime_utc": e.get("datetime_utc"),
                        "forecast": e.get("forecast"), "prior": e.get("prior")})
        except Exception:
            continue

    eg = row.get("engine") or {}
    ws = eng.get("weighted_score")
    edir = eg.get("direction") or eng.get("direction")
    eng_sign = 1 if edir == "Bullish" else -1 if edir == "Bearish" else 0
    disagree = bool(eng_sign and lean and eng_sign != lean)
    votes = eng.get("factor_votes") or {}
    vnames = {"seasonal": "seasonality", "regime": "risk regime", "macro": "macro", "cot": "COT",
              "momentum": "momentum", "relval": "relative value", "pcr": "put/call"}
    eb = [vnames.get(k, k) for k, v in votes.items() if v > 0]
    es = [vnames.get(k, k) for k, v in votes.items() if v < 0]
    et = (f"The 1 to 10 composite is {_f(ws)} ({eng.get('bias') or 'n/a'}). Its engine leans "
          f"{(edir or 'neutral').lower()} at conviction {_f(eg.get('conviction'), 2, True)}, tier {eg.get('tier') or 'n/a'}"
          + (f"; bullish votes: {_join(eb)}" if eb else "") + (f"; bearish votes: {_join(es)}" if es else "") + ".")
    if disagree:
        et += (f" That differs from the read above, which takes direction from macro context "
               f"rather than the full vote. Both are shown; neither is adjusted to match.")
    out = {
        "id": row.get("id"),
        "headline": headline,
        "stance": {"key": ck, "label": cls.get("label"), "tone": cls.get("tone"), "side": side,
                   "lean": lean, "lean_basis": ("side" if side else "macro" if P["macro"]["sign"] else
                                                 "pillar count" if lean else None),
                   "trigger": trig, "with": row.get("with"), "against": row.get("against")},
        "summary": summary,
        "pillars": pillars,
        "watch": watch,
        "risks": risks,
        "events": evs,
        "engine": {"text": et, "composite": ws, "direction": edir, "disagrees": disagree},
    }
    return _clean(out)


def _clean(x):
    """House style: no em dashes in generated prose."""
    if isinstance(x, str):
        return x.replace(" \u2014 ", "; ").replace("\u2014", "; ")
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x
