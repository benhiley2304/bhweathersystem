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
  headline   one line: side, setup type and why
  stance     grade key/label, side, lean, setup type, for/against lists
  summary    2-4 sentences: what kind of trade, what is for and against, sizing
  pillars    positioning, valuation, trend, seasonal, macro: sign, rel, role, facts[]
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
def _pil(row, k) -> Dict[str, Any]:
    return (row.get("pillars") or {}).get(k) or {}


def _positioning(row) -> Dict[str, Any]:
    p = row.get("positioning")
    pb = row.get("playbook") or {}
    pil = _pil(row, "positioning")
    R = row.get("positioning_read") or {}
    facts: List[str] = []
    if not p:
        facts.append("No house COT for this market (crosses and some contracts have none).")
        return {"key": "positioning", "title": "Positioning", "sign": 0, "rel": "na", "role": pil.get("role"),
                "word": "n/a", "facts": facts, "tab": "cot"}
    facts.append(f"Commercials {_ord(p.get('commercials'))}, fund managers {_ord(p.get('managers'))}, "
                 f"retail {_ord(p.get('retail'))} percentile (futures-only, 156-report mid-rank, "
                 f"observed {_date(p.get('observed'))}).")
    ch = (p.get("changes") or {}).get("managers") or {}
    if ch.get("chg_1") is not None:
        facts.append(f"Fund managers {_f(ch.get('chg_1'), 1, True)} pts on the latest report, "
                     f"{_f(ch.get('chg_4'), 1, True)} over four.")
    dv = row.get("divergence") or {}
    if dv.get("dir"):
        facts.append(dv.get("text"))
    if pb.get("explain"):
        facts.append(pb["explain"])
    if R.get("equity"):
        facts.append("Equity index: positioning is a size warning here, never the side.")
    word = R.get("word") or "no extreme"
    return {"key": "positioning", "title": "Positioning", "sign": pil.get("sign") or 0,
            "rel": pil.get("rel") or "none", "role": pil.get("role"), "word": word, "facts": facts, "tab": "cot"}


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


_TREND_WORD = {"running": "trending", "pullback": "pulling back inside the trend",
               "extended": "extended and stalling", "range": "rangebound", "n/a": "n/a"}


def _trend(row, eng) -> Dict[str, Any]:
    pil = _pil(row, "trend")
    T = row.get("trend") or {}
    facts = []
    if T.get("sma200_diff") is not None:
        facts.append(f"Price {_f(abs(T['sma200_diff']))}% {'above' if T['sma200_diff'] >= 0 else 'below'} "
                     f"the 200-day; {_ord(T.get('pct_52w'))} percentile of its 52-week range.")
    if T.get("roc26w") is not None:
        if T.get("horizon_basis") == "daily_bars":
            facts.append(f"Change over 4 sessions {_f(T.get('roc4w'), 1, True)}%, 13 sessions {_f(T.get('roc13w'), 1, True)}%, "
                         f"26 sessions {_f(T.get('roc26w'), 1, True)}% (weekly horizons unavailable).")
        else:
            facts.append(f"Change over 4 weeks {_f(T.get('roc4w'), 1, True)}%, 13 weeks {_f(T.get('roc13w'), 1, True)}%, "
                         f"26 weeks {_f(T.get('roc26w'), 1, True)}%.")
    if T.get("state") == "extended":
        facts.append("Outer 15% of the 52-week range with the 13-week move stalled: the trend looks tired, "
                     "so it does not count against a fade.")
    elif T.get("state") == "pullback":
        facts.append("Last four weeks against an intact trend: a pullback, the entry window for trend trades.")
    if T.get("state_text"):
        facts.append(f"Engine trend state: {T['state_text']}.")
    d = T.get("dir") or 0
    dw = "up" if d > 0 else "down" if d < 0 else ""
    st = T.get("state") or "n/a"
    word = (f"{dw}, " if dw and st != "range" else "") + _TREND_WORD.get(st, st)
    return {"key": "trend", "title": "Trend", "sign": pil.get("sign") or 0, "rel": pil.get("rel") or "neutral",
            "role": pil.get("role"), "state": st, "word": word, "facts": facts, "tab": "momentum"}


def _seasonal(row) -> Dict[str, Any]:
    pil = _pil(row, "seasonal")
    S = row.get("seasonal_fwd") or {}
    fw = S.get("fwd") or {}
    facts = []
    if not fw:
        s = row.get("seasonal") or {}
        if s.get("median_pct") is None:
            facts.append("No seasonal path for this market.")
            return {"key": "seasonal", "title": "Seasonality", "sign": 0, "rel": "neutral", "role": pil.get("role"),
                    "word": "n/a", "facts": facts, "tab": "seasonal"}
        w = s.get("window_td") or []
        facts.append(f"Median {_f(s['median_pct'], 2, True)}% over trading days {w[0] if w else '?'} to "
                     f"{w[-1] if w else '?'} (positive means price tends to rise).")
        return {"key": "seasonal", "title": "Seasonality", "sign": 0, "rel": "neutral", "role": pil.get("role"),
                "word": "window only", "facts": facts, "tab": "seasonal"}
    bits = []
    for h in ("30", "60", "90"):
        x = fw.get(h) or fw.get(int(h)) if isinstance(fw, dict) else None
        if x and x.get("median") is not None:
            bits.append(f"{h} TD {_f(x['median'], 2, True)}% (up {_f(x.get('hit_up'), 0)}%)")
    if bits:
        n = (fw.get("60") or fw.get(60) or {}).get("n")
        facts.append("Forward path from today, weighted median of past years: " + ", ".join(bits)
                     + (f", {n} years" if n else "") + ". Positive means price tends to rise.")
    facts.append("In the setup score, seasonality is the timing gate: a path against the side over 4 to 8 weeks is a veto, and a dip before a window means wait.")
    return {"key": "seasonal", "title": "Seasonality", "sign": pil.get("sign") or 0,
            "rel": pil.get("rel") or "neutral", "role": pil.get("role"), "word": S.get("word") or "n/a",
            "facts": facts, "tab": "seasonal"}


def _valuation(row) -> Dict[str, Any]:
    pil = _pil(row, "valuation")
    v = row.get("valuation") or {}
    R = row.get("valuation_read") or {}
    facts = []
    if not R.get("refs"):
        facts.append("Valuation Tool lines unavailable for this market.")
    else:
        facts.append("Valuation Tool, daily (D) and weekly (W): " + R["refs"] + ". "
                     "Below \u221240 is cheap, above +40 rich.")
        if R.get("split"):
            facts.append(_join([k.upper() for k in R["split"]]) + (" reads" if len(R["split"]) == 1 else " read")
                         + " opposite ways on the two timeframes, so it counts for neither side.")
        ctx = {k: x for k, x in (R.get("weekly_context") or {}).items() if x is not None}
        if ctx:
            facts.append("Weekly context only: " + ", ".join(f"{k.upper()} {_f(x, 0, True)}" for k, x in ctx.items()) + ".")
    bp = v.get("daily_buy_point") or {}
    if bp.get("active"):
        facts.append(f"Daily buy point is live (bar {bp.get('bars_held')} of 10): ZB {_f(bp.get('zb'), 0, True)}, "
                     f"DXY {_f(bp.get('dxy'), 0, True)}. {bp.get('record_note') or ''}".strip())
    word = R.get("word") or "n/a"
    if bp.get("active"):
        word += ", buy point live"
    return {"key": "valuation", "title": "Valuation", "sign": pil.get("sign") or 0, "rel": pil.get("rel") or "neutral",
            "role": pil.get("role"), "word": word, "facts": facts, "tab": "relval", "buy_point": bool(bp.get("active"))}


# ── assembly ─────────────────────────────────────────────────────────────
_NAMES = {"positioning": "positioning", "macro": "macro", "trend": "trend",
          "seasonal": "seasonality", "valuation": "valuation"}


def _why(row, pillars) -> str:
    P = {p["key"]: p for p in pillars}
    bits = []
    for k in ("positioning", "valuation", "trend", "seasonal"):
        p = P.get(k) or {}
        if p.get("rel") in ("with", "with_pullback", "exhausted") or (k == "trend" and p.get("rel") == "with_late"):
            w = p.get("word") or ""
            if k == "positioning":
                dv = row.get("divergence") or {}
                w = "fund managers diverging" if dv.get("dir") == row.get("side") else w.split(":")[0]
            elif k == "seasonal":
                w = "seasonal " + w
            elif k == "trend":
                up = "uptrend" if (row.get("trend") or {}).get("dir", 0) > 0 else "downtrend"
                w = {"extended": "trend extended", "pullback": up + " intact"}.get(p.get("state"), up + " running")
            bits.append(w)
    return ", ".join(bits)


_ARCH_NOTE = {
    "Crowd fade": "Fading a positioning extreme with commercials on the other side.",
    "Divergence fade": "Fading a trend that has run on while fund managers stopped adding.",
    "Exhaustion fade": "Fading a stretched trend that has stalled.",
    "Pullback in trend": "Buying the dip (or selling the bounce) inside a healthy trend.",
    "Trend with value": "With the trend, and the valuation agrees.",
    "With trend": "With the trend; the payoff case is thinner.",
    "Counter-trend": "Against a trend that is still running: wait for it to turn.",
    "Range": "No trend to lean on.",
}


def build_read(row: Dict[str, Any], eng: Optional[Dict[str, Any]], fed: Optional[Dict[str, Any]] = None,
               now: Optional[datetime] = None) -> Dict[str, Any]:
    eng = eng or {}
    cls = row.get("class") or {}
    ck = cls.get("key")
    side = int(row.get("side") or 0)
    lean = int(row.get("lean") or side or 0)
    pillars = [_positioning(row), _valuation(row), _trend(row, eng), _seasonal(row), _macro(row, eng, fed)]
    P = {p["key"]: p for p in pillars}
    arch = row.get("archetype")
    sw = _SIDE.get(side or lean, "flat").capitalize()
    why = _why(row, pillars)
    names = lambda xs: _join([_NAMES.get(x, x) for x in xs or []])

    if ck in ("prime", "setup", "developing") and side:
        lead = {"prime": "", "setup": "", "developing": "Developing "}[ck]
        headline = f"{lead}{sw.lower() if lead else sw}: {(arch or 'setup').lower()}." + (f" {why[0].upper() + why[1:]}." if why else "")
    elif ck == "backdrop" and side:
        headline = f"{sw} backdrop: trend and macro agree, no payoff case yet."
    else:
        fl, al = row.get("for_list") or [], row.get("against_list") or []
        headline = "No setup this week." + (f" {names(fl).capitalize()} lean {_SIDE.get(lean, '')}" if fl and lean else "")
        if headline.endswith(("long", "short")):
            headline += "."

    summary: List[str] = []
    if arch and ck != "none":
        summary.append(_ARCH_NOTE.get(arch, ""))
    if row.get("for_list"):
        summary.append(f"For the {_SIDE.get(side or lean, '')}: {names(row['for_list'])}."
                       + (f" Against: {names(row.get('against_list'))}." if row.get("against_list") else ""))
    elif row.get("against_list"):
        summary.append(f"Against: {names(row['against_list'])}.")
    for d in row.get("downgrades") or []:
        summary.append(f"Marked down a grade: {d}.")
    sz = [g for g in (row.get("gates") or []) if "size" in g or "starter" in g or "adds" in g]
    if sz:
        summary.append("Sizing: " + "; ".join(sz) + ".")
    if P["macro"].get("marginal"):
        summary.append("The macro lean is marginal: the vote only just clears the engine's deadband.")
    summary = [x for x in summary if x]

    watch: List[str] = []
    cf = row.get("confirm") or {}
    if cf.get("text"):
        watch.append(f"{cf.get('state') or 'Confirmation'}: {cf['text']}")
    elif row.get("trigger"):
        watch.append(f"Playbook: {row['trigger']}.")
    pos = row.get("positioning")
    if pos and pos.get("managers") is not None and EXT_LO < pos["managers"] < EXT_HI and not (row.get("divergence") or {}).get("dir"):
        m = pos["managers"]
        near = EXT_LO if m < 50 else EXT_HI
        watch.append(f"Fund managers at the {_ord(m)}: a fresh extreme needs the {_ord(near)} or beyond "
                     f"({_f(abs(m - near), 0)} pts away).")
    T = row.get("trend") or {}
    if T.get("state") == "extended" and side and side != T.get("dir"):
        watch.append("The fade needs the trend to stay stalled: a fresh 13-week push in the trend direction undoes it.")
    if T.get("state") == "pullback" and side == T.get("dir"):
        watch.append("The pullback case breaks if price closes through the 200-day.")
    if P["macro"].get("marginal"):
        watch.append("A small move in the risk regime or macro score would remove the macro lean.")
    watch.append("Zones and stops are yours; the Desk does not draw levels.")

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
        et += (" That differs from the read above, which weighs positioning and valuation as the payoff and "
               "treats an extended trend as tired rather than as a vote. Both are shown; neither is adjusted to match.")
    out = {
        "id": row.get("id"),
        "headline": headline,
        "archetype": arch,
        "stance": {"key": ck, "label": cls.get("label"), "tone": cls.get("tone"), "side": side,
                   "lean": lean, "lean_basis": "grade" if side else ("evidence" if lean else None),
                   "trigger": row.get("trigger"), "with": row.get("with"), "against": row.get("against"),
                   "archetype": arch, "for": row.get("for_list"), "against_list": row.get("against_list")},
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
