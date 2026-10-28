"""House strategy read: how Ben finds and judges a swing setup, as rules.

Built from the BH Macro issues (Issue 07 copper divergence, Issue 10 "The
Edge", Issue 11 playbook, the equity-index notes) and Ben's 28 Oct note on
the dollar and Nasdaq. The point is that trend is context, not the verdict:

    positioning   sets the payoff: a crowd extreme, or fund managers making a
                  lower high in net length against a higher high in price
                  (and the mirror), or commercials hard against the crowd.
                  For equity indices it is a size warning, never the side
                  ("the crowd is long with the book").
    valuation     BHWS Valuation Tool on daily bars against the references
                  that matter for the asset. Cheap or rich is a payoff source
                  of its own.
    trend         healthy, pulling back, or extended and stalling. A pullback
                  in a healthy trend is a timing gift for the with-trend side.
                  An extended, stalling trend no longer counts against a fade.
    macro         lifts or lowers the hit rate. Against means smaller, not no.
    seasonality   a minor tilt, read off the forward path at 30 and 60
                  trading days (positive means price tends to rise).
    zones         are Ben's. Nothing here draws or proposes a level.

Grades are rules, not a score: Prime, Setup, Developing, Backdrop, None. The
alignment number that orders markets inside a grade is a ranking aid and
is labelled as not a probability. None of these rules has been backtested
as a whole; the divergence and exhaustion tests are heuristics.

Every function is pure: callers pass data in, nothing is fetched here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import percentile

EQUITY_IDS = {"ES", "NQ", "YM", "RTY", "Z"}

# Role weights for the ordering number only (Issue 10 hierarchy).
W = {"positioning": 3.0, "valuation": 2.5, "trend": 2.0, "macro": 1.5, "seasonal": 1.0}
W_POS_EQUITY = 1.0

LIVE = ("Not turned", "One more week", "Turned", "Past test window", "Add eligible", "Hold; no add")
LAPSED = ("Extreme persisting", "Expired")

GRADES = [
    # key, label, rank, tone, note
    ("prime", "Prime setup", 0, "prime",
     "Two payoff sources agree (positioning, valuation) or one plus a healthy trend and the seasonal, "
     "and the trend is not running hard the other way."),
    ("setup", "Setup", 1, "neutral",
     "One payoff source with support from trend, seasonality or macro, and no healthy trend against."),
    ("developing", "Developing", 2, "warn",
     "A payoff source is there but it is fighting a healthy trend or has nothing behind it yet."),
    ("backdrop", "Backdrop only", 3, "muted",
     "Trend and macro agree, but there is no crowd and no valuation edge. Your level or nothing."),
    ("none", "Nothing to do", 4, "muted", "No payoff source and no aligned backdrop."),
]
GRADE = {k: dict(key=k, label=l, rank=r, tone=t, note=n) for k, l, r, t, n in GRADES}
ORDER = [g[0] for g in GRADES]


def _sgn(x) -> int:
    if x is None:
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _num(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def is_equity(market: Dict[str, Any]) -> bool:
    return market.get("id") in EQUITY_IDS or market.get("category") == "equity"


# ── trend ────────────────────────────────────────────────────────────────
def trend_profile(engine: Dict[str, Any]) -> Dict[str, Any]:
    """Direction plus condition. Uses the engine momentum detail only.

    state: running | pullback | extended | range | n/a
      pullback  trend intact (right side of the 200-day) but the last four
                weeks went the other way
      extended  price in the outer 15% of its 52-week range and the
                13-week move has stalled (under a quarter of the 26-week
                move, or the wrong sign) or the engine calls it flat
    """
    ts = str(engine.get("trend_state") or "")
    md = (((engine.get("scores") or {}).get("momentum") or {}).get("detail")) or {}
    d = 1 if ts.startswith("Uptrend") else -1 if ts.startswith("Downtrend") else 0
    if not ts:
        v = (engine.get("factor_votes") or {}).get("momentum")
        d = _sgn(v) if v is not None else 0
    pr = _num(md.get("pct_range"))
    r4 = _num(md.get("roc4w_pct"))
    r13 = _num(md.get("roc13w_pct"))
    r26 = _num(md.get("roc26w_pct"))
    sma = _num(md.get("sma200_pct_diff"))
    er = _num(md.get("efficiency_ratio"))
    out = {"dir": d, "state_text": ts or None, "pct_52w": pr, "roc4w": r4, "roc13w": r13, "roc26w": r26,
           "sma200_diff": sma, "efficiency": er, "stretched": False, "stalling": False, "state": "n/a"}
    if not md and not ts:
        return out
    if d == 0:
        out["state"] = "range"
        return out
    stretched = pr is not None and ((d > 0 and pr >= 85) or (d < 0 and pr <= 15))
    stall = False
    if r13 is not None and r26 is not None:
        stall = (_sgn(r13) != d) or (abs(r13) < 0.25 * abs(r26))
    if "flat" in ts:
        stall = True
    pull = ("pullback" in ts or "bounce" in ts) or (r4 is not None and _sgn(r4) == -d and abs(r4) >= 0.5)
    if pull and sma is not None and _sgn(sma) != d:
        pull = False          # through the 200-day is not a pullback
    out["stretched"], out["stalling"] = bool(stretched), bool(stall)
    if stretched and stall and not pull:
        out["state"] = "extended"
    elif pull:
        out["state"] = "pullback"
    else:
        out["state"] = "running"
    return out


# ── positioning ──────────────────────────────────────────────────────────
def manager_divergence(prices: Sequence[float], mgr_pct: Sequence[Optional[float]],
                       lookback: int = 26, min_gap: int = 3, peak_min: float = 80.0,
                       drop_min: float = 10.0, near_high: float = 0.02) -> Dict[str, Any]:
    """Fund manager divergence on weekly report dates (Issue 07).

    Bearish: price is within `near_high` of its lookback high and above where
    it stood when manager net length peaked (a higher high), while managers
    peaked at or above the `peak_min` percentile at least `min_gap` reports
    ago and now sit `drop_min` points or more below that peak (a lower high).
    Bullish is the mirror. Inputs are aligned per report; None means no data.
    Returns dir 0 when there is no divergence.
    """
    n = min(len(prices), len(mgr_pct))
    if n < lookback + 1:
        return {"dir": 0, "why": "not enough history"}
    P = list(prices)[-lookback:]
    M = list(mgr_pct)[-lookback:]
    if any(x is None for x in P) or M[-1] is None:
        return {"dir": 0, "why": "gaps in the window"}
    now_p, now_m = P[-1], M[-1]
    hi, lo = max(P), min(P)
    cand = [(i, m) for i, m in enumerate(M[:-min_gap]) if m is not None]
    if not cand:
        return {"dir": 0}
    i_hi, m_hi = max(cand, key=lambda t: t[1])
    i_lo, m_lo = min(cand, key=lambda t: t[1])
    out = {"dir": 0}
    if (m_hi >= peak_min and now_m <= m_hi - drop_min and now_p >= hi * (1 - near_high)
            and now_p > P[i_hi]):
        out = {"dir": -1, "kind": "bearish", "peak_pct": round(m_hi, 1), "now_pct": round(now_m, 1),
               "reports_since_peak": lookback - 1 - i_hi, "price_at_peak": P[i_hi], "price_now": now_p}
    elif (m_lo <= 100 - peak_min and now_m >= m_lo + drop_min and now_p <= lo * (1 + near_high)
            and now_p < P[i_lo]):
        out = {"dir": 1, "kind": "bullish", "peak_pct": round(m_lo, 1), "now_pct": round(now_m, 1),
               "reports_since_peak": lookback - 1 - i_lo, "price_at_peak": P[i_lo], "price_now": now_p}
    if out["dir"]:
        chg = (out["price_now"] / out["price_at_peak"] - 1) * 100 if out["price_at_peak"] else None
        if out["dir"] < 0:
            out["text"] = (f"Fund managers made a lower high: net length peaked at the "
                           f"{_o(out['peak_pct'])} percentile {out['reports_since_peak']} reports ago and is now "
                           f"{_o(out['now_pct'])}, while price made a higher high"
                           + (f" ({chg:+.1f}% since that peak)" if chg is not None else "") + ".")
        else:
            out["text"] = (f"Fund managers made a higher low: net length troughed at the "
                           f"{_o(out['peak_pct'])} percentile {out['reports_since_peak']} reports ago and is now "
                           f"{_o(out['now_pct'])}, while price made a lower low"
                           + (f" ({chg:+.1f}% since that trough)" if chg is not None else "") + ".")
        out["chg_since_peak_pct"] = None if chg is None else round(chg, 2)
    return out


def _o(x) -> str:
    if x is None:
        return "n/a"
    n = int(round(float(x)))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def positioning_read(pos: Optional[Dict[str, Any]], pb: Optional[Dict[str, Any]],
                     div: Optional[Dict[str, Any]], equity: bool) -> Dict[str, Any]:
    """Which side the crowd structure pays, and how strongly (0..1)."""
    if not pos:
        return {"sign": None, "strength": 0.0, "parts": [], "word": "n/a", "equity": equity}
    mg, cm, rt = _num(pos.get("managers")), _num(pos.get("commercials")), _num(pos.get("retail"))
    pb = pb or {}
    parts: List[Tuple[str, int, float, str]] = []
    st, fade = pb.get("status"), pb.get("direction") or 0
    if st in LIVE and fade:
        conf = cm is not None and ((fade > 0 and cm >= 70) or (fade < 0 and cm <= 30))
        parts.append(("extreme", fade, 1.0 if conf else 0.8,
                      "fund manager extreme" + (", commercials on the other side" if conf else "")))
    if div and div.get("dir"):
        d = div["dir"]
        conf = cm is not None and ((d > 0 and cm >= 60) or (d < 0 and cm <= 40))
        parts.append(("divergence", d, 0.8 if conf else 0.65,
                      "fund manager divergence" + (", commercials against the crowd" if conf else "")))
    if cm is not None:
        crowd_long = (mg is not None and mg >= 70) or (rt is not None and rt >= 75)
        crowd_short = (mg is not None and mg <= 30) or (rt is not None and rt <= 25)
        if cm <= 20 and crowd_long:
            parts.append(("structure", -1, 0.5, "commercials short against a long crowd"))
        elif cm >= 80 and crowd_short:
            parts.append(("structure", 1, 0.5, "commercials long against a short crowd"))
    if not parts:
        word = "no extreme"
        if st in LAPSED:
            word = "extreme outlasted the clock (a trend, not a fade)"
        return {"sign": 0, "strength": 0.0, "parts": [], "word": word, "equity": equity, "lapsed": st in LAPSED}
    score = {1: 0.0, -1: 0.0}
    best = {1: 0.0, -1: 0.0}
    for _, d, s, _t in parts:
        best[d] = max(best[d], s)
        score[d] += s
    sgn = 1 if best[1] > best[-1] else -1 if best[-1] > best[1] else _sgn(score[1] - score[-1])
    if sgn == 0:
        return {"sign": 0, "strength": 0.0, "parts": [p[3] for p in parts], "word": "mixed", "equity": equity}
    n_same = sum(1 for p in parts if p[1] == sgn)
    strength = min(1.0, best[sgn] + 0.1 * (n_same - 1)) - 0.5 * best[-sgn]
    strength = max(0.0, round(strength, 2))
    kinds = [p[0] for p in parts if p[1] == sgn]
    txt = [p[3] for p in parts if p[1] == sgn]
    if any("commercials" in t for t in txt if not t.startswith("commercials")):
        txt = [t for t in txt if not t.startswith("commercials")]   # structure already said
    word = "crowd " + ("short" if sgn > 0 else "long") + ": " + "; ".join(txt)
    return {"sign": sgn if strength > 0 else 0, "strength": strength, "kinds": kinds,
            "parts": [p[3] for p in parts], "word": word, "equity": equity}


# ── valuation ────────────────────────────────────────────────────────────
def _ref_state(d: Optional[float], w: Optional[float]) -> Tuple[int, Optional[float], str]:
    """One reference across both timeframes. Returns (sign, depth, how).
    sign -1 = rich, +1 = cheap (the side valuation favours).

    both    daily and weekly on the same side, one at 40+, the other at 20+
    daily   daily at 40+, weekly quiet (inside +/-20) or missing
    weekly  weekly at 40+, daily on the same side at 20+
    split   the two timeframes point opposite ways (either beyond 20)
    """
    if d is None and w is None:
        return 0, None, "n/a"
    if d is not None and w is not None and ((d >= 20 and w <= -20) or (d <= -20 and w >= 20)):
        return 0, None, "split"
    for sgn in (1, -1):                      # +1 cheap (negative line), -1 rich
        dd = None if d is None else -d * sgn
        ww = None if w is None else -w * sgn
        if dd is not None and ww is not None and max(dd, ww) >= 40 and min(dd, ww) >= 20:
            return sgn, max(dd, ww), "both"
        if dd is not None and dd >= 40 and (ww is None or abs(ww) < 20):
            return sgn, dd, "daily"
        if ww is not None and ww >= 40 and dd is not None and dd >= 20:
            return sgn, ww, "weekly"
    return 0, None, "fair"


def valuation_read(val: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Cheap/rich per reference from the Valuation Tool on BOTH the daily
    and weekly bars (Ben reads either chart). A reference counts when the
    timeframes agree, or one is strong and the other quiet; opposite reads
    are reported as split and count for neither side. Two qualifying
    references and none opposing = full read; one qualifying and none
    opposing = reduced strength. Weekly-only references (e.g. gold for
    indices) are context and are not counted."""
    if not val:
        return {"sign": None, "strength": 0.0, "basis": None, "word": "n/a"}
    daily = (val.get("daily") or {}).get("lines") or {}
    weekly = val.get("lines") or {}
    keys = list(daily.keys()) or list(weekly.keys())
    per, xs = {}, []
    for k in keys:
        d = _num((daily.get(k) or {}).get("value"))
        w = _num((weekly.get(k) or {}).get("value"))
        if d is None and w is None:
            continue
        sg, depth, how = _ref_state(d, w)
        per[k] = {"daily": d, "weekly": w, "sign": sg, "depth": depth, "how": how}
        xs.append(k)
    if not xs:
        return {"sign": None, "strength": 0.0, "basis": None, "word": "n/a"}
    cheap = [k for k in xs if per[k]["sign"] > 0]
    rich = [k for k in xs if per[k]["sign"] < 0]
    split = [k for k in xs if per[k]["how"] == "split"]
    n = len(xs)
    sign, use = 0, []
    if cheap and not rich:
        sign, use = 1, cheap
    elif rich and not cheap:
        sign, use = -1, rich
    strength = 0.0
    if sign:
        depth = min(1.0, sum(per[k]["depth"] for k in use) / len(use) / 75.0)
        strength = 0.35 * (len(use) / n) + 0.65 * depth
        if len(use) == 1 and n >= 2:
            strength *= 0.75
        strength = round(min(1.0, strength), 2)

    def _f(k):
        x = per[k]
        a = [] if x["daily"] is None else [f"D {x['daily']:+.0f}"]
        if x["weekly"] is not None:
            a.append(f"W {x['weekly']:+.0f}")
        return f"{k.upper()} " + "/".join(a)
    refs = ", ".join(_f(k) for k in xs)
    if sign:
        word = ("cheap vs " if sign > 0 else "rich vs ") + " and ".join(k.upper() for k in use)
        if split:
            word += " (" + ", ".join(k.upper() for k in split) + " split across timeframes)"
    else:
        word = "split" if (cheap and rich) or split else "fair"
    ctx = {k: _num(v.get("value")) for k, v in weekly.items() if k not in per}
    return {"sign": sign, "strength": strength, "basis": "daily+weekly", "per_ref": per,
            "lines": {k: per[k]["daily"] if per[k]["daily"] is not None else per[k]["weekly"] for k in xs},
            "refs": refs, "word": word, "cheap": cheap, "rich": rich, "split": split,
            "weekly_context": ctx}


# ── seasonality ──────────────────────────────────────────────────────────
def _wq(pts: List[Tuple[float, float]], q: float) -> Optional[float]:
    pts = sorted(pts)
    tot = sum(w for _, w in pts)
    if tot <= 0:
        return None
    acc = 0.0
    for v, w in pts:
        acc += w
        if acc >= q * tot:
            return v
    return pts[-1][0]


def seasonal_forward(years: Dict[str, Sequence[Optional[float]]], weights: Dict[str, float],
                     current_td: int, current_year: int,
                     horizons: Sequence[int] = (20, 30, 60, 90)) -> Dict[int, Dict[str, Any]]:
    """Weighted per-year forward returns from today's trading day, crossing
    the year end where the next year's path exists. `years` holds cumulative
    % paths from the first close of each year (1-based trading days, as the
    backend stores them). Completed years only; the current year never
    contributes. Returns {h: {median, p25, p75, hit_up, n}} in percent."""
    a = max(1, int(current_td)) - 1
    out: Dict[int, Dict[str, Any]] = {}
    for h in horizons:
        pts = []
        for ys, w in weights.items():
            try:
                y = int(ys)
            except (TypeError, ValueError):
                continue
            if y >= current_year or w <= 0:
                continue
            p = years.get(str(y))
            if not p or a >= len(p) or p[a] is None:
                continue
            j = a + h
            base = 1 + p[a] / 100.0
            if j < len(p):
                if p[j] is None:
                    continue
                r = (1 + p[j] / 100.0) / base - 1
            else:
                q = years.get(str(y + 1))
                k = j - len(p)
                if y + 1 >= current_year or not q or k >= len(q) or q[k] is None or p[-1] is None:
                    continue
                r = (1 + p[-1] / 100.0) / base * (1 + q[k] / 100.0) - 1
            pts.append((r * 100.0, float(w)))
        if len(pts) < 8:
            out[h] = {"median": None, "n": len(pts)}
            continue
        tot = sum(w for _, w in pts)
        up = sum(w for v, w in pts if v > 0) / tot * 100
        out[h] = {"median": round(_wq(pts, 0.5), 2), "p25": round(_wq(pts, 0.25), 2),
                  "p75": round(_wq(pts, 0.75), 2), "hit_up": round(up, 0), "n": len(pts)}
    return out


def seasonal_read(fwd: Optional[Dict[int, Dict[str, Any]]], primary: int = 60,
                  check: int = 30) -> Dict[str, Any]:
    """Minor tilt from the forward path. Positive median means price tends
    to rise. A tilt needs the 60 TD median to be a meaningful share of its
    own spread and the 30 TD median not to point the other way."""
    if not fwd or (fwd.get(primary) or {}).get("median") is None:
        return {"sign": None, "strength": 0.0, "word": "n/a", "fwd": fwd}
    f60, f30 = fwd[primary], fwd.get(check) or {}
    m60, m30 = f60["median"], f30.get("median")
    half = max(1e-6, (f60["p75"] - f60["p25"]) / 2.0)
    ratio = m60 / half
    hit = f60.get("hit_up")
    out = {"fwd": fwd, "primary": primary, "ratio": round(ratio, 2),
           "convention": "positive median = price tends to rise"}
    if abs(ratio) < 0.15 or abs(m60) < 0.2:
        out.update(sign=0, strength=0.0,
                   word=("flat to down" if (m60 < 0 or (m30 is not None and m30 < 0)) else
                         "flat to up" if (m60 > 0 or (m30 is not None and m30 > 0)) else "flat"))
        return out
    if m30 is not None and _sgn(m30) != _sgn(m60) and abs(m30) >= 0.2:
        out.update(sign=0, strength=0.0, word="mixed (30 and 60 TD disagree)")
        return out
    s = _sgn(m60)
    agree = hit is not None and ((s > 0 and hit >= 55) or (s < 0 and hit <= 45))
    strength = min(1.0, abs(ratio) / 0.5) * (1.0 if agree else 0.6)
    out.update(sign=s, strength=round(strength, 2), word=("tends up" if s > 0 else "tends down"))
    return out


# ── the read ─────────────────────────────────────────────────────────────
def _rel_simple(sign, side) -> str:
    if sign is None:
        return "na"
    if not side or not sign:
        return "neutral"
    return "with" if sign == side else "against"


def evaluate_side(side: int, *, P, V, T, M, S, div) -> Dict[str, Any]:
    """Pillar relations and the ordering number for one candidate side."""
    rel: Dict[str, str] = {}
    pts: Dict[str, float] = {}
    wp = W_POS_EQUITY if P.get("equity") else W["positioning"]
    for k, R, w in (("positioning", P, wp), ("valuation", V, W["valuation"]),
                    ("macro", M, W["macro"]), ("seasonal", S, W["seasonal"])):
        r = _rel_simple(R.get("sign"), side)
        rel[k] = r
        st = R.get("strength") or 0.0
        pts[k] = w * st if r == "with" else -w * st if r == "against" else 0.0
    d = T.get("dir") or 0
    if not d or T.get("state") in ("range", "n/a"):
        rel["trend"], pts["trend"] = ("neutral" if T.get("state") != "n/a" else "na"), 0.0
    elif d == side:
        if T.get("state") == "extended":
            rel["trend"], pts["trend"] = "with_late", 0.3 * W["trend"]
        else:
            rel["trend"], pts["trend"] = ("with_pullback" if T.get("state") == "pullback" else "with"), W["trend"]
    else:
        # A trend only counts as exhausted on its own evidence: stretched and
        # stalling, or stretched with fund managers already diverging. Cheap
        # valuation alone against a trend that is still running is a falling
        # knife, not exhaustion.
        exhausted = (T.get("state") == "extended"
                     or (T.get("stretched") and (div or {}).get("dir") == side))
        if exhausted:
            rel["trend"], pts["trend"] = "exhausted", 0.0
        else:
            rel["trend"], pts["trend"] = "against", -W["trend"]
    return {"side": side, "rel": rel, "pts": pts, "total": round(sum(pts.values()), 2)}


def _is_for(rel: str) -> bool:
    return rel in ("with", "with_pullback", "with_late")


def grade(ev: Dict[str, Any], *, P, V, S, M, T, pb_live: bool, conf: str) -> Dict[str, Any]:
    rel, side = ev["rel"], ev["side"]
    anchors = []
    if rel["positioning"] == "with" and (P.get("strength") or 0) >= 0.5 and not P.get("equity"):
        anchors.append("positioning")
    if rel["valuation"] == "with" and (V.get("strength") or 0) >= 0.5:
        anchors.append("valuation")
    trend_for = rel["trend"] in ("with", "with_pullback")
    veto = rel["trend"] == "against"
    seas_for, seas_against = rel["seasonal"] == "with", rel["seasonal"] == "against"
    supports = sum([trend_for, seas_for, rel["macro"] == "with"])
    if veto:
        k = "developing" if anchors else "none"
    elif (len(anchors) >= 2 or (anchors and trend_for and seas_for)) and not seas_against:
        k = "prime"
    elif anchors and (supports >= 1 or rel["trend"] == "exhausted" or pb_live):
        k = "setup"
    elif anchors:
        k = "developing"
    elif trend_for and rel["macro"] == "with":
        k = "backdrop"
    elif trend_for and seas_for and rel["valuation"] != "against":
        k = "developing"
    else:
        k = "none"
    downs = []
    # A payoff pillar firmly on the other side (index positioning excepted)
    # costs a grade: the reports never take a trade into a rich-and-crowded
    # mirror image of their own case.
    opp = []
    if rel["positioning"] == "against" and (P.get("strength") or 0) >= 0.5 and not P.get("equity"):
        opp.append("positioning")
    if rel["valuation"] == "against" and (V.get("strength") or 0) >= 0.5:
        opp.append("valuation")
    if k in ("prime", "setup") and opp:
        downs.append(" and ".join(opp) + " firmly on the other side")
    if k in ("prime", "setup", "developing") and rel["macro"] == "against" and seas_against:
        downs.append("macro and seasonality both against")
    if k in ("prime", "setup", "developing") and conf in ("Low", "Insufficient"):
        downs.append(f"data confidence {conf.lower()}")
    for _ in downs:
        k = ORDER[min(ORDER.index(k) + 1, ORDER.index("developing"))] if k != "developing" else "developing"
    return {"key": k, "anchors": anchors, "veto": veto, "downgrades": downs}


def archetype(ev, *, P, V, T, pb_live: bool, div) -> Optional[str]:
    rel, side = ev["rel"], ev["side"]
    if rel["positioning"] == "with" and "extreme" in (P.get("kinds") or []) and pb_live and not P.get("equity"):
        base = "Crowd fade"
    elif rel["trend"] == "exhausted":
        base = "Divergence fade" if (div or {}).get("dir") == side else "Exhaustion fade"
    elif rel["trend"] == "with_pullback":
        base = "Pullback in trend"
    elif rel["trend"] in ("with", "with_late") and rel["valuation"] == "with":
        base = "Trend with value"
    elif rel["trend"] in ("with", "with_late"):
        base = "With trend"
    elif rel["trend"] == "against":
        base = "Counter-trend"
    else:
        base = "Range"
    return base


def assess(*, market: Dict[str, Any], engine: Dict[str, Any], pos: Optional[Dict[str, Any]],
           pb: Optional[Dict[str, Any]], div: Optional[Dict[str, Any]], val: Optional[Dict[str, Any]],
           seas_fwd: Optional[Dict[int, Dict[str, Any]]], macro: Optional[int], macro_marginal: bool,
           conf: str, weekly: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    eq = is_equity(market)
    P = positioning_read(pos, pb, div, eq)
    V = valuation_read(val)
    T = trend_profile(engine or {})
    S = seasonal_read(seas_fwd)
    M = {"sign": macro if macro is not None else None, "strength": 0.5 if macro_marginal else 1.0,
         "marginal": bool(macro_marginal)}
    pb_live = (pb or {}).get("status") in LIVE and bool((pb or {}).get("direction"))
    evs = {s: evaluate_side(s, P=P, V=V, T=T, M=M, S=S, div=div) for s in (1, -1)}
    best = max(evs.values(), key=lambda e: e["total"])
    other = evs[-best["side"]]
    if abs(best["total"] - other["total"]) < 0.25:
        # Genuinely split: grade both and keep the better-graded side.
        g1 = grade(evs[1], P=P, V=V, S=S, M=M, T=T, pb_live=pb_live, conf=conf)
        g2 = grade(evs[-1], P=P, V=V, S=S, M=M, T=T, pb_live=pb_live, conf=conf)
        best = evs[1] if ORDER.index(g1["key"]) <= ORDER.index(g2["key"]) else evs[-1]
    g = grade(best, P=P, V=V, S=S, M=M, T=T, pb_live=pb_live, conf=conf)
    side = best["side"] if g["key"] != "none" else 0
    lean = best["side"] if best["total"] >= 1.0 else 0
    arch = archetype(best, P=P, V=V, T=T, pb_live=pb_live, div=div) if side else None
    rel = best["rel"]
    for_list = [k for k in ("positioning", "valuation", "trend", "seasonal", "macro") if _is_for(rel[k])]
    against_list = [k for k in ("positioning", "valuation", "trend", "seasonal", "macro") if rel[k] == "against"]
    sizing = []
    if side:
        if rel["macro"] == "against":
            sizing.append("macro against: lower hit rate, size down")
        if eq and rel["positioning"] == "against":
            sizing.append("crowd on the same side: size warning only for an index")
        if rel["trend"] == "exhausted":
            sizing.append("fading a trend: starter size until it confirms")
        if rel["seasonal"] == "against":
            sizing.append("seasonality against: no adds")
    confirm = confirmation(side, arch, pb, weekly, val, T)
    return {"grade": dict(GRADE[g["key"]]), "side": side, "lean": lean, "archetype": arch,
            "total": best["total"], "other_side_total": evs[-best["side"]]["total"],
            "rel": rel, "pts": best["pts"], "anchors": g["anchors"], "downgrades": g["downgrades"],
            "for": for_list, "against": against_list, "sizing": sizing, "confirm": confirm,
            "reads": {"positioning": P, "valuation": V, "trend": T, "seasonal": S, "macro": M},
            "divergence": div if (div or {}).get("dir") else None}


def confirmation(side: int, arch: Optional[str], pb: Optional[Dict[str, Any]],
                 weekly: Optional[Dict[str, Any]], val: Optional[Dict[str, Any]],
                 T: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What turns the setup into a trade (Issue 10 step 5). States only what
    the data shows; the zone and the entry are Ben's."""
    if not side:
        return None
    pb = pb or {}
    if arch == "Crowd fade" and pb.get("status"):
        return {"state": pb.get("status"), "text": None, "kind": "playbook"}
    w = weekly or {}
    c, a = _num(w.get("close")), _num(w.get("sma10"))
    bp = (val or {}).get("daily_buy_point") or {}
    if arch in ("Exhaustion fade", "Divergence fade", "Counter-trend"):
        if c is not None and a is not None:
            done = (c < a) if side < 0 else (c > a)
            return {"state": "Confirmed" if done else "Waiting", "kind": "close_10w",
                    "text": (f"Weekly close {'below' if side < 0 else 'above'} the 10-week average "
                             + ("is in" if done else "not yet") + f" (close {c:g}, average {a:.4g}). "
                             "Until then it is a starter at your zone at most.")}
        return {"state": "Waiting", "kind": "close_10w",
                "text": "Wants a weekly close back through the 10-week average before adding."}
    if arch in ("Pullback in trend", "Trend with value", "With trend"):
        if bp:
            if bp.get("active"):
                return {"state": "Confirmed", "kind": "buy_point",
                        "text": f"Valuation buy point live on the daily (bar {bp.get('bars_held')} of 10)."}
            zz, dd = _num(bp.get("zb")), _num(bp.get("dxy"))
            if zz is not None and dd is not None and max(zz, dd) < -40 and side > 0:
                return {"state": "In zone", "kind": "buy_point",
                        "text": f"Both daily lines below -40 (ZB {zz:+.0f}, DXY {dd:+.0f}): inside the buy zone; "
                                "the Pine engine's own trade is not open (it holds 10 bars, then re-arms)."}
        if c is not None and a is not None:
            back = (c > a) if side > 0 else (c < a)
            return {"state": "Confirmed" if back else "Waiting", "kind": "close_10w",
                    "text": (f"Weekly close back {'above' if side > 0 else 'below'} the 10-week average "
                             + ("is in" if back else "not yet") + f" (close {c:g}, average {a:.4g}).")}
    return None
