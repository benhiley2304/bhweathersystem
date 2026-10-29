"""Setup score: one transparent 0-10 number for the Desk's side, plus the
seasonal timing state ("sweet spot") and the relation to the engine bias.

Three layers, one read:
  Bias   - the engine composite (directional weather). Context, never inverted.
  Setup  - this score: is the Desk's side a favourable 2-8 week trade?
  Timing - where the seasonal path puts the entry: in window, approaching,
           wait for the window, against, or flat.

Points were set on the 2012-2018 point-in-time replay only (shrunk marginal
means of the 4- and 8-week vol-normalised return after per-market drift,
one point per 0.03 R, half-point steps). 2019-2026 was held out. See
research/score/REPORT.md. Macro and the composite are not point-in-time
replayable, so they carry no points here; they are shown as context.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

K = 80          # trading days of forward seasonal path used
MIN_YEARS = 8

# ── points table (2012-2018 design sample) ─────────────────────────────────
PTS_STRUCTURE = {
    ("Counter-trend", "Waiting"): 2.0, ("Counter-trend", "Confirmed"): 0.5,
    ("Crowd fade", "Not turned"): 1.0, ("Crowd fade", "One more week"): 0.0,
    ("Crowd fade", "Past test window"): -0.5, ("Crowd fade", "Turned"): -0.5,
    ("Divergence fade", "Waiting"): 0.0, ("Divergence fade", "Confirmed"): 0.5,
    ("Exhaustion fade", "Waiting"): 1.0, ("Exhaustion fade", "Confirmed"): 0.5,
    ("Pullback in trend", "Waiting"): 0.0, ("Pullback in trend", "Confirmed"): 1.5,
    ("Pullback in trend", "In zone"): 1.5,
    ("Trend with value", "Waiting"): 1.5, ("Trend with value", "Confirmed"): -2.0,
    ("With trend", "Waiting"): 0.0, ("With trend", "Confirmed"): -2.0,
    ("Range", None): -2.0,
}
PTS_TREND = {"against": 1.5, "exhausted": 0.0, "neutral": -1.0, "with": -1.5,
             "with_late": 0.0, "with_pullback": 0.0}
PTS_POSITIONING = {"with": 1.0, "against": 0.5, "neutral": -0.5, "na": -1.5}
PTS_VALUATION = {"with": 0.0, "against": 0.0, "neutral": -0.5}
# timing pools: in window / with / flat score the same (in-sample they did not
# separate); waiting for a window ahead and outright against both cost 1.5 now.
PTS_TIMING = {"ok": 1.5, "approaching": -0.5, "wait": -1.5, "against": -1.5}
POOL = {"in_window": "ok", "with": "ok", "flat": "ok", "approaching": "approaching",
        "wait": "wait", "against": "against"}

TIMING_LABEL = {
    "in_window": "In window", "with": "Seasonal with", "flat": "Seasonal flat",
    "approaching": "Window approaching", "wait": "Wait for window",
    "against": "Seasonal against", "na": "No seasonal path",
}

EVIDENCE = {
    "against": "Held in both halves: 4-8 wk R about -0.15 (2012-18) and -0.19 (2019-26). Treated as a veto.",
    "wait": "Entering now lost; waiting for the dip improved 4-8 wk R by +0.09 (2012-18, CI above 0) "
            "and +0.06 (2019-26, CI spans 0).",
    "approaching": "Mixed: delaying to the window helped 2012-18 and hurt 2019-26.",
    "ok": "No veto. In-window did not beat flat after drift in either half.",
}

STRUCT_EVIDENCE = {
    ("Counter-trend", "Waiting"): "Best-sorting combination in both halves (+0.06 / +0.14 R after drift).",
    ("Crowd fade", "Not turned"): "Positive in both halves; the crowd-turned version was negative in both.",
    ("Crowd fade", "Turned"): "Negative in both halves: the move is usually spent once managers turn.",
    ("Trend with value", "Confirmed"): "Negative in both halves (-0.11 R after drift).",
    ("With trend", "Confirmed"): "Mostly market drift; negative after drift in 2012-18.",
    ("Range", None): "Range rows lost in 2012-18 and gained in 2019-26. Unstable.",
    ("Pullback in trend", "Confirmed"): "Raw R positive in both halves, but mostly drift.",
}


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ── seasonal path ─────────────────────────────────────────────────────────
def seasonal_path(years: Dict[str, Sequence[Optional[float]]], weights: Dict[str, float],
                  current_td: int, current_year: int, k_max: int = K) -> Optional[Dict[str, Any]]:
    """Weighted-median forward path F(k) in % for k = 0..k_max trading days,
    from completed years only, crossing the year end. Same construction as
    the replay (research/score/seas_pit.py)."""
    a = max(1, int(current_td)) - 1
    rows: List[List[Optional[float]]] = []
    ws: List[float] = []
    for ys, w in (weights or {}).items():
        try:
            y = int(ys)
        except (TypeError, ValueError):
            continue
        if y >= current_year or not w or w <= 0:
            continue
        p = years.get(str(y))
        if not p or a >= len(p) or p[a] is None:
            continue
        ext = [None if v is None else 1 + v / 100.0 for v in p]
        q = years.get(str(y + 1))
        if y + 1 < current_year and q and ext[-1] is not None:
            ext = ext + [None if v is None else ext[-1] * (1 + v / 100.0) for v in q]
        base = ext[a]
        seg = ext[a:a + k_max + 1]
        r = [None if v is None else (v / base - 1) * 100 for v in seg]
        r += [None] * (k_max + 1 - len(r))
        rows.append(r)
        ws.append(float(w))
    if len(rows) < MIN_YEARS:
        return None
    F: List[Optional[float]] = []
    H: List[Optional[float]] = []
    for k in range(k_max + 1):
        pts = [(r[k], w) for r, w in zip(rows, ws) if r[k] is not None]
        if len(pts) < MIN_YEARS:
            F.append(None); H.append(None)
            continue
        pts.sort()
        tot = sum(w for _, w in pts); acc = 0.0; med = pts[-1][0]
        for v, w in pts:
            acc += w
            if acc >= 0.5 * tot:
                med = v; break
        F.append(round(med, 4))
        H.append(round(sum(w for v, w in pts if v > 0) / tot, 4))
    return {"F": F, "H": H, "n": len(rows), "td": int(current_td)}


def timing_state(path: Optional[Dict[str, Any]], side: int, sd_weekly: Optional[float]) -> Dict[str, Any]:
    """Classify the seasonal path for `side` into the replay-tested states.
    g(k) is the side-signed path in units of one weekly standard deviation."""
    if not path or not side or not sd_weekly or sd_weekly <= 0:
        return {"state": "na", "label": TIMING_LABEL["na"]}
    F = path.get("F") or []
    if len(F) < 66 or any(F[k] is None for k in range(66)):
        return {"state": "na", "label": TIMING_LABEL["na"]}
    g = [None if v is None else side * v / 100.0 / sd_weekly for v in F]
    e = min(range(26), key=lambda k: g[k])
    hi = [(g[k], k) for k in range(e + 10, min(e + 41, len(g))) if g[k] is not None]
    pk_v, pk_k = max(hi) if hi else (g[e], e)
    leg = pk_v - g[e]
    nowleg = max(v for v in g[5:41] if v is not None)
    g20, g40 = g[20], g[40]
    appr = e >= 4 and leg >= 1.5
    if g20 <= -0.4 and appr:
        st = "wait"
    elif g40 <= -1 or g20 <= -1:
        st = "against"
    elif e <= 3 and nowleg >= 1.0:
        st = "in_window"
    elif appr:
        st = "approaching"
    elif g20 >= 0.4:
        st = "with"
    else:
        st = "flat"
    # leg in % for display (price terms, side-signed)
    leg_pct = side * (F[pk_k] - F[e])
    if st in ("wait", "approaching"):
        entry_td, peak_td = e, pk_k
    elif st == "in_window":
        entry_td = 0
        peak_td = max(range(5, 41), key=lambda k: g[k])
        leg_pct = side * F[peak_td]
    else:
        entry_td, peak_td = None, None
    hold = None
    if peak_td is not None and entry_td is not None:
        wk = (peak_td - entry_td) / 5.0
        hold = [max(2, int(math.floor(wk * 0.75))), max(2, min(8, int(math.ceil(wk))))]
        if hold[0] > hold[1]:
            hold[0] = hold[1]
    return {"state": st, "pool": POOL[st], "label": TIMING_LABEL[st], "entry_td": entry_td,
            "peak_td": peak_td, "leg_sd": round(leg, 2), "leg_pct": round(leg_pct, 2),
            "near_sd": round(g20, 2), "dip_pct": round(side * F[e], 2), "hold_weeks": hold,
            "years": path.get("n"), "evidence": EVIDENCE[POOL[st]]}


def _bias(row: Dict[str, Any]) -> Dict[str, Any]:
    eng = row.get("engine") or {}
    d = (eng.get("direction") or "").lower()
    s = 1 if d.startswith("bull") else -1 if d.startswith("bear") else 0
    return {"dir": s, "label": eng.get("direction") or "Neutral", "conviction": eng.get("conviction")}


def _relation(bias: int, side: int, arch: Optional[str]) -> Dict[str, str]:
    bw = {1: "bullish", -1: "bearish", 0: "neutral"}[bias]
    sw = {1: "long", -1: "short", 0: "none"}[side]
    if not side:
        return {"key": "none", "text": f"Bias {bw}, no setup"}
    if bias == side:
        return {"key": "aligned", "text": f"Bias {bw}, setup {sw}: aligned"}
    if bias == 0:
        return {"key": "neutral", "text": f"Bias neutral, setup {sw}"}
    why = "counter-trend" if arch in ("Counter-trend", "Crowd fade", "Exhaustion fade", "Divergence fade") else "against the weather"
    return {"key": "counter", "text": f"Bias {bw}, setup {sw} ({why})"}


def _add_days(now: datetime, td: int) -> str:
    d = now; n = 0
    while n < td:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d.date().isoformat()


def compute(row: Dict[str, Any], path: Optional[Dict[str, Any]], sd_weekly: Optional[float],
            now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    side = int(row.get("side") or 0)
    arch = row.get("archetype")
    cf = row.get("confirm") or {}
    cstate = (cf.get("state") or cf.get("status") or cf.get("label")) if isinstance(cf, dict) else cf
    pil = row.get("pillars") or {}
    bias = _bias(row)
    rel = _relation(bias["dir"], side, arch)
    lean = int(row.get("lean") or 0)
    tside = side or lean
    tm = timing_state(path, tside, sd_weekly)
    if tm.get("entry_td") is not None:
        tm["entry_date"] = _add_days(now, tm["entry_td"]) if tm["entry_td"] else now.date().isoformat()
    if not side:
        return {"score": None, "side": 0, "ready": {"key": "none", "label": "No setup"},
                "timing": tm, "timing_side": tside, "bias": bias, "relation": rel, "parts": []}

    key = (arch, None if arch == "Range" else cstate)
    p_s = PTS_STRUCTURE.get(key, 0.0)
    trel = (pil.get("trend") or {}).get("rel")
    prel = (pil.get("positioning") or {}).get("rel")
    vrel = (pil.get("valuation") or {}).get("rel")
    p_t = PTS_TREND.get(trel, 0.0)
    p_p = PTS_POSITIONING.get(prel, 0.0)
    p_v = PTS_VALUATION.get(vrel, 0.0)
    pool = tm.get("pool")
    p_m = PTS_TIMING.get(pool, 0.0) if pool else 0.0

    def word(r):
        return {"with": "with", "against": "against", "neutral": "neutral", "na": "no COT",
                "exhausted": "exhausted", "with_late": "late", "with_pullback": "pullback"}.get(r, r or "n/a")
    parts = [
        {"key": "structure", "label": "Structure", "detail": f"{arch or 'none'} · {cstate or 'n/a'}",
         "pts": p_s, "evidence": STRUCT_EVIDENCE.get(key)},
        {"key": "trend", "label": "Trend location", "detail": word(trel), "pts": p_t,
         "evidence": "Fading the trend sorted better than joining it in both halves." if trel in ("against", "with") else None},
        {"key": "positioning", "label": "Positioning payoff", "detail": word(prel), "pts": p_p, "evidence": None},
        {"key": "valuation", "label": "Valuation", "detail": word(vrel), "pts": p_v,
         "evidence": "No measurable 2-8 week effect in the replay; shown as context." if vrel != "neutral" else None},
        {"key": "timing", "label": "Seasonal timing", "detail": tm["label"], "pts": p_m, "evidence": tm.get("evidence")},
    ]
    raw = 5.0 + sum(x["pts"] for x in parts)
    score = max(0.0, min(10.0, raw))
    caps = []
    conf = row.get("data_confidence")
    if conf in ("Low", "Insufficient") and score > 6:
        score = 6.0
        caps.append(f"data confidence {str(conf).lower()}: capped at 6")
    score = round(score * 2) / 2
    projected = None
    if pool in ("wait", "approaching") and tm.get("entry_td"):
        pj = max(0.0, min(10.0, raw - p_m + PTS_TIMING["ok"]))
        if conf in ("Low", "Insufficient"):
            pj = min(pj, 6.0)
        projected = {"score": round(pj * 2) / 2, "in_td": tm["entry_td"], "date": tm.get("entry_date"),
                     "note": "if structure, trend and positioning hold until the window"}
    events = [g for g in (row.get("gates") or []) if "this week" in str(g)]
    if pool == "against":
        ready = {"key": "veto", "label": "Stand aside: seasonal against"}
    elif score < 4:
        ready = {"key": "weak", "label": "Weak setup"}
    elif pool in ("wait", "approaching") and tm.get("entry_td"):
        ready = {"key": "wait", "label": f"Wait {tm['entry_td']} TD for window"}
    elif score >= 6:
        ready = {"key": "ready", "label": "Trade-ready" + (" · event this week" if events else "")}
    else:
        ready = {"key": "building", "label": "Building"}
    return {"score": score, "side": side, "parts": parts, "caps": caps, "projected": projected,
            "ready": ready, "timing": tm, "timing_side": tside, "bias": bias, "relation": rel,
            "event_flags": events,
            "basis": "Points set on the 2012-18 replay only; 2019-26 held out. Macro and the composite carry no points."}
