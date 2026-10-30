"""BHWS crowd map API (additive): GET /api/complex.

/api/complex?market=NQ
    Where the crowd sits across the market's complex (US indices, G10 FX, metals, energy, grains, softs,
    livestock, Treasuries, crypto): each peer's latest house percentiles (managers, commercials, retail;
    futures-only, trailing 156 reports, mid-rank), its Desk class and side, plus the market's own crowd
    history from research/positioning/complex_evidence.json: what longs or shorts paid over 13 weeks when
    the managers were at the extreme on the same side, and when they were at the extreme on the other side,
    against all weeks in that market. The within-complex relative test is included with its pooled result
    first, because the pooled result is weak and single pairs can look strong by chance.

Everything is history with overlapping windows. The response carries its own definitions and caveats so the
panel can quote them; nothing here is a probability for a trade.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi.responses import JSONResponse

from bhws_core import crowd as CR

STATE_WORD = {
    "crowded_long": "crowded long",
    "crowded_short": "crowded short",
    "managers_high": "managers at 90 or above",
    "managers_low": "managers at 10 or below",
    "none": "no 90/10 extreme",
}


def _peer_row(r: Dict[str, Any]) -> Dict[str, Any]:
    pos = r.get("positioning") or {}
    ch = pos.get("changes") or {}
    cls = r.get("class") or {}
    mgr, comm, ret = pos.get("managers"), pos.get("commercials"), pos.get("retail")
    return {
        "id": r.get("id"), "name": r.get("name"),
        "managers": mgr, "commercials": comm, "retail": ret,
        "managers_chg_4": (ch.get("managers") or {}).get("chg_4"),
        "observed": pos.get("observed"), "released": pos.get("released"), "freshness": pos.get("freshness"),
        "state": CR.crowd_state(mgr, comm),
        "class": cls.get("key"), "class_label": cls.get("label"), "side": r.get("side"), "archetype": r.get("archetype"),
        "confirm": ((r.get("confirm") or {}).get("state") if isinstance(r.get("confirm"), dict) else r.get("confirm")),
    }


def _fmt(x: Optional[float], sign: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.1f}%" if sign else f"{x:.1f}%"


def _pos(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{round(100 * x):.0f}%"


def _ord(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _flip(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A short-side block seen from the long side (sign of the mean flips, positive rate complements)."""
    if not block or not block.get("n"):
        return block
    out = dict(block)
    if block.get("mean_pct") is not None:
        out["mean_pct"] = round(-block["mean_pct"], 2)
    if block.get("median_pct") is not None:
        out["median_pct"] = round(-block["median_pct"], 2)
    if block.get("pos") is not None:
        out["pos"] = round(1 - block["pos"], 3)
    return out


def build(mid: str, rows: List[Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ckey = CR.complex_of(mid)
    if not ckey:
        return {"market": mid, "complex": None, "peers": [], "note": "No CFTC series for this market; crowd history is read on the legs."}
    ids = CR.complex_ids(ckey)
    rmap = {r.get("id"): r for r in rows}
    peers = []
    for pid in ids:
        r = rmap.get(pid)
        if r:
            peers.append(_peer_row(r))
        else:
            m = by_id.get(pid) or {}
            peers.append({"id": pid, "name": m.get("name"), "managers": None, "commercials": None, "retail": None, "state": "none"})
    peers.sort(key=lambda p: (-(p.get("managers") if p.get("managers") is not None else -1)))
    me = next((p for p in peers if p["id"] == mid), None)
    h = CR.history(mid) or {}
    hist = None
    if h:
        hist = {
            "n_weeks": h.get("n_weeks"), "from": h.get("from"), "to": h.get("to"),
            "all_long": h.get("all_13"), "crowded_long": h.get("long_crowded_13"), "n_long_thin": h.get("n_long_crowded_thin"),
            # a long taken against a crowded short = minus the short's return in those weeks
            "fade_short_as_long": _flip(h.get("short_crowded_13")), "n_short_thin": h.get("n_short_crowded_thin"),
            "all_short": h.get("all_short_13"), "crowded_short": h.get("short_crowded_13"),
            "fade_long_as_short": _flip(h.get("long_crowded_13")),
        }
    pooled = CR.pooled(ckey) or {}
    rel = CR.relative(ckey) or {}
    pairs = {}
    for k, v in (rel.get("pairs") or {}).items():
        a, b = k.split(">")
        if mid in (a, b):
            pairs[k] = {"n": v.get("n"), "n_thin": v.get("n_thin"), "base_13": v.get("base_13"), "cond_13": v.get("cond_13"),
                        "base_8": v.get("base_8"), "cond_8": v.get("cond_8")}
    # ranking words
    ranked = [p for p in peers if p.get("managers") is not None]
    most = ranked[0] if ranked else None
    least = ranked[-1] if ranked else None
    read: List[str] = []
    if me and me.get("managers") is not None:
        read.append(f"{mid}: managers {_ord(me['managers'])}, commercials {_ord(me['commercials'])}, retail "
                    f"{_ord(me['retail'])} percentile ({STATE_WORD.get(me['state'], me['state'])}), observed {me.get('observed')}.")
    if most and least and most["id"] != least["id"]:
        read.append(f"Most crowded long in the complex: {most['id']} (managers {_ord(most['managers'])}). Least: "
                    f"{least['id']} (managers {_ord(least['managers'])}).")
    if hist and hist.get("crowded_long") and hist["crowded_long"].get("n"):
        cl, al = hist["crowded_long"], hist["all_long"]
        read.append(f"Crowd history here, longs: with the managers crowded long, {_fmt(cl.get('mean_pct'))} over 13 weeks "
                    f"against {_fmt(al.get('mean_pct'))} for all weeks ({_pos(cl.get('pos'))} vs {_pos(al.get('pos'))} positive, "
                    f"{cl.get('n')} weeks, about {hist.get('n_long_thin')} independent).")
    if hist and hist.get("fade_short_as_long") and hist["fade_short_as_long"].get("n"):
        fs, al = hist["fade_short_as_long"], hist["all_long"]
        read.append(f"Longs against a crowded short: {_fmt(fs.get('mean_pct'))} against {_fmt(al.get('mean_pct'))} "
                    f"({_pos(fs.get('pos'))} vs {_pos(al.get('pos'))} positive, {fs.get('n')} weeks).")
    c13, b13 = rel.get("cond_13") or {}, rel.get("base_13") or {}
    if c13.get("n"):
        m13 = c13.get("mean_pct") or 0.0
        if m13 > 0.5:
            verdict = f"the less crowded market beat the more crowded by {_fmt(m13)} over 13 weeks"
        elif m13 < -0.5:
            verdict = f"the more crowded market kept winning, by {_fmt(-m13)} over 13 weeks"
        else:
            verdict = f"no difference over 13 weeks ({_fmt(m13)})"
        read.append(f"Relative test, all pairs in this complex: when one market's managers sat 50 or more points above "
                    f"another's, {verdict} (less crowded ahead in {_pos(c13.get('pos'))} of {c13.get('n')} overlapping weeks, "
                    f"about {rel.get('n_thin')} independent). Single pairs swing both ways; the map is location, not a signal.")
    return {
        "market": mid, "complex": {"key": ckey, "label": CR.complex_label(ckey), "ids": ids},
        "peers": peers, "me": me, "history": hist, "pooled": pooled or None,
        "relative": {"pooled": {"cond_13": rel.get("cond_13"), "base_13": rel.get("base_13"), "cond_8": rel.get("cond_8"),
                                "n_thin": rel.get("n_thin"), "gap_points": rel.get("gap_points")}, "pairs": pairs},
        "read": read,
        "definitions": {
            "percentile": "CFTC legacy futures-only, trailing 156 reports, mid-rank; the same number as the BHWS TradingView COT indicators.",
            "crowded_long": "managers at or above the 90th percentile with commercials at or below the 10th.",
            "crowded_short": "managers at or below the 10th with commercials at or above the 90th.",
            "history": "Forward 13-week return from the Friday after the observation, 2012 to 2026, overlapping weeks, no costs. "
                       "'About n independent' keeps every eighth qualifying week. History for this market, not a probability for a trade.",
            "relative": "Within the complex, weeks where A's managers percentile exceeded B's by 50 points or more; B's forward "
                        "return minus A's against the pair's unconditional spread.",
        },
    }


def register(app, *, markets, desk_state_getter: Callable[[], Optional[Dict[str, Any]]]) -> None:
    by_id = {m["id"]: m for m in markets}

    @app.get("/api/complex")
    def complex_view(market: str):
        mid = (market or "").upper().strip()
        if mid not in by_id:
            return JSONResponse({"error": f"unknown market {mid}"}, status_code=404)
        st = desk_state_getter() or {}
        data = st.get("data") or {}
        rows = data.get("rows") or []
        if not rows:
            return JSONResponse({"status": "building", "market": mid}, status_code=202, headers={"Retry-After": "10"})
        out = build(mid, rows, by_id)
        out["desk_built"] = st.get("built")
        out["evidence_built"] = (CR.load() or {}).get("built")
        return JSONResponse(out, headers={"Cache-Control": "public, max-age=600"})
