"""Crowd history: what a positioning state paid in this market and its complex.

Reads research/positioning/complex_evidence.json (built by research/positioning/complex.py). Everything is
history over 2012 to 2026 with overlapping forward windows; nothing is a probability for a live trade. The
loader is lazy and tolerant: if the file is missing every helper returns None and callers fall back to the
generic wording.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_PATH = Path(__file__).resolve().parents[1] / "research" / "positioning" / "complex_evidence.json"
_EV: Optional[Dict[str, Any]] = None
_LOADED = False


def load() -> Optional[Dict[str, Any]]:
    global _EV, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            _EV = json.loads(_PATH.read_text())
        except Exception:
            _EV = None
    return _EV


def complex_of(mid: str) -> Optional[str]:
    ev = load()
    return (ev or {}).get("market_complex", {}).get(mid) if ev else None


def complex_ids(key: str) -> List[str]:
    ev = load()
    return list(((ev or {}).get("complexes", {}).get(key) or {}).get("ids") or [])


def complex_label(key: Optional[str]) -> Optional[str]:
    ev = load()
    if not ev or not key:
        return None
    return ((ev.get("complexes") or {}).get(key) or {}).get("label")


def crowd_state(mgr: Optional[float], comm: Optional[float]) -> str:
    """Classify the current observation with the same thresholds as the research."""
    if mgr is None or comm is None:
        return "none"
    if mgr >= 90 and comm <= 10:
        return "crowded_long"
    if mgr <= 10 and comm >= 90:
        return "crowded_short"
    if mgr >= 90:
        return "managers_high"
    if mgr <= 10:
        return "managers_low"
    return "none"


def history(mid: str) -> Optional[Dict[str, Any]]:
    ev = load()
    return ((ev or {}).get("haircut") or {}).get(mid) if ev else None


def pooled(key: Optional[str]) -> Optional[Dict[str, Any]]:
    ev = load()
    return ((ev or {}).get("haircut_pooled") or {}).get(key) if (ev and key) else None


def relative(key: Optional[str]) -> Optional[Dict[str, Any]]:
    ev = load()
    return ((ev or {}).get("relative") or {}).get(key) if (ev and key) else None


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def _pos(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{round(100 * x):.0f}%"


def side_history(mid: str, side: int, horizon: int = 13) -> Optional[Dict[str, Any]]:
    """Forward return of a trade on `side` when the managers were crowded on that same side, against all weeks.

    For a long: managers >= 90th and commercials <= 10th. For a short: managers <= 10th and commercials >= 90th.
    Returns the two stat blocks plus counts, or None when the market has no COT history.
    """
    h = history(mid)
    if not h or side not in (1, -1):
        return None
    if side == 1:
        crowd, base = h.get(f"long_crowded_{horizon}"), h.get(f"all_{horizon}")
        n_thin = h.get("n_long_crowded_thin")
    else:
        crowd, base = h.get(f"short_crowded_{horizon}"), h.get(f"all_short_{horizon}")
        n_thin = h.get("n_short_crowded_thin")
    if not crowd or not base or not crowd.get("n"):
        return None
    return {"horizon": horizon, "crowd": crowd, "base": base, "n": crowd.get("n"), "n_thin": n_thin,
            "lift_pct": round((crowd.get("mean_pct") or 0) - (base.get("mean_pct") or 0), 2)}


def same_side_text(mid: str, side: int, horizon: int = 13) -> Optional[str]:
    """One sentence for the gate or fact list when the crowd is on the trade's side."""
    sh = side_history(mid, side, horizon)
    if not sh:
        return None
    word = "longs" if side == 1 else "shorts"
    c, b = sh["crowd"], sh["base"]
    return (f"crowd on the same side: {word} with managers at this extreme returned "
            f"{_pct(c.get('mean_pct'))} over {horizon} wk vs {_pct(b.get('mean_pct'))} all weeks "
            f"({_pos(c.get('pos'))} vs {_pos(b.get('pos'))} positive, {sh['n']} wk, ~{sh['n_thin']} independent). "
            f"{'Size down.' if (sh['lift_pct'] or 0) < 0 else 'No measured penalty; the crowd has tended to persist here.'}")


def fade_text(mid: str, side: int, horizon: int = 13) -> Optional[str]:
    """One sentence when the trade fades the crowd (managers at the extreme on the other side)."""
    sh = side_history(mid, -side, horizon)
    if not sh:
        return None
    c, b = sh["crowd"], sh["base"]
    # the opposite side's crowded return, sign flipped, is this trade's return in those weeks
    mean = None if c.get("mean_pct") is None else -c["mean_pct"]
    pos = None if c.get("pos") is None else 1 - c["pos"]
    bmean = None if b.get("mean_pct") is None else -b["mean_pct"]
    bpos = None if b.get("pos") is None else 1 - b["pos"]
    word = "longs" if side == 1 else "shorts"
    lift = (mean or 0) - (bmean or 0)
    if lift > 0.5:
        verdict = "a measured lift" if (mean or 0) > 0 else "better than all weeks but still not positive"
    elif lift >= -0.5:
        verdict = "no measured difference"
    else:
        verdict = "a measured drag"
    return (f"fading the crowd: {word} with managers at the opposite extreme returned "
            f"{_pct(mean)} over {horizon} wk vs {_pct(bmean)} all weeks ({_pos(pos)} vs {_pos(bpos)} positive, "
            f"{sh['n']} wk, ~{sh['n_thin']} independent): {verdict} on this crude test.")
