"""The house percentile: trailing 156 weekly reports, mid-rank.

This reproduces f_midrank() in the BHWS TradingView COT indicators exactly:

    pct = 100 * (less + 0.5 * (equal - 1)) / (n - 1)

where the window holds the current report plus up to 155 earlier ones, `less`
counts earlier values strictly below the current one and `equal` includes the
current value itself. At least 20 reports are required, as in Pine.

The legacy backend uses a Briese min-max index over 520 weeks (and 156 in
places). Min-max is driven by two points only and reads very differently near
the ends of the range, so the two are not interchangeable. Both are exposed
here with their method named, so nothing is relabelled quietly.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import math

HOUSE_WINDOW = 156
HOUSE_MIN_REPORTS = 20
EXTREME_HI = 90.0
EXTREME_LO = 10.0
HOUSE_METHOD = "midrank_156w"


def _clean(values: Iterable) -> List[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            out.append(f)
    return out


def midrank_percentile(values: Sequence, window: int = HOUSE_WINDOW,
                       min_reports: int = HOUSE_MIN_REPORTS) -> Optional[float]:
    """House percentile of the LAST value in `values` (oldest first)."""
    a = _clean(values)[-window:]
    n = len(a)
    if n < max(2, min_reports):
        return None
    v = a[-1]
    less = sum(1 for x in a if x < v)
    eq = sum(1 for x in a if x == v)
    return 100.0 * (less + 0.5 * (eq - 1)) / (n - 1)


def midrank_series(values: Sequence, window: int = HOUSE_WINDOW,
                   min_reports: int = HOUSE_MIN_REPORTS) -> List[Optional[float]]:
    """Rolling house percentile, point-in-time (each value uses only its past)."""
    a = _clean(values)
    return [midrank_percentile(a[: i + 1], window, min_reports) for i in range(len(a))]


def minmax_index(values: Sequence, window: int = 520) -> Optional[float]:
    """Legacy Briese-style index (what /api/cot-history serves at window=520)."""
    a = _clean(values)[-window:]
    if len(a) < 2:
        return None
    lo, hi = min(a), max(a)
    if hi == lo:
        return 50.0
    return 100.0 * (a[-1] - lo) / (hi - lo)


def extreme_label(pct: Optional[float], hi: float = EXTREME_HI, lo: float = EXTREME_LO) -> str:
    if pct is None:
        return "insufficient history"
    if pct >= hi:
        return "extreme high"
    if pct <= lo:
        return "extreme low"
    return "inside range"


def describe(values: Sequence, basis: str, window: int = HOUSE_WINDOW) -> dict:
    """Labelled house reading plus the legacy min-max for comparison."""
    a = _clean(values)
    p = midrank_percentile(a, window)
    return {
        "method": HOUSE_METHOD if window == HOUSE_WINDOW else f"midrank_{window}w",
        "basis": basis,
        "window_reports": min(len(a), window),
        "percentile": None if p is None else round(p, 1),
        "label": extreme_label(p),
        "net": None if not a else a[-1],
        "legacy_minmax_520": None if minmax_index(a, 520) is None else round(minmax_index(a, 520), 1),
    }
