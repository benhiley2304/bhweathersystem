"""Refine a higher-timeframe supply or demand zone with the bars one step down.

The method one timeframe down is the same three-step read: the lower-timeframe base is the last run of bars (up to
six) whose bodies sat inside the higher-timeframe zone before price left it for good within the leg-out window;
the bar after that run is the lower-timeframe leg out. The refined proximal edge is the body extreme of those bars
facing price and the refined distal edge their wick extreme away from price, both clipped to the higher-timeframe
zone. A refinement is reported only when it tightens the zone to 85% of its height or less;
otherwise the lower timeframe agrees with the daily edges and there is nothing to add.

Ben's rule: a level that is not visible on the daily (or weekly) is not traded. So this never produces zones of
its own; it only tightens a daily, two-day or weekly zone, which is where the entry and the stop go.

Weekly zones are refined with daily bars; daily and two-day zones with four-hour bars.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from bhws_core.prices import session_days

TIGHTEN_MAX = 0.85        # refined height / zone height; above this the lower timeframe adds nothing
BASE_MAX = 6              # lower-timeframe base bars, same cap as the method's base
BODY_MAX = 0.60           # a base bar's body may be at most this fraction of the zone height
PAD = 0.15                # body midpoints may sit this fraction of the height outside the zone band

# how many calendar days a higher-timeframe bar labelled on its first day spans, by timeframe
_SPAN_DAYS = {"W": 6, "2D": 1, "D": 0}


def _bar_days(df: pd.DataFrame, intraday: bool, cutoff: int) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index)
    return session_days(idx, cutoff) if intraday else idx.normalize()


def refine(zone: Dict[str, Any], lower: Optional[pd.DataFrame], lower_tf: str, *, cutoff: int = 18) -> Optional[Dict[str, Any]]:
    """Refined edges for `zone` (a zones_v2 Zone.to_dict()) from `lower` bars, or None when the bars do not cover it."""
    if lower is None or len(lower) < 20:
        return None
    tf = zone.get("tf", "D")
    kind = zone.get("kind")
    d = 1 if kind == "demand" else -1
    lo, hi = float(zone["lo"]), float(zone["hi"])
    height = hi - lo
    if not np.isfinite(height) or height <= 0:
        return None
    base_start = pd.Timestamp(zone["base_start"])
    legout_start = pd.Timestamp(zone["legout_start"])
    legout_end = pd.Timestamp(zone["legout_end"]) + pd.Timedelta(days=_SPAN_DAYS.get(tf, 0))
    intraday = lower_tf.upper().endswith("H")
    lower_name = {"D": "daily", "4H": "4h", "8H": "8h"}.get(lower_tf, lower_tf)
    days = _bar_days(lower, intraday, cutoff)
    if days[0] > base_start:
        return {"tf": lower_tf, "available": False, "note": f"{lower_name} bars start {days[0]:%d %b %Y}, after this base formed"}
    win = (days >= base_start) & (days <= legout_end)
    if win.sum() < 2:
        return {"tf": lower_tf, "available": False, "note": f"no {lower_name} bars inside this base"}
    w = lower.loc[win]
    wdays = days[win]
    O, H, L, C = (w[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    # the lower-timeframe base is the last run of bars whose bodies sat inside the level before price left it for
    # good within the leg-out window; the bar after it is the lower-timeframe leg out
    pad = PAD * height
    mids = (O + C) / 2.0
    inside = (mids >= lo - pad) & (mids <= hi + pad)
    idx_inside = np.where(inside)[0]
    if not len(idx_inside):
        return {"tf": lower_tf, "available": True, "tightens": False, "note": f"no {lower_name} bar body sits inside the level"}
    last = int(idx_inside[-1])
    if last == len(w) - 1:
        return {"tf": lower_tf, "available": True, "tightens": False, "note": f"price had not left the level on {lower_name} bars by the end of the leg-out bar"}
    j = last + 1
    base_idx = []
    for i in range(last, -1, -1):
        if not inside[i] or abs(C[i] - O[i]) > BODY_MAX * height:
            break
        base_idx.append(i)
        if len(base_idx) >= BASE_MAX:
            break
    if not base_idx:
        return {"tf": lower_tf, "available": True, "tightens": False, "note": f"the last {lower_name} bar inside the level is a single wide bar, not a base"}
    b = np.array(sorted(base_idx))
    bodies_hi = np.maximum(O[b], C[b])
    bodies_lo = np.minimum(O[b], C[b])
    if d > 0:
        prox, dist = float(bodies_hi.max()), float(L[b].min())
    else:
        prox, dist = float(bodies_lo.min()), float(H[b].max())
    prox = min(max(prox, lo), hi)
    dist = min(max(dist, lo), hi)
    r_lo, r_hi = (dist, prox) if d > 0 else (prox, dist)
    r_h = r_hi - r_lo
    tf_name = {"W": "weekly", "2D": "two-day", "D": "daily"}.get(tf, tf)
    fmt = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
    rng_ref = float(np.nanmedian((lower["High"] - lower["Low"]).to_numpy(float)[-240:]))
    out = {"tf": lower_tf, "available": True, "bars": int(len(b)), "lo": round(r_lo, 6), "hi": round(r_hi, 6),
           "proximal": round(prox, 6), "distal": round(dist, 6), "height": round(r_h, 6),
           "frac": round(r_h / height, 3) if height else None, "thin": False,
           "base_from": w.index[b[0]].strftime(fmt), "base_to": w.index[b[-1]].strftime(fmt), "legout_at": w.index[j].strftime(fmt)}
    if r_h <= 0.02 * height:
        out.update(tightens=False, note=f"the last {lower_name} bars before the leg out sat at the edge of the level rather than inside it; the {tf_name} edges stand")
    elif r_h / height > TIGHTEN_MAX:
        out.update(tightens=False, note=f"the {lower_name} base spans {int(round(100 * r_h / height))}% of the level; the {tf_name} edges stand")
    else:
        part = "upper" if (r_lo + r_hi) / 2 > (lo + hi) / 2 else "lower"
        thin = np.isfinite(rng_ref) and rng_ref > 0 and r_h < rng_ref
        out.update(tightens=True, thin=bool(thin),
                   note=f"{lower_name} base of {len(b)} bar{'s' if len(b) != 1 else ''} sits in the {part} part of the level: "
                        f"{int(round(100 * r_h / height))}% of its height" + (f"; thinner than a typical {lower_name} bar, so a stop just beyond it sits inside normal noise" if thin else ""))
    return out
