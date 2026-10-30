"""Supply and demand zone detection on daily and weekly OHLC bars.

Rules are a mechanical reading of how Ben marks zones on his own charts
(BH Macro Sunday Setup issues 00a to 11, July to October 2026):

  leg in  -> base -> leg out

* The base is a cluster of consecutive bars that sit inside a band no taller
  than the scale's height cap (tight: 1.1 ATR weekly / 2 ATR daily; wide:
  3 ATR weekly / 5 ATR daily). It is built backwards from the leg-out bar
  and stops at a bar whose own range spans the band so far (that bar is a
  leg, not base). Both scales are kept when they differ: the tight base is
  the entry zone, the wide one the shelf around it.
* The leg out is a run of one to three same-direction bars that starts by
  closing beyond the base bodies and travels at least ``impulse_atr`` ATRs
  and at least ``impulse_base`` base heights from the proximal edge.
* The leg in is the run of same-direction bars immediately before the base.
  A falling leg in gives D (drop), a rising one R (rally).
* Demand zone (leg out up): distal = lowest low of the base, proximal =
  highest body (max of open and close) in the base.  Types DBR and RBR.
* Supply zone (leg out down): distal = highest high of the base, proximal =
  lowest body in the base.  Types RBD and DBD.
* A test is a visit into the zone (bar range touches the proximal edge) that
  does not close through the distal edge.  Consecutive bars inside count once.
* The zone is broken when a bar closes through the distal edge.

Nothing here is a trade signal. Entries, stops and which zone to trust remain
Ben's. The detector exists to put candidate levels on the chart and to let the
replay measure what happened at them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


DEFAULTS = {
    # scales: (label, max base height in ATRs, max base bars). A tight base and a wide
    # shelf can both be valid; Ben draws both ("supply within supply", "larger base beneath").
    "W": dict(atr_n=14, base_min=1, scales=(("tight", 1.10, 8), ("wide", 3.00, 16)),
              leg_range_frac=1.00, leg_min_atr=0.60,
              impulse_atr=1.00, impulse_base=0.90, impulse_bars=3, first_bar_body_atr=0.35),
    "D": dict(atr_n=14, base_min=2, scales=(("tight", 2.00, 15), ("wide", 5.00, 45)),
              leg_range_frac=1.00, leg_min_atr=0.60,
              impulse_atr=1.20, impulse_base=0.80, impulse_bars=3, first_bar_body_atr=0.50),
}


@dataclass
class Zone:
    tf: str                 # 'D' or 'W'
    scale: str              # 'tight' or 'wide'
    kind: str               # 'demand' or 'supply'
    pattern: str            # DBR, RBR, RBD, DBD (or ?BR / ?BD when the leg in is flat)
    proximal: float         # body edge facing price
    distal: float           # wick edge away from price
    proximal_wick: float    # wick extreme on the proximal side (Ben sometimes draws to this)
    base_start: pd.Timestamp
    base_end: pd.Timestamp
    legout_start: pd.Timestamp
    legout_end: pd.Timestamp
    base_bars: int
    height: float
    height_atr: float       # zone height in ATRs at formation
    departure: float        # leg-out travel beyond the proximal edge, price units
    departure_atr: float
    departure_base: float   # departure / zone height
    legin_atr: float        # signed leg-in travel in ATRs (negative = drop)
    atr: float
    tests: int = 0
    last_test: Optional[pd.Timestamp] = None
    broken: Optional[pd.Timestamp] = None
    flipped: bool = False
    score: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def lo(self) -> float:
        return min(self.proximal, self.distal)

    @property
    def hi(self) -> float:
        return max(self.proximal, self.distal)

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k in ("base_start", "base_end", "legout_start", "legout_end", "last_test", "broken"):
            v = d.get(k)
            d[k] = None if v is None else pd.Timestamp(v).strftime("%Y-%m-%d")
        d["lo"] = float(self.lo)
        d["hi"] = float(self.hi)
        return d


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(3, n // 2)).mean()


def _run(C: np.ndarray, start: int, direction: int, max_bars: int) -> int:
    """Length of the run of bars from ``start`` whose close-to-close change has ``direction``.

    Close-to-close rather than open-to-close: free daily feeds often print a bad open
    (open equal to close), which would hide the impulse.
    """
    n = 0
    i = start
    while 0 < i < len(C) and n < max_bars:
        if np.sign(C[i] - C[i - 1]) != direction:
            break
        n += 1
        i += 1
    return n


def detect(df: pd.DataFrame, tf: str = "D", params: Optional[dict] = None) -> List[Zone]:
    """Return zones found in ``df`` (columns Open High Low Close, DatetimeIndex)."""
    p = dict(DEFAULTS[tf])
    if params:
        p.update(params)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    a = atr(df, p["atr_n"])
    O, H, L, C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    A = a.to_numpy(float)
    idx = df.index
    n = len(df)
    zones: List[Zone] = []

    for j in range(p["atr_n"] + 2, n):
        atr_j = A[j - 1]
        if not np.isfinite(atr_j) or atr_j <= 0:
            continue
        chg = C[j] - C[j - 1]
        direction = 1 if chg > 0 else (-1 if chg < 0 else 0)
        if direction == 0:
            continue
        if abs(chg) < p["first_bar_body_atr"] * atr_j:
            continue
        # leg-out run
        m = _run(C, j, direction, p["impulse_bars"])
        if m == 0:
            continue
        legout_end = j + m - 1
        far_close = C[j:legout_end + 1].max() if direction > 0 else C[j:legout_end + 1].min()

        for scale, h_max, b_max in p["scales"]:
            z = _build(df, O, H, L, C, idx, j, m, legout_end, far_close, direction, atr_j, p, scale, h_max, b_max, tf)
            if z is not None:
                zones.append(z)

    zones = _dedupe(zones)
    for z in zones:
        z.score = _score(z)
    return zones


def _build(df, O, H, L, C, idx, j, m, legout_end, far_close, direction, atr_j, p, scale, h_max, b_max, tf):
    """Build the base backwards from bar j-1 at one scale; return a Zone or None.

    Every extension length is a candidate; the longest one that still passes the
    departure test is kept, so a wide shelf does not fail just because one more
    bar made it too tall for its leg out.
    """
    n = len(df)
    base = []
    hi, lo = -np.inf, np.inf
    k = j - 1
    best = None
    while k >= 0 and len(base) < b_max:
        rng_k = H[k] - L[k]
        if base:
            cur = hi - lo
            # a bar that on its own spans the whole band is a leg, not base
            # (only once the band is real: at least two bars and half an ATR tall)
            if (len(base) >= max(2, p["base_min"]) and cur >= 0.5 * atr_j
                    and rng_k >= p["leg_range_frac"] * cur and rng_k >= p["leg_min_atr"] * atr_j):
                break
        nhi, nlo = max(hi, H[k]), min(lo, L[k])
        if nhi - nlo > h_max * atr_j:
            break
        base.append(k)
        hi, lo = nhi, nlo
        k -= 1
        if len(base) < p["base_min"]:
            continue
        b0, b1 = base[-1], base[0]
        bodies_hi = np.maximum(O[b0:b1 + 1], C[b0:b1 + 1]).max()
        bodies_lo = np.minimum(O[b0:b1 + 1], C[b0:b1 + 1]).min()
        if direction > 0:
            proximal, distal, pwick = bodies_hi, lo, hi
            if C[j] <= proximal:
                continue
            departure = far_close - proximal
        else:
            proximal, distal, pwick = bodies_lo, hi, lo
            if C[j] >= proximal:
                continue
            departure = proximal - far_close
        height = abs(proximal - distal)
        if height <= 0:
            continue
        if departure < p["impulse_atr"] * atr_j or departure < p["impulse_base"] * height:
            continue
        best = (b0, b1, float(proximal), float(distal), float(pwick), float(height), float(departure), len(base))
    if best is None:
        return None
    b0, b1, proximal, distal, pwick, height, departure, nbase = best

    # leg in: the run of same-direction bars immediately before the base
    li_end = b0 - 1
    legin = 0.0
    if li_end >= 1:
        d_in = int(np.sign(C[li_end] - C[li_end - 1]))
        if d_in != 0:
            cnt = 0
            i = li_end
            while i >= 1 and cnt < 3:
                if np.sign(C[i] - C[i - 1]) != d_in:
                    break
                cnt += 1
                i -= 1
            legin = (C[li_end] - C[i]) / atr_j
    # Ben's taxonomy has four types, so the leg in is read as drop or rally by sign;
    # a weak leg in (under a quarter ATR) is kept in the notes.
    pat_in = "D" if legin < 0 else "R"
    pat = f"{pat_in}B{'R' if direction > 0 else 'D'}"
    notes = ["weak leg in"] if abs(legin) < 0.25 else []

    z = Zone(tf=tf, scale=scale, kind="demand" if direction > 0 else "supply", pattern=pat,
             proximal=float(proximal), distal=float(distal), proximal_wick=float(pwick),
             base_start=idx[b0], base_end=idx[b1], legout_start=idx[j], legout_end=idx[legout_end],
             base_bars=nbase, height=float(height), height_atr=float(height / atr_j),
             departure=float(departure), departure_atr=float(departure / atr_j),
             departure_base=float(departure / height), legin_atr=float(legin), atr=float(atr_j), notes=notes)
    inside_prev = False
    for t in range(legout_end + 1, n):
        if direction > 0:
            touched = L[t] <= proximal
            broke = C[t] < distal
        else:
            touched = H[t] >= proximal
            broke = C[t] > distal
        if broke:
            z.broken = idx[t]
            break
        if touched and not inside_prev:
            z.tests += 1
            z.last_test = idx[t]
        inside_prev = touched
    return z


def _overlap(a: Zone, b: Zone) -> float:
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    if hi <= lo:
        return 0.0
    return (hi - lo) / min(a.hi - a.lo, b.hi - b.lo)


def _iou(a: Zone, b: Zone) -> float:
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    if hi <= lo:
        return 0.0
    union = max(a.hi, b.hi) - min(a.lo, b.lo)
    return (hi - lo) / union if union > 0 else 0.0


def _dedupe(zones: List[Zone]) -> List[Zone]:
    """Same kind, overlapping price band, formed within a few bars: keep the stronger leg out."""
    zones = sorted(zones, key=lambda z: z.legout_start)
    out: List[Zone] = []
    for z in zones:
        merged = False
        for i, o in enumerate(out):
            if o.kind != z.kind:
                continue
            ov = _overlap(o, z)
            same_scale = o.scale == z.scale
            if ((same_scale and ov >= 0.5) or _iou(o, z) >= 0.85) and abs((z.legout_start - o.legout_start).days) <= (40 if z.tf == "W" else 12):
                if (z.departure_atr, -z.height) > (o.departure_atr, -o.height):
                    out[i] = z
                merged = True
                break
        if not merged:
            out.append(z)
    return out


def _score(z: Zone) -> float:
    """0-100 quality score. Ordinal only: it ranks zones, it is not a probability."""
    s = 0.0
    s += 35 * min(z.departure_base, 3.0) / 3.0           # strength of the leg out against the base
    s += 20 * min(z.departure_atr, 3.0) / 3.0            # and in volatility units
    s += 15 * (1.0 - min(z.height_atr, 2.0) / 2.0)       # tighter base scores higher
    s += 10 * (1.0 if z.pattern[0] != z.pattern[-1] else 0.5)  # reversal (DBR/RBD) over continuation
    fresh = {0: 20, 1: 12, 2: 6}.get(z.tests, 0)
    s += fresh
    if z.broken is not None:
        s *= 0.25
    return round(s, 1)


def mark_nesting(daily: List[Zone], weekly: List[Zone]) -> None:
    """Flag daily zones that sit inside a live weekly zone of the same kind."""
    for d in daily:
        for w in weekly:
            if w.kind == d.kind and w.broken is None and d.lo >= w.lo - 0.1 * (w.hi - w.lo) and d.hi <= w.hi + 0.1 * (w.hi - w.lo):
                d.notes.append(f"inside weekly {w.kind} {w.lo:g}-{w.hi:g}")
                break


def live(zones: List[Zone], price: float, atr_now: float, max_dist_atr: float = 6.0) -> List[Zone]:
    """Unbroken zones within ``max_dist_atr`` ATRs of price, nearest first."""
    out = []
    for z in zones:
        if z.broken is not None:
            continue
        if z.kind == "demand":
            dist = (price - z.proximal) / atr_now
        else:
            dist = (z.proximal - price) / atr_now
        if dist < -1.0:       # price has gone through the proximal edge and beyond
            continue
        if dist <= max_dist_atr:
            z.notes = [n for n in z.notes if not n.startswith("dist")] + [f"dist {dist:.2f} atr"]
            out.append((dist, z))
    out.sort(key=lambda t: abs(t[0]))
    return [z for _, z in out]


def reaction(df: pd.DataFrame, z: Zone, horizon: int, start_after: Optional[pd.Timestamp] = None) -> Optional[Dict]:
    """What price did at the first visit to the zone after it formed (or after ``start_after``).

    Returns None if the zone was never visited. Otherwise:
      touch        date of the first bar whose range reached the proximal edge
      held         True if, within ``horizon`` bars after the touch, the close travelled at least
                   one zone height away from the proximal edge in the zone's direction before any
                   close through the distal edge
      failed       True if a close through the distal edge came first
      mfe_h        best excursion in the zone's direction within the horizon, in zone heights
      mae_h        worst excursion through the zone within the horizon, in zone heights (negative)
    Point in time by construction: the zone exists before the visit.
    """
    H, L, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    idx = df.index
    start = idx.searchsorted(z.legout_end, side="right")
    if start_after is not None:
        start = max(start, idx.searchsorted(start_after, side="right"))
    sgn = 1 if z.kind == "demand" else -1
    h = z.hi - z.lo
    for t in range(start, len(df)):
        touched = (L[t] <= z.proximal) if sgn > 0 else (H[t] >= z.proximal)
        if not touched:
            continue
        held = failed = False
        mfe = mae = 0.0
        # Trade view: entry at the proximal edge, stop a quarter height beyond the distal edge
        # (Ben's convention), intrabar first passage of 1R/2R/3R against the stop.
        stop = z.distal - sgn * 0.25 * h
        risk = abs(z.proximal - stop)
        hit = {1: False, 2: False, 3: False}
        stopped = False
        resolved = False
        exit_r = None
        for u in range(t, min(len(df), t + horizon + 1)):
            ex = sgn * (C[u] - z.proximal) / h
            mfe = max(mfe, ex)
            mae = min(mae, ex)
            if not resolved:
                stop_hit = (L[u] <= stop) if sgn > 0 else (H[u] >= stop)
                best = (H[u] - z.proximal) / risk if sgn > 0 else (z.proximal - L[u]) / risk
                for k in (1, 2, 3):
                    if best >= k and not stop_hit:
                        hit[k] = True
                if stop_hit:
                    # same-bar ambiguity resolved against the trade
                    stopped = True
                    resolved = True
                    exit_r = -1.0
                elif hit[3]:
                    resolved = True
                    exit_r = 3.0
                if not resolved:
                    exit_r = sgn * (C[u] - z.proximal) / risk
            broke = (C[u] < z.distal) if sgn > 0 else (C[u] > z.distal)
            if broke and not failed and not held:
                failed = True
            if ex >= 1.0 and not held and not failed:
                held = True
            if (held or failed) and resolved:
                break
        return {"touch": idx[t], "held": held, "failed": failed, "mfe_h": round(mfe, 3), "mae_h": round(mae, 3),
                "hit_1r": hit[1], "hit_2r": hit[2], "hit_3r": hit[3], "stopped": stopped,
                "exit_r": None if exit_r is None else round(float(exit_r), 3)}
    return None
