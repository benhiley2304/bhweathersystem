"""Supply and demand zones, second detector: the three-step method as taught by Bernd Skorupinski,
calibrated to the zones Ben draws on his own TradingView charts.

Method (from the Supply and Demand Pro Trader Series, lessons 3, 6, the zone-qualifier lessons and the
level-on-top-of-level lesson):

  step 1  find the LEG OUT: an explosive candle. "The body has to be bigger than 70% of the entire range"
          and, more important, "the candle has to be abnormally bigger than the previous few candles";
          alternatively "abnormally bigger candles that are decisive if they're followed by another decisive
          candle". We measure size close to close as well as by range because several free feeds print bad
          opens.
  step 2  find the LEG IN: the decisive move into the base. Its sign names the shape: drop-base-rally and
          rally-base-rally are demand, rally-base-drop and drop-base-drop are supply.
  step 3  find the BASE: "small bodies build the base, a big body is the exit". One or more indecisive candles
          between leg in and leg out, judged against the surrounding candles. A short base is best ("two,
          three, doesn't matter"); beyond six candles the zone is disregarded in his later material, so a base
          of seven to ten bars is kept only as a wide zone and marked.

  lines   demand: proximal across the top of the base bodies, distal under the base wicks.
          supply: proximal across the bottom of the base bodies, distal over the base wicks.
          The preferred (tight) version uses the last one or two base candles; the wider version the whole
          base. Ben draws the wider version body-to-wick, so that is the zone here, and the preferred proximal
          edge is carried for the freshness rule.

  five zone qualifiers (scored, never a probability):
          departure   how far and how fast price left: the leg-out run in ATR and the whole swing in zone heights.
          base        how long price paused: fewer candles is better.
          freshness   measured to the right only. Zero touches best. On weekly and monthly charts a test of about
                      25% into the preferred version still counts as fresh; on lower timeframes it does not.
          profit margin  room to the next opposing zone, reported in multiples of the zone risk.
          arrival     not modelled (it is read live on the chart).
          originality a base that formed inside an older zone of the same kind is a reaction, not an original
                      move; marked and scored down, not removed.

  level on top of level: two zones of one kind whose gap is no taller than either zone are one structure.
  location on the curve: where price sits between the nearest weekly demand below and weekly supply above,
          split at 0.33 and 0.66 into low, equilibrium and high (very low inside the demand, very high inside the
          supply). Direction on the weekly from the last three swing highs and lows: two higher lows is up, two
          lower highs is down, otherwise sideways. The action matrix (demand only when low, or at equilibrium
          with an uptrend; supply only when high, or at equilibrium with a downtrend; never demand when high,
          never supply when low) is reported as the method's rule, not as evidence.

Nothing here is a trade signal. Zones are candidates for Ben's chart; entries, stops and targets are his.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULTS = {
    # Sizes are in units of the typical range (rolling median true range over `typ_n` bars; a median so that one
    # crash bar does not hide the next base). explosive: range at least `expl_range` times the median range of the
    # previous `ref_n` bars and a close-to-close move of at least `expl_move`; decisive2: a `dec_move` bar followed by
    # another in the same direction; run_strong: the run out of the base travels `run_strong` within `run_max` bars
    # (Ben's zones include bases left by a sustained move rather than one explosive candle).
    "W": dict(typ_n=20, ref_n=5, expl_range=1.35, expl_move=0.80, dec_move=0.55, run_max=4, run_min=1.40,
              run_strong=2.0, run_counter=0.35, slow_bars=6, slow_travel=2.5, swing_bars=26, swing_travel=3.5, base_max=12, base_pref=6,
              base_body=1.7, base_height=3.0, base_small=1.0, base_pad=0.35, legin_bars=4, min_height=0.12, hard_pen=0.25),
    "D": dict(typ_n=20, ref_n=6, expl_range=1.35, expl_move=0.85, dec_move=0.60, run_max=6, run_min=1.60,
              run_strong=2.2, run_counter=0.35, slow_bars=10, slow_travel=2.5, swing_bars=40, swing_travel=3.5, base_max=12, base_pref=6,
              base_body=1.7, base_height=3.0, base_small=1.0, base_pad=0.35, legin_bars=4, min_height=0.12, hard_pen=0.0),
}
DEFAULTS["2D"] = dict(DEFAULTS["D"], run_max=5, slow_bars=8, swing_bars=20)


@dataclass
class Zone:
    tf: str
    kind: str                 # demand | supply
    pattern: str              # DBR RBR RBD DBD (first letter '?' when the leg in is weak)
    proximal: float           # body edge facing price
    distal: float             # wick edge away from price
    pref_proximal: float      # body edge of the last one or two base candles (preferred version)
    base_start: pd.Timestamp
    base_end: pd.Timestamp
    legout_start: pd.Timestamp
    legout_end: pd.Timestamp
    base_bars: int
    height: float
    height_atr: float
    atr: float
    run_atr: float            # close-to-close travel of the leg-out run, in ATR
    run_bars: int
    swing_h: float            # furthest excursion before the first return, in zone heights (capped at 30)
    body_ratio: Optional[float]  # body / range of the first leg-out candle, None when the open is unusable
    explosive: bool           # met the range-and-size test (else the two-decisive-candles test)
    base_small: float         # body band of the base / travel of the leg-out run (smaller is tighter)
    legin_atr: float          # signed leg-in travel in ATR (negative = drop)
    tests: int = 0            # completed prior visits that reached the proximal edge
    visiting_now: bool = False  # price is at or inside the zone on the latest bars (the arrival in progress)
    wide_touches: int = 0     # prior visits that reached the wider band (proximal edge) at all
    deep_touches: int = 0     # bars that reached past the preferred proximal edge
    pen_max: float = 0.0      # deepest penetration as a fraction of the zone height (0 untouched, 1 reached distal)
    last_test: Optional[pd.Timestamp] = None
    broken: Optional[pd.Timestamp] = None
    fresh_state: str = "fresh"   # fresh | shallow | tested | broken
    original: bool = True
    origin_note: str = ""
    origin_ref: Optional[str] = None                    # base_start of the older zone a reaction formed at
    score: float = 0.0
    notes: List[str] = field(default_factory=list)
    parts: List[Dict] = field(default_factory=list)   # component zones when stacked levels are shown as one structure
    inside_older: Optional[str] = None                  # an older zone of the same kind this one formed inside (shown instead)

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
        d["lo"], d["hi"] = float(self.lo), float(self.hi)
        return d


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(3, n // 2)).mean()


def typical_range(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Rolling median true range: the size yardstick for 'abnormally bigger than the previous candles'."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(5, n // 3)).median()


def looks_like_spot_fx_defect(df: pd.DataFrame) -> bool:
    """Yahoo spot FX daily bars print the open equal to the close and lag the close by one bar."""
    if len(df) < 40:
        return False
    body = (df["Close"] - df["Open"]).abs().median()
    rng = (df["High"] - df["Low"]).median()
    return bool(rng > 0 and body / rng < 0.08)


def repair_spot_fx(df: pd.DataFrame) -> pd.DataFrame:
    """Repair the Yahoo spot FX daily defect: the printed close is the bar's open (the next bar's print is the close).

    open := printed close, close := next printed close (last bar keeps its print), high/low widened to hold both.
    """
    out = df.copy()
    o = out["Close"].to_numpy(float)
    c = np.roll(o, -1)
    c[-1] = o[-1]
    out["Open"], out["Close"] = o, c
    out["High"] = np.maximum(out["High"].to_numpy(float), np.maximum(o, c))
    out["Low"] = np.minimum(out["Low"].to_numpy(float), np.minimum(o, c))
    return out


def resample_2d(df: pd.DataFrame) -> pd.DataFrame:
    """Two-day bars from consecutive trading days, paired from a fixed anchor: the first bar on or after 1 January
    of the earliest full calendar year in the series. Pairing therefore does not shift from one day to the next
    (a right-anchored pairing would re-pair every bar each session). Bars before the anchor pair backwards from it.
    Each bar is indexed by its last day."""
    n = len(df)
    if n < 4:
        return df.copy()
    first_year = df.index[0].year + (0 if (df.index[0].month == 1 and df.index[0].day <= 7) else 1)
    anchor = int(df.index.searchsorted(pd.Timestamp(year=first_year, month=1, day=1)))
    if anchor >= n - 2:
        anchor = 0
    pos = np.arange(n) - anchor
    grp = np.where(pos >= 0, pos // 2, -((-pos + 1) // 2))
    g = df.groupby(grp)
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(), "Close": g["Close"].last(),
    })
    if "Volume" in df.columns:
        out["Volume"] = g["Volume"].sum()
    last_idx = df.index.to_series().groupby(grp).last()
    out.index = pd.DatetimeIndex(last_idx.values)
    out = out.sort_index()
    return out


def _run(C: np.ndarray, start: int, direction: int, max_bars: int, counter_max: float = 0.0) -> int:
    """Bars in the leg-out run: same-direction closes, allowing one counter close no bigger than ``counter_max``.
    The run ends on the bar with the furthest close."""
    n, i, used = 0, start, False
    best_n, best = 0, C[start - 1] if start > 0 else C[start]
    while 0 < i < len(C) and n < max_bars:
        mv = (C[i] - C[i - 1]) * direction
        if mv < 0:
            if used or -mv > counter_max:
                break
            used = True
        n += 1
        if (C[i] - best) * direction > 0:
            best, best_n = C[i], n
        i += 1
    return best_n


def _body_ratio(O, H, L, C, i) -> Optional[float]:
    r = H[i] - L[i]
    if r <= 0 or not (L[i] <= O[i] <= H[i]) or O[i] == C[i]:
        return None
    return abs(C[i] - O[i]) / r


def detect(df: pd.DataFrame, tf: str = "D", params: Optional[dict] = None) -> List[Zone]:
    p = dict(DEFAULTS[tf])
    if params:
        p.update(params)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    O, H, L, C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    T = typical_range(df, p["typ_n"]).to_numpy(float)
    R = H - L
    idx = df.index
    n = len(df)
    zones: List[Zone] = []
    for j in range(8, n):
        a = T[j - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        mv = C[j] - C[j - 1]
        d = 1 if mv > 0 else (-1 if mv < 0 else 0)
        if d == 0:
            continue
        ref = float(np.median(R[max(0, j - p["ref_n"]):j]))
        ref = ref if ref > 0 else a
        explosive = R[j] >= p["expl_range"] * ref and abs(mv) >= p["expl_move"] * a
        decisive2 = (abs(mv) >= p["dec_move"] * a and R[j] >= 1.1 * ref and j + 1 < n
                     and np.sign(C[j + 1] - C[j]) == d and abs(C[j + 1] - C[j]) >= p["dec_move"] * a)
        m = _run(C, j, d, p["run_max"], p["run_counter"] * a)
        if m == 0:
            continue
        le = j + m - 1
        run = abs(C[le] - C[j - 1]) / a
        run_strong = run >= p["run_strong"] and abs(mv) >= 0.4 * a
        # a slower but sustained departure: within slow_bars the close travels slow_travel without coming back
        w_end = min(n, j + p["slow_bars"])
        ex_slow = float(((C[j:w_end] - C[j - 1]) * d).max() / a)
        held = float(((C[j:w_end] - C[j - 1]) * d).min() / a) >= -0.3
        run_slow = ex_slow >= p["slow_travel"] and held and abs(mv) >= 0.3 * a
        # the origin of a big move: before any close comes back through the launch close, price travels swing_travel
        s_end = min(n, j + p["swing_bars"])
        rel = (C[j:s_end] - C[j - 1]) * d
        back = np.where(rel < -0.3 * a)[0]
        rel = rel[:back[0]] if len(back) else rel
        run_swing = len(rel) >= 3 and float(rel.max() / a) >= p["swing_travel"] and abs(mv) >= 0.3 * a and rel[:3].max() > 0
        if not (explosive or decisive2 or run_strong or run_slow or run_swing):
            continue
        if run < p["run_min"] and not (run_slow or run_swing):
            continue
        if (run_slow or run_swing) and not (explosive or decisive2 or run_strong):
            # extend the run to the bar with the furthest close inside the slow window
            le = j + int(np.argmax((C[j:w_end] - C[j - 1]) * d))
            run = abs(C[le] - C[j - 1]) / a
        r_out = float(R[j:le + 1].max())
        # step 3: the base, backwards from the bar before the leg out. Bars join the base while the band of their
        # bodies stays narrow (a consolidation) and the wick band stays within reach; a decisive bar that widens the
        # body band is the leg in and ends the base.
        base: List[int] = []
        bhi, blo, whi, wlo = -np.inf, np.inf, -np.inf, np.inf
        pad = p["base_pad"] * a
        k = j - 1
        while k >= 1 and len(base) < p["base_max"]:
            bh = min(max(O[k], C[k]), H[k])
            bl = max(min(O[k], C[k]), L[k])
            if base:
                # the bar's body must sit mostly inside the body band so far (padded): bodies overlapping is what
                # makes a consolidation; a bar whose body lies beyond it belongs to the leg in
                lo_e, hi_e = blo - pad, bhi + pad
                body = max(bh - bl, 1e-12)
                ov = max(0.0, min(bh, hi_e) - max(bl, lo_e)) / body
                if ov < 0.5 and not (bh - bl < 0.1 * a and lo_e <= (bh + bl) / 2 <= hi_e):
                    break
            nbhi, nblo, nwhi, nwlo = max(bhi, bh), min(blo, bl), max(whi, H[k]), min(wlo, L[k])
            if nbhi - nblo > p["base_body"] * a or nwhi - nwlo > p["base_height"] * a:
                break
            mk = abs(C[k] - C[k - 1])
            if not base and R[k] >= r_out:
                break                      # the bar before the leg out is as big as the leg out: no base here
            if base and mk >= 1.0 * a and (nbhi - nblo) - (bhi - blo) > 0.4 * a:
                break                      # a decisive candle that widens the band: that is the leg in
            base.append(k)
            bhi, blo, whi, wlo = nbhi, nblo, nwhi, nwlo
            k -= 1
        if not base:
            continue
        b1, b0 = base[0], base[-1]          # b1 is the last base bar (next to the leg out)
        base_small = float((bhi - blo) / (run * a)) if run > 0 else 9.9
        if base_small > p["base_small"]:
            continue                       # the base is as tall as the move that left it: not a base
        bodies_hi = np.minimum(np.maximum(O[b0:b1 + 1], C[b0:b1 + 1]), H[b0:b1 + 1])
        bodies_lo = np.maximum(np.minimum(O[b0:b1 + 1], C[b0:b1 + 1]), L[b0:b1 + 1])
        pref_i = max(b0, b1 - 1) - b0       # preferred version: the last one or two base candles
        if d > 0:
            proximal, distal = float(bodies_hi.max()), float(wlo)
            pref = float(bodies_hi[pref_i:].max())
            if C[j] <= proximal:
                continue
        else:
            proximal, distal = float(bodies_lo.min()), float(whi)
            pref = float(bodies_lo[pref_i:].min())
            if C[j] >= proximal:
                continue
        height = abs(proximal - distal)
        if height < p["min_height"] * a:
            continue
        # step 2: the leg in
        legin = 0.0
        li = b0 - 1
        if li >= 1:
            din = int(np.sign(C[li] - C[li - 1]))
            if din != 0:
                cnt, i = 0, li
                while i >= 1 and cnt < p["legin_bars"] and np.sign(C[i] - C[i - 1]) == din:
                    cnt += 1
                    i -= 1
                legin = (C[li] - C[i]) / a
        pat_in = "?" if abs(legin) < 0.35 else ("D" if legin < 0 else "R")
        pat = f"{pat_in}B{'R' if d > 0 else 'D'}"
        z = Zone(tf=tf, kind="demand" if d > 0 else "supply", pattern=pat, proximal=proximal, distal=distal,
                 pref_proximal=pref, base_start=idx[b0], base_end=idx[b1], legout_start=idx[j], legout_end=idx[le],
                 base_bars=len(base), height=float(height), height_atr=float(height / a), atr=float(a),
                 run_atr=float(run), run_bars=m, swing_h=0.0, body_ratio=_body_ratio(O, H, L, C, j),
                 explosive=bool(explosive), base_small=base_small, legin_atr=float(legin))
        if len(base) > p["base_pref"]:
            z.notes.append(f"wide base: {len(base)} bars")
        if not explosive:
            z.notes.append("leg out by a sustained move, not one explosive candle" if (run_strong or decisive2) else "slow departure: the move left over several bars")
        if pat_in == "?":
            z.notes.append("weak leg in")
        _walk_forward(z, H, L, C, idx, le, d, p)
        zones.append(z)
    zones = _dedupe(zones, tf)
    _mark_originality(zones)
    for z in zones:
        z.score = score(z)
    return zones


def _walk_forward(z: Zone, H, L, C, idx, le: int, d: int, p: dict) -> None:
    """Freshness to the right: completed visits, deepest penetration, break; the swing excursion before the first
    return; and whether price is visiting the zone now (the visit in progress is not counted as a test)."""
    n = len(C)
    h = z.height
    inside_prev = False
    swing = 0.0
    first_touch = None
    visits: List[list] = []               # [start, end, reached the preferred version] per visit
    for t in range(le + 1, n):
        if d > 0:
            pen = (z.proximal - L[t]) / h
            deep = L[t] < z.pref_proximal
            ex = (H[t] - z.proximal) / h
            broke = C[t] < z.distal
            left = (L[t] - z.proximal) / h >= 0.5
        else:
            pen = (H[t] - z.proximal) / h
            deep = H[t] > z.pref_proximal
            ex = (z.proximal - L[t]) / h
            broke = C[t] > z.distal
            left = (z.proximal - H[t]) / h >= 0.5
        if first_touch is None:
            swing = max(swing, ex)
        touched = pen > 0
        if touched:
            if first_touch is None:
                first_touch = t
            z.pen_max = max(z.pen_max, float(pen))
            if deep:
                z.deep_touches += 1
            if not inside_prev:
                visits.append([t, t, deep])
            else:
                visits[-1][1] = t
                visits[-1][2] = visits[-1][2] or deep
        if broke:
            z.broken = idx[t]
            z.pen_max = max(z.pen_max, 1.0)
            break
        # a visit lasts until price has left the proximal edge by half a zone height (hysteresis), so bars that
        # hover just outside the zone extend the visit instead of opening a new one
        if touched:
            inside_prev = True
        elif left:
            inside_prev = False
        elif inside_prev:
            visits[-1][1] = t
    far = (C[le] - z.proximal) / h if d > 0 else (z.proximal - C[le]) / h
    z.swing_h = float(min(30.0, max(swing, far)))
    # the visit in progress (touching within the last two bars, unbroken) is the arrival, not a prior test
    if visits and z.broken is None and visits[-1][1] >= n - 2:
        z.visiting_now = True
        prior = visits[:-1]
    else:
        prior = visits
    z.wide_touches = len(prior)
    prior = [v for v in prior if v[2]]    # a test reaches the preferred (tight) version; a wick into the wider band is noted only
    z.tests = len(prior)
    if prior:
        z.last_test = idx[prior[-1][1]]
    if z.broken is not None:
        z.fresh_state = "broken"
    elif not prior:
        z.fresh_state = "fresh"
        if z.wide_touches:
            z.notes.append(f"wider band touched {z.wide_touches}x without reaching the preferred version")
    else:
        # depth of the prior tests measured against the preferred version
        pref_h = abs(z.pref_proximal - z.distal)
        end = prior[-1][1] + 1
        if d > 0:
            deep = max(0.0, (z.pref_proximal - float(np.min(L[le + 1:end]))) / pref_h) if pref_h > 0 else 1.0
        else:
            deep = max(0.0, (float(np.max(H[le + 1:end])) - z.pref_proximal) / pref_h) if pref_h > 0 else 1.0
        z.fresh_state = "shallow" if (p["hard_pen"] > 0 and deep <= p["hard_pen"]) else "tested"
        z.notes.append(f"deepest prior test {int(round(100 * min(z.pen_max, 1.0)))}% into the zone")


def _iou(a: Zone, b: Zone) -> float:
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    if hi <= lo:
        return 0.0
    union = max(a.hi, b.hi) - min(a.lo, b.lo)
    return (hi - lo) / union if union > 0 else 0.0


def _dedupe(zones: List[Zone], tf: str) -> List[Zone]:
    """One zone per base: same kind, same base end or heavily overlapping band within a few bars.

    Keeps the zone with the stronger swing, then the stronger run.
    """
    zones = sorted(zones, key=lambda z: (z.legout_start, -z.run_atr))
    out: List[Zone] = []
    near_days = 45 if tf == "W" else 14
    for z in zones:
        merged = False
        for i, o in enumerate(out):
            if o.kind != z.kind:
                continue
            if o.legout_start <= z.base_start and z.base_end <= o.legout_end and z.base_end < o.legout_end:
                merged = True              # a pause inside another zone's leg-out run: the origin zone is the level
                break
            same_base = o.base_end == z.base_end or o.legout_start == z.legout_start
            shared = z.base_start <= o.base_end and o.base_start <= z.base_end   # the bases share bars
            close = abs((z.legout_start - o.legout_start).days) <= near_days
            if same_base or shared or (close and _iou(o, z) >= 0.6):
                # one zone per consolidation: the wider base (Ben boxes the whole consolidation), then the stronger
                if (z.base_bars, z.swing_h, z.run_atr) > (o.base_bars, o.swing_h, o.run_atr):
                    out[i] = z
                merged = True
                break
        if not merged:
            out.append(z)
    return out


def _mark_originality(zones: List[Zone]) -> None:
    """A base that formed inside, or whose wick extreme sits at, an older zone of the same kind that was still
    unbroken at the time is a reaction, not an original move. Looks back three years on the weekly and eighteen
    months on the daily and two-day. Marked and scored down, never removed."""
    for z in zones:
        back = pd.Timedelta(days=3 * 365 if z.tf == "W" else 548)
        for o in zones:
            if o is z or o.kind != z.kind or o.legout_end >= z.base_start or z.base_start - o.legout_end > back:
                continue
            if o.broken is not None and o.broken < z.base_start:
                continue
            ov_lo, ov_hi = max(o.lo, z.lo), min(o.hi, z.hi)
            inside = ov_hi > ov_lo and (ov_hi - ov_lo) >= 0.5 * (z.hi - z.lo)
            # or the base's wick extreme sits in the older band: the move left from the older zone (a reaction)
            pad = 0.25 * o.height
            at_zone = (o.lo - pad) <= z.distal <= (o.hi + pad)
            if inside or at_zone:
                z.original = False
                z.origin_note = f"formed {'inside' if inside else 'at'} the {o.kind} zone from {pd.Timestamp(o.base_start):%b %Y}"
                z.origin_ref = pd.Timestamp(o.base_start).strftime("%Y-%m-%d")
                break


def score(z: Zone) -> float:
    """0 to 100, ordinal. Follows the five qualifiers; it ranks candidates and is not a probability."""
    s = 0.0
    s += 25 * min(z.swing_h, 6.0) / 6.0 + 10 * min(z.run_atr, 3.0) / 3.0          # departure (35)
    s += 20 if z.base_bars <= 3 else 15 if z.base_bars <= 6 else 9 if z.base_bars <= 9 else 4   # base duration (20)
    s += {"fresh": 25, "shallow": 19}.get(z.fresh_state, 10 if z.tests == 1 else 4 if z.tests == 2 else 0)  # freshness (25)
    br = z.body_ratio
    s += 6 if br is None else 10 if br >= 0.7 else 6 if br >= 0.5 else 3                 # leg-out candle (10)
    s += 10 if z.original else 2                                                         # originality (10): a reaction from an older zone is a weak origin
    if z.broken is not None:
        s *= 0.2
    return round(min(100.0, s), 1)


# ----------------------------------------------------------------------------------------------- context

def live(zones: List[Zone], price: float, atr_now: float, max_dist_atr: float = 8.0) -> List[Zone]:
    """Unbroken zones whose proximal edge is within ``max_dist_atr`` of price, or that price sits inside."""
    out = []
    for z in zones:
        if z.broken is not None:
            continue
        inside = z.lo <= price <= z.hi
        dist = (price - z.proximal) / atr_now if z.kind == "demand" else (z.proximal - price) / atr_now
        if not inside and dist < 0:
            continue                      # price is through the proximal edge: the zone is being traded through
        if inside or dist <= max_dist_atr:
            out.append(z)
    return out


def stack(zones: List[Zone], price: float) -> List[Zone]:
    """Skorupinski's level on top of level: unbroken zones of one kind whose gap is no taller than either zone are
    one structure, drawn as one band with each base boxed. A newer zone that formed inside an older zone of the
    same kind (overlap of at least 60%) is the level; the older one is kept as a candidate and named on the newer.
    Returns new Zone objects for merged structures; untouched zones are returned as they are."""
    out: List[Zone] = []
    for kind in ("demand", "supply"):
        zs = sorted([z for z in zones if z.kind == kind and z.broken is None], key=lambda z: z.lo)
        # overlap suppression first
        keep: List[Zone] = []
        for z in zs:
            dup = None
            for o in keep:
                if _iou(o, z) >= 0.6:
                    dup = o
                    break
            if dup is None:
                keep.append(z)
                continue
            newer, older = (z, dup) if z.base_start > dup.base_start else (dup, z)
            newer.inside_older = f"formed inside the {older.kind} zone from {pd.Timestamp(older.base_start):%b %Y} ({older.fresh_state})"
            keep[keep.index(dup)] = newer
        keep.sort(key=lambda z: z.lo)
        # stacking: walk outward from price. A neighbour joins when the gap is no taller than either zone, the
        # structure stays within 3.5 typical ranges and 2.5 times its tallest part, the bases formed within about a
        # year of each other on the weekly (200 days on the daily and two-day), and at most three bases.
        span_days = 400 if zs and zs[0].tf == "W" else 200
        ordered = sorted(keep, key=lambda z: abs(z.proximal - price))
        used: set = set()
        groups: List[List[Zone]] = []
        for z in ordered:
            if id(z) in used:
                continue
            g = [z]
            used.add(id(z))
            while len(g) < 3:
                lo, hi = min(x.lo for x in g), max(x.hi for x in g)
                best = None
                for o in keep:
                    if id(o) in used:
                        continue
                    gap = max(0.0, o.lo - hi, lo - o.hi)
                    if gap > min(o.height, min(x.height for x in g)):
                        continue
                    nlo, nhi = min(lo, o.lo), max(hi, o.hi)
                    if (nhi - nlo) > 3.5 * z.atr or (nhi - nlo) > 2.5 * max(x.height for x in g + [o]):
                        continue
                    if max(abs((pd.Timestamp(o.base_start) - pd.Timestamp(x.base_start)).days) for x in g) > span_days:
                        continue
                    if best is None or gap < best[0]:
                        best = (gap, o)
                if best is None:
                    break
                g.append(best[1])
                used.add(id(best[1]))
            g.sort(key=lambda x: x.lo)
            groups.append(g)
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue
            top, bot = g[-1], g[0]
            part_refs = {pd.Timestamp(z.base_start).strftime("%Y-%m-%d") for z in g}   # a reaction to another base of the same structure is not a reaction
            near = top if kind == "demand" else bot          # the component price reaches first
            far = bot if kind == "demand" else top
            m = Zone(tf=near.tf, kind=kind, pattern=near.pattern,
                     proximal=near.proximal, distal=far.distal, pref_proximal=near.pref_proximal,
                     base_start=min(z.base_start for z in g), base_end=max(z.base_end for z in g),
                     legout_start=near.legout_start, legout_end=near.legout_end,
                     base_bars=sum(z.base_bars for z in g), height=abs(near.proximal - far.distal),
                     height_atr=abs(near.proximal - far.distal) / near.atr if near.atr else 0.0, atr=near.atr,
                     run_atr=max(z.run_atr for z in g), run_bars=near.run_bars, swing_h=max(z.swing_h for z in g),
                     body_ratio=near.body_ratio, explosive=any(z.explosive for z in g), base_small=near.base_small,
                     legin_atr=near.legin_atr, tests=near.tests, pen_max=near.pen_max, last_test=near.last_test,
                     fresh_state=near.fresh_state, visiting_now=near.visiting_now, wide_touches=near.wide_touches,
                     deep_touches=near.deep_touches, original=all(z.original or z.origin_ref in part_refs for z in g),
                     origin_note=near.origin_note if not (near.original or near.origin_ref in part_refs) else "",
                     score=max(z.score for z in g), inside_older=near.inside_older)
            m.notes = [f"level on level: {len(g)} bases ({', '.join(pd.Timestamp(z.base_start).strftime('%b %y') for z in g)})"] + list(near.notes)
            m.parts = [z.to_dict() for z in g]
            out.append(m)
    return out


def select(zones: List[Zone], price: float, per_side: int = 2, typ: Optional[float] = None) -> Tuple[List[Zone], List[Zone]]:
    """Key levels for the chart and the remaining candidates.

    A zone price sits inside is always shown. Per side, the anchor is the inside zone of that kind if there is one,
    otherwise the nearest fresh (or shallow-tested) zone. One more zone joins when it sits close behind the anchor
    (fresh within four typical ranges, tested within two) or when it is tested and closer to price than the anchor.
    This keeps the chart to the one or two levels per side that matter, the way the zones are drawn by hand.
    """
    inside = [z for z in zones if z.lo <= price <= z.hi]
    typ = typ or (max(z.atr for z in zones) if zones else 1.0)
    dem = [z for z in zones if z.kind == "demand" and z not in inside and z.proximal <= price]
    sup = [z for z in zones if z.kind == "supply" and z not in inside and z.proximal >= price]

    def gap(a: Zone, b: Zone) -> float:
        return max(0.0, b.lo - a.hi, a.lo - b.hi)

    def pick(side: List[Zone], kind: str, key):
        side = sorted(side, key=key)
        anchors = [z for z in inside if z.kind == kind]
        chosen: List[Zone] = list(anchors)    # inside zones count towards the per-side limit
        anchor = anchors[0] if anchors else None
        if anchor is None:
            fresh = [z for z in side if z.fresh_state in ("fresh", "shallow")]
            if fresh:
                anchor = fresh[0]
                chosen.append(anchor)
            elif side:
                anchor = side[0]
                chosen.append(anchor)
        for z in side:
            if len(chosen) >= per_side or anchor is None:
                break
            if z in chosen:
                continue
            fresh_z = z.fresh_state in ("fresh", "shallow")
            near = gap(anchor, z) <= (4.0 if fresh_z else 2.0) * typ
            closer = (not fresh_z) and key(z) < key(anchor)
            if near or closer:
                chosen.append(z)
        return sorted(chosen, key=key)

    shown = pick(dem, "demand", lambda z: price - z.proximal) + pick(sup, "supply", lambda z: z.proximal - price)
    shown = [z for z in inside if z not in shown] + shown
    rest = [z for z in zones if z not in shown]
    return shown, rest


def level_on_level(zones: List[Zone]) -> Dict[int, int]:
    """Group index per zone (id -> group) for same-kind zones whose gap is no taller than either zone."""
    groups: Dict[int, int] = {}
    g = 0
    for kind in ("demand", "supply"):
        zs = sorted([z for z in zones if z.kind == kind], key=lambda z: z.lo)
        for a, b in zip(zs, zs[1:]):
            gap = max(0.0, b.lo - a.hi)
            if gap <= min(a.height, b.height):
                ga = groups.get(id(a))
                if ga is None:
                    g += 1
                    ga = g
                    groups[id(a)] = ga
                groups[id(b)] = ga
    return groups


def room(z: Zone, opposing: List[Zone], stop_frac: float = 0.25) -> Optional[Dict]:
    """Profit margin: distance from the proximal edge to the nearest opposing zone's proximal edge, in R
    (risk = zone height plus the stop buffer beyond the distal edge)."""
    risk = z.height * (1 + stop_frac)
    if risk <= 0:
        return None
    if z.kind == "demand":
        ahead = [o for o in opposing if o.kind == "supply" and o.proximal > z.proximal]
        if not ahead:
            return None
        o = min(ahead, key=lambda o: o.proximal)
        dist = o.proximal - z.proximal
    else:
        ahead = [o for o in opposing if o.kind == "demand" and o.proximal < z.proximal]
        if not ahead:
            return None
        o = max(ahead, key=lambda o: o.proximal)
        dist = z.proximal - o.proximal
    return {"to": float(o.proximal), "r": round(float(dist / risk), 2), "state": o.fresh_state, "tf": o.tf}


def pivots(df: pd.DataFrame, k: int = 4) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    """Swing highs and lows: a bar whose high (low) is the extreme of the k bars either side."""
    H, L = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    idx = df.index
    highs, lows = [], []
    for i in range(k, len(df) - k):
        if H[i] >= H[i - k:i].max() and H[i] >= H[i + 1:i + k + 1].max():
            highs.append((idx[i], float(H[i])))
        if L[i] <= L[i - k:i].min() and L[i] <= L[i + 1:i + k + 1].min():
            lows.append((idx[i], float(L[i])))
    return highs, lows


def direction(df: pd.DataFrame, k: int = 4) -> Dict:
    """Skorupinski's rule: two higher lows is an uptrend, two lower highs a downtrend, otherwise sideways.
    Uses the six most recent swings (three highs, three lows)."""
    highs, lows = pivots(df, k)
    hs, ls = [v for _, v in highs[-3:]], [v for _, v in lows[-3:]]
    up = len(ls) == 3 and ls[2] > ls[1] > ls[0]
    down = len(hs) == 3 and hs[2] < hs[1] < hs[0]
    if up and not down:
        lab = "up"
    elif down and not up:
        lab = "down"
    else:
        lab = "sideways"
    return {"label": lab, "highs": [[d.strftime("%Y-%m-%d"), round(v, 6)] for d, v in highs[-3:]],
            "lows": [[d.strftime("%Y-%m-%d"), round(v, 6)] for d, v in lows[-3:]],
            "rule": "two higher lows is up, two lower highs is down, from the last three swing highs and lows"}


def curve(price: float, weekly: List[Zone]) -> Dict:
    """Location between the nearest unbroken weekly demand below and supply above, split at 0.33 and 0.66."""
    dem = [z for z in weekly if z.kind == "demand" and z.broken is None and z.proximal <= price]
    sup = [z for z in weekly if z.kind == "supply" and z.broken is None and z.proximal >= price]
    inside_dem = [z for z in weekly if z.kind == "demand" and z.broken is None and z.lo <= price <= z.hi]
    inside_sup = [z for z in weekly if z.kind == "supply" and z.broken is None and z.lo <= price <= z.hi]
    out: Dict = {"demand": None, "supply": None, "frac": None, "label": "unknown"}
    if inside_dem:
        out["label"] = "very low"
        out["demand"] = inside_dem[0].to_dict()
    if inside_sup:
        out["label"] = "very high"
        out["supply"] = inside_sup[0].to_dict()
    if dem:
        d = max(dem, key=lambda z: z.proximal)
        out["demand"] = out["demand"] or d.to_dict()
    if sup:
        s = min(sup, key=lambda z: z.proximal)
        out["supply"] = out["supply"] or s.to_dict()
    if dem and sup and not (inside_dem or inside_sup):
        d, s = max(dem, key=lambda z: z.proximal), min(sup, key=lambda z: z.proximal)
        span = s.proximal - d.proximal
        f = (price - d.proximal) / span if span > 0 else 0.5
        out["frac"] = round(float(f), 3)
        out["label"] = "low" if f < 0.33 else "high" if f > 0.66 else "equilibrium"
    elif not (inside_dem or inside_sup):
        if dem and not sup:
            out["label"] = "no weekly supply above"
        elif sup and not dem:
            out["label"] = "no weekly demand below"
    return out


ACTION = {
    # (zone kind, location, direction) -> (allowed, tier, wording). Skorupinski's action matrix.
    ("demand", "low", "up"): (True, "best", "demand at a low location in an uptrend: the method's best case"),
    ("demand", "low", "sideways"): (True, "medium", "demand at a low location in a sideways market: allowed, medium case"),
    ("demand", "low", "down"): (True, "weak", "demand at a low location in a downtrend: allowed, the weakest case (counter-trend)"),
    ("demand", "equilibrium", "up"): (True, "medium", "demand at equilibrium with an uptrend: allowed but not ideal"),
    ("demand", "equilibrium", "sideways"): (False, "none", "demand at equilibrium in a sideways market: the method says stand aside"),
    ("demand", "equilibrium", "down"): (False, "none", "demand at equilibrium in a downtrend: no case in the method"),
    ("demand", "high", None): (False, "none", "demand at a high location: the method never buys here"),
    ("supply", "high", "down"): (True, "best", "supply at a high location in a downtrend: the method's best case"),
    ("supply", "high", "sideways"): (True, "medium", "supply at a high location in a sideways market: allowed, medium case"),
    ("supply", "high", "up"): (True, "weak", "supply at a high location in an uptrend: allowed, the weakest case (counter-trend)"),
    ("supply", "equilibrium", "down"): (True, "medium", "supply at equilibrium with a downtrend: allowed but not ideal"),
    ("supply", "equilibrium", "sideways"): (False, "none", "supply at equilibrium in a sideways market: the method says stand aside"),
    ("supply", "equilibrium", "up"): (False, "none", "supply at equilibrium in an uptrend: no case in the method"),
    ("supply", "low", None): (False, "none", "supply at a low location: the method never sells here"),
}


def action(kind: str, location: str, trend: str) -> Dict:
    loc = {"very low": "low", "very high": "high"}.get(location, location)
    if loc not in ("low", "equilibrium", "high"):
        return {"allowed": None, "tier": "unknown", "text": f"location on the weekly curve is {location}; the matrix needs a weekly zone on both sides"}
    key = (kind, loc, trend)
    if key not in ACTION:
        key = (kind, loc, None)
    allowed, tier, text = ACTION.get(key, (None, "unknown", "no rule"))
    return {"allowed": allowed, "tier": tier, "text": text}


def covered(z: Zone, weekly: List[Zone]) -> Optional[Zone]:
    """The weekly zone of the same kind this lower-timeframe zone sits in (Skorupinski: 'always trade covered').

    Covered means at least half of the lower-timeframe zone's height lies inside the weekly zone."""
    for w in weekly:
        if w.kind != z.kind or w.broken is not None:
            continue
        overlap = min(z.hi, w.hi) - max(z.lo, w.lo)
        if overlap >= 0.5 * max(1e-12, z.hi - z.lo):
            return w
    return None
