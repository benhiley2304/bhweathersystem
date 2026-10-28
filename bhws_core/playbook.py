"""Issue 11 playbook as a transparent state, not a signal.

"Extreme, level, turn, add" (BH Macro Issue 11, 26 Oct 2026):
  1 Crowd offside: fund managers at or beyond the 90th / 10th percentile of
    the trailing 156 weeks, against a view that fits the macro bias.
  2 Starter: half size at a weekly supply/demand zone, stop beyond the level
    by about 25% of the zone's depth.
  3 Add on the turn: managers moving against their extreme two reports
    running AND a weekly close through the 10-week average.
  4 No turn, no trade: no add within six weeks and the starter goes.

What this module does NOT do: it never draws a zone. Zones are hand-drawn and
Issue 11 states they were not tested. If no zone is supplied the state stops
at "Wait for the zone" and no stop is produced. That follows the brief:
do not manufacture exact levels the data cannot support.

The statuses reuse the Issue 11 "Who is turning" vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

from .percentile import midrank_series, EXTREME_HI, EXTREME_LO

TURN_REPORTS = 2
SMA_WEEKS = 10
EXPIRY_REPORTS = 6
REARM_REPORTS = 13
STOP_BUFFER_FRACTION = 0.25


@dataclass
class Zone:
    """A hand-drawn weekly zone. proximal is the edge price reaches first."""
    proximal: float
    distal: float
    source: str = "manual"   # always manual; kept explicit for the UI label

    @property
    def depth(self) -> float:
        return abs(self.proximal - self.distal)

    def contains(self, price: float) -> bool:
        lo, hi = sorted((self.proximal, self.distal))
        return lo <= price <= hi


def _sma(xs: Sequence[float], n: int) -> Optional[float]:
    if len(xs) < n:
        return None
    return sum(xs[-n:]) / n


def find_events(pcts: Sequence[Optional[float]], hi: float = EXTREME_HI,
                lo: float = EXTREME_LO, rearm: int = REARM_REPORTS) -> List[Tuple[int, int]]:
    """(index, fade_direction) for each first crossing into an extreme.

    fade_direction is +1 (fade a crowded short = long) or -1 (fade a crowded
    long = short). One event per side per `rearm` reports, as in the Issue 11
    test design.
    """
    events, last = [], {1: -10 ** 9, -1: -10 ** 9}
    prev = None
    for i, p in enumerate(pcts):
        if p is None:
            prev = p
            continue
        side = -1 if p >= hi else (1 if p <= lo else 0)
        prev_side = 0 if prev is None else (-1 if prev >= hi else (1 if prev <= lo else 0))
        if side != 0 and side != prev_side and i - last[side] >= rearm:
            events.append((i, side))
            last[side] = i
        prev = p
    return events


def evaluate(obs_dates: Sequence[date], manager_net: Sequence[float],
             weekly_close: Sequence[float], *, zone: Optional[Zone] = None,
             macro_bias: Optional[int] = None, book_side: Optional[int] = None,
             starter_date: Optional[date] = None) -> dict:
    """Playbook state for the latest report.

    obs_dates / manager_net: COT reports, oldest first (non-commercial net,
      futures-only to match the TradingView indicator).
    weekly_close: weekly closes aligned to the same weeks, each taken AFTER
      that report's release (the caller aligns; see cot_calendar).
    macro_bias: +1 / -1 / 0 / None (None = not assessed).
    book_side: +1 long / -1 short / None if not on the book.
    starter_date: observation date of the first report after the starter went
      on. Issue 11 says "no add within six weeks: the starter goes", so the
      live clock runs from the starter. The Issue 11 test (and the BHWS
      point-in-time study) ran it from the extreme. Without a starter date
      BHWS cannot know the live clock, so it reports "Past test window"
      rather than calling the idea expired.
    """
    n = len(manager_net)
    if n == 0 or len(weekly_close) != n or len(obs_dates) != n:
        return {"status": "Insufficient data", "reason": "series missing or misaligned"}
    pcts = midrank_series(manager_net)
    cur = pcts[-1]
    if cur is None:
        return {"status": "Insufficient data", "reason": "fewer than 20 reports"}

    events = find_events(pcts)
    out = {
        "as_of_report": obs_dates[-1].isoformat(),
        "managers_pct": round(cur, 1),
        "managers_net": manager_net[-1],
        "rules": "Issue 11: extreme 90/10 of 156w; add = 2 reports turning + weekly close through 10w avg; expiry 6 reports",
    }
    sma = _sma(list(weekly_close), SMA_WEEKS)
    px = weekly_close[-1]
    out["price"] = px
    out["sma10"] = None if sma is None else round(sma, 6)

    live = [e for e in events if n - 1 - e[0] <= EXPIRY_REPORTS + 6]
    if not live:
        if cur >= EXTREME_HI or cur <= EXTREME_LO:
            f = -1 if cur >= EXTREME_HI else 1
            t = 0
            for k in range(n - 1, 0, -1):
                if (manager_net[k] - manager_net[k - 1]) * f > 0:
                    t += 1
                else:
                    break
            out["managers_turning_reports"] = t
            out.update(status="Extreme persisting", direction=f,
                       explain=("Managers are still at an extreme, but the first crossing is more than "
                                "12 reports old, so the playbook clock has run out. A crowd that stays "
                                "extreme this long and is right is a trend."))
        else:
            out.update(status="No extreme", direction=0,
                       explain="Managers are inside their 156-week range; nothing for the playbook.")
        return out
    idx, fade = live[-1]
    age = n - 1 - idx
    out["direction"] = fade
    out["fade_side"] = "long" if fade > 0 else "short"
    out["extreme_report"] = obs_dates[idx].isoformat()
    out["reports_since_extreme"] = age

    # Managers moving against their extreme: covering shorts (net up) when the
    # fade is long; selling longs (net down) when the fade is short.
    turning = 0
    for k in range(n - 1, max(idx, 0), -1):
        if (manager_net[k] - manager_net[k - 1]) * fade > 0:
            turning += 1
        else:
            break
    out["managers_turning_reports"] = turning
    through = None if sma is None else ((px - sma) * fade > 0)
    out["close_through_10w"] = through

    if macro_bias is None:
        out["macro_fit"] = "not assessed"
    else:
        out["macro_fit"] = "fits" if macro_bias == fade else ("against" if macro_bias == -fade else "neutral")

    if book_side is not None and book_side == -fade:
        out.update(status="With the crowd",
                   explain="The book is on the crowded side; positioning gives no reason to hold on.")
        return out

    add_ok = turning >= TURN_REPORTS and through is True
    if add_ok:
        status = "Add eligible" if book_side == fade else "Turned"
        explain = "Managers have turned two reports running and price closed through the 10-week average."
    elif starter_date is not None and sum(1 for d in obs_dates if d > starter_date) > EXPIRY_REPORTS:
        status = "Expired"
        explain = "No add within six reports of the starter. Playbook says the starter goes."
    elif starter_date is None and age > EXPIRY_REPORTS and EXTREME_LO < cur < EXTREME_HI:
        status = "No extreme"
        explain = (f"Managers have left the extreme first reached {age} reports ago "
                   f"and are now at the {round(cur)}th percentile.")
    elif starter_date is None and age > EXPIRY_REPORTS:
        status = "Past test window"
        explain = (f"First crossing was {age} reports ago, beyond the six the Issue 11 test allowed. "
                   "The live clock runs from the starter, which BHWS does not know. "
                   f"Managers turning {turning} report(s).")
    elif turning == 1 or (turning >= 2 and through is not True):
        status = "One more week"
        explain = (f"Managers turning {turning} report(s); price "
                   + ("has" if through else "has not") + " closed through the 10-week average.")
    else:
        status = "Not turned"
        explain = "Extreme in place, managers not yet moving against it."

    # Starter eligibility needs a hand-drawn zone. Without one, no level.
    if zone is None:
        out["starter"] = {"state": "Wait for the zone", "stop": None,
                          "note": "No zone supplied. BHWS does not draw zones."}
    else:
        stop = zone.distal - fade * STOP_BUFFER_FRACTION * zone.depth
        at_zone = zone.contains(px)
        out["starter"] = {
            "state": "Starter ready" if at_zone and age <= EXPIRY_REPORTS else "Wait for the zone",
            "zone": {"proximal": zone.proximal, "distal": zone.distal, "source": zone.source},
            "stop": round(stop, 6), "stop_rule": "beyond distal by 25% of zone depth",
        }
    out["clock"] = "from starter" if starter_date is not None else "from first crossing (test design)"
    if book_side == fade and status not in ("Add eligible", "Expired"):
        status = "Hold; no add"
    out["status"] = status
    out["explain"] = explain
    return out
